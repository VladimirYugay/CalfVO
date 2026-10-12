""" This module reads windows of frames and poses from the prepared corpora. """
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchcodec.decoders import VideoDecoder

from src.utils.data_utils import (create_overlapping_windows,
                                  create_strided_windows, sample_random_window)


class BaseDataset(torch.utils.data.Dataset):
    """ Decodes windows of frames and pairs them with normalised relative poses.
        Every corpus is stored as a video.mp4 plus a camera-to-world poses.npz where
        frame i of one matches pose i of the other, so they all read the same way.
        It serves either a fixed grid of windows or an endless stream of random ones,
        depending on whether virtual_length is set.
    """

    # Max open VideoDecoder objects per worker. Each H.264 decoder holds its Decoded
    # Picture Buffer in memory, about 2 MB for 640x480 and 12 MB for 1920x1440. At 50
    # decoders of 12 MB across 64 workers that is roughly 38 GB per node.
    _DECODER_CACHE_MAXSIZE = 50

    def __init__(self,
                 dataset_path: str,
                 scene_names: list,
                 window_size: int,
                 overlap: int,
                 transforms,
                 pose_stats: dict,
                 stride: int,
                 virtual_length: int = None,
                 scene_sampling: str = "proportional"):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.scene_names = scene_names
        self.window_size = window_size
        self.transform = transforms

        # An int, or a [lo, hi] range each window draws its spacing from.
        if isinstance(stride, (list, tuple)):
            self.stride_range = (int(stride[0]), int(stride[1]))
        else:
            self.stride_range = (int(stride), int(stride))

        # Streaming reads every frame, so the drawn spacing doesn't compound with stride.
        self.anchor_stride = (self.stride_range[0] + self.stride_range[1]) // 2
        streaming = virtual_length is not None
        stride = 1 if streaming else self.stride_range[0]

        # Only the translation is normalised
        self.trans_mean = pose_stats['trans_mean']
        self.trans_std = pose_stats['trans_std']

        self.stride = stride
        self._sample_rng = None

        # When set, the dataset is an endless stream of random windows of this length.
        self.virtual_length = virtual_length
        self.scene_sampling = scene_sampling
        self.overlap = overlap
        self._decoder_cache = OrderedDict()
        self._read_poses()
        self._create_samples()
        self._prepare_scene_sampling()

    def _read_poses(self):
        """ Reads every scene's ground truth poses into self.scenes.
            The poses are made relative to each sequence's first frame.
        """
        self.scenes = {}
        for scene_name in self.scene_names:
            video_reader = VideoDecoder(
                self.dataset_path / scene_name / "video.mp4")
            c2ws = np.load(
                self.dataset_path / scene_name / "poses.npz")["poses"]

            possible_number_of_frames = [len(video_reader), c2ws.shape[0]]

            number_of_frames = min(possible_number_of_frames)

            frame_ids = np.arange(0, number_of_frames, self.stride)

            selected_c2ws = c2ws[frame_ids].copy()
            selected_c2ws = np.linalg.inv(selected_c2ws[0]) @ selected_c2ws

            self.scenes[scene_name] = {
                "frame_ids": frame_ids,
                "c2ws": selected_c2ws
            }

    def _create_samples(self):
        self.samples = []
        for scene_name in self.scenes:
            num_frames = len(self.scenes[scene_name]["frame_ids"])
            # The anchor grid always has a raw spacing of anchor_stride. When frames
            # were read at full rate that spacing has to be applied here. When they
            # were read at `stride` the index grid already carries it.
            if self.stride == 1 and self.anchor_stride > 1:
                windows = create_strided_windows(
                    num_frames, window_size=self.window_size,
                    overlap=self.overlap, stride=self.anchor_stride)
            else:
                windows = create_overlapping_windows(
                    np.arange(num_frames),
                    window_size=self.window_size,
                    overlap=self.overlap,
                )

            for window in windows:
                self.samples.append({
                    "scene_name": scene_name,
                    "frame_ids": self.scenes[scene_name]["frame_ids"][window],
                    "c2ws": self.scenes[scene_name]["c2ws"][window]
                })

    def _prepare_scene_sampling(self):
        """ Lists the sequences eligible for the endless stream and weights them.
            A sequence needs at least window_size + 1 frames to yield any window. The
            weights are proportional to length and stored as a cumulative sum.
        """
        lengths = {s: len(self.scenes[s]["frame_ids"]) for s in self.scenes}
        self.sampling_scene_names = [
            s for s, n in lengths.items() if n >= self.window_size + 1]
        weights = np.array(
            [lengths[s] for s in self.sampling_scene_names], dtype=np.float64)
        self.scene_cumweights = (
            np.cumsum(weights / weights.sum()) if weights.size else weights)

    def _rng(self):
        """ Returns this worker's random generator for the stream.
            It is seeded from torch's worker seed, so every worker and every rank
            draws a different stream. Workers are persistent, so the stream continues
            rather than restarting on each pass.
        Returns:
            The numpy generator, created on first use.
        """
        if self._sample_rng is None:
            self._sample_rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        return self._sample_rng

    def _draw_scene(self):
        """ Draws one sequence for the endless stream.
            "proportional" weights each sequence by its length, so every frame in the
            dataset is equally likely, which is the distribution the fixed grid also
            produces. "uniform" weights sequences equally and over-weights short ones.
        Returns:
            The drawn scene name.
        """
        names = self.sampling_scene_names
        if self.scene_sampling == "uniform":
            return names[int(self._rng().integers(0, len(names)))]
        # searchsorted on cumulative weights: O(log n), unlike rng.choice(p=...)
        # which rebuilds its alias table on every call.
        return names[int(np.searchsorted(
            self.scene_cumweights, self._rng().random(), side="right"))]

    def get_sample(self, idx):
        """ Returns the sample to serve for an index.
            In stream mode the index is ignored, and a sequence is drawn at random
            followed by a random window inside it. Otherwise this is the precomputed
            window.
        Args:
            idx: Index into the sample list, unused in stream mode.
        Returns:
            A dict with the scene name, the frame ids and the poses.
        """
        if self.virtual_length is None:
            return self.samples[idx]

        scene_name = self._draw_scene()
        scene = self.scenes[scene_name]
        window = sample_random_window(
            len(scene["frame_ids"]), self.window_size, self.stride_range, self._rng())
        return {"scene_name": scene_name, "frame_ids": scene["frame_ids"][window],
                "c2ws": scene["c2ws"][window]}

    def _get_decoder(self, scene_name: str) -> VideoDecoder:
        if scene_name in self._decoder_cache:
            self._decoder_cache.move_to_end(scene_name)
            return self._decoder_cache[scene_name]
        decoder = VideoDecoder(self.dataset_path / scene_name / "video.mp4")
        self._decoder_cache[scene_name] = decoder
        if len(self._decoder_cache) > self._DECODER_CACHE_MAXSIZE:
            self._decoder_cache.popitem(last=False)  # evict least-recently-used
        return decoder

    def apply_transform(self, images):
        """ Transforms one window's frames.
            A window level transform gets the whole list in one call, so it can draw
            its augmentation parameters once for the window. Per-frame transforms are
            mapped over the frames instead.
        Args:
            images: The window's frames as PIL images.
        Returns:
            The transformed frames as a list of tensors.
        """
        if getattr(self.transform, "per_window", False):
            return self.transform(images)
        return [self.transform(img) for img in images]

    def get_scene_c2ws(self, scene_name):
        """ Returns one scene's poses and the frame ids they belong to.
        Args:
            scene_name: The scene to look up.
        Returns:
            Its camera-to-world poses and its frame ids.
        """
        return self.scenes[scene_name]["c2ws"], self.scenes[scene_name]["frame_ids"]

    def preprocess_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """ Normalises the translation part of ground truth poses.
        Args:
            poses: Ground truth poses of shape (M, 4, 4) with metric translations.
        Returns:
            The same poses with normalised translations.
        """
        final_poses = poses.clone()
        t = poses[:, :3, 3]
        t = (t - self.trans_mean) / self.trans_std
        final_poses[:, :3, 3] = t
        return final_poses

    def postprocess_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """ Denormalises predicted poses back to metric translations.
        Args:
            poses: Predicted poses of any shape flattenable to (M, 4, 4).
        Returns:
            The same poses with metric translations, flattened to (M, 4, 4).
        """
        poses = poses.flatten(0, 1)
        M = poses.shape[0]
        device, dtype = poses.device, poses.dtype
        final_poses = torch.eye(
            4, device=device, dtype=dtype).unsqueeze(0).repeat(M, 1, 1)
        final_poses[:, :3, :3] = poses[:, :3, :3]
        t = poses[:, :3, 3]
        t = (t * self.trans_std.to(device, dtype=dtype)
             + self.trans_mean.to(device, dtype=dtype))
        final_poses[:, :3, 3] = t
        return final_poses

    def __len__(self):
        # In stream mode the length is how many draws the loop wants before it stops
        # to validate. It says nothing about how much data exists.
        if self.virtual_length is not None:
            return self.virtual_length
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        c2ws = sample["c2ws"]
        frame_ids, scene_name = sample["frame_ids"], sample["scene_name"]

        frames = self._get_decoder(scene_name).get_frames_at(indices=frame_ids)
        images = frames.data.permute(0, 2, 3, 1).numpy()

        images_transformed = [Image.fromarray(img, mode="RGB") for img in images]
        images_transformed = self.apply_transform(images_transformed)
        # (C, N, H, W)
        images_transformed = torch.stack(
            images_transformed, dim=0).permute(1, 0, 2, 3)

        rel_poses = np.linalg.inv(c2ws[:-1]) @ c2ws[1:]
        final_poses = self.preprocess_poses(torch.from_numpy(rel_poses))

        return images_transformed, final_poses, torch.from_numpy(frame_ids)
