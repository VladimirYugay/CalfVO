""" This module reads the ScanNet corpus. """
import numpy as np
from torchcodec.decoders import VideoDecoder

from src.datasets.base_dataset import BaseDataset
from src.utils.data_utils import create_overlapping_windows


class ScanNetMatrixDataset(BaseDataset):
    """ ScanNet, written by scripts/data/process_scannet.py.
        Its poses jump where tracking was lost, so windows holding a frame to frame
        translation above MAX_FRAME_TRANSLATION metres are dropped, both from the
        fixed grid and from the stream.
    """

    # A handheld scan can't move 1.5 m between frames, so a bigger step is a tracking
    # failure in the ground truth. These are finite, so the inf truncation misses them.
    MAX_FRAME_TRANSLATION = 1.5

    def _read_poses(self):
        """ Reads every scene's ground truth poses into self.scenes.
            Each sequence is truncated at its first inf pose, which is how ScanNet
            marks lost tracking. The poses are then made relative to its first frame.
        """
        self.scenes = {}
        for scene_name in self.scene_names:
            video_reader = VideoDecoder(
                self.dataset_path / scene_name / "video.mp4")
            c2ws = np.load(
                self.dataset_path / scene_name / "poses.npz")["poses"]

            possible_number_of_frames = [len(video_reader), c2ws.shape[0]]

            # ScanNet writes inf poses where tracking was lost so keep only the part before.
            inf_ids = np.where(np.isinf(c2ws).any(axis=(1, 2)))[0]
            if inf_ids.size > 0:
                possible_number_of_frames.append(inf_ids.min())

            number_of_frames = min(possible_number_of_frames)

            frame_ids = np.arange(0, number_of_frames, self.stride)

            selected_c2ws = c2ws[frame_ids].copy()
            selected_c2ws = np.linalg.inv(selected_c2ws[0]) @ selected_c2ws

            self.scenes[scene_name] = {
                "frame_ids": frame_ids,
                "c2ws": selected_c2ws
            }

    def _has_jump(self, c2ws: np.ndarray) -> bool:
        """ Checks a window for a frame to frame step above MAX_FRAME_TRANSLATION.
        Args:
            c2ws: The window's camera-to-world poses.
        Returns:
            True if the window holds a tracking failure.
        """
        steps = np.linalg.inv(c2ws[:-1]) @ c2ws[1:]
        return np.abs(steps[:, :3, 3]).max() > self.MAX_FRAME_TRANSLATION

    def _create_samples(self):
        """ Builds the window list, skipping any window with an implausible jump. """
        self.samples = []
        for scene_name in self.scenes:
            windows = create_overlapping_windows(
                np.arange(len(self.scenes[scene_name]["frame_ids"])),
                window_size=self.window_size,
                overlap=self.overlap
            )
            for window in windows:
                if self._has_jump(self.scenes[scene_name]["c2ws"][window]):
                    continue

                self.samples.append({
                    "scene_name": scene_name,
                    "frame_ids": self.scenes[scene_name]["frame_ids"][window],
                    "c2ws": self.scenes[scene_name]["c2ws"][window]
                })

    def get_sample(self, idx):
        """ Returns the sample to serve for an index.
            In stream mode a few sequences keep implausible jumps after the inf
            truncation, so the draw is repeated rather than serve a tracking failure.
            Bounded so a pathological sequence cannot spin here forever.
        """
        sample = super().get_sample(idx)
        if self.virtual_length is None:
            return sample
        for _ in range(7):
            if not self._has_jump(sample["c2ws"]):
                break
            sample = super().get_sample(idx)
        return sample
