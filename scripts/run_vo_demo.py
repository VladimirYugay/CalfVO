""" This script runs the model over one video and writes out its trajectory. """
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import BaseDataset  # noqa: E402
from src.model import get_model  # noqa: E402
from src.utils import io_utils, training_utils  # noqa: E402


class SingleVideoDataset(BaseDataset):
    """ One video file, windowed and decoded as a prepared sequence would be.
        The poses are identity placeholders, so the gt the loader returns carries
        no information and nothing here scores against it.
    """

    def __init__(self, video_path, max_frames=-1, **kwargs):
        self.video_path = Path(video_path)
        self.max_frames = max_frames
        super().__init__(dataset_path=self.video_path.parent,
                         scene_names=[self.video_path.stem], **kwargs)

    def _read_poses(self):
        decoder = VideoDecoder(self.video_path)
        num_frames = len(decoder)
        if self.max_frames > 0:
            num_frames = min(num_frames, self.max_frames)
        frame_ids = np.arange(0, num_frames, self.stride)
        self.scenes = {self.scene_names[0]: {
            "frame_ids": frame_ids,
            "c2ws": np.eye(4)[None].repeat(len(frame_ids), axis=0),
        }}

    def _get_decoder(self, scene_name):
        if scene_name not in self._decoder_cache:
            self._decoder_cache[scene_name] = VideoDecoder(self.video_path)
        return self._decoder_cache[scene_name]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--video", required=True, help="Any video file.")
    p.add_argument("--output_dir", required=True,
                   help="Directory for the trajectory this writes.")
    p.add_argument("--checkpoint", default=None,
                   help="A local checkpoint. Defaults to the released one "
                        "for this config's backbone, fetched from the Hub.")
    p.add_argument("--stride", type=int, default=1,
                   help="Subsample the video. Raise it for a clip that moves faster "
                        "than the roughly 30 fps footage the model was trained on.")
    p.add_argument("--overlap", type=int, default=5)
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def main():
    args = get_args()
    config = io_utils.read_yaml_file(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = training_utils.get_transforms(
        config["model"]["img_size"], train=False,
        norm_mean=config["model"].get("norm_mean"),
        norm_std=config["model"].get("norm_std"))

    name = Path(args.video).stem
    dataset = SingleVideoDataset(
        args.video, window_size=config["num_views"], overlap=args.overlap,
        transforms=transforms, max_frames=args.max_frames,
        pose_stats={k: torch.tensor(v) for k, v in config["pose_stats"].items()},
        stride=args.stride, virtual_length=None)
    print(f"{name}: {len(dataset)} windows")
    if len(dataset) == 0:
        raise SystemExit(
            f"no windows: the video needs at least {config['num_views']} frames "
            f"at stride {args.stride}")

    config["model"]["num_views"] = config["num_views"]
    model = get_model(config["model"]).to(device)
    checkpoint = io_utils.resolve_checkpoint(
        args.checkpoint, config["model"].get("backbone", "crocov2_dust3r"))
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    # A release checkpoint carries the whole model, so this loads strictly.
    state_dict = {k.replace("_orig_mod.", "", 1): v
                  for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded step {ckpt.get('step')} from {checkpoint}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)

    pred_poses, acc_frame_ids, acc_confidences = [], [], []
    for i, (images, _, frame_ids) in enumerate(loader):
        images = images.to(device)
        with torch.no_grad():
            output = model(images.float())
        pred_poses.append(output["relative_poses"])
        acc_confidences.append(output["confidences"])
        acc_frame_ids.append(frame_ids)
        if i % 10 == 0:
            print(f"  batch {i}/{len(loader)}", flush=True)

    pred_poses = torch.cat(pred_poses, dim=0).detach()
    acc_frame_ids = torch.vstack(acc_frame_ids)
    acc_confidences = torch.vstack(acc_confidences)
    pred_poses_unnorm = dataset.postprocess_poses(pred_poses)

    acc_confidences = acc_confidences.reshape(pred_poses_unnorm.shape[0], -1).cpu().numpy()
    acc_confidences = np.exp(-acc_confidences)

    pred_trajectory = training_utils.recover_trajectory(
        pred_poses_unnorm.cpu().numpy(), acc_frame_ids.cpu().numpy(), acc_confidences)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_trajectory.npy", pred_trajectory)
    np.save(out / "confidence.npy", acc_confidences)
    # The scene's own frame_ids, not acc_frame_ids reshaped - that's every window's
    # ids concatenated batch-major, which only matches pred_trajectory's rows for
    # the first window.
    frame_ids = dataset.scenes[dataset.scene_names[0]]["frame_ids"][:pred_trajectory.shape[0]]
    np.save(out / "frame_ids.npy", frame_ids)

    print(f"wrote {pred_trajectory.shape[0]} poses to {out}")
    print("Visualise it with:")
    print(f"  python scripts/visualize_trajectory.py \\")
    print(f"      --trajectory_dir {out} \\")
    print(f"      --video {args.video} --output_file out/{name}.rrd")


if __name__ == "__main__":
    main()
