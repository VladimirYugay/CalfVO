#!/usr/bin/env python3
""" This script replays one predicted trajectory next to its video. """
import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from torchcodec.decoders import VideoDecoder

TRAJECTORY_COLOR = [0, 158, 115]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory_dir", required=True,
                   help="Directory holding pred_trajectory*.npy and frame_ids*.npy.")
    p.add_argument("--video", required=True, help="The video those poses came from.")
    p.add_argument("--output_file", required=True,
                   help="The .rrd file to write.")

    p.add_argument("--stride", type=int, default=1,
                   help="Log every Nth video frame.")
    p.add_argument("--jpeg_quality", type=int, default=90,
                   help="JPEG quality for the logged frames. Pass 0 to store raw.")
    p.add_argument("--line_width", type=float, default=2.0,
                   help="Trajectory thickness in UI points.")
    p.add_argument("--frustum_frac", type=float, default=0.03,
                   help="Frustum size as a fraction of the trajectory's extent.")
    return p.parse_args()


def set_frame(index):
    """ Sets the frame index on the timeline, under either spelling of the call.
    Args:
        index: The frame number to set.
    """
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=index)
    else:
        rr.set_time_sequence("frame", index)


def find_one(directory, prefix):
    """ Finds the newest file in a directory whose name starts with a prefix.
        Evaluation runs tag their files with the training step, so sorting by name
        picks the latest.
    Args:
        directory: Where to look.
        prefix: The file name stem to match, such as "pred_trajectory".
    Returns:
        The path, or None if nothing matches.
    """
    matches = sorted(directory.glob(f"{prefix}*.npy"))
    return matches[-1] if matches else None


def load_trajectory(directory):
    """ Reads one predicted trajectory and the frame ids that go with it.
    Args:
        directory: The directory a run wrote the sequence to.
    Returns:
        The trajectory and the frame ids, trimmed to the same length.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise SystemExit(f"{directory} does not exist")

    pred_path = find_one(directory, "pred_trajectory")
    frame_ids_path = find_one(directory, "frame_ids")
    if pred_path is None or frame_ids_path is None:
        raise SystemExit(
            f"{directory} needs a pred_trajectory*.npy and a frame_ids*.npy")

    pred = np.load(pred_path)
    frame_ids = np.load(frame_ids_path)
    n = min(len(pred), len(frame_ids))
    return pred[:n], frame_ids[:n]


def main():
    args = get_args()
    pred, frame_ids = load_trajectory(args.trajectory_dir)
    decoder = VideoDecoder(Path(args.video))
    name = Path(args.video).stem

    pred_xyz = pred[:, :3, 3].astype(np.float32)

    # Sized from the whole trajectory, so the frustum reads the same on a room and
    # on a drive.
    diag = float(np.linalg.norm(pred_xyz.max(0) - pred_xyz.min(0))) or 1.0
    frustum = diag * args.frustum_frac

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="/world/camera/image", name="input frames"),
            rrb.Spatial3DView(origin="/world", name="trajectory"),
            column_shares=[1, 1],
        ),
        collapse_panels=True,
    )
    rr.init(f"calfvo/{name}", spawn=False, default_blueprint=blueprint)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_path))

    # The poses are camera-to-world in OpenCV axes, x right, y down, z forward.
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    first = decoder.get_frames_at(indices=[int(frame_ids[0])]).data
    height, width = int(first.shape[2]), int(first.shape[3])
    focal = 0.9 * width

    logged = 0
    for i in range(len(pred)):
        set_frame(i)

        # The trajectory grows as it is traced out. A negative radius is in UI
        # points, which hold their size under zoom.
        rr.log("/world/pred",
               rr.LineStrips3D([pred_xyz[:i + 1]], colors=[TRAJECTORY_COLOR],
                               radii=-args.line_width))

        rr.log("/world/camera", rr.Transform3D(
            translation=pred[i, :3, 3].astype(np.float32),
            mat3x3=pred[i, :3, :3].astype(np.float32)))
        rr.log("/world/camera", rr.Pinhole(
            focal_length=focal, width=width, height=height,
            image_plane_distance=frustum))

        if i % args.stride == 0:
            frame = decoder.get_frames_at(indices=[int(frame_ids[i])]).data
            img = frame[0].permute(1, 2, 0).numpy()
            image = rr.Image(img)
            if args.jpeg_quality:
                image = image.compress(jpeg_quality=args.jpeg_quality)
            rr.log("/world/camera/image", image)
            logged += 1

    rr.rerun_shutdown()
    size_mb = out_path.stat().st_size / 1e6
    print(f"{name}: {len(pred)} poses, {logged} frames -> {out_path} "
          f"({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
