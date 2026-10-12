"""KITTI odometry to one video and pose file per sequence.

Expects the odometry benchmark as it unpacks, with images under
<input>/sequences/{00..21}/image_2 and ground truth under <input>/poses. Only the
sequences that have a pose file are processed, so 00 to 10. They stay at native
resolution, since both transforms take the same shortest edge square crop.

    python scripts/data/process_kitti.py /raw/kitti_odometry /path/to/datasets/KittiVideos
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from prep_utils import encode_video


def create_video_from_images(image_paths, output_video, fps=30):
    """ Encodes one sequence's frames into a video.
    Args:
        image_paths: The frames in order.
        output_video: Where to write the mp4.
        fps: Frame rate to declare on the stream.
    """
    encode_video((Image.open(p).convert('RGB') for p in image_paths),
                 output_video, fps=fps)


def parse_poses(pose_path):
    """ Reads a KITTI odometry pose file.
        Rows are a row-major 3x4 camera-to-world matrix, one per frame.
    Args:
        pose_path: The sequence's pose file.
    Returns:
        Camera-to-world poses of shape (N, 4, 4).
    """
    poses = []
    with open(pose_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :4] = np.array(values).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)


def main(input_path, output_path):
    """ Converts every sequence that has ground truth.
        Sequences 11 to 21 ship no poses, so they are skipped. Files that already
        exist are left alone, so an interrupted run can be rerun.
    Args:
        input_path: The unpacked odometry benchmark root.
        output_path: Where the processed sequences go.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    image_folders = sorted((input_path / "sequences").glob("*"))
    pose_paths = sorted((input_path / "poses").glob("*"))

    # Sequences 11-21 have no ground truth; zip stops at the shorter list.
    image_folders = image_folders[:len(pose_paths)]

    for image_folder, pose_path in zip(image_folders, pose_paths):
        scene_name = image_folder.parts[-1]
        print(f"Processing sequence {scene_name}")

        scene_output_path = output_path / scene_name
        scene_output_path.mkdir(parents=True, exist_ok=True)

        video_file = scene_output_path / "video.mp4"
        if not video_file.exists():
            image_paths = sorted((image_folder / "image_2").glob("*"))
            create_video_from_images(image_paths, str(video_file), fps=30)

        pose_file = scene_output_path / "poses.npz"
        if not pose_file.exists():
            np.savez(pose_file, poses=parse_poses(pose_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('input_path', type=str,
                        help="KITTI odometry root (contains sequences/ and poses/).")
    parser.add_argument('output_path', type=str,
                        help="Directory to write processed sequences into.")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
