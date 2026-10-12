#!/usr/bin/env python3
"""TartanAir V1 to one video, pose file and intrinsics per trajectory.

Pulls one environment and difficulty zip from HuggingFace at a time and deletes it
before moving on, so peak disk is one archive rather than the whole corpus.

    python process_tartanv1.py output_path --scenes ocean soulcity --difficulties Hard
    python process_tartanv1.py output_path --list-scenes
"""

import argparse
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from prep_utils import (TRAINING_READY_SIZE, encode_video, is_sequence_valid,
                        square_crop_resize, square_intrinsics)


HF_DATASET_ID = "theairlabcmu/tartanair"
DIFFICULTY_LEVELS = ["Easy", "Hard"]
# image_left.zip also carries pose_left.txt, so it is the only archive needed.
ARCHIVE = "image_left.zip"
POSE_FILE = "pose_left.txt"

# 640x480 pinhole: fx=fy=320, cx=320, cy=240
INTRINSICS = np.array(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

# NED to CV cyclic permutation. V1 and V2 share the pose convention.
NED_PERM = [1, 2, 0, 4, 5, 3, 6]


def discover_scenes() -> list[str]:
    """ Lists the environments the HuggingFace dataset holds.
    Returns:
        The environment names, sorted.
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=HF_DATASET_ID, repo_type="dataset")
    scenes: set[str] = set()
    for f in files:
        if f.startswith(".") or "/" not in f:
            continue
        scene = f.split("/")[0]
        if scene not in ("README.md", ".gitattributes") and not scene.startswith("."):
            scenes.add(scene)
    return sorted(scenes)


def create_video_from_zip_images(
    zf: zipfile.ZipFile,
    image_files: list[str],
    output_path: Path,
    fps: int = 30,
    square_size: int | None = None,
) -> None:
    """ Encodes one trajectory's frames straight out of the archive.
        The images are decoded from the zip in memory, so nothing is extracted to
        disk.
    Args:
        zf: The opened archive.
        image_files: Names of the frames inside it, in order.
        output_path: Where to write the mp4.
        fps: Frame rate to declare on the stream.
        square_size: Square edge to crop and resize to, or None for native size.
    """
    def frames():
        for fname in image_files:
            with Image.open(io.BytesIO(zf.read(fname))) as img:
                img = img.convert("RGB")
                yield square_crop_resize(img, square_size) if square_size else img

    with Image.open(io.BytesIO(zf.read(image_files[0]))) as img:
        width, height = img.size
    if square_size is not None:
        width = height = square_size

    encode_video(frames(), output_path, fps=fps, size=(width, height))


def parse_poses(pose_bytes: bytes) -> np.ndarray:
    """ Reads a TartanAir V1 pose file.
        Rows are a translation and a scalar last quaternion in NED axes, which are
        permuted to the usual computer vision ones here.
    Args:
        pose_bytes: The raw file, as read out of the archive.
    Returns:
        Camera-to-world poses of shape (N, 4, 4).
    """
    lines = [
        ln.strip()
        for ln in pose_bytes.decode("utf-8").strip().split("\n")
        if ln.strip()
    ]
    poses = np.array([[float(v) for v in ln.split()] for ln in lines])
    assert poses.shape[1] == 7, f"Expected 7 columns in pose file, got {poses.shape[1]}"
    poses = poses[:, NED_PERM]
    rot = Rotation.from_quat(poses[:, 3:], scalar_first=False).as_matrix()
    c2ws = np.eye(4, dtype=np.float32)[None].repeat(len(poses), axis=0)
    c2ws[:, :3, :3] = rot
    c2ws[:, :3, 3] = poses[:, :3]
    return c2ws


def _in_traj(name: str, traj_id: str) -> bool:
    """ Says whether an archive member belongs to a trajectory.
    Args:
        name: The member's name inside the archive.
        traj_id: The trajectory to match, such as P000.
    Returns:
        True if the member is part of that trajectory.
    """
    return f"/{traj_id}/" in name and not name.startswith("__MACOSX")


def process_trajectory(
    zip_path: Path,
    traj_id: str,
    output_dir: Path,
    square_size: int | None = None,
) -> bool:
    """ Extracts one trajectory into a video, poses and intrinsics.
        Files that already exist are left alone, so an interrupted run can be rerun.
    Args:
        zip_path: The downloaded archive.
        traj_id: Which trajectory inside it to convert.
        output_dir: Where this trajectory's files go.
        square_size: Square edge to crop and resize to, or None for native size.
    Returns:
        True if the video and the poses are both on disk afterwards.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        image_files = sorted(
            n for n in names
            if _in_traj(n, traj_id) and n.lower().endswith((".png", ".jpg"))
        )
        pose_candidates = [
            n for n in names if _in_traj(n, traj_id) and n.endswith(POSE_FILE)
        ]

        video_path = output_dir / "video.mp4"
        if not video_path.exists() and image_files:
            create_video_from_zip_images(zf, image_files, video_path, square_size=square_size)

        poses_path = output_dir / "poses.npz"
        intrinsics_path = output_dir / "intrinsics.npy"
        if not poses_path.exists() and pose_candidates:
            c2ws = parse_poses(zf.read(pose_candidates[0]))
            assert len(c2ws) == len(image_files), (
                f"{output_dir.name}: {len(c2ws)} poses vs {len(image_files)} images"
            )
            np.savez(poses_path, poses=c2ws)
            if not intrinsics_path.exists():
                K = np.tile(INTRINSICS, (c2ws.shape[0], 1, 1))
                if square_size is not None and image_files:
                    with Image.open(io.BytesIO(zf.read(image_files[0]))) as img:
                        K = square_intrinsics(K, (img.height, img.width), square_size)
                np.save(intrinsics_path, K)

    return (output_dir / "video.mp4").exists() and (output_dir / "poses.npz").exists()


def discover_trajectories_in_zip(zip_path: Path) -> list[str]:
    """ Lists the trajectories an archive holds.
        Keyed off the pose files rather than every member name, so each trajectory
        appears once and is known to have the poses this script needs.
    Args:
        zip_path: The downloaded archive.
    Returns:
        The trajectory ids, sorted.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        return sorted(
            n.split("/")[-2] for n in zf.namelist()
            if n.endswith(POSE_FILE) and not n.startswith("__MACOSX")
        )


def process_scene(
    scene_name: str,
    difficulties: list[str],
    work_dir: Path,
    output_path: Path,
    hf_token: str | None = None,
    square_size: int | None = None,
) -> None:
    """ Downloads one environment, converts every trajectory in it, then cleans up.
        The archive and the download cache are deleted afterwards, so peak disk is
        one archive rather than the whole corpus. Trajectories that are already valid
        on disk are skipped.
    Args:
        scene_name: The environment to fetch.
        difficulties: Which difficulties to convert.
        work_dir: Scratch directory for the download.
        output_path: Where the processed trajectories go.
        hf_token: HuggingFace token, if the dataset needs one.
        square_size: Square edge to crop and resize to, or None for native size.
    """
    scene_work = work_dir / scene_name
    scene_work.mkdir(parents=True, exist_ok=True)
    hf_cache = work_dir / ".cache"

    for difficulty in difficulties:
        hf_filename = f"{scene_name}/{difficulty}/{ARCHIVE}"

        local_zip = Path(
            hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=hf_filename,
                repo_type="dataset",
                local_dir=str(scene_work),
                token=hf_token,
            )
        )

        for traj_id in discover_trajectories_in_zip(local_zip):
            seq_name = f"{scene_name}_{difficulty}_{traj_id}"
            seq_out = output_path / seq_name

            if is_sequence_valid(seq_out):
                print(f"  Skipping {seq_name} (valid)")
                continue

            process_trajectory(local_zip, traj_id, seq_out, square_size=square_size)
            print(f"  Done: {seq_name}")

        local_zip.unlink(missing_ok=True)

    shutil.rmtree(scene_work, ignore_errors=True)
    shutil.rmtree(hf_cache, ignore_errors=True)


def main() -> int:
    """ Converts the requested environments, one at a time.
    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Process TartanAir V1: download -> video + poses, one env at a time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_path", type=str,
                        help="Directory to write processed sequences.")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Environments to process (default: all).")
    parser.add_argument("--difficulties", nargs="+", default=DIFFICULTY_LEVELS,
                        choices=DIFFICULTY_LEVELS,
                        help="Difficulties to process (default: both).")
    parser.add_argument(
        "--square-size", type=int, default=TRAINING_READY_SIZE,
        help=(
            f"Square edge in pixels, {TRAINING_READY_SIZE} by default. Each frame is "
            "centre cropped to its shortest edge and resized to it, and the "
            "intrinsics follow. This is the form the configs read."
        ),
    )
    parser.add_argument(
        "--full-resolution", action="store_true",
        help="Keep the native resolution. The shipped configs do not read this form.")
    parser.add_argument("--work-dir", type=str, default="/tmp/tartanv1_work",
                        help="Scratch directory for downloads (default: /tmp/tartanv1_work).")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace access token (if the dataset requires login).")
    parser.add_argument("--list-scenes", action="store_true",
                        help="List all available environments and exit.")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(work_dir / ".cache" / "huggingface")

    print("Discovering environments from HuggingFace...")
    all_scenes = discover_scenes()
    print(f"Found {len(all_scenes)} environments.")

    if args.list_scenes:
        for s in all_scenes:
            print(f"  {s}")
        return 0

    scenes_to_process = args.scenes if args.scenes else all_scenes
    unknown = [s for s in scenes_to_process if s not in all_scenes]
    if unknown:
        print(f"Error: unknown environments {unknown}")
        return 1

    for i, scene in enumerate(tqdm(scenes_to_process, desc="Environments"), 1):
        print(f"\n[{i}/{len(scenes_to_process)}] Environment: {scene}")
        process_scene(
            scene, args.difficulties, work_dir, output_path, hf_token=args.token,
            square_size=None if args.full_resolution else args.square_size)

    print(f"\n{'=' * 50}")
    print("All environments processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
