#!/usr/bin/env python3
"""TartanAir V2 to one video, pose file and intrinsics per trajectory.

Pulls one scene zip from HuggingFace at a time and deletes it before moving on, so
peak disk is one archive rather than the whole corpus.

    python process_tartanv2.py output_path --scenes AbandonedFactory Hospital
    python process_tartanv2.py output_path --list-scenes
"""

import argparse
import io
import os
import re
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


HF_DATASET_ID = "theairlabcmu/tartanair2"
DIFFICULTY_LEVELS = ["Data_easy", "Data_hard"]
CAMERA = "lcam_front"

# fx=fy=320, cx=cy=320 for 640x640 pinhole images
INTRINSICS = np.array(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

# NED-to-CV cyclic permutation (same as TartanAir V1)
NED_PERM = [1, 2, 0, 4, 5, 3, 6]


def discover_scenes() -> list[str]:
    """ Lists the scenes the HuggingFace dataset holds.
    Returns:
        The scene names, sorted.
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
    """ Reads a TartanAir V2 pose file.
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
    in_traj = f"/{traj_id}/" in name or name.startswith(f"{traj_id}/")
    return in_traj and not name.startswith("__MACOSX")


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
            n for n in names
            if _in_traj(n, traj_id) and n.endswith(f"pose_{CAMERA}.txt")
        ]

        video_path = output_dir / "video.mp4"
        if not video_path.exists() and image_files:
            create_video_from_zip_images(zf, image_files, video_path, square_size=square_size)

        poses_path = output_dir / "poses.npz"
        intrinsics_path = output_dir / "intrinsics.npy"
        if not poses_path.exists() and pose_candidates:
            c2ws = parse_poses(zf.read(pose_candidates[0]))
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
    Args:
        zip_path: The downloaded archive.
    Returns:
        The trajectory ids, sorted.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        traj_ids: set[str] = set()
        for name in zf.namelist():
            for part in name.split("/"):
                if re.match(r"^P\d{3,}$", part):
                    traj_ids.add(part)
        return sorted(traj_ids)


def process_scene(
    scene_name: str,
    work_dir: Path,
    output_path: Path,
    hf_token: str | None = None,
    square_size: int | None = None,
) -> None:
    """ Downloads one scene, converts every trajectory in it, then cleans up.
        The archive and the download cache are deleted afterwards, so peak disk is
        one scene rather than the whole corpus. Trajectories that are already valid
        on disk are skipped.
    Args:
        scene_name: The scene to fetch.
        work_dir: Scratch directory for the download.
        output_path: Where the processed trajectories go.
        hf_token: HuggingFace token, if the dataset needs one.
        square_size: Square edge to crop and resize to, or None for native size.
    """
    scene_work = work_dir / scene_name
    scene_work.mkdir(parents=True, exist_ok=True)
    hf_cache = work_dir / ".cache"

    for difficulty in DIFFICULTY_LEVELS:
        # Matches the split files, lowercase scene and capitalised difficulty,
        # rather than the HF repo's own casing.
        diff_label = difficulty.replace("Data_", "").capitalize()
        hf_filename = f"{scene_name}/{difficulty}/image_{CAMERA}.zip"

        local_zip = Path(
            hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=hf_filename,
                repo_type="dataset",
                local_dir=str(scene_work),
                local_dir_use_symlinks=False,
                token=hf_token,
            )
        )

        traj_ids = discover_trajectories_in_zip(local_zip)

        for traj_id in traj_ids:
            seq_name = f"{scene_name.lower()}_{diff_label}_{traj_id}"
            seq_out = output_path / seq_name

            if is_sequence_valid(seq_out):
                print(f"  Skipping {seq_name} (valid)")
                continue

            process_trajectory(local_zip, traj_id, seq_out, square_size=square_size)
            print(f"  Done: {seq_name}")

        local_zip.unlink(missing_ok=True)

    # Delete scene work dir and HF cache to free disk space
    shutil.rmtree(scene_work, ignore_errors=True)
    shutil.rmtree(hf_cache, ignore_errors=True)


def main() -> int:
    """ Converts the requested scenes, one at a time.
    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Process TartanAir V2: download → video + poses, one scene at a time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Directory to write processed sequences.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Specific scenes to process (default: all).",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=TRAINING_READY_SIZE,
        help=(
            f"Square edge in pixels, {TRAINING_READY_SIZE} by default. Each frame is "
            "centre cropped to its shortest edge and resized to it, and the "
            "intrinsics follow. This is the form the configs read."
        ),
    )
    parser.add_argument(
        "--full-resolution", action="store_true",
        help="Keep the native resolution. The shipped configs do not read this form.")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="/tmp/tartanv2_work",
        help="Scratch directory for downloads (default: /tmp/tartanv2_work).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Skip all scenes before this one.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (if the dataset requires login).",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List all available scenes and exit.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(work_dir / ".cache" / "huggingface")

    print("Discovering scenes from HuggingFace...")
    all_scenes = discover_scenes()
    print(f"Found {len(all_scenes)} scenes.")

    if args.list_scenes:
        for s in all_scenes:
            print(f"  {s}")
        return 0

    scenes_to_process = args.scenes if args.scenes else all_scenes

    if args.resume_from:
        if args.resume_from not in scenes_to_process:
            print(f"Error: --resume-from scene '{args.resume_from}' not in list.")
            return 1
        idx = scenes_to_process.index(args.resume_from)
        scenes_to_process = scenes_to_process[idx:]
        print(f"Resuming from '{args.resume_from}'.")

    for i, scene in enumerate(tqdm(scenes_to_process, desc="Scenes"), 1):
        print(f"\n[{i}/{len(scenes_to_process)}] Scene: {scene}")
        process_scene(
            scene, work_dir, output_path, hf_token=args.token,
            square_size=None if args.full_resolution else args.square_size)

    print(f"\n{'=' * 50}")
    print("All scenes processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
