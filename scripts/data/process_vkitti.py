#!/usr/bin/env python3
"""Virtual KITTI 2 to one video, pose file and intrinsics per sequence.

Expects the vkitti_2.0.3_rgb and vkitti_2.0.3_textgt tarballs extracted and merged.

    python process_vkitti.py <input_path> <output_path>
    python process_vkitti.py <in> <out> --variants clone fog --cameras 0 1
    python process_vkitti.py <in> <out> --split-out splits
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from prep_utils import (TRAINING_READY_SIZE, encode_video, is_sequence_valid,
                        square_crop_resize, square_intrinsics)


SCENES = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
VARIANTS = [
    "clone",
    "15-deg-left",
    "15-deg-right",
    "30-deg-left",
    "30-deg-right",
    "fog",
    "morning",
    "overcast",
    "rain",
    "sunset",
]
# Virtual KITTI 2 renders at KITTI's capture rate. Nothing downstream reads it.
FPS = 10


def _read_table(path: Path) -> np.ndarray:
    """ Reads one of the whitespace separated .txt tables, dropping the header.
    Args:
        path: The table to read.
    Returns:
        Its rows as floats.
    """
    return np.loadtxt(path, skiprows=1, dtype=np.float64)


def parse_poses(extrinsic_path: Path, camera_id: int) -> np.ndarray:
    """ Reads a sequence's extrinsics for one camera.
        Rows are a frame id, a camera id and a row-major 4x4 world-to-camera matrix.
        The camera frame is already OpenCV, so no basis change is applied anywhere.
    Args:
        extrinsic_path: The variant's extrinsic.txt.
        camera_id: Which of the two cameras to read.
    Returns:
        Camera-to-world poses of shape (N, 4, 4), ordered by frame.
    """
    table = _read_table(extrinsic_path)
    rows = table[table[:, 1] == camera_id]
    rows = rows[np.argsort(rows[:, 0], kind="stable")]

    frames = rows[:, 0].astype(np.int64)
    assert np.array_equal(frames, np.arange(len(frames))), (
        f"{extrinsic_path}: camera {camera_id} frame ids are not 0..N-1 contiguous"
    )

    w2c = rows[:, 2:].reshape(-1, 4, 4)
    assert np.allclose(w2c[:, 3, :], [0, 0, 0, 1]), f"{extrinsic_path}: bad bottom row"

    # Built from R^T and -R^T t rather than a general inverse, so the result stays
    # exactly rigid where the file's printed digits leave R only near orthonormal.
    rot = w2c[:, :3, :3]
    trans = w2c[:, :3, 3]
    c2w = np.tile(np.eye(4, dtype=np.float64), (len(w2c), 1, 1))
    c2w[:, :3, :3] = rot.transpose(0, 2, 1)
    c2w[:, :3, 3] = -np.einsum("nij,nj->ni", rot.transpose(0, 2, 1), trans)
    return c2w.astype(np.float32)


def parse_intrinsics(intrinsic_path: Path, camera_id: int) -> np.ndarray:
    """ Reads a sequence's intrinsics for one camera.
        Rows are a frame id, a camera id, then fx, fy, cx and cy.
    Args:
        intrinsic_path: The variant's intrinsic.txt.
        camera_id: Which of the two cameras to read.
    Returns:
        Camera matrices of shape (N, 3, 3), ordered by frame.
    """
    table = _read_table(intrinsic_path)
    rows = table[table[:, 1] == camera_id]
    rows = rows[np.argsort(rows[:, 0], kind="stable")]

    n = len(rows)
    K = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    K[:, 0, 0] = rows[:, 2]  # fx
    K[:, 1, 1] = rows[:, 3]  # fy
    K[:, 0, 2] = rows[:, 4]  # cx
    K[:, 1, 2] = rows[:, 5]  # cy
    return K


def adjust_intrinsics(K: np.ndarray, src_hw, dst_hw) -> np.ndarray:
    """ Rescales camera matrices for a resize.
        A pure crop is a no-op, because cropping here only removes rows and columns
        at the bottom and right edge, which leaves the principal point alone.
    Args:
        K: Camera matrices of shape (N, 3, 3).
        src_hw: Height and width the matrices belong to.
        dst_hw: Height and width after the resize.
    Returns:
        The adjusted matrices.
    """
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K = K.copy()
    K[:, 0, 0] *= sx
    K[:, 1, 1] *= sy
    K[:, 0, 2] *= sx
    K[:, 1, 2] *= sy
    return K


def create_video_from_images(
    image_paths: list,
    output_path: Path,
    fps: int = FPS,
    resize_hw=None,
    threads: int = 2,
    square_size: int | None = None,
) -> tuple:
    """ Encodes one sequence's frames into a video.
        yuv420p needs even dimensions and Virtual KITTI 2 renders 1242x375, so the
        odd row is dropped from the bottom. That is an edge crop, so it leaves the
        camera matrix unchanged.
    Args:
        image_paths: The frames in order.
        output_path: Where to write the mp4.
        fps: Frame rate to declare on the stream.
        resize_hw: Height and width to resize to, or None to keep the native size.
        threads: Encoder thread count.
        square_size: Square edge to crop and resize to, which wins over resize_hw.
    Returns:
        The height and width actually written.
    """
    with Image.open(image_paths[0]) as img:
        src_w, src_h = img.size

    if square_size is not None:
        dst_h = dst_w = square_size          # takes precedence over --resize
    elif resize_hw is not None:
        dst_h, dst_w = resize_hw
    else:
        dst_h, dst_w = src_h, src_w
    dst_h -= dst_h % 2
    dst_w -= dst_w % 2

    def frames():
        for image_path in image_paths:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if square_size is not None:
                    img = square_crop_resize(img, square_size)
                elif resize_hw is not None:
                    img = img.resize((dst_w, dst_h), Image.LANCZOS)
                elif (img.height, img.width) != (dst_h, dst_w):
                    img = img.crop((0, 0, dst_w, dst_h))
                yield img

    # Atomic, so a run killed mid-encode leaves no truncated video.mp4 for the
    # resume check to accept.
    encode_video(frames(), output_path, fps=fps, size=(dst_w, dst_h),
                 threads=threads, atomic=True)
    return dst_h, dst_w


def sequence_name(scene: str, variant: str, camera_id: int) -> str:
    """ Builds the output directory name for one sequence.
    Args:
        scene: The scene, such as Scene01.
        variant: The appearance or camera angle variant.
        camera_id: Which of the two cameras.
    Returns:
        The name, matching what the split files hold.
    """
    return f"{scene.lower()}_{variant.replace('-', '')}_cam{camera_id}"


def process_sequence(task: dict) -> dict:
    """ Converts one scene, variant and camera triple. Runs in a worker process.
        Every failure is caught and reported in the result, so one bad sequence does
        not kill a run over hundreds of them.
    Args:
        task: The work item, holding the paths and the conversion options.
    Returns:
        A result dict whose status is ok, skipped or failed.
    """
    scene, variant, camera_id = task["scene"], task["variant"], task["camera_id"]
    input_path = Path(task["input_path"])
    seq_out = Path(task["output_path"]) / sequence_name(scene, variant, camera_id)
    name = seq_out.name

    if not task["overwrite"] and is_sequence_valid(seq_out):
        return {"name": name, "status": "skipped"}

    variant_dir = input_path / scene / variant
    frames_dir = variant_dir / "frames" / "rgb" / f"Camera_{camera_id}"

    try:
        image_paths = sorted(frames_dir.glob("rgb_*.jpg"))
        if not image_paths:
            return {"name": name, "status": "failed", "error": f"no frames in {frames_dir}"}

        c2ws = parse_poses(variant_dir / "extrinsic.txt", camera_id)
        K = parse_intrinsics(variant_dir / "intrinsic.txt", camera_id)

        if not (len(image_paths) == len(c2ws) == len(K)):
            return {
                "name": name,
                "status": "failed",
                "error": (
                    f"count mismatch: {len(image_paths)} frames, "
                    f"{len(c2ws)} poses, {len(K)} intrinsics"
                ),
            }

        seq_out.mkdir(parents=True, exist_ok=True)

        with Image.open(image_paths[0]) as img:
            src_hw = (img.height, img.width)

        video_path = seq_out / "video.mp4"
        if task["overwrite"] or not video_path.exists():
            dst_hw = create_video_from_images(
                image_paths,
                video_path,
                fps=task["fps"],
                resize_hw=task["resize_hw"],
                threads=task["threads"],
                square_size=task["square_size"],
            )
        else:
            if task["square_size"] is not None:
                dst_hw = (task["square_size"], task["square_size"])
            else:
                dst_hw = task["resize_hw"] or src_hw
            dst_hw = (dst_hw[0] - dst_hw[0] % 2, dst_hw[1] - dst_hw[1] % 2)

        if task["square_size"] is not None:
            K = square_intrinsics(K, src_hw, task["square_size"])
        elif task["resize_hw"] is not None:
            K = adjust_intrinsics(K, src_hw, dst_hw)

        np.savez_compressed(seq_out / "poses.npz", poses=c2ws)
        np.save(seq_out / "intrinsics.npy", K)

        return {"name": name, "status": "ok", "frames": len(image_paths), "hw": dst_hw}
    except Exception as exc:  # keep one bad sequence from killing the whole run
        return {"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def write_splits(output_path: Path, tasks: list, split_dir: Path) -> None:
    """ Writes train.txt holding every valid sequence.
        Train only, since VKITTI is synthetic and evaluation stays on the real
        corpora. A held-out split would have to be cut by scene and never by variant,
        because a scene's six appearance variants are one trajectory re-rendered.
    Args:
        output_path: Where the processed sequences were written.
        tasks: Every work item that was scheduled.
        split_dir: Where to write train.txt.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(
        name
        for name in (
            sequence_name(t["scene"], t["variant"], t["camera_id"]) for t in tasks
        )
        if is_sequence_valid(output_path / name)
    )
    (split_dir / "train.txt").write_text("\n".join(names) + "\n")
    print(f"Split written to {split_dir / 'train.txt'}: {len(names)} sequences (train only)")


def main() -> int:
    """ Converts the requested scenes, variants and cameras in parallel.
    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Process Virtual KITTI 2 into video + poses + intrinsics sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Extracted VKITTI2 root (the directory containing Scene01, Scene02, ...).",
    )
    parser.add_argument(
        "output_path", type=str, help="Directory to write processed sequences."
    )
    parser.add_argument(
        "--scenes", nargs="+", default=SCENES, choices=SCENES, help="Scenes to process."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=VARIANTS,
        choices=VARIANTS,
        help="Weather/rotation variants to process.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        type=int,
        default=[0],
        choices=[0, 1],
        help="Stereo cameras: 0 = left, 1 = right (default: left only).",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=TRAINING_READY_SIZE,
        help=(
            f"Square edge in pixels, {TRAINING_READY_SIZE} by default. Each frame is "
            "centre cropped to its shortest edge and resized to it, and the "
            "intrinsics follow. Takes precedence over --resize."
        ),
    )
    parser.add_argument(
        "--full-resolution", action="store_true",
        help="Keep the native resolution. The shipped configs do not read this form.")
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Resize frames to HxW and rescale K (default: keep native 1242x375, "
             "cropping the odd bottom row that yuv420p cannot take).",
    )
    parser.add_argument("--fps", type=int, default=FPS,
                        help=f"Container frame rate (default: {FPS}).")
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Sequences encoded concurrently (default: 2).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="libx264 threads per job (default: 2). jobs*threads is the core budget.",
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=19,
        help="Niceness for the workers (default: 19, i.e. yield to training).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-encode sequences that already look valid."
    )
    parser.add_argument(
        "--split-out",
        type=str,
        default=None,
        help="Also write a train.txt listing every valid sequence to this directory.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List the work to be done and exit."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if not input_path.is_dir():
        print(f"Error: input path {input_path} is not a directory.", file=sys.stderr)
        return 1
    output_path.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "scene": scene,
            "variant": variant,
            "camera_id": cam,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "resize_hw": tuple(args.resize) if args.resize else None,
            "square_size": (None if args.full_resolution
                            else args.square_size),
            "fps": args.fps,
            "threads": args.threads,
            "overwrite": args.overwrite,
        }
        for scene in args.scenes
        for variant in args.variants
        for cam in args.cameras
    ]

    pending = [
        t
        for t in tasks
        if args.overwrite
        or not is_sequence_valid(output_path / sequence_name(
            t["scene"], t["variant"], t["camera_id"]))
    ]
    print(f"{len(tasks)} sequences requested, {len(pending)} to process "
          f"({len(tasks) - len(pending)} already valid).")

    if args.dry_run:
        for t in pending:
            print(f"  {sequence_name(t['scene'], t['variant'], t['camera_id'])}")
        return 0

    results = []
    if pending:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=max(1, args.jobs),
            initializer=_init_worker,
            initargs=(args.nice,),
        ) as pool:
            for res in tqdm(
                pool.imap_unordered(process_sequence, pending),
                total=len(pending),
                desc="Sequences",
            ):
                results.append(res)
                if res["status"] == "failed":
                    tqdm.write(f"  FAILED {res['name']}: {res['error']}")
                elif res["status"] == "ok":
                    tqdm.write(
                        f"  done {res['name']}: {res['frames']} frames @ "
                        f"{res['hw'][1]}x{res['hw'][0]}"
                    )

    failed = [r for r in results if r["status"] == "failed"]
    done = [r for r in results if r["status"] == "ok"]
    print(f"\n{'=' * 60}")
    print(f"Processed {len(done)} sequences, {len(failed)} failed.")
    for r in failed:
        print(f"  FAILED {r['name']}: {r['error']}")

    if args.split_out:
        write_splits(output_path, tasks, Path(args.split_out))

    (output_path / "dataset_info.json").write_text(
        json.dumps(
            {
                "dataset": "virtual_kitti_2.0.3",
                "pose_convention": ("camera-to-world, OpenCV camera axes "
                                    "(x right, y down, z forward)"),
                "source_extrinsics": "world-to-camera; inverted here",
                "scenes": args.scenes,
                "variants": args.variants,
                "cameras": args.cameras,
                "fps": args.fps,
            },
            indent=2,
        )
        + "\n"
    )
    return 1 if failed else 0


def _init_worker(niceness: int) -> None:
    """ Sets up a worker process.
        The thread limits matter, because PIL and numpy would otherwise each fan out
        over every core in every worker.
    Args:
        niceness: How far to lower the worker's priority.
    """
    try:
        os.nice(niceness)
    except OSError:
        pass
    # PIL/numpy can otherwise fan out over all 96 cores per worker.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


if __name__ == "__main__":
    sys.exit(main())
