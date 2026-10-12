import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from prep_utils import encode_video

# Camera intrinsics (3x3) per TUM freiburg camera at 640x480
TUM_INTRINSICS = {
    "freiburg1": np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]]),
    "freiburg2": np.array([[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]]),
    "freiburg3": np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]]),
}


def parse_tum_txt(txt_file):
    """ Reads a TUM index file into timestamped relative image paths.
    Args:
        txt_file: rgb.txt from a sequence directory.
    Returns:
        A list of (timestamp, image path) pairs, comments skipped.
    """
    entries = []
    with open(txt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            entries.append((float(parts[0]), parts[1]))
    return entries


def parse_groundtruth(gt_file):
    """ Reads the ground truth trajectory file.
        Rows are a timestamp, a translation and a scalar last quaternion.
    Args:
        gt_file: groundtruth.txt from a sequence directory.
    Returns:
        A list of (ts, tx, ty, tz, qx, qy, qz, qw) tuples.
    """
    poses = []
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            poses.append((ts, tx, ty, tz, qx, qy, qz, qw))
    return poses


def associate(rgb_entries, gt_poses, max_diff=0.02):
    """ Pairs every image with the ground truth pose closest to it in time.
        The two streams are logged independently, so an image whose nearest pose is
        further away than max_diff is dropped rather than given a stale one.
    Args:
        rgb_entries: The timestamped image paths.
        gt_poses: The timestamped ground truth poses.
        max_diff: Largest timestamp gap to accept, in seconds.
    Returns:
        The kept image paths and their matched poses, in order.
    """
    gt_ts = np.array([p[0] for p in gt_poses])
    matched_imgs, matched_poses = [], []
    for ts, img_path in rgb_entries:
        idx = np.argmin(np.abs(gt_ts - ts))
        if abs(gt_ts[idx] - ts) <= max_diff:
            matched_imgs.append(img_path)
            matched_poses.append(gt_poses[idx])
    return matched_imgs, matched_poses


def build_c2ws(matched_poses):
    """ Turns matched TUM poses into camera-to-world matrices.
    Args:
        matched_poses: The tuples associate returned.
    Returns:
        Camera-to-world poses of shape (N, 4, 4).
    """
    c2ws = []
    for _, tx, ty, tz, qx, qy, qz, qw in matched_poses:
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = [tx, ty, tz]
        c2ws.append(c2w)
    return np.array(c2ws)


def create_video(image_paths, scene_dir, output_video, fps=30):
    """ Encodes one sequence's RGB frames into a video.
    Args:
        image_paths: The frames in order, relative to scene_dir.
        scene_dir: The raw sequence directory.
        output_video: Where to write the mp4.
        fps: Frame rate to declare on the stream.
    """
    encode_video((Image.open(scene_dir / p).convert('RGB') for p in image_paths),
                 output_video, fps=fps)


def get_intrinsics(scene_name):
    """ Looks up the camera matrix for a sequence.
        TUM ships three cameras, and the sequence name says which one recorded it.
    Args:
        scene_name: The sequence directory name.
    Returns:
        The 3x3 camera matrix.
    """
    for key in TUM_INTRINSICS:
        if key in scene_name:
            return TUM_INTRINSICS[key]
    raise ValueError(f"Unknown camera for scene: {scene_name}")


def process_scene(scene_path, output_path):
    """ Converts one raw sequence into a video, poses and intrinsics.
        Files that already exist are left alone, so an interrupted run can be rerun.
    Args:
        scene_path: The raw sequence directory.
        output_path: Where the processed sequences go.
    """
    scene_name = scene_path.name
    scene_out = output_path / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)

    rgb_entries = parse_tum_txt(scene_path / "rgb.txt")
    gt_poses = parse_groundtruth(scene_path / "groundtruth.txt")
    matched_imgs, matched_poses = associate(rgb_entries, gt_poses)

    if not matched_imgs:
        print(f"  No matched frames for {scene_name}, skipping")
        return

    video_file = scene_out / "video.mp4"
    if not video_file.exists():
        create_video(matched_imgs, scene_path, str(video_file))

    pose_file = scene_out / "poses.npz"
    if not pose_file.exists():
        c2ws = build_c2ws(matched_poses)
        np.savez(pose_file, poses=c2ws)

    intrinsics_file = scene_out / "intrinsics.npz"
    if not intrinsics_file.exists():
        K = get_intrinsics(scene_name)
        np.savez(intrinsics_file, intrinsics=K)


def main(input_path, output_path):
    """ Converts every sequence under the dataset root.
    Args:
        input_path: The TUM RGB-D root holding the rgbd_dataset_* directories.
        output_path: Where the processed sequences go.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    scene_paths = sorted(p for p in input_path.glob('*') if p.is_dir())
    for scene_path in tqdm(scene_paths):
        print(f"Processing: {scene_path.name}")
        process_scene(scene_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',
                        help="Path to the TUM RGB-D dataset root "
                             "(contains rgbd_dataset_* folders)")
    parser.add_argument('output_path', help="Path to the output directory")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
