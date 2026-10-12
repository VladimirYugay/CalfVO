""" This script converts EuRoC MAV sequences into the layout the loaders read.

Reads the raw dataset directly, so no pre-decimated ground truth file is needed.

    python scripts/data/process_euroc.py /raw/euroc /path/to/datasets/EuRoCVideos
"""
import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from prep_utils import encode_video
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def create_video_from_images(image_paths, output_video, fps=30):
    """ Encodes one sequence's cam0 frames into a video.
        cam0 is 752x480 monochrome, so the three encoded channels are identical.
    Args:
        image_paths: The frames in order.
        output_video: Where to write the mp4.
        fps: Frame rate to declare on the stream.
    """
    encode_video((Image.open(p).convert('RGB') for p in image_paths),
                 output_video, fps=fps)


def load_extrinsics(scene_path):
    """ Reads the cam0 to body transform from the sequence's own calibration.
        T_BS maps a point in the cam0 frame to the body frame, so p_body is
        T_BS times p_cam0. Every EuRoC sequence was recorded on one physical rig
        and carries the same value, but reading it per scene is more robust than
        hardcoding it.
    Args:
        scene_path: The raw sequence directory.
    Returns:
        The 4x4 transform.
    """
    with open(scene_path / "mav0/cam0/sensor.yaml") as f:
        calib = yaml.safe_load(f)
    return np.array(calib["T_BS"]["data"], dtype=np.float64).reshape(4, 4)


def nearest_indices(query_ts, ref_ts):
    """ For each query timestamp, finds the index of the closest reference one.
    Args:
        query_ts: Timestamps to match, sorted ascending.
        ref_ts: Timestamps to match against, sorted ascending.
    Returns:
        One index into ref_ts per query timestamp.
    """
    idx = np.searchsorted(ref_ts, query_ts)
    idx = np.clip(idx, 1, len(ref_ts) - 1)
    left_closer = (query_ts - ref_ts[idx - 1]) <= (ref_ts[idx] - query_ts)
    return idx - left_closer


def load_poses_from_groundtruth(scene_path, image_paths):
    """ Pairs every camera frame with the ground truth pose closest to it in time.
        state_groundtruth_estimate0 runs at about 200 Hz against the camera's 20 Hz,
        so the two have to be matched on timestamps. The body frame pose is then
        taken to the camera frame through the cam0 extrinsic. Quaternion columns are
        scalar first, not TUM's scalar last.
    Args:
        scene_path: The raw sequence directory.
        image_paths: The cam0 frames in order. Their stems are the timestamps.
    Returns:
        Camera-to-world poses of shape (N, 4, 4), one per frame.
    """
    gt = np.loadtxt(scene_path / "mav0/state_groundtruth_estimate0/data.csv",
                    delimiter=",", skiprows=1)
    gt_ts = gt[:, 0]

    image_ts = np.array([int(p.stem) for p in image_paths], dtype=np.float64)
    matched = nearest_indices(image_ts, gt_ts)

    trans = gt[matched, 1:4]
    quat_wxyz = gt[matched, 4:8]
    rot = Rotation.from_quat(quat_wxyz, scalar_first=True).as_matrix()

    T_w_i = np.eye(4)[None].repeat(len(image_paths), axis=0)
    T_w_i[:, :3, :3] = rot
    T_w_i[:, :3, 3] = trans

    T_i_c0 = load_extrinsics(scene_path)
    return (T_w_i @ T_i_c0[None]).astype(np.float32)


def main(input_path, output_path):
    """ Converts every EuRoC sequence under the input directory.
    Args:
        input_path: Directory holding the unpacked EuRoC sequences.
        output_path: Directory that receives one subdirectory per sequence.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for scene_path in tqdm(sorted(Path(input_path).glob('*'))):
        scene_name = scene_path.parts[-1]
        print("Processing scene:", scene_name)
        scene_output_path = output_path / scene_name
        scene_output_path.mkdir(parents=True, exist_ok=True)

        image_paths = sorted((scene_path / "mav0/cam0/data").glob("*"))

        video_file = scene_output_path / "video.mp4"
        if not video_file.exists():
            create_video_from_images(image_paths, str(video_file), fps=30)

        pose_file = scene_output_path / "poses.npz"
        if not pose_file.exists():
            c2ws = load_poses_from_groundtruth(scene_path, image_paths)
            np.savez(pose_file, poses=c2ws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert EuRoC MAV sequences into videos and poses.")
    parser.add_argument('input_path', type=str,
                        help="Directory holding the unpacked EuRoC sequences.")
    parser.add_argument('output_path', type=str,
                        help="Directory that receives the processed sequences.")

    args = parser.parse_args()
    main(args.input_path, args.output_path)
