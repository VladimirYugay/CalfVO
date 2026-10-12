import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from SensorData import SensorData
from tqdm import tqdm

from prep_utils import (TRAINING_READY_SIZE, encode_video, square_crop_resize,
                        square_intrinsics)

IMG_HEIGHT, IMG_WIDTH = 968, 1296


def create_intrinsics_file_from_sens(sensor_data, output_dir, img_size=None, square_size=None):
    """ Writes the colour camera matrix, adjusted for whichever resize is in use.
    Args:
        sensor_data: The opened .sens reader.
        output_dir: Path of the npz to write.
        img_size: Target height and width, when the frames are plainly resized.
        square_size: Square edge, when the frames are cropped and resized.
    """
    intrinsics = sensor_data.intrinsic_color[:3, :3]
    if square_size is not None:
        # Adjusted from the native resolution: the square path crops and scales the
        # original frame directly, it does not go through img_size.
        intrinsics = square_intrinsics(intrinsics, (IMG_HEIGHT, IMG_WIDTH), square_size)
    elif img_size is not None:
        sx, sy = img_size[1] / IMG_WIDTH, img_size[0] / IMG_HEIGHT
        new_intrinsics = intrinsics[:3, :3].copy()
        new_intrinsics[0, 0] *= sx
        new_intrinsics[1, 1] *= sy
        new_intrinsics[0, 2] *= sx
        new_intrinsics[1, 2] *= sy
        intrinsics = new_intrinsics.copy()
    np.savez_compressed(output_dir, poses=intrinsics)


def create_video_from_images(sensor_data, output_video, fps=30, img_size=None, square_size=None):

    """ Encodes one scene's colour frames into a video.
    Args:
        sensor_data: The opened .sens reader.
        output_video: Where to write the mp4.
        fps: Frame rate to declare on the stream.
        img_size: Target height and width, or None to keep the native one.
        square_size: Square edge, which takes precedence over img_size.
    """
    if square_size is not None:
        height = width = square_size
    elif img_size is None:
        first_image = sensor_data.frames[0].decompress_color(
            sensor_data.color_compression_type)
        first_image = Image.fromarray(first_image)
        width, height = first_image.size
        print(f"Inferred resolution: {width}x{height}")
    else:
        height, width = img_size

    def frames():
        for frame_id in range(len(sensor_data.frames)):
            image = sensor_data.frames[frame_id].decompress_color(
                sensor_data.color_compression_type)
            if square_size is not None:
                # Straight from the native frame, since going via img_size would
                # resample twice.
                yield square_crop_resize(
                    Image.fromarray(image).convert('RGB'), square_size)
            else:
                if img_size is not None:
                    image = cv2.resize(
                        image, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
                yield Image.fromarray(image).convert('RGB')

    encode_video(frames(), output_video, fps=fps, size=(width, height))


def create_pose_file_from_sens(sensor_data, output_dir):
    """ Writes every frame's camera-to-world pose.
    Args:
        sensor_data: The opened .sens reader.
        output_dir: Path of the npz to write.
    """
    poses = []
    for i in range(len(sensor_data.frames)):
        poses.append(sensor_data.frames[i].camera_to_world)
    np.savez_compressed(output_dir, poses=np.array(poses))


def main(input_path, output_path, square_size=None):
    """ Converts every .sens file under the input directory.
        Files that already exist are left alone, so an interrupted run can be rerun.
    Args:
        input_path: Directory holding one subdirectory per scan.
        output_path: Where the processed sequences go.
        square_size: Square edge to crop and resize to, or None for native size.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for scene_path in tqdm(sorted((Path(input_path).glob('*')))):

        scene_name = scene_path.parts[-1]
        sens_file_path = scene_path / (scene_name + ".sens")
        sensor_data = SensorData(sens_file_path)

        scene_output_path = output_path / scene_name
        scene_output_path.mkdir(parents=True, exist_ok=True)

        img_size = (480, 640)

        video_file = scene_output_path / "video.mp4"
        if not video_file.exists():
            create_video_from_images(sensor_data, str(video_file), fps=30,
                                     img_size=img_size, square_size=square_size)

        pose_file = scene_output_path / "poses.npz"
        if not pose_file.exists():
            create_pose_file_from_sens(sensor_data, str(pose_file))

        intrinsics_file = scene_output_path / "intrinsics.npz"
        if not intrinsics_file.exists():
            create_intrinsics_file_from_sens(sensor_data, str(intrinsics_file), img_size,
                                             square_size=square_size)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Process .sens files in a given directory.")
    parser.add_argument('input_path', type=str,
                        help="Path to the input directory containing .sens files.")
    parser.add_argument('output_path', type=str,
                        help="Path to the output directory where results will be stored.")

    parser.add_argument(
        '--square-size', type=int, default=TRAINING_READY_SIZE,
        help=(f"Square edge in pixels, {TRAINING_READY_SIZE} by default. Each frame "
              "is centre cropped to its shortest edge and resized to it, and the "
              "intrinsics follow. This is the form the configs read."))
    parser.add_argument(
        '--full-resolution', action='store_true',
        help="Keep the native resolution. The shipped configs do not read this form.")

    args = parser.parse_args()

    main(args.input_path, args.output_path,
         square_size=None if args.full_resolution else args.square_size)
