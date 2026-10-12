"""Shared helpers for the dataset processing scripts."""
import os
from itertools import chain
from pathlib import Path

import av
import numpy as np
from PIL import Image

# Baking training's centre crop into the video makes decoding cheaper without
# changing what the model sees. Stays above 224/0.8 so the random zoom downsamples.
TRAINING_READY_SIZE = 288

# All-intra, because training reads scattered windows and a long GOP forces a
# forward decode from the nearest keyframe on every seek. Every corpus is encoded
# with these settings, so keep them in step or the corpora stop being comparable.
X264_ALL_INTRA = {"crf": "18", "g": "1", "bf": "0", "refs": "1"}


def encode_video(frames, output_path, fps=30, size=None, threads=None, atomic=False):
    """ Encodes an iterable of PIL images into an all-intra H.264 mp4.
        libx264 refuses odd dimensions under yuv420p, so those are rounded down and
        the frames cropped to match. It is an edge crop, so the intrinsics are
        unaffected.
    Args:
        frames: PIL images in order, all the same size.
        output_path: Where to write the mp4.
        fps: Frame rate to declare on the stream.
        size: Width and height to declare, defaulting to the first frame's.
        threads: Thread count for the stream and for x264, or None for the default.
        atomic: Write to a .partial file and rename on success, so a run killed
            mid-encode leaves nothing a resume check would accept.
    """
    frames = iter(frames)
    try:
        first = next(frames)
    except StopIteration:
        raise ValueError(f"{output_path}: no frames to encode") from None

    width, height = size if size is not None else first.size
    width, height = width - width % 2, height - height % 2

    def cropped(source):
        for img in source:
            if img.size != (width, height):
                img = img.crop((0, 0, width, height))
            yield img

    frames = cropped(frames)
    if first.size != (width, height):
        first = first.crop((0, 0, width, height))
    output_path = str(output_path)
    target = output_path + ".partial" if atomic else output_path

    # `format` is explicit because an atomic write's filename ends in .partial,
    # which av cannot map to a container on its own.
    container = av.open(target, mode="w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        options = dict(X264_ALL_INTRA)
        if threads is not None:
            stream.thread_count = threads
            options["threads"] = str(threads)
        stream.options = options

        for img in chain([first], frames):
            for packet in stream.encode(av.VideoFrame.from_image(img)):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except BaseException:
        container.close()
        if atomic and os.path.exists(target):
            os.unlink(target)
        raise

    if atomic:
        os.replace(target, output_path)


def square_crop_resize(img, size):
    """ Centre-crops an image to its shortest edge and resizes it to a square.
    Args:
        img: The image to crop.
        size: Edge of the square, in pixels.
    Returns:
        The cropped and resized image.
    """
    w, h = img.size
    edge = min(w, h)
    left, top = (w - edge) // 2, (h - edge) // 2
    img = img.crop((left, top, left + edge, top + edge))
    return img.resize((size, size), Image.BILINEAR)


def square_intrinsics(K, src_hw, size):
    """ Rescales a camera matrix for square_crop_resize.
        The centre crop shifts the principal point, then a uniform scale multiplies
        the focal length and the principal point alike.
    Args:
        K: A camera matrix of shape (3, 3) or a stack of shape (N, 3, 3).
        src_hw: Height and width of the frames K belongs to.
        size: Edge of the square, in pixels.
    Returns:
        The adjusted camera matrix, in the shape it came in.
    """
    h, w = src_hw
    edge = min(w, h)
    left, top = (w - edge) // 2, (h - edge) // 2
    scale = size / edge
    K = K.copy()
    K[..., 0, 2] -= left
    K[..., 1, 2] -= top
    K[..., :2, :] *= scale
    return K


def is_sequence_valid(output_dir: Path) -> bool:
    """ Says whether a processed sequence is complete.
        This is the resume check for every script that can be interrupted, so it is
        strict about the shapes agreeing. The common failure is a run killed between
        writing the poses and the video, and a corrupt npz counts as invalid rather
        than raising, so the sequence is redone.
    Args:
        output_dir: The sequence directory to check.
    Returns:
        True if the video, the poses and the intrinsics are all there and agree on
        the frame count.
    """
    output_dir = Path(output_dir)
    video = output_dir / "video.mp4"
    poses_file = output_dir / "poses.npz"
    intrinsics_file = output_dir / "intrinsics.npy"
    if not (video.exists() and poses_file.exists() and intrinsics_file.exists()):
        return False
    try:
        poses = np.load(poses_file)["poses"]
        intrinsics = np.load(intrinsics_file)
    except (OSError, ValueError, KeyError):
        return False
    return (
        poses.ndim == 3
        and poses.shape[1:] == (4, 4)
        and intrinsics.ndim == 3
        and intrinsics.shape[1:] == (3, 3)
        and poses.shape[0] == intrinsics.shape[0]
        and poses.shape[0] > 0
    )
