""" This module cuts sequences into the windows the model is trained on. """
import numpy as np


def create_strided_windows(num_frames, window_size, overlap, stride):
    """ Cuts overlapping windows over raw frame indices at an explicit spacing.
        This is equivalent to create_overlapping_windows applied to
        np.arange(0, num_frames, stride), including its strict stopping condition, so
        a dataset built this way has the same number of windows as a fixed stride
        dataset at the same stride.
    Args:
        num_frames: Number of raw frames in the sequence.
        window_size: Frames per window.
        overlap: Frames shared between consecutive windows.
        stride: Raw frame spacing within a window.
    Returns:
        An array of shape (num_windows, window_size) of raw frame indices.
    """
    num_strided = -(-num_frames // stride)  # ceil, i.e. len(arange(0, n, stride))
    step = window_size - overlap

    windows = []
    start = 0
    while start < num_strided - window_size:
        first = start * stride
        windows.append(np.arange(first, first + window_size * stride, stride))
        start += step

    if not windows:
        return np.empty((0, window_size), dtype=int)
    return np.vstack(windows)


def sample_random_window(num_frames, window_size, stride_range, rng):
    """ Draws one window with a random frame spacing and a random start offset.
        The stride is drawn uniformly from stride_range and clamped down to whatever
        the sequence can accommodate. The start is then uniform over every position
        that fits, which is what lets the stream reach frames the fixed window grid
        never lands on.
    Args:
        num_frames: Number of frames in the sequence.
        window_size: Frames per window.
        stride_range: The (low, high) spacing to draw from, inclusive.
        rng: The generator to draw with.
    Returns:
        The window's frame indices.
    """
    lo, hi = stride_range
    hi = min(hi, max(lo, (num_frames - 2) // max(window_size - 1, 1)))
    stride = int(rng.integers(lo, hi + 1))

    max_start = num_frames - 1 - (window_size - 1) * stride
    if max_start <= 0:
        stride = 1
        max_start = num_frames - 1 - (window_size - 1)
    start = int(rng.integers(0, max_start))

    return np.arange(start, start + window_size * stride, stride)


def create_overlapping_windows(arr, window_size, overlap):
    """ Cuts overlapping windows over an index array.
    Args:
        arr: The indices to cut, usually a frame id array.
        window_size: Frames per window.
        overlap: Frames shared between consecutive windows.
    Returns:
        An array of shape (num_windows, window_size).
    """
    arr = np.asarray(arr)
    n = arr.shape[0]

    step = window_size - overlap
    windows = []

    start = 0
    # Strict stopping condition, kept so window counts match earlier runs.
    while start < n - window_size:
        windows.append(arr[start:start + window_size])
        start += step

    if not windows:
        return np.empty((0, window_size), dtype=arr.dtype)
    return np.vstack(windows)
