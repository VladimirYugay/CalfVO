""" This module exposes the dataset loaders and the factory that builds them. """
import torch

from src.datasets.base_dataset import BaseDataset
from src.datasets.scannet import ScanNetMatrixDataset

__all__ = ["BaseDataset", "ScanNetMatrixDataset", "get_dataset"]


def get_dataset(dataset_name: str, dataset_config: dict, transforms,
                scene_names: list = []):
    """ Builds the loader for one corpus.
    Args:
        dataset_name: The corpus's config key, which also selects the class.
        dataset_config: That corpus's section of the run config.
        transforms: The image transform to apply to every window.
        scene_names: Sequences to read. Defaults to every name in the split file.
    Returns:
        The dataset.
    """
    if not scene_names:
        with open(dataset_config["split_file_path"], 'r') as f:
            scene_names = sorted([line.strip() for line in f.readlines()])

    # Only ScanNet needs its own loader. Every other corpus reads the same way.
    # EuRoC's ground truth covers a shorter span than the image stream, so the frames
    # outside it hold a frozen copy of the nearest pose and read as zero motion. That
    # is why EuRoC is a test set here rather than a training one.
    if dataset_name == "scannet_matrix":
        dataset_class = ScanNetMatrixDataset
    else:
        dataset_class = BaseDataset

    pose_stats = {key: torch.tensor(value)
                  for key, value in dataset_config["pose_stats"].items()}

    return dataset_class(
        dataset_path=dataset_config["dataset_path"],
        scene_names=scene_names,
        window_size=dataset_config["num_views"],
        overlap=dataset_config["overlap"],
        transforms=transforms,
        pose_stats=pose_stats,
        stride=dataset_config["stride"],
        virtual_length=dataset_config.get("virtual_length", None),
        scene_sampling=dataset_config.get("scene_sampling", "proportional")
    )
