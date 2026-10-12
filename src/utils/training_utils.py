""" This module holds the schedule, the transforms and the trajectory helpers. """
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as R
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms


def unwrap_model(model):
    """ Returns the bare module behind the DDP and torch.compile wrappers.
        torch.compile returns an OptimizedModule that holds the real model as a
        submodule named _orig_mod and does not override state_dict, so saving the
        wrapper's state_dict prefixes every key with "_orig_mod.". Unwrapping first
        writes keys that load into a freshly built model.
    Args:
        model: A model, wrapped in DDP or torch.compile or neither.
    Returns:
        The innermost module.
    """
    model = model.module if isinstance(model, DDP) else model
    return getattr(model, "_orig_mod", model)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, step: int, config: dict) -> float:
    """ Sets the learning rate from a cosine schedule with linear warm-up.
        Training has no epochs, so the rate moves every iteration. Warm-up starts at
        lr / warmup_steps rather than at 0.
    Args:
        optimizer: The optimizer whose param groups are updated in place.
        step: The global optimizer step.
        config: The optimization section, read for lr, min_lr, warmup_steps and
            train_steps.
    Returns:
        The rate that was set.
    """
    if step < config["warmup_steps"]:
        lr = config["lr"] * (step + 1) / config["warmup_steps"]
    else:
        progress = (step - config["warmup_steps"]) / max(
            1, config["train_steps"] - config["warmup_steps"])
        progress = min(1.0, progress)
        lr = config["min_lr"] + (config["lr"] - config["min_lr"]) * 0.5 * (
            1. + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr


def setup_output_paths(output_path: str, folder_names: list):
    """ Creates the run directory and the subdirectories listed.
        Only needed for writers that do not create their own parents, which in this
        repo is torch.save.
    Args:
        output_path: The run directory.
        folder_names: Subdirectories to create inside it.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for folder_name in folder_names:
        (output_path / folder_name).mkdir(exist_ok=True)


def split_sequences(sequences, sizes, rank, world_size):
    """ Splits sequences across ranks by greedy longest-first bin packing.
        Each rank ends up with a similar total frame count rather than a similar
        number of sequences, so the ranks finish evaluation at roughly the same time.
    Args:
        sequences: The sequence names to split.
        sizes: Each sequence's window count, in the same order.
        rank: The rank whose share to return.
        world_size: Number of ranks to split across.
    Returns:
        The sequence names assigned to this rank.
    """
    sorted_indices = np.argsort(sizes)[::-1]
    sorted_sequences = [sequences[i] for i in sorted_indices]
    sorted_sizes = [sizes[i] for i in sorted_indices]

    worker_bins = [[] for _ in range(world_size)]
    worker_loads = np.zeros(world_size)

    for i in range(len(sorted_sequences)):
        least_loaded_worker = np.argmin(worker_loads)
        worker_bins[least_loaded_worker].append(sorted_sequences[i])
        worker_loads[least_loaded_worker] += sorted_sizes[i]

    return worker_bins[rank]


def average_SE3(transforms, conf):
    """ Averages several estimates of the same edge under their confidences.
    Args:
        transforms: Estimates of one edge, of shape (K, 4, 4).
        conf: Their weights, either one column or separate rotation and translation
            columns.
    Returns:
        The averaged transform, of shape (4, 4).
    """
    if conf.shape[1] == 2:
        rot_weights = conf[:, 0] / (np.sum(conf[:, 0]) + 1e-6)
        trans_weights = conf[:, 1] / (np.sum(conf[:, 1]) + 1e-6)
    else:
        rot_weights = conf / (np.sum(conf) + 1e-6)
        trans_weights = conf / (np.sum(conf) + 1e-6)
    rot_weights = rot_weights[:, np.newaxis]
    trans_weights = trans_weights[:, np.newaxis]

    t = transforms[:, :3, 3]
    t_avg = np.sum(t * trans_weights, axis=0)

    Rmats = transforms[:, :3, :3]
    Rmats_avg = R.from_matrix(Rmats).mean(weights=rot_weights[:, 0]).as_matrix()

    T_avg = np.eye(4)
    T_avg[:3, :3] = Rmats_avg
    T_avg[:3, 3] = t_avg
    return T_avg


def recover_trajectory(poses, frame_ids, confidences):
    """ Chains predicted relative poses into one trajectory.
        Windows overlap, so an edge can be predicted more than once. Duplicate
        estimates are averaged under their confidences before the chain is built.
    Args:
        poses: Predicted relative poses of shape (n, 4, 4).
        frame_ids: The frame ids of each window, of shape (m, d).
        confidences: One weight per predicted edge.
    Returns:
        The trajectory, of shape (number of edges + 1, 4, 4).
    """
    num_tuples, tuple_size = frame_ids.shape
    edges = {}  # (src, dst) -> [(transform, conf), ...]
    idx = 0
    for i in range(num_tuples):
        ids = frame_ids[i]
        for j in range(tuple_size - 1):
            src = int(ids[j].item())
            dst = int(ids[j + 1].item())
            conf = confidences[idx]
            Tij = poses[idx]
            if (src, dst) not in edges:
                edges[(src, dst)] = []
            edges[(src, dst)].append((Tij, conf))
            idx += 1

    averaged = {}
    for edge, data in edges.items():
        if len(data) > 1:
            pred_poses = np.stack([d[0] for d in data])
            conf = np.array([d[1] for d in data])
            pose = average_SE3(pred_poses, conf)
        else:
            pose, _ = data[0]
        averaged[edge] = pose

    traj = [np.eye(4)]
    for _, T_rel in averaged.items():
        traj.append(traj[-1] @ T_rel)

    return np.stack(traj)


class AspectPreservingCropAndResize(torch.nn.Module):
    """ Crops to a square on the shortest edge before resizing.
        This keeps the x and y scaling uniform, so the resize does not distort the
        optical geometry the poses were measured in.
    """

    def __init__(self, output_size, scale=(0.8, 1.0), train=False):
        super().__init__()
        self.scale = scale
        self.is_train = train
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, img):
        w, h = img.size

        min_edge = min(w, h)
        img = TF.center_crop(img, (min_edge, min_edge))

        if self.is_train:
            scale_factor = random.uniform(self.scale[0], self.scale[1])
            crop_size = int(min_edge * scale_factor)
            img = TF.center_crop(img, (crop_size, crop_size))

        return TF.resize(img, self.output_size,
                         interpolation=TF.InterpolationMode.BILINEAR, antialias=True)


class WindowTransform(torch.nn.Module):
    """ Training augmentation drawn once per window rather than once per frame.
        Zooming each frame independently reads as forward or backward motion the
        ground truth does not contain, and a window mixing grayscale and colour frames
        is something no camera produces. Exposure and focus drift gradually too, so
        colour jitter and blur are drawn per window as well.
    """

    # Read by BaseDataset.apply_transform to pass the whole window in one call.
    per_window = True

    def __init__(self, img_size, norm_mean, norm_std, scale=(0.8, 1.0),
                 photometric_per_frame=False, geometry_per_frame=False):
        super().__init__()
        # train=False: the zoom is drawn per window in _apply instead.
        self.crop_resize = AspectPreservingCropAndResize(
            output_size=img_size, scale=scale, train=False)
        self.scale = scale
        self.jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.blur_p, self.blur_kernel, self.blur_sigma = 0.5, 3, (0.05, 0.7)
        self.gray_p = 0.2
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
        self.photometric_per_frame = photometric_per_frame
        self.geometry_per_frame = geometry_per_frame

    def _sample_geometry(self):
        return random.uniform(self.scale[0], self.scale[1])

    def _sample_photometric(self):
        # get_params returns the op order plus the drawn factors, which is how
        # ColorJitter.forward randomises internally. Reusing it keeps the jitter
        # identical in distribution to the per-frame version.
        params = self.jitter.get_params(
            self.jitter.brightness, self.jitter.contrast,
            self.jitter.saturation, self.jitter.hue)
        blur_sigma = (random.uniform(*self.blur_sigma)
                      if random.random() < self.blur_p else None)
        return {"jitter": params, "blur_sigma": blur_sigma,
                "gray": random.random() < self.gray_p}

    def _apply(self, img, scale_factor, photometric):
        # Square crop -> zoom -> resize, mirroring AspectPreservingCropAndResize.
        w, h = img.size
        min_edge = min(w, h)
        img = TF.center_crop(img, (min_edge, min_edge))
        img = TF.center_crop(img, (int(min_edge * scale_factor),) * 2)
        img = TF.resize(img, self.crop_resize.output_size,
                        interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        fn_idx, brightness, contrast, saturation, hue = photometric["jitter"]
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                img = TF.adjust_brightness(img, brightness)
            elif fn_id == 1 and contrast is not None:
                img = TF.adjust_contrast(img, contrast)
            elif fn_id == 2 and saturation is not None:
                img = TF.adjust_saturation(img, saturation)
            elif fn_id == 3 and hue is not None:
                img = TF.adjust_hue(img, hue)

        if photometric["blur_sigma"] is not None:
            img = TF.gaussian_blur(img, [self.blur_kernel, self.blur_kernel],
                                   [photometric["blur_sigma"], photometric["blur_sigma"]])
        if photometric["gray"]:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        return self.normalize(self.to_tensor(img))

    def forward(self, images):
        geometry = None if self.geometry_per_frame else self._sample_geometry()
        photometric = None if self.photometric_per_frame else self._sample_photometric()
        return [self._apply(img,
                            self._sample_geometry() if geometry is None else geometry,
                            self._sample_photometric() if photometric is None else photometric)
                for img in images]


DEFAULT_NORM_MEAN = [0.34721234, 0.36705238, 0.36066107]
DEFAULT_NORM_STD = [0.30737526, 0.31515116, 0.32020183]


def get_transforms(img_size, train=False, per_window=True, photometric_per_frame=False,
                   norm_mean=None, norm_std=None):
    """ Builds the train or the evaluation image transform.
        The normalisation defaults to the training mixture's statistics. A frozen
        encoder pretrained on another input distribution needs its own, which is the
        only part of the transform that varies per backbone.
    Args:
        img_size: Output resolution.
        train: Whether to include the training augmentations.
        per_window: Whether to draw the augmentation once per window.
        photometric_per_frame: Whether to draw colour jitter and blur per frame.
        norm_mean: Channel means, defaulting to the training mixture's.
        norm_std: Channel standard deviations, defaulting to the training mixture's.
    Returns:
        The transform.
    """

    norm_mean = DEFAULT_NORM_MEAN if norm_mean is None else list(norm_mean)
    norm_std = DEFAULT_NORM_STD if norm_std is None else list(norm_std)

    if train and per_window:
        return WindowTransform(img_size, norm_mean, norm_std, scale=(0.8, 1.0),
                               photometric_per_frame=photometric_per_frame)

    if train:
        return transforms.Compose([
            # Crop/resize first so the per-pixel augmentations below run on the
            # ~224px output rather than the native (e.g. ~640px) source image.
            AspectPreservingCropAndResize(
                output_size=img_size, scale=(0.8, 1.0), train=True),

            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([
                # kernel_size/sigma scaled down from (5, (0.1, 2.0)) by ~224/640 so
                # blur strength relative to the image survives running post-resize.
                transforms.GaussianBlur(kernel_size=3, sigma=(0.05, 0.7))
            ], p=0.5),

            # Simulates a grayscale sensor, which is needed to generalise to EuRoC.
            transforms.RandomGrayscale(p=0.2),

            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        return transforms.Compose([
            AspectPreservingCropAndResize(output_size=img_size, train=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])


def setup_seed(seed: int) -> None:
    """ Seeds every generator the training loop touches and pins cuDNN.
        deterministic is on and benchmark is off, which is what lets a rerun of a
        config on fixed hardware reproduce bit-identically. Nothing else in the repo
        changes either flag, so whatever this sets is what the run uses.
    Args:
        seed: The base seed. Callers offset it by the rank.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_dict(loss_dict: dict):
    """ Averages the list-valued entries of a dictionary.
    Args:
        loss_dict: Metric name to either a list of values or a single float.
    Returns:
        The same keys with one float each. Entries of any other type are dropped.
    """
    avg_dict = defaultdict(float)
    for key, value in loss_dict.items():
        if isinstance(value, list):
            avg_dict[key] = float(np.mean(np.array(value)))
        elif isinstance(value, float):
            avg_dict[key] = value
    return avg_dict
