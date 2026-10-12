""" This module trains the pose model across GPUs and evaluates the best checkpoint. """
import argparse
import os
from collections import defaultdict
from copy import copy, deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
import torch.optim as optim
import wandb
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets import get_dataset
from src.losses import PoseLoss
from src.model import get_model
from src.utils import eval_utils, io_utils, training_utils


def get_train_val_data(config: dict, steps_per_cycle: int, val_portion: float = 0.1):
    """ Builds the endless training stream and the fixed validation set.
        The training datasets draw a random sequence and a random window on every
        access, so they have no sample list to exhaust and there is no epoch. Their
        length is how many draws happen between validations. Validation reads a twin
        with the stream switched off, so it holds the same windows every time. A set
        that resampled every pass could not be compared across runs.
    Args:
        config: The full run config.
        steps_per_cycle: Optimizer steps between validations, which sets the stream
            length so a cycle is the same regardless of how many datasets are mixed.
        val_portion: Fraction of each dataset's anchor windows held out for validation.
    Returns:
        The concatenated training stream, and a dict of dataset name to its validation
        subset.
    """
    train_transforms = training_utils.get_transforms(
        config['model']["img_size"], train=True,
        norm_mean=config['model'].get("norm_mean"),
        norm_std=config['model'].get("norm_std"))
    batch_size = config["optimization"]["batch_size"]
    total_draws = steps_per_cycle * batch_size

    streams, val_datasets, weights = [], {}, []
    for dataset_name in config["training_datasets"]:
        ds_config = dict(config["training_datasets"][dataset_name])
        ds_config["virtual_length"] = total_draws  # replaced below by the share
        stream_dataset = get_dataset(dataset_name, ds_config, train_transforms)

        val_source = copy(stream_dataset)
        val_source.virtual_length = None  # back to the fixed anchor windows
        # Seeded from the config rather than the per-rank torch seed, so every rank
        # holds out the same windows.
        generator = torch.Generator().manual_seed(config["seed"])
        order = torch.randperm(len(val_source), generator=generator)
        nb_val = round(val_portion * len(val_source))

        streams.append(stream_dataset)
        val_datasets[dataset_name] = torch.utils.data.Subset(
            val_source, order[:nb_val].tolist())
        # mix_weight splits the per-cycle draws between the datasets. It defaults to
        # the dataset's frame count, which makes every frame in the corpus equally
        # likely to be drawn.
        weights.append(float(ds_config.get(
            "mix_weight",
            sum(len(s["frame_ids"]) for s in stream_dataset.scenes.values()))))

    # Largest remainder apportionment, so the shares sum to total_draws.
    exact = [total_draws * w / sum(weights) for w in weights]
    shares = [int(e) for e in exact]
    for i in sorted(range(len(shares)),
                    key=lambda i: exact[i] - shares[i], reverse=True):
        if sum(shares) >= total_draws:
            break
        shares[i] += 1
    for stream_dataset, share in zip(streams, shares):
        stream_dataset.virtual_length = share

    if not dist.is_initialized() or dist.get_rank() == 0:
        for name, share in zip(config["training_datasets"], shares):
            print(f"Mixture: {name} -> {share}/{total_draws} draws per cycle "
                  f"({100 * share / total_draws:.1f}%)")

    # Validation stays split by dataset. A blended number cannot show which corpus
    # moved, and one dataset holds most of the anchors, so it would dominate. The
    # selection criterion below still pools them.
    return torch.utils.data.ConcatDataset(streams), val_datasets


def load_sequence_datasets_from_config(config: dict, global_rank: int,
                                       world_size: int, transforms):
    """ Builds one ordered dataset per test sequence and shards them across ranks.
        Sequences that yield no windows are skipped with a log line.
    Args:
        config: The test_datasets section of the run config.
        global_rank: Rank of this process.
        world_size: Total number of processes, used to split the sequences by size.
        transforms: The evaluation image transform.
    Returns:
        The list of datasets this rank is responsible for, each tagged with the config
        key it came from.
    """
    datasets = []
    for dataset_name in config:
        with open(config[dataset_name]["split_file_path"], 'r') as f:
            scene_names = sorted([line.strip() for line in f.readlines()])

        valid_scene_names = []
        valid_scene_sizes = []
        for scene_name in scene_names:
            dataset = get_dataset(
                dataset_name, config[dataset_name], transforms, [scene_name])
            size = len(dataset)
            if size > 0:
                valid_scene_names.append(scene_name)
                valid_scene_sizes.append(size)
            else:
                print("Skipping the sequence", scene_name, global_rank,
                      config[dataset_name]["split_file_path"])

        local_scene_names = training_utils.split_sequences(
            valid_scene_names, valid_scene_sizes, global_rank, world_size)
        print("Scene names of the rank", global_rank, local_scene_names)
        for local_scene_name in local_scene_names:
            dataset = get_dataset(
                dataset_name, config[dataset_name], transforms, [local_scene_name])
            dataset.config_key = dataset_name
            datasets.append(dataset)
    return datasets


def validate(model, val_loader, criterion, log_every_percentage: int = 10):
    """ Runs one validation pass on this GPU and reduces the result across ranks.
        The sum and the count are returned separately rather than as a mean, so the
        caller can report each dataset on its own and still pool them into one figure
        with the weighting a single concatenated loader would have given.
    Args:
        model: The model being trained.
        val_loader: Loader over this rank's shard of one dataset's validation windows.
        criterion: The pose loss.
        log_every_percentage: How often to print progress, in percent of the loader.
    Returns:
        The loss sum and the batch count, both already summed over all ranks.
    """
    model.eval()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    val_loss_sum = 0.0
    iter_count = 0

    log_interval = max(1, len(val_loader) * log_every_percentage // 100)

    with torch.no_grad():
        for batch_id, (images, gt, _) in enumerate(val_loader):
            images, gt = images.to(device), gt.to(device)
            output = model(images.float())
            loss = criterion(output, gt)
            val_loss_sum += loss.item()
            iter_count += 1

            if dist.get_rank() == 0 and batch_id % log_interval == 0:
                print("Validation Batch: {}/{}, Loss: {}".format(
                    batch_id, len(val_loader), loss.item()))

    loss_tensor = torch.tensor(val_loss_sum, device=device)
    batch_tensor = torch.tensor(iter_count, device=device)

    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(batch_tensor, op=dist.ReduceOp.SUM)

    return loss_tensor.item(), batch_tensor.item()


def train_cycle(model, train_loader, criterion, optimizer, scaler, opt_cfg: dict,
                global_step: int, log_every_percentage: int = 10):
    """ Runs one cycle of training, i.e. len(train_loader) optimizer steps off the stream.
        The learning rate is set from the global step before every iteration.
    Args:
        model: The model being trained.
        train_loader: Loader over this rank's share of the endless training stream.
        criterion: The pose loss.
        optimizer: The optimizer, whose learning rate this updates every iteration.
        scaler: The gradient scaler for mixed precision.
        opt_cfg: The optimization section of the run config, read by the LR schedule.
        global_step: Optimizer step this cycle starts from.
        log_every_percentage: How often to print progress, in percent of the loader.
    Returns:
        The mean training loss over all ranks, and the global step after this cycle.
    """
    model.train()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    cycle_loss = 0.0
    iter_count = 0
    log_interval = max(1, len(train_loader) * log_every_percentage // 100)

    for batch_id, (images, gt, _) in enumerate(train_loader):
        training_utils.adjust_learning_rate(
            optimizer, global_step + batch_id, opt_cfg)
        optimizer.zero_grad()

        images, gt = images.to(device), gt.to(device)

        with autocast("cuda"):
            output = model(images)
            loss = criterion(output, gt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        cycle_loss += loss.item()
        iter_count += 1

        if dist.get_rank() == 0 and batch_id % log_interval == 0:
            print("Step: {}, Loss: {}".format(global_step + batch_id, loss.item()))

    loss_tensor = torch.tensor(cycle_loss, device=device)
    batch_tensor = torch.tensor(iter_count, device=device)

    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(batch_tensor, op=dist.ReduceOp.SUM)

    total_train_loss = loss_tensor.item() / batch_tensor.item()

    return total_train_loss, global_step + iter_count


def train(model, train_loader, val_loaders, criterion, optimizer, config: dict,
          output_path: Path, step: int = 0, best_val: float = np.inf):
    """ Trains the model, validating and checkpointing on step boundaries.
        The loader is an endless stream of randomly drawn sequences and windows, so
        there is no epoch, and the loop runs for train_steps optimizer steps. The
        held-out sets are not touched here. main() evaluates once, after this returns.
    Args:
        model: The model to train.
        train_loader: Loader over this rank's share of the endless training stream.
        val_loaders: Dict of dataset name to that dataset's validation loader.
        criterion: The pose loss.
        optimizer: The optimizer.
        config: The full run config.
        output_path: Run directory, whose checkpoints/ subdirectory receives the
            checkpoints.
        step: Optimizer step to start from. It is non-zero when resuming and has to
            come from the caller, since restarting at 0 would re-run the schedule.
        best_val: Lowest pooled validation loss so far. Resuming with inf would
            overwrite checkpoint_best with a worse checkpoint on the next cycle.
    """
    opt_cfg = config["optimization"]
    output_path = Path(output_path)
    checkpoint_path = output_path / "checkpoints"
    train_steps, eval_every = opt_cfg["train_steps"], opt_cfg["eval_every"]
    scaler = GradScaler("cuda")

    # One pass of the loader is eval_every steps, because its dataset's virtual_length
    # is eval_every * batch_size. That keeps every rank on the same step count.
    # Uneven step counts would deadlock the all_reduce below.
    while step < train_steps:
        train_loss, step = train_cycle(model, train_loader, criterion, optimizer,
                                       scaler, opt_cfg, step)
        val_by_dataset, loss_sum, batch_count = {}, 0.0, 0
        for name, loader in val_loaders.items():
            per_sum, per_count = validate(model, loader, criterion)
            val_by_dataset[name] = per_sum / per_count
            loss_sum, batch_count = loss_sum + per_sum, batch_count + per_count
        # Pooled as a single concatenated loader would have been, so the split does
        # not change the selection criterion.
        val_loss = loss_sum / batch_count
        # Reports the learning rate the next cycle starts from. It also sets it on the
        # optimizer, which the next cycle's first iteration would do anyway.
        current_lr = training_utils.adjust_learning_rate(
            optimizer, min(step, train_steps - 1), opt_cfg)

        if dist.get_rank() == 0:
            per_dataset = "  ".join(
                f"{n}: {v:.4f}" for n, v in val_by_dataset.items())
            print(f"Step: {step}/{train_steps} - loss: {train_loss:.4f} - "
                  f"val_loss: {val_loss:.4f} ({per_dataset}) - lr: {current_lr:.3e}\n")

            # Unwrapped, so the keys load into a freshly built model without renaming.
            state = {
                "step": step,
                "model_state_dict": training_utils.unwrap_model(
                    model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
            }

            if val_loss < best_val:
                print(f"Saving new best model, loss decreased from "
                      f"{best_val:.6f} to {val_loss:.6f} \n")
                best_val = val_loss
                state["best_val"] = best_val
                torch.save(state, checkpoint_path / "checkpoint_best.pth")

            if step % opt_cfg["save_ckpt_every"] < eval_every:
                torch.save(state, checkpoint_path / f"checkpoint_{step:07}.pth")

            torch.save(state, checkpoint_path / "checkpoint_last.pth")

            wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                       "step": step, "lr": current_lr,
                       **{f"val_loss/{n}": v for n, v in val_by_dataset.items()}})


def test(model, test_datasets: list, batch_size: int, step: int,
         metrics_path: Path, trajectories_path: Path):
    """ Rolls out each test sequence and writes its metrics, plots and trajectories.
        Runs one sequence at a time, chains the predicted relative poses into a
        trajectory and scores it both unaligned and Sim(3) aligned. Results are
        grouped by config key and then scene, so the same scene listed under several
        test entries does not overwrite itself.
    Args:
        model: The model to evaluate.
        test_datasets: This rank's sequence datasets, from
            load_sequence_datasets_from_config.
        batch_size: Windows per forward pass.
        step: Training step the weights come from, used to name the written files.
        metrics_path: Directory that receives the per-scene metrics json.
        trajectories_path: Directory that receives the per-scene plots and arrays.
    """
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model.eval()
    for dataset in test_datasets:
        # num_workers is 0 because forking once the main process has used torchcodec
        # heavily segfaults. Test runs once, so decoding here costs little.
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=0)
        sequence = dataset.scene_names[0]
        # Results are grouped by config key and then scene. The same scene can be
        # listed under several test entries, such as tum_matrix_stride_1 and
        # tum_matrix_stride_3, and would otherwise overwrite each other.
        sequence_path = Path(getattr(dataset, "config_key", "")) / sequence

        pred_poses, acc_frame_ids, acc_confidences = [], [], []
        for images, gt, batch_frame_ids in test_loader:
            images, gt = images.to(device), gt.to(device)

            with torch.no_grad():
                output = model(images.float())
                pred_poses.append(output["relative_poses"])
                acc_confidences.append(output["confidences"])
                acc_frame_ids.append(batch_frame_ids)

        pred_poses = torch.cat(pred_poses, dim=0).detach()
        acc_frame_ids = torch.vstack(acc_frame_ids)
        acc_confidences = torch.vstack(acc_confidences)
        pred_poses_unnorm = dataset.postprocess_poses(pred_poses)

        # Log-variance to confidence.
        acc_confidences = acc_confidences.reshape(
            pred_poses_unnorm.shape[0], -1).cpu().numpy()
        acc_confidences = np.exp(-acc_confidences)

        pred_trajectory = training_utils.recover_trajectory(
            pred_poses_unnorm.cpu().numpy(),
            acc_frame_ids.cpu().numpy(),
            acc_confidences)

        gt_poses, scene_frame_ids = dataset.get_scene_c2ws(sequence)

        metrics, traj_plot = eval_utils.eval_trajectory(
            pred_trajectory, gt_poses[:pred_trajectory.shape[0]],
            np.arange(len(pred_trajectory)), align=False)

        if pred_trajectory.shape[0] >= 3:  # Sim(3) alignment needs at least 3 poses
            metrics_aligned, traj_plot_aligned = eval_utils.eval_trajectory(
                pred_trajectory, gt_poses[:pred_trajectory.shape[0]],
                np.arange(len(pred_trajectory)), align=True)
        else:
            metrics_aligned = metrics
            traj_plot_aligned = traj_plot

        all_metrics = deepcopy(metrics)
        for metric_name, metric_value in metrics_aligned.items():
            all_metrics[f"aligned_{metric_name}"] = metric_value

        io_utils.save_dict_to_json(all_metrics, f"metrics_step_{step:07}.json",
                                   directory=metrics_path / sequence_path)

        plot_img = Image.new(
            "RGB",
            (traj_plot.width + traj_plot_aligned.width, traj_plot.height))
        plot_img.paste(traj_plot, (0, 0))
        plot_img.paste(traj_plot_aligned, (traj_plot.width, 0))
        plot_path = trajectories_path / sequence_path
        plot_path.mkdir(parents=True, exist_ok=True)
        plot_img.save(plot_path / f"plot_step_{step:07}.png")
        np.save(plot_path / f"pred_trajectory_step_{step:07}.npy", pred_trajectory)
        np.save(plot_path / f"confidence_step_{step:07}.npy", acc_confidences)
        np.save(plot_path / f"frame_ids_step_{step:07}.npy",
                scene_frame_ids[:pred_trajectory.shape[0]])
        np.save(plot_path / f"gt_trajectory_step_{step:07}.npy",
                gt_poses[:pred_trajectory.shape[0]])


def summarize_metrics_per_dataset(datasets_config: dict, metrics_path: Path,
                                  output_path: Path, wandb_key: str):
    """ Averages the per-scene metrics of each dataset into one summary json.
        Reads the newest metrics file per scene, so a scene measured more than once
        contributes only its latest numbers.
    Args:
        datasets_config: The test_datasets section of the run config.
        metrics_path: Directory test() wrote the per-scene metrics to.
        output_path: Directory that receives the per-dataset summary json.
        wandb_key: Suffix for the wandb keys the summary is logged under.
    """
    for dataset_name in datasets_config:
        summary_metrics = defaultdict(list)
        with open(datasets_config[dataset_name]["split_file_path"], 'r') as f:
            scene_names = sorted([line.strip() for line in f.readlines()])

        dataset_metrics_path = metrics_path / dataset_name
        print(dataset_name, len(scene_names), dataset_metrics_path)

        for scene_path in sorted(dataset_metrics_path.glob("*")):
            if scene_path.parts[-1] not in scene_names:
                continue

            # A scene directory with no json is one test() did not finish, so it
            # is skipped rather than taken as the newest file.
            written = sorted(scene_path.glob("*.json"))
            if not written:
                print("No metrics for the sequence", scene_path.parts[-1])
                continue

            scene_metrics = io_utils.read_json_file(written[-1])
            for metric_name, metric_value in scene_metrics.items():
                summary_metrics[metric_name].append(metric_value)

        summary_metrics = training_utils.average_dict(summary_metrics)

        io_utils.save_dict_to_json(summary_metrics,
                                   f"{dataset_name}_test_metrics.json",
                                   directory=output_path)
        io_utils.log_dict_to_wandb(summary_metrics, f"{dataset_name}_{wandb_key}")


def main():
    """ Entry point for multi-node multi-GPU training.
        Sets up the process group, builds the model and the data, trains, then
        evaluates checkpoint_best on the held-out sets and summarises the result.
    """
    # A long timeout rather than the 10 minute NCCL default. Test sequences are split
    # by size, so ranks waiting at the barrier can wait a long time for the slowest.
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    config = io_utils.read_yaml_file(args.config_path)
    training_utils.setup_seed(config["seed"] + global_rank)

    # Propagate the window length from the model to every dataset.
    config["model"]["num_views"] = config["num_views"]
    for dataset_type in ["training_datasets", "test_datasets"]:
        for dataset_name in config[dataset_type]:
            config[dataset_type][dataset_name]["num_views"] = config["num_views"]
            config[dataset_type][dataset_name]["pose_stats"] = config["pose_stats"]

    logging_cfg = config["logging"]
    output_path = (Path(logging_cfg["output_dir"])
                   / logging_cfg["experiment_group"]
                   / logging_cfg["experiment_name"])
    if global_rank == 0:
        io_utils.init_wandb(
            logging_cfg,
            project=logging_cfg["project_name"],
            config=config,
            name=logging_cfg["experiment_name"],
            group=logging_cfg["experiment_group"],
            settings=wandb.Settings(code_dir="."))
        training_utils.setup_output_paths(output_path, ["checkpoints"])
        io_utils.save_dict_to_yaml(config, "config.yaml", directory=output_path)

    opt_cfg = config["optimization"]
    model = get_model(config['model']).to(device)
    optimizer = optim.Adam(model.parameters(), opt_cfg["lr"])

    step_init, best_val = 0, np.inf
    if config.get("checkpoint") is not None:
        checkpoint_path = config["checkpoint"]
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step_init = checkpoint.get("step", 0)
        best_val = checkpoint.get("best_val", np.inf)
        print(f"Resuming training from step {step_init}, "
              f"best val loss: {best_val:.6f}")

    # The compiled RoPE2D CUDA kernel in croco/models/curope is wrapped in a custom
    # torch.autograd.Function, which DDPOptimizer's graph splitting cannot decompose.
    # It fails with a "higher order op" error.
    torch._dynamo.config.optimize_ddp = False

    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Ranks on a node share its CPUs, so the divisor is the local world size, not the
    # global one. sched_getaffinity respects a container's CPU mask, cpu_count does not.
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", world_size))
    num_cpus = len(os.sched_getaffinity(0))
    num_workers = min(16, max(2, num_cpus // local_world_size))

    train_data, val_datasets = get_train_val_data(
        config, steps_per_cycle=opt_cfg["eval_every"])
    # No DistributedSampler, because every item is an independent random draw.
    # shuffle=True mixes the datasets, which ConcatDataset lays out end to end.
    train_loader = DataLoader(train_data, batch_size=opt_cfg["batch_size"],
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=True)

    # One loader per dataset so validation can be reported separately. Each is
    # sharded across ranks as the single concatenated loader was.
    val_loaders, val_sizes = {}, {}
    for name, dataset in val_datasets.items():
        sampler = DistributedSampler(dataset)
        val_loaders[name] = DataLoader(
            dataset, batch_size=opt_cfg["batch_size"], sampler=sampler,
            num_workers=num_workers, pin_memory=True, persistent_workers=True)
        val_sizes[name] = len(sampler)

    print(f"Rank {global_rank} - Draws per cycle: {len(train_data)}, "
          f"Validation samples: {val_sizes} (total {sum(val_sizes.values())})")

    test_transforms = training_utils.get_transforms(
        config['model']["img_size"], train=False,
        norm_mean=config['model'].get("norm_mean"),
        norm_std=config['model'].get("norm_std"))

    test_sequence_datasets = load_sequence_datasets_from_config(
        config["test_datasets"], global_rank, world_size, test_transforms)

    print(f"Rank {global_rank} - Test samples: "
          f"{sum(len(d) for d in test_sequence_datasets)}")

    criterion = PoseLoss()

    if global_rank == 0:
        print("Using GPUS: ", world_size)
        print("Model size: ", sum(p.numel() for p in model.parameters()))
        print("Model opt size: ",
              sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("Steps:", opt_cfg["train_steps"],
              "| draws per cycle:", len(train_data))
        print("Number of validation samples:",
              {n: len(d) for n, d in val_datasets.items()})
        print("Number of test samples:",
              sum(len(d) for d in test_sequence_datasets))
        print("Using workers", num_workers)
        print(20 * "--" + " Training " + 20 * "--")
    train(model, train_loader, val_loaders, criterion, optimizer, config,
          output_path, step=step_init, best_val=best_val)

    dist.barrier()

    if global_rank == 0:
        print(20 * "--" + " Testing " + 20 * "--")
    best_ckpt_path = output_path / "checkpoints" / "checkpoint_best.pth"
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    training_utils.unwrap_model(model).load_state_dict(
        best_ckpt["model_state_dict"])
    test(model, test_sequence_datasets, opt_cfg["batch_size"],
         best_ckpt.get("step", opt_cfg["train_steps"]),
         output_path / "test_metrics", output_path / "test_trajectories")

    dist.barrier()
    if global_rank == 0:
        summarize_metrics_per_dataset(
            config["test_datasets"], output_path / "test_metrics", output_path,
            "test_metrics")
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
