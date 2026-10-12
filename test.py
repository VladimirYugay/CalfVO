""" This module evaluates a trained checkpoint on the config's test datasets. """
import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch._dynamo
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model import get_model
from src.utils import io_utils, training_utils
from train import (load_sequence_datasets_from_config,
                   summarize_metrics_per_dataset, test)


def main():
    """ Entry point for multi-node multi-GPU evaluation.
        Sets up the process group, builds the model, loads checkpoint_best and scores
        every test sequence, then summarises the result. This is the same pass train.py
        runs at the end of a run, against a checkpoint chosen on the command line.
    """
    # Arguments first, so --help works without torchrun.
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="Run output directory, or the checkpoints/ dir inside it. "
                             "Without it the released checkpoint for the config's "
                             "backbone is used, which needs --config_path.")
    parser.add_argument('--config_path', type=str, default=None,
                        help="Config to evaluate with. Defaults to the run's own "
                             "config.yaml. Point it at another file to score a trained "
                             "model under a different test protocol, such as other "
                             "splits, stride or overlap.")
    parser.add_argument('--output_path', type=str, default=None,
                        help="Where to write metrics and trajectories. Defaults to "
                             "the config's run directory with _testing appended, so "
                             "scoring a run does not overwrite what it wrote itself.")
    args = parser.parse_args()

    # A long timeout rather than the 10 minute NCCL default. Test sequences are split
    # by size, so ranks waiting at the barrier can wait a long time for the slowest.
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
        # Accept either the run directory or its checkpoints/ subdirectory.
        best_ckpt_path = checkpoint_path / "checkpoint_best.pth"
        if not best_ckpt_path.exists():
            best_ckpt_path = checkpoint_path / "checkpoints" / "checkpoint_best.pth"
        config_file = (Path(args.config_path) if args.config_path
                       else checkpoint_path / "config.yaml")
    elif args.config_path is None:
        raise SystemExit("pass --config_path to score a released checkpoint, or "
                         "--checkpoint_path to score a run of your own")
    else:
        best_ckpt_path = None
        config_file = Path(args.config_path)

    config = io_utils.read_yaml_file(config_file)
    training_utils.setup_seed(config["seed"] + global_rank)

    # One rank resolves first, so a release that is not in the cache yet is
    # downloaded once rather than by every rank at the same time.
    backbone = config["model"].get("backbone", "crocov2_dust3r")
    if global_rank == 0:
        best_ckpt_path = io_utils.resolve_checkpoint(best_ckpt_path, backbone)
    dist.barrier()
    best_ckpt_path = io_utils.resolve_checkpoint(best_ckpt_path, backbone)

    # Propagate the window length from the model to every dataset.
    config["model"]["num_views"] = config["num_views"]
    for dataset_name in config["test_datasets"]:
        config["test_datasets"][dataset_name]["num_views"] = config["num_views"]
        config["test_datasets"][dataset_name]["pose_stats"] = config["pose_stats"]

    logging_cfg = config["logging"]
    run_path = (Path(logging_cfg["output_dir"])
                / logging_cfg["experiment_group"]
                / logging_cfg["experiment_name"])
    output_path = (Path(args.output_path) if args.output_path
                   else Path(str(run_path) + "_testing"))
    if global_rank == 0:
        print("Output path", output_path)
        io_utils.init_wandb(
            logging_cfg,
            project=logging_cfg["project_name"],
            config=config,
            name=output_path.name,
            group=logging_cfg.get("experiment_group", "default_group"),
            settings=wandb.Settings(code_dir="."))
        # No setup_output_paths call, because every writer below creates its own
        # directory and nothing here writes a checkpoint.
        io_utils.save_dict_to_yaml(config, "config.yaml", directory=output_path)

    opt_cfg = config["optimization"]
    model = get_model(config['model']).to(device)

    # The compiled RoPE2D CUDA kernel in croco/models/curope is wrapped in a custom
    # torch.autograd.Function, which DDPOptimizer's graph splitting cannot decompose.
    # It fails with a "higher order op" error.
    torch._dynamo.config.optimize_ddp = False

    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Must follow the backbone's normalisation, same as train.py. Otherwise a
    # standalone evaluation feeds the frozen encoder a different input distribution
    # than the one it was trained under.
    test_transforms = training_utils.get_transforms(
        config['model']["img_size"], train=False,
        norm_mean=config['model'].get("norm_mean"),
        norm_std=config['model'].get("norm_std"))

    test_sequence_datasets = load_sequence_datasets_from_config(
        config["test_datasets"], global_rank, world_size, test_transforms)

    print(f"Rank {global_rank} - Test samples: "
          f"{sum(len(d) for d in test_sequence_datasets)}")

    print(f"Loading {best_ckpt_path}")
    best_ckpt = torch.load(
        best_ckpt_path, map_location=device, weights_only=False)
    training_utils.unwrap_model(model).load_state_dict(
        best_ckpt["model_state_dict"])

    if global_rank == 0:
        print("Using GPUS: ", world_size)
        print("Model size: ", sum(p.numel() for p in model.parameters()))
        print("Number of test samples:",
              sum(len(d) for d in test_sequence_datasets))
        print(20 * "--" + " Testing " + 20 * "--")

    # Tagged with the checkpoint's own step, matching train.py's final pass.
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
