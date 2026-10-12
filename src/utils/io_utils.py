""" This module reads and writes the run's configs, metrics and wandb logs. """
import json
import os
import re
from pathlib import Path
from typing import Union

import plotly
import wandb
import yaml
from PIL import Image

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# The released checkpoints, one per backbone. Each carries the whole model, so a
# downloader needs nothing else.
RELEASE_REPO = "voviktyl/CalfVO"
RELEASE_CHECKPOINTS = {
    "crocov2_dust3r": "calfvo_croco.pth",
    "dinov2_large": "calfvo_dinov2.pth",
}


def resolve_checkpoint(path, backbone="crocov2_dust3r"):
    """ Returns a local checkpoint file, downloading the released one if needed.
        Passing a path uses that file. Passing nothing fetches the release for this
        backbone from the Hub, which is cached after the first call, so scoring or
        demoing the released model takes no manual download.
    Args:
        path: A local checkpoint, or None to use the release.
        backbone: Which release to fetch when path is None.
    Returns:
        The path to a checkpoint on disk.
    """
    if path is not None:
        local = Path(path)
        if not local.exists():
            raise SystemExit(f"{local} does not exist")
        return local

    if backbone not in RELEASE_CHECKPOINTS:
        raise SystemExit(
            f"no released checkpoint for backbone {backbone!r}. Pass one explicitly.")

    from huggingface_hub import hf_hub_download
    name = RELEASE_CHECKPOINTS[backbone]
    print(f"Fetching {name} from {RELEASE_REPO}")
    return Path(hf_hub_download(RELEASE_REPO, name))


def _expand_env(value, file_path):
    """ Substitutes ${VAR} from the environment throughout a loaded config.
        A missing variable raises rather than expanding to an empty string. A config
        whose dataset_path became "" would resolve to a relative path and fail much
        later, naming the split file rather than the unset variable. Paths in the
        shipped config are written out in full, so this only applies to a config that
        opts into ${VAR}.
    Args:
        value: Any part of a loaded config, walked recursively.
        file_path: The config's path, named in the error.
    Returns:
        The same structure with every reference expanded.
    """
    if isinstance(value, dict):
        return {k: _expand_env(v, file_path) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v, file_path) for v in value]
    if isinstance(value, str):
        missing = sorted({m.group(1) for m in _ENV_PATTERN.finditer(value)
                          if m.group(1) not in os.environ})
        if missing:
            raise KeyError(
                f"{file_path}: {', '.join(missing)} referenced in the config but not set "
                f"in the environment (in {value!r}). See the README for the variables "
                f"the configs expect.")
        return _ENV_PATTERN.sub(lambda m: os.environ[m.group(1)], value)
    return value


def read_yaml_file(file_path: Union[str, Path]) -> dict:
    """ Reads a YAML config and expands its environment references.
        ${VAR} anywhere in the file is substituted from the environment, so paths
        specific to one machine stay out of the config.
    Args:
        file_path: Path to the config.
    Returns:
        The parsed config.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")
    return _expand_env(config, file_path)


def read_json_file(file_path: Union[str, Path]) -> dict:
    """ Reads a JSON file into a dictionary.
    Args:
        file_path: Path to the file.
    Returns:
        The parsed contents.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error parsing JSON file {file_path}: {e}")


def mkdir_decorator(func):
    """ Creates the wrapped function's `directory` argument before calling it.
    Args:
        func: A function taking a keyword-only `directory`.
    Returns:
        The wrapped function.
    """
    def wrapper(*args, **kwargs):
        output_path = Path(kwargs["directory"])
        output_path.mkdir(parents=True, exist_ok=True)
        kwargs["directory"] = output_path
        return func(*args, **kwargs)
    return wrapper


@mkdir_decorator
def save_dict_to_yaml(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary as YAML, creating the directory if needed.
    Args:
        dictionary: The data to write.
        file_name: Name of the file inside the directory.
        directory: Where to write it.
    """
    with open(directory / file_name, "w") as f:
        yaml.dump(dictionary, f)


@mkdir_decorator
def save_dict_to_json(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary as JSON, creating the directory if needed.
    Args:
        dictionary: The data to write.
        file_name: Name of the file inside the directory.
        directory: Where to write it.
    """
    with open(directory / file_name, "w") as f:
        json.dump(dictionary, f)


def init_wandb(logging_cfg: dict, **init_kwargs) -> str:
    """ Starts the wandb run and states which mode it is running in.
        WANDB_MODE takes precedence over the config, so a run without wandb
        credentials can be started with WANDB_MODE=offline or WANDB_MODE=disabled.
        The mode and where it came from are printed, because an offline or disabled
        run sends nothing to the server and the variable is easy to leave set by
        accident. An online run with no credentials stops here rather than at an
        interactive prompt that torchrun cannot answer.
    Args:
        logging_cfg: The logging section of the run config.
        init_kwargs: The rest of the arguments for wandb.init.
    Returns:
        The mode the run was started in.
    """
    from_env = os.environ.get("WANDB_MODE")
    mode = from_env or ("offline" if logging_cfg["offline"] else "online")
    source = "WANDB_MODE" if from_env else "logging.offline in the config"
    print(f"wandb mode is {mode}, set by {source}")
    if mode != "online":
        print(f"Nothing will be sent to the wandb server in {mode} mode.")

    try:
        wandb.init(mode=mode, **init_kwargs)
    except Exception as error:
        raise SystemExit(
            f"wandb.init failed in {mode} mode with: {error}\n"
            "Run `wandb login`, or set WANDB_API_KEY, or start the run with "
            "WANDB_MODE=offline to keep the logs on disk.")
    return mode


def log_dict_to_wandb(dictionary: dict, prefix: str) -> None:
    """ Logs the floats, plotly figures and images in a dictionary to wandb.
        Values of any other type are skipped.
    Args:
        dictionary: The values to log.
        prefix: Prefix for the wandb key of every value.
    """
    wandb_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, float):
            wandb_dict[f"{prefix}/{key}"] = value
        elif isinstance(value, plotly.graph_objects.Figure):
            wandb_dict[f"{prefix}/{key}"] = value
        elif isinstance(value, Image.Image):
            wandb_dict[f"{prefix}/{key}"] = wandb.Image(value)
    wandb.log(wandb_dict)
