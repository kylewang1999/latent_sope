"""Shared utilities for training, rollout, and checkpoint code.

This file consolidates the legacy ``src.latent_sope.utils.common`` and
``src.latent_sope.utils.misc`` helpers into a single import surface.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import rich.logging as rlogging

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_D4RL_DATA_DIR = Path(os.environ["D4RL_DATASET_DIR"]) if "D4RL_DATASET_DIR" in os.environ else _REPO_ROOT / "data" / "d4rl"


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path = _REPO_ROOT
    src_root: Path = _REPO_ROOT / "src"
    scripts_root: Path = _REPO_ROOT / "scripts"
    docs_root: Path = _REPO_ROOT / "docs"
    logs_dir: Path = _REPO_ROOT / "logs"
    configs_dir: Path = _REPO_ROOT / "configs"
    third_party_root: Path = _REPO_ROOT / "third_party"
    safediffuser_root: Path = _REPO_ROOT / "third_party" / "safe_diffuser"
    safediffuser_d4rl_path: Path = safediffuser_root / "diffuser" / "datasets" / "d4rl.py"
    d4rl_root: Path = _REPO_ROOT / "third_party" / "d4rl"
    d4rl_data_dir: Path = _D4RL_DATA_DIR
    robomimic_root: Path = _REPO_ROOT / "third_party" / "robomimic"
    robomimic_diffusion_models_root: Path = robomimic_root / "diffusion_policy_trained_models"
    default_rollout_latents_path: Path = (
        robomimic_diffusion_models_root / "test" / "20260130145148" / "rollout_latents.h5"
    )


PATHS = ProjectPaths()


def get_repo_root() -> Path:
    return PATHS.repo_root


def make_log_dir(description: str = "", verbose: bool = True) -> str:
    suffix = f"{description}_" if description else ""
    log_dir = PATHS.logs_dir / f"{suffix}{time.strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        CONSOLE_LOGGER.info("Log directory created at %s", log_dir)
    return str(log_dir)


def get_console_logger(name: str = "wkt_sope", level: str = "INFO") -> logging.Logger:
    """Return a colored console logger without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = rlogging.RichHandler(level=level, markup=True)
    formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    for noisy_logger in ("jax", "orbax", "flax", "genesis"):
        logging.getLogger(noisy_logger).propagate = False
    return logger


def load_configs(
    config_dir: str | os.PathLike[str] | None = None,
    config_name: str = "base",
):
    import hydra

    config_root = Path(config_dir) if config_dir is not None else PATHS.configs_dir
    with hydra.initialize_config_dir(config_dir=str(config_root), version_base=None):
        cfg = hydra.compose(config_name=config_name)
    return cfg


def save_configs(cfg: Any, save_path: str | os.PathLike[str] = "./configs.yaml") -> None:
    from omegaconf import OmegaConf

    path = Path(save_path)
    path.write_text(OmegaConf.to_yaml(cfg))
    CONSOLE_LOGGER.info("Configs saved to %s", path)


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        CONSOLE_LOGGER.info("%s took %.2f seconds to execute", func.__name__, execution_time)
        return result

    return wrapper


def catch_keyboard_interrupt(message: str = "Training interrupted by user") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to handle KeyboardInterrupt gracefully."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                print(message)
                return None

        return wrapper

    return decorator


def display_video(imgs_array: np.ndarray, save: bool = False, save_path: str = "./video.mp4"):
    import imageio
    from IPython.display import HTML

    imgs_array = imgs_array.astype(np.uint8)
    if imgs_array.shape[-1] == 1:
        imgs_array = np.repeat(imgs_array, 3, axis=-1)

    if isinstance(imgs_array, np.ndarray):
        if imgs_array.ndim == 3:
            imgs_array = [imgs_array]
        elif imgs_array.ndim == 4:
            imgs_array = list(imgs_array)
        else:
            raise ValueError(f"Unexpected image array shape: {imgs_array.shape}")

    buf = io.BytesIO()
    imageio.mimsave(buf, imgs_array, fps=30, format="mp4")
    buf.seek(0)

    if save:
        if not save_path.endswith(".mp4"):
            raise ValueError("save_path must end with .mp4")
        imageio.mimsave(save_path, imgs_array, fps=30, format="mp4")
        CONSOLE_LOGGER.info("Video saved to %s", save_path)

    data = base64.b64encode(buf.getvalue()).decode("ascii")
    html = f'<video src="data:video/mp4;base64,{data}" controls></video>'
    return HTML(html)


def wandb_log_artifact(run: "Run", type: str, path: str | os.PathLike[str]) -> None:
    import wandb

    artifact = wandb.Artifact(f"{run.name}_{type}", type=type)
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def save_nnx_module(model: Any, save_dir: str | os.PathLike[str]) -> None:
    import flax.nnx as nnx
    import orbax.checkpoint as ocp

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    save_path = Path(ocp.test_utils.erase_and_create_empty(save_path))
    _, state, _ = nnx.split(model, nnx.Not(nnx.RngState), nnx.RngState)

    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(save_path / "state", args=ocp.args.StandardSave(state))
    checkpointer.wait_until_finished()
    CONSOLE_LOGGER.info("NNX module saved to %s", save_path)


def restore_nnx_module(
    build_fn: Callable[[], Any],
    ckpt_dir: str | Path,
    *,
    name: str = "state",
    rng_seed: int = 0,
) -> Any:
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    del rng_seed

    def _is_abstract(x: Any) -> bool:
        return isinstance(x, jax.ShapeDtypeStruct)

    def _to_concrete_like(template_leaf: Any) -> Any:
        return jnp.zeros(template_leaf.shape, template_leaf.dtype)

    def _canon_leaf(x: Any) -> Any:
        if isinstance(x, (int, float, bool)):
            return jnp.asarray(x)
        return x

    del _to_concrete_like
    del _canon_leaf

    path = Path(ckpt_dir) / name
    checkpointer = ocp.StandardCheckpointer()

    abstract_model = nnx.eval_shape(build_fn)
    graphdef, state_norng, state_rng = nnx.split(abstract_model, nnx.Not(nnx.RngState), nnx.RngState)
    state_restored = checkpointer.restore(path, state_norng, strict=False)
    model = nnx.merge(graphdef, state_restored, state_rng)
    return model


def save_dict_to_json(dict_to_save: dict[str, Any], save_path: str | os.PathLike[str], verbose: bool = False) -> None:
    path = Path(save_path)
    path.write_text(json.dumps(dict_to_save, indent=2))
    if verbose:
        CONSOLE_LOGGER.info("Dictionary saved to %s", path)


def set_global_seed(seed: int) -> None:
    """Best-effort global seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


@dataclass(frozen=True)
class DeviceConfig:
    """Simple device selection helper."""

    device: str = "cuda"
    dtype: str = "float32"

    def torch_dtype(self):
        import torch

        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32


def resolve_device(prefer_cuda: bool = True) -> str:
    """Return ``cuda`` if available and preferred, else ``cpu``."""
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


CONSOLE_LOGGER = get_console_logger()


__all__ = [
    "CONSOLE_LOGGER",
    "DeviceConfig",
    "PATHS",
    "ProjectPaths",
    "catch_keyboard_interrupt",
    "display_video",
    "get_console_logger",
    "get_repo_root",
    "load_configs",
    "make_log_dir",
    "resolve_device",
    "restore_nnx_module",
    "save_configs",
    "save_dict_to_json",
    "save_nnx_module",
    "set_global_seed",
    "timeit",
    "wandb_log_artifact",
]
