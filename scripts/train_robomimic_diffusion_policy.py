#!/usr/bin/env python3
"""Train and flatten the Lift MH robomimic diffusion-policy artifacts.

This script launches robomimic diffusion-policy training for the Lift MH task,
then flattens the raw robomimic run directory into a single output directory
with selected checkpoints and rollout summaries.

Example commands:

1. Run with defaults, which trains on
   `data/robomimic/lift/mh/image_v15.hdf5` using config
   `data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json`
   and writes a date-stamped output directory under `data/policy/`:

   ```bash
   python3 scripts/train_robomimic_diffusion_policy.py
   ```

2. Train for 100 epochs and write artifacts to a fixed output directory:

   ```bash
   python3 scripts/train_robomimic_diffusion_policy.py \
       --dataset data/robomimic/lift/mh/image_v15.hdf5 \
       --config data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json \
       --epochs 100 \
       --device cuda:0 \
       --output-dir data/policy/rmimic-lift-mh-image-v15-diffusion_custom \
       --overwrite
   ```
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOMIMIC_ROOT = REPO_ROOT / "third_party" / "robomimic"
for candidate in (REPO_ROOT, ROBOMIMIC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "robomimic" / "lift" / "mh" / "image_v15.hdf5"
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "data" / "policy" / "rmimic-lift-mh-image-v15-diffusion_260123" / "config.json"
)
DEFAULT_OUTPUT_STEM = "rmimic-lift-mh-image-v15-diffusion"
TARGET_POLICY_EPOCHS = (40, 60, 80, 100, 200, 300)
ROLLOUT_RATE_EPOCHS = 20


def _default_output_dir() -> Path:
    return REPO_ROOT / "data" / "policy" / f"{DEFAULT_OUTPUT_STEM}_{datetime.now().strftime('%y%m%d')}"


def _resolve_file(path: Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _resolve_output_dir(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_config(config_path: Path) -> Any:
    from robomimic.config import config_factory

    ext_cfg = json.loads(config_path.read_text())
    config = config_factory(ext_cfg["algo_name"])
    with config.unlocked():
        config.update(ext_cfg)
    return config


def _override_train_data(config: Any, dataset_path: Path) -> None:
    dataset_entry = {"path": str(dataset_path)}
    current = config.train.data
    if isinstance(current, list) and current:
        dataset_entry.update(dict(current[0]))
        dataset_entry["path"] = str(dataset_path)
    config.train.data = [dataset_entry]


def _configure_run(
    config: Any,
    *,
    dataset_path: Path,
    staging_output_root: Path,
    total_epochs: int,
) -> list[int]:
    selected_epochs = [epoch for epoch in TARGET_POLICY_EPOCHS if epoch <= total_epochs]
    with config.values_unlocked():
        _override_train_data(config, dataset_path)
        config.train.output_dir = str(staging_output_root)
        config.train.num_epochs = int(total_epochs)
        config.experiment.logging.terminal_output_to_txt = True
        config.experiment.logging.log_wandb = False
        config.experiment.save.enabled = True
        config.experiment.save.every_n_seconds = None
        config.experiment.save.every_n_epochs = None
        config.experiment.save.epochs = list(selected_epochs)
        config.experiment.save.on_best_validation = False
        config.experiment.save.on_best_rollout_return = False
        config.experiment.save.on_best_rollout_success_rate = False
        config.experiment.rollout.enabled = True
        config.experiment.rollout.rate = ROLLOUT_RATE_EPOCHS
    return selected_epochs


def _resolve_device(config: Any, device_arg: str | None) -> "torch.device":
    import torch
    import robomimic.utils.torch_utils as TorchUtils

    if device_arg is None:
        return TorchUtils.get_torch_device(try_to_use_cuda=bool(config.train.cuda))

    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device {device_arg!r}, but torch.cuda.is_available() is False."
            )
        torch.backends.cudnn.benchmark = True
    return device


def _install_diffusers_ema_compat() -> None:
    import inspect

    import torch
    import diffusers.training_utils as diffusers_training_utils

    signature = inspect.signature(diffusers_training_utils.EMAModel.__init__)
    if "model" in signature.parameters:
        return

    base_ema_model = diffusers_training_utils.EMAModel

    class LegacyEMAModel(base_ema_model):
        def __init__(self, parameters=None, *args, model=None, **kwargs):
            module = None
            if model is not None:
                module = model
                parameters = model.parameters()
            elif isinstance(parameters, torch.nn.Module):
                module = parameters
                parameters = parameters.parameters()

            if parameters is None:
                raise TypeError("LegacyEMAModel requires `parameters` or `model`.")

            super().__init__(parameters, *args, **kwargs)

            self.averaged_model = copy.deepcopy(module) if module is not None else None
            if self.averaged_model is not None:
                self.averaged_model.requires_grad_(False)
                self.averaged_model.eval()
                self._sync_averaged_model(module)

        def _sync_averaged_model(self, module) -> None:
            if self.averaged_model is None or module is None:
                return

            for averaged_param, shadow_param in zip(
                self.averaged_model.parameters(), self.shadow_params
            ):
                averaged_param.copy_(shadow_param)
            for averaged_buffer, live_buffer in zip(
                self.averaged_model.buffers(), module.buffers()
            ):
                averaged_buffer.copy_(live_buffer)

        @torch.no_grad()
        def step(self, parameters):
            module = parameters if isinstance(parameters, torch.nn.Module) else None
            super().step(parameters)
            self._sync_averaged_model(module)

        def to(self, *args, **kwargs):
            result = super().to(*args, **kwargs)
            if self.averaged_model is not None:
                self.averaged_model.to(*args, **kwargs)
            return result

    diffusers_training_utils.EMAModel = LegacyEMAModel


def _find_run_dir(staging_output_root: Path, experiment_name: str) -> Path:
    experiment_dir = staging_output_root / experiment_name
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Expected robomimic experiment directory at {experiment_dir}")

    candidates = sorted(path for path in experiment_dir.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No timestamped robomimic runs found under {experiment_dir}")
    return candidates[-1]


def _parse_rollout_success_rates(log_path: Path) -> dict[int, float]:
    success_rates: dict[int, float] = {}
    current_epoch: int | None = None
    epoch_pattern = re.compile(r"^Epoch (\d+) Rollouts took ")
    success_pattern = re.compile(r'"Success_Rate": ([0-9eE+\-.]+)')

    for line in log_path.read_text().splitlines():
        epoch_match = epoch_pattern.match(line)
        if epoch_match is not None:
            current_epoch = int(epoch_match.group(1))
            continue

        if current_epoch is None:
            continue

        success_match = success_pattern.search(line)
        if success_match is not None:
            success_rates[current_epoch] = float(success_match.group(1))
            current_epoch = None

    return success_rates


def _write_report_csv(
    output_path: Path,
    *,
    experiment_name: str,
    env_key: str,
    success_rates: dict[int, float],
) -> None:
    metric_name = f"{experiment_name} - Rollout/Success_Rate/{env_key}/mean"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(
            [
                "Step",
                metric_name,
                f"{metric_name}__MIN",
                f"{metric_name}__MAX",
            ]
        )
        for epoch in sorted(success_rates):
            value = success_rates[epoch]
            writer.writerow([epoch, value, value, value])


def _copy_selected_videos(videos_src: Path, videos_dst: Path, *, env_key: str, epochs: list[int]) -> None:
    videos_dst.mkdir(parents=True, exist_ok=True)
    if not videos_src.is_dir():
        return

    for epoch in epochs:
        src_path = videos_src / f"{env_key}_epoch_{epoch}.mp4"
        if src_path.is_file():
            shutil.copy2(src_path, videos_dst / src_path.name)


def _build_inference_only_checkpoint(ckpt_dict: dict[str, Any]) -> dict[str, Any]:
    required_top_level = ("algo_name", "config", "shape_metadata", "model")
    missing = [key for key in required_top_level if key not in ckpt_dict]
    if missing:
        raise KeyError(f"Checkpoint missing required inference keys: {missing}")

    model_dict = ckpt_dict["model"]
    if not isinstance(model_dict, dict):
        raise TypeError(f"Expected checkpoint['model'] to be a dict, got {type(model_dict)}")

    inference_model = {
        key: value
        for key, value in model_dict.items()
        if key not in {"optimizers", "lr_schedulers"}
    }
    if "nets" not in inference_model:
        raise KeyError("Checkpoint model payload is missing required 'nets' weights.")
    inference_model.setdefault("optimizers", {})
    inference_model.setdefault("lr_schedulers", {})

    inference_ckpt = {
        "algo_name": ckpt_dict["algo_name"],
        "config": ckpt_dict["config"],
        "shape_metadata": ckpt_dict["shape_metadata"],
        "model": inference_model,
    }
    for optional_key in (
        "env_metadata",
        "obs_normalization_stats",
        "action_normalization_stats",
    ):
        if optional_key in ckpt_dict:
            inference_ckpt[optional_key] = ckpt_dict[optional_key]
    return inference_ckpt


def _save_inference_only_checkpoint(src_path: Path, dst_path: Path) -> None:
    import torch

    ckpt_dict = torch.load(src_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt_dict, dict):
        raise TypeError(f"Expected checkpoint {src_path} to deserialize into a dict.")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_build_inference_only_checkpoint(ckpt_dict), dst_path)


def _copy_selected_checkpoints(
    models_src: Path,
    models_dst: Path,
    *,
    epochs: list[int],
    success_rates: dict[int, float],
) -> None:
    models_dst.mkdir(parents=True, exist_ok=True)
    if not models_src.is_dir():
        raise FileNotFoundError(f"Expected checkpoint directory at {models_src}")

    for epoch in epochs:
        src_path = models_src / f"model_epoch_{epoch}.pth"
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing target-policy checkpoint: {src_path}")
        if epoch not in success_rates:
            raise ValueError(f"Missing rollout success rate for epoch {epoch} in training log.")

        success_token = f"{success_rates[epoch]:.2f}"
        dst_name = f"model_epoch_{epoch}_success{success_token}.pth"
        _save_inference_only_checkpoint(src_path, models_dst / dst_name)


def _materialize_flat_output(
    run_dir: Path,
    output_dir: Path,
    *,
    experiment_name: str,
    env_key: str,
    selected_epochs: list[int],
    success_rates: dict[int, float],
) -> None:
    flat_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_dir.name}_flat_", dir=str(output_dir.parent))
    ).resolve()
    try:
        shutil.copy2(run_dir / "config.json", flat_dir / "config.json")
        _save_inference_only_checkpoint(run_dir / "last.pth", flat_dir / "last.pth")
        _save_inference_only_checkpoint(run_dir / "last_bak.pth", flat_dir / "last_bak.pth")
        shutil.copytree(run_dir / "logs", flat_dir / "logs")
        _copy_selected_checkpoints(
            run_dir / "models",
            flat_dir / "models",
            epochs=selected_epochs,
            success_rates=success_rates,
        )
        _copy_selected_videos(
            run_dir / "videos",
            flat_dir / "videos",
            env_key=env_key,
            epochs=selected_epochs,
        )
        _write_report_csv(
            flat_dir / "report.csv",
            experiment_name=experiment_name,
            env_key=env_key,
            success_rates=success_rates,
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        flat_dir.rename(output_dir)
    except Exception:
        if flat_dir.exists():
            shutil.rmtree(flat_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Lift MH robomimic diffusion policy and flatten the run artifacts."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    mpl_config_dir = Path(tempfile.gettempdir()) / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    dataset_path = _resolve_file(args.dataset, "Robomimic dataset")
    config_path = _resolve_file(args.config, "Training config")
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
        )

    config = _load_config(config_path)
    total_epochs = int(config.train.num_epochs if args.epochs is None else args.epochs)
    if total_epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {total_epochs}.")

    staging_root = Path(
        tempfile.mkdtemp(prefix=f"{output_dir.name}_raw_", dir=str(output_dir.parent))
    ).resolve()
    cleanup_staging = False
    try:
        selected_epochs = _configure_run(
            config,
            dataset_path=dataset_path,
            staging_output_root=staging_root,
            total_epochs=total_epochs,
        )
        device = _resolve_device(config, args.device)
        with config.values_unlocked():
            config.train.cuda = device.type == "cuda"
        config.lock()

        print(f"Training dataset: {dataset_path}")
        print(f"Reference config: {config_path}")
        print(f"Final output dir: {output_dir}")
        print(f"Raw robomimic staging dir: {staging_root}")
        print(f"Target checkpoints: {selected_epochs}")
        print(f"Device: {device}")

        _install_diffusers_ema_compat()
        from third_party.robomimic.robomimic.scripts.train import train as robomimic_train

        robomimic_train(config, device=device, resume=False)

        run_dir = _find_run_dir(staging_root, config.experiment.name)
        log_path = run_dir / "logs" / "log.txt"
        success_rates = _parse_rollout_success_rates(log_path)
        _materialize_flat_output(
            run_dir,
            output_dir,
            experiment_name=config.experiment.name,
            env_key=dataset_path.stem,
            selected_epochs=selected_epochs,
            success_rates=success_rates,
        )
        cleanup_staging = True
        print(f"Saved flattened policy run to {output_dir}")
    except Exception:
        print(f"Training artifacts preserved in staging dir for debugging: {staging_root}")
        raise
    finally:
        if cleanup_staging and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    main()
