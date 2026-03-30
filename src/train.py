from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import ConcatDataset

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import resolve_device, set_global_seed

if TYPE_CHECKING:
    from src.robomimic_interface.dataset import (
        NormalizationStats as DatasetNormalizationStats,
        RolloutChunkDatasetConfig,
    )
    from src.sope_diffuser import (
        NormalizationStats as SopeNormalizationStats,
        SopeDiffuser,
        SopeDiffusionConfig,
    )

PathLike = Union[str, Path]


@dataclass(frozen=True)
class TrainingConfig:
    data: Sequence[PathLike]
    checkpoint_dir: Optional[PathLike] = None
    train_fraction: float = 0.8
    epochs: int = 10
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 3e-4
    lr_scheduler_enabled: bool = True
    lr_scheduler_type: str = "cosine"
    lr_scheduler_min_lr: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_steps: Optional[int] = None
    num_saves: int = 10
    num_evals: int = 50
    seed: int = 0
    device: Optional[str] = None
    prefer_cuda: bool = True
    wandb_project: Optional[str] = "wkt_sope.main"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ()


def _to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value is None or key == "metadata":
            out[key] = value
            continue
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _convert_stats(
    stats: Optional[DatasetNormalizationStats],
) -> Optional[SopeNormalizationStats]:
    from src.sope_diffuser import NormalizationStats as SopeNormalizationStats

    if stats is None:
        return None
    return SopeNormalizationStats(mean=stats.mean, std=stats.std)


def _serialize_training_config(cfg_training: TrainingConfig) -> dict[str, Any]:
    payload = asdict(cfg_training)
    payload["data"] = [str(path) for path in _as_paths(cfg_training.data)]
    payload["checkpoint_dir"] = (
        str(Path(cfg_training.checkpoint_dir))
        if cfg_training.checkpoint_dir is not None
        else None
    )
    return payload


def _save_checkpoint(
    path: Path,
    diffuser: SopeDiffuser,
    epoch: int,
    step: int,
    cfg_diffusion: SopeDiffusionConfig,
    cfg_dataset: RolloutChunkDatasetConfig,
    cfg_training: TrainingConfig,
    stats: Optional[DatasetNormalizationStats],
) -> None:
    payload = {
        "diffusion_state_dict": diffuser.diffusion.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "diffusion_config": asdict(cfg_diffusion),
        "dataset_config": asdict(cfg_dataset),
        "training_config": _serialize_training_config(cfg_training),
        "normalization_stats": (
            None if stats is None else {"mean": stats.mean, "std": stats.std}
        ),
    }
    torch.save(payload, str(path))


def _save_configs(
    path: Path,
    cfg_diffusion: SopeDiffusionConfig,
    cfg_dataset: RolloutChunkDatasetConfig,
    cfg_training: TrainingConfig,
) -> None:
    payload = {
        "diffusion_config": asdict(cfg_diffusion),
        "dataset_config": asdict(cfg_dataset),
        "training_config": _serialize_training_config(cfg_training),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _as_paths(paths: Sequence[PathLike]) -> list[Path]:
    return [p if isinstance(p, Path) else Path(p) for p in paths]


def _resolve_rollout_paths(paths: Sequence[PathLike]) -> list[Path]:
    resolved: list[Path] = []
    for p in _as_paths(paths):
        if p.is_dir():
            for ext in ("*.npz", "*.h5", "*.hdf5"):
                resolved.extend(sorted(p.glob(ext)))
        else:
            resolved.append(p)
    resolved = [p for p in resolved if p.is_file()]
    if not resolved:
        raise FileNotFoundError("No rollout files found.")
    return resolved


def _flatten_scalar_metrics(info: Any) -> dict[str, float]:
    if not isinstance(info, dict):
        return {}

    flat: dict[str, float] = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                flat[key] = float(value.item())
        elif np.isscalar(value):
            flat[key] = float(value)
    return flat


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg_training: TrainingConfig,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if not cfg_training.lr_scheduler_enabled:
        return None

    scheduler_type = cfg_training.lr_scheduler_type.lower()
    if scheduler_type != "cosine":
        raise ValueError(
            f"Unsupported lr_scheduler_type={cfg_training.lr_scheduler_type!r}; expected 'cosine'."
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps, 1),
        eta_min=cfg_training.lr_scheduler_min_lr,
    )


def _resolve_save_every(cfg_training: TrainingConfig) -> int:
    if cfg_training.num_saves <= 0:
        raise ValueError(f"num_saves must be positive, got {cfg_training.num_saves}.")
    return max(1, math.ceil(cfg_training.epochs / cfg_training.num_saves))


def _resolve_eval_every(cfg_training: TrainingConfig) -> Optional[int]:
    if cfg_training.num_evals <= 0:
        return None
    return max(1, cfg_training.epochs // cfg_training.num_evals)


def _split_rollout_paths(
    paths: Sequence[PathLike],
    seed: int,
    train_fraction: float = 0.8,
) -> tuple[list[Path], list[Path]]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(
            f"train_fraction must be strictly between 0 and 1, got {train_fraction}."
        )
    rollout_paths = _resolve_rollout_paths(paths)
    if len(rollout_paths) < 2:
        return rollout_paths, []

    rng = np.random.default_rng(seed)
    shuffled = list(rollout_paths)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_fraction)
    split_idx = min(max(split_idx, 1), len(shuffled) - 1)
    return shuffled[:split_idx], shuffled[split_idx:]


def _assign_dataset_stats(dataset: Any, stats: Optional[DatasetNormalizationStats]) -> None:
    if dataset is None:
        return
    if isinstance(dataset, ConcatDataset):
        for subdataset in dataset.datasets:
            _assign_dataset_stats(subdataset, stats)
        return
    if hasattr(dataset, "stats"):
        dataset.stats = stats
    if hasattr(dataset, "normalize"):
        dataset.normalize = stats is not None


def train_sope(
    cfg_dataset: RolloutChunkDatasetConfig,
    cfg_diffusion: SopeDiffusionConfig,
    cfg_training: TrainingConfig,
) -> None:
    from src.eval import evaluate_sope
    from src.robomimic_interface.dataset import make_rollout_chunk_dataloader
    from src.sope_diffuser import SopeDiffuser, cross_validate_configs
    import wandb

    set_global_seed(cfg_training.seed)

    device_str = cfg_training.device or resolve_device(
        prefer_cuda=cfg_training.prefer_cuda
    )
    device = torch.device(device_str)

    if cfg_dataset.source == "obs":
        raise ValueError(
            "source='obs' requires an encoder; train_sope currently supports source='latents' only."
        )

    cross_validate_configs(cfg_dataset, cfg_diffusion)

    train_paths, eval_paths = _split_rollout_paths(
        paths=cfg_training.data,
        seed=cfg_training.seed,
        train_fraction=cfg_training.train_fraction,
    )

    loader, stats = make_rollout_chunk_dataloader(
        paths=train_paths,
        config=cfg_dataset,
        batch_size=cfg_training.batch_size,
        num_workers=cfg_training.num_workers,
        shuffle=True,
        drop_last=True,
        encoder=None,
        obs_keys=None,
        encoder_device=device_str,
    )
    eval_loader = None
    if eval_paths:
        eval_loader, _ = make_rollout_chunk_dataloader(
            paths=eval_paths,
            config=cfg_dataset,
            batch_size=cfg_training.batch_size,
            num_workers=cfg_training.num_workers,
            shuffle=False,
            drop_last=False,
            encoder=None,
            obs_keys=None,
            encoder_device=device_str,
        )
        _assign_dataset_stats(eval_loader.dataset, stats if cfg_dataset.normalize else None)

    diffuser = SopeDiffuser(
        cfg=cfg_diffusion,
        normalization_stats=_convert_stats(stats),
        device=device_str,
    )
    optimizer = torch.optim.Adam(
        diffuser.diffusion.parameters(),
        lr=cfg_training.lr,
        weight_decay=cfg_training.weight_decay,
    )
    steps_per_epoch = len(loader)
    max_steps = (
        cfg_training.max_steps
        if cfg_training.max_steps and cfg_training.max_steps > 0
        else None
    )
    total_training_steps = (
        min(cfg_training.epochs * steps_per_epoch, max_steps)
        if max_steps
        else cfg_training.epochs * steps_per_epoch
    )
    scheduler = _build_lr_scheduler(
        optimizer=optimizer,
        cfg_training=cfg_training,
        total_steps=total_training_steps,
    )
    save_every = _resolve_save_every(cfg_training)
    eval_every = _resolve_eval_every(cfg_training)

    checkpoint_dir = (
        Path(cfg_training.checkpoint_dir) if cfg_training.checkpoint_dir else None
    )
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _save_configs(checkpoint_dir / "configs.json", cfg_diffusion, cfg_dataset, cfg_training)

    run = None
    if cfg_training.wandb_project:
        run = wandb.init(
            project=cfg_training.wandb_project,
            entity=cfg_training.wandb_entity,
            name=cfg_training.wandb_run_name,
            group=cfg_training.wandb_group,
            mode=cfg_training.wandb_mode,
            tags=list(cfg_training.wandb_tags),
            config={
                "dataset": asdict(cfg_dataset),
                "diffusion": asdict(cfg_diffusion),
                "training": {
                    **asdict(cfg_training),
                    "data": [str(path) for path in _as_paths(cfg_training.data)],
                    "train_paths": [str(path) for path in train_paths],
                    "eval_paths": [str(path) for path in eval_paths],
                    "checkpoint_dir": (
                        str(checkpoint_dir) if checkpoint_dir is not None else None
                    ),
                    "save_every": save_every,
                    "eval_every": eval_every,
                },
            },
        )

    step = 0
    epoch_iterator = tqdm(
        range(1, cfg_training.epochs + 1), desc="train_sope", unit="epoch"
    )
    try:
        for epoch in epoch_iterator:
            epoch_loss_sum = 0.0
            epoch_batches = 0

            # Train
            for batch in loader:
                step += 1
                batch_t = _to_device(batch, device)
                loss_out = diffuser.loss(batch_t)
                if isinstance(loss_out, tuple):
                    loss, info = loss_out
                else:
                    loss, info = loss_out, {}

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg_training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        diffuser.diffusion.parameters(), cfg_training.grad_clip
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                loss_value = float(loss.item())
                epoch_loss_sum += loss_value
                epoch_batches += 1
                current_lr = float(optimizer.param_groups[0]["lr"])

                if run is not None:
                    batch_metrics = {
                        "train/loss": loss_value,
                        "train/lr": current_lr,
                        "train/epoch": float(epoch),
                        "train/step": float(step),
                    }
                    for key, value in _flatten_scalar_metrics(info).items():
                        batch_metrics[f"train/{key}"] = value
                    run.log(batch_metrics, step=step)

                if max_steps and step >= max_steps:
                    break

            avg_epoch_loss = epoch_loss_sum / max(epoch_batches, 1)
            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_iterator.set_postfix(
                loss=f"{avg_epoch_loss:.2e}", lr=f"{current_lr:.2e}"
            )

            if run is not None:
                run.log({"train_epoch_avg/loss": avg_epoch_loss}, step=step)
            
            # Evaluate on held-out data
            eval_now = eval_loader is not None and eval_every is not None \
                and (epoch % eval_every == 0 or epoch == cfg_training.epochs)
            save_now = checkpoint_dir is not None \
                and (epoch % save_every == 0 or epoch == cfg_training.epochs)
            
            if eval_now:
                evaluate_sope(
                    diffuser,
                    eval_loader,
                    device,
                    epoch=epoch,
                    step=step,
                    run=run,
                    epoch_iterator=epoch_iterator,
                    avg_epoch_loss=avg_epoch_loss,
                    current_lr=current_lr,
                )
            
            if save_now:
                ckpt_path = checkpoint_dir / f"sope_diffuser_epoch_{epoch:04d}.pt"
                _save_checkpoint(
                    ckpt_path,
                    diffuser,
                    epoch,
                    step,
                    cfg_diffusion,
                    cfg_dataset,
                    cfg_training,
                    stats,
                )
                latest_path = checkpoint_dir / "sope_diffuser_latest.pt"
                _save_checkpoint(
                    latest_path,
                    diffuser,
                    epoch,
                    step,
                    cfg_diffusion,
                    cfg_dataset,
                    cfg_training,
                    stats,
                )

            if max_steps and step >= max_steps:
                break
    finally:
        if run is not None:
            run.finish()


def train_reward(*args, **kwargs) -> None:
    """Placeholder for future per-step reward predictor training."""
    raise NotImplementedError("train_reward() is not implemented yet.")


def train(
    cfg_dataset: RolloutChunkDatasetConfig,
    cfg_diffusion: SopeDiffusionConfig,
    cfg_training: TrainingConfig,
) -> None:
    train_sope(
        cfg_dataset=cfg_dataset,
        cfg_diffusion=cfg_diffusion,
        cfg_training=cfg_training,
    )


__all__ = [
    "PathLike",
    "TrainingConfig",
    "train",
    "train_reward",
    "train_sope",
]
