from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union

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

PathLike = Union[str, Path]


@dataclass(frozen=True)
class TrainingConfig:
    data: Sequence[PathLike]
    data_kind: str = "rollout"
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
    diffuser: Any,
    epoch: int,
    step: int,
    cfg_diffusion: Any,
    cfg_dataset: Any,
    cfg_training: TrainingConfig,
    stats: Optional["DatasetNormalizationStats"],
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


def _save_reward_checkpoint(
    path: Path,
    reward_predictor: Any,
    epoch: int,
    step: int,
    cfg_reward: Any,
    cfg_dataset: Any,
    cfg_training: TrainingConfig,
    input_stats: Optional["DatasetNormalizationStats"] = None,
) -> None:
    payload = {
        "reward_state_dict": reward_predictor.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "reward_config": asdict(cfg_reward),
        "dataset_config": asdict(cfg_dataset),
        "training_config": _serialize_training_config(cfg_training),
        "input_normalization_stats": (
            None if input_stats is None else {"mean": input_stats.mean, "std": input_stats.std}
        ),
    }
    torch.save(payload, str(path))


def _save_configs(
    path: Path,
    cfg_diffusion: Any,
    cfg_dataset: Any,
    cfg_training: TrainingConfig,
) -> None:
    payload = {
        "diffusion_config": asdict(cfg_diffusion),
        "dataset_config": asdict(cfg_dataset),
        "training_config": _serialize_training_config(cfg_training),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _save_reward_configs(
    path: Path,
    cfg_reward: Any,
    cfg_dataset: Any,
    cfg_training: TrainingConfig,
) -> None:
    payload = {
        "reward_config": asdict(cfg_reward),
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
                resolved.extend(sorted(p.rglob(ext)))
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


def _format_train_info_metrics(info: Any) -> dict[str, float]:
    formatted: dict[str, float] = {}
    for key, value in _flatten_scalar_metrics(info).items():
        if key.startswith("chunk_rmse_"):
            formatted[f"train_normalized/{key}"] = value
        else:
            formatted[f"train/{key}"] = value
    return formatted


def _format_reward_info_metrics(info: Any) -> dict[str, float]:
    return {f"reward_train/{key}": value for key, value in _flatten_scalar_metrics(info).items()}


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


def _extract_dataset_stats(dataset: Any) -> Optional["DatasetNormalizationStats"]:
    if dataset is None:
        return None
    if isinstance(dataset, ConcatDataset):
        for subdataset in dataset.datasets:
            sub_stats = _extract_dataset_stats(subdataset)
            if sub_stats is not None:
                return sub_stats
        return None
    return getattr(dataset, "stats", None)


def _stringify_refs(refs: Optional[Sequence[Any]]) -> list[str]:
    if refs is None:
        return []
    return [str(ref) for ref in refs]


def _resolve_training_device(
    cfg_training: TrainingConfig,
) -> tuple[str, torch.device]:
    device_str = cfg_training.device or resolve_device(
        prefer_cuda=cfg_training.prefer_cuda
    )
    return device_str, torch.device(device_str)


def _resolve_checkpoint_dir(cfg_training: TrainingConfig) -> Optional[Path]:
    return Path(cfg_training.checkpoint_dir) if cfg_training.checkpoint_dir else None


def derive_phase_training_config(
    cfg_training: TrainingConfig,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    wandb_run_name_suffix: Optional[str] = None,
    wandb_tags: Sequence[str] = (),
) -> TrainingConfig:
    """Build a phase-local TrainingConfig for diffusion or reward training.

    Diffusion consumes `cfg_training.lr` when constructing its optimizer.
    Reward training uses `cfg_training.epochs` and `cfg_training.batch_size`,
    but its optimizer LR comes from `RewardPredictorConfig.lr` instead.
    """
    run_name = cfg_training.wandb_run_name
    if wandb_run_name_suffix and run_name is not None:
        run_name = f"{run_name}_{wandb_run_name_suffix}"

    return replace(
        cfg_training,
        epochs=cfg_training.epochs if epochs is None else epochs,
        batch_size=cfg_training.batch_size if batch_size is None else batch_size,
        lr=cfg_training.lr if lr is None else lr,
        wandb_run_name=run_name,
        wandb_tags=cfg_training.wandb_tags + tuple(wandb_tags),
    )


def _build_wandb_training_payload(
    cfg_training: TrainingConfig,
    *,
    checkpoint_dir: Optional[Path],
    train_data_refs: Optional[Sequence[Any]] = None,
    eval_data_refs: Optional[Sequence[Any]] = None,
    save_every: Optional[int] = None,
    eval_every: Optional[int] = None,
) -> dict[str, Any]:
    return {
        **asdict(cfg_training),
        "data": [str(path) for path in _as_paths(cfg_training.data)],
        "train_data_refs": _stringify_refs(train_data_refs),
        "eval_data_refs": _stringify_refs(eval_data_refs),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "save_every": save_every,
        "eval_every": eval_every,
    }


def _collect_optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> list[torch.Tensor]:
    params: list[torch.Tensor] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if not torch.is_tensor(param):
                continue
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    return params


def _set_model_training_mode(model: Any, training: bool) -> None:
    train_fn = getattr(model, "train", None)
    if callable(train_fn):
        train_fn(training)
        return

    diffusion = getattr(model, "diffusion", None)
    diffusion_train_fn = getattr(diffusion, "train", None)
    if callable(diffusion_train_fn):
        diffusion_train_fn(training)


@dataclass(frozen=True)
class TrainLoopCallbacks:
    name: str
    build_optimizer: Callable[[Any, TrainingConfig], torch.optim.Optimizer]
    compute_loss: Callable[
        [Any, Dict[str, torch.Tensor], int, int, int, Optional[int]],
        tuple[torch.Tensor, dict[str, Any]],
    ]
    format_batch_metrics: Callable[[dict[str, Any], int, int, float], dict[str, float]]
    evaluate: Optional[Callable[..., Optional[dict[str, float]]]] = None
    save_configs: Optional[Callable[[Path], None]] = None
    save_checkpoint: Optional[Callable[[Path, Any, int, int], None]] = None
    checkpoint_stem: str = "model"
    epoch_avg_metric_key: str = "train_epoch_avg/loss"
    wandb_config: Optional[dict[str, Any]] = None


def train_general(
    *,
    model: Any,
    loader: torch.utils.data.DataLoader,
    cfg_training: TrainingConfig,
    callbacks: TrainLoopCallbacks,
    eval_loader: Optional[torch.utils.data.DataLoader] = None,
) -> None:
    import wandb

    set_global_seed(cfg_training.seed)

    _, device = _resolve_training_device(cfg_training)
    optimizer = callbacks.build_optimizer(model, cfg_training)
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

    checkpoint_dir = _resolve_checkpoint_dir(cfg_training)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if callbacks.save_configs is not None:
            callbacks.save_configs(checkpoint_dir)

    run = None
    if cfg_training.wandb_project:
        run = wandb.init(
            project=cfg_training.wandb_project,
            entity=cfg_training.wandb_entity,
            name=cfg_training.wandb_run_name,
            group=cfg_training.wandb_group,
            mode=cfg_training.wandb_mode,
            tags=list(cfg_training.wandb_tags),
            config=callbacks.wandb_config or {},
        )

    step = 0
    epoch_iterator = tqdm(
        range(1, cfg_training.epochs + 1), desc=callbacks.name, unit="epoch"
    )
    try:
        for epoch in epoch_iterator:
            _set_model_training_mode(model, True)
            epoch_loss_sum = 0.0
            epoch_batches = 0

            for batch_idx, batch in enumerate(loader):
                step += 1
                batch_t = _to_device(batch, device)
                loss, info = callbacks.compute_loss(
                    model,
                    batch_t,
                    batch_idx,
                    step,
                    steps_per_epoch,
                    max_steps,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg_training.grad_clip > 0:
                    params = _collect_optimizer_parameters(optimizer)
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, cfg_training.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                loss_value = float(loss.item())
                epoch_loss_sum += loss_value
                epoch_batches += 1
                current_lr = float(optimizer.param_groups[0]["lr"])

                if run is not None:
                    batch_info = dict(info)
                    batch_info["loss"] = loss_value
                    batch_metrics = callbacks.format_batch_metrics(
                        batch_info,
                        epoch,
                        step,
                        current_lr,
                    )
                    if batch_metrics:
                        run.log(batch_metrics, step=step)

                if max_steps and step >= max_steps:
                    break

            avg_epoch_loss = epoch_loss_sum / max(epoch_batches, 1)
            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_iterator.set_postfix(
                loss=f"{avg_epoch_loss:.2e}", lr=f"{current_lr:.2e}"
            )

            if run is not None:
                run.log({callbacks.epoch_avg_metric_key: avg_epoch_loss}, step=step)

            eval_now = eval_loader is not None and eval_every is not None and (
                epoch % eval_every == 0 or epoch == cfg_training.epochs
            )
            save_now = checkpoint_dir is not None and callbacks.save_checkpoint is not None and (
                epoch % save_every == 0 or epoch == cfg_training.epochs
            )

            if eval_now and callbacks.evaluate is not None:
                eval_metrics = callbacks.evaluate(
                    model,
                    eval_loader,
                    device,
                    epoch,
                    step,
                    run,
                    epoch_iterator,
                    avg_epoch_loss,
                    current_lr,
                )
                if run is not None and eval_metrics:
                    run.log(eval_metrics, step=step)

            if save_now:
                ckpt_path = checkpoint_dir / f"{callbacks.checkpoint_stem}_epoch_{epoch:04d}.pt"
                callbacks.save_checkpoint(ckpt_path, model, epoch, step)
                latest_path = checkpoint_dir / f"{callbacks.checkpoint_stem}_latest.pt"
                callbacks.save_checkpoint(latest_path, model, epoch, step)

            if max_steps and step >= max_steps:
                break
    finally:
        if run is not None:
            run.finish()


def _validate_sope_training_inputs(
    cfg_dataset: Any,
    cfg_diffusion: Any,
) -> None:
    from src.diffusion import cross_validate_configs

    if getattr(cfg_dataset, "source", None) == "obs":
        raise ValueError(
            "source='obs' requires an encoder; train_sope currently supports source='latents' only."
        )

    dataset_state_dim = getattr(cfg_dataset, "latents_dim", None)
    if dataset_state_dim is None:
        dataset_state_dim = getattr(cfg_dataset, "state_dim", None)
    if dataset_state_dim is None:
        raise AttributeError(
            f"{type(cfg_dataset).__name__} must define latents_dim or state_dim."
        )
    dataset_state_dim = int(dataset_state_dim)
    dataset_state_projection = getattr(cfg_dataset, "state_projection", "full")
    if cfg_diffusion.diffuser_eef_pos_only:
        if dataset_state_projection == "eef_pos":
            if dataset_state_dim != 3:
                raise ValueError(
                    "state_projection='eef_pos' requires the effective dataset state_dim to be 3."
                )
        elif dataset_state_dim < 13:
            raise ValueError(
                "diffuser_eef_pos_only requires either state_projection='eef_pos' or a low-dim robomimic state with robot0_eef_pos at slice [10:13]."
            )

    cross_validate_configs(cfg_dataset, cfg_diffusion)


def _make_sope_train_callbacks(
    *,
    cfg_dataset: Any,
    cfg_diffusion: Any,
    cfg_training: TrainingConfig,
    stats: Optional["DatasetNormalizationStats"],
    train_data_refs: Optional[Sequence[Any]] = None,
    eval_data_refs: Optional[Sequence[Any]] = None,
) -> TrainLoopCallbacks:
    from src.eval import evaluate_sope

    checkpoint_dir = _resolve_checkpoint_dir(cfg_training)
    save_every = _resolve_save_every(cfg_training)
    eval_every = _resolve_eval_every(cfg_training)

    def _build_optimizer(
        diffuser: Any,
        training_cfg: TrainingConfig,
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            diffuser.diffusion.parameters(),
            lr=training_cfg.lr,
            weight_decay=training_cfg.weight_decay,
        )

    def _compute_loss(
        diffuser: Any,
        batch_t: Dict[str, torch.Tensor],
        batch_idx: int,
        step: int,
        steps_per_epoch: int,
        max_steps: Optional[int],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        compute_batch_rmse = (
            batch_idx == (steps_per_epoch - 1)
            or (max_steps is not None and step >= max_steps)
        )
        loss, info = diffuser.loss(
            batch_t,
            compute_batch_rmse=compute_batch_rmse,
        )
        return loss, info

    def _format_batch_metrics(
        info: dict[str, Any],
        epoch: int,
        step: int,
        lr: float,
    ) -> dict[str, float]:
        metrics = {
            "train/lr": lr,
            "train/epoch": float(epoch),
            "train/step": float(step),
        }
        metrics.update(_format_train_info_metrics(info))
        return metrics

    def _evaluate(
        diffuser: Any,
        eval_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epoch: int,
        step: int,
        run: Optional[Any],
        epoch_iterator: Optional[Any],
        avg_epoch_loss: float,
        current_lr: float,
    ) -> Optional[dict[str, float]]:
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
        return None

    def _save_configs_to_dir(checkpoint_root: Path) -> None:
        _save_configs(
            checkpoint_root / "configs.json",
            cfg_diffusion,
            cfg_dataset,
            cfg_training,
        )

    def _save_diffusion_checkpoint(
        path: Path,
        diffuser: Any,
        epoch: int,
        step: int,
    ) -> None:
        _save_checkpoint(
            path,
            diffuser,
            epoch,
            step,
            cfg_diffusion,
            cfg_dataset,
            cfg_training,
            stats,
        )

    return TrainLoopCallbacks(
        name="train_sope",
        build_optimizer=_build_optimizer,
        compute_loss=_compute_loss,
        format_batch_metrics=_format_batch_metrics,
        evaluate=_evaluate,
        save_configs=_save_configs_to_dir,
        save_checkpoint=_save_diffusion_checkpoint,
        checkpoint_stem="sope_diffuser",
        epoch_avg_metric_key="train_epoch_avg/loss",
        wandb_config={
            "dataset": asdict(cfg_dataset),
            "diffusion": asdict(cfg_diffusion),
            "training": _build_wandb_training_payload(
                cfg_training,
                checkpoint_dir=checkpoint_dir,
                train_data_refs=train_data_refs,
                eval_data_refs=eval_data_refs,
                save_every=save_every,
                eval_every=eval_every,
            ),
        },
    )


def train_sope(
    cfg_dataset: Any,
    cfg_diffusion: Any,
    cfg_training: TrainingConfig,
    *,
    loader: Optional[torch.utils.data.DataLoader] = None,
    eval_loader: Optional[torch.utils.data.DataLoader] = None,
    stats: Optional["DatasetNormalizationStats"] = None,
    train_data_refs: Optional[Sequence[Any]] = None,
    eval_data_refs: Optional[Sequence[Any]] = None,
) -> None:
    """Train chunk diffusion with phase-local TrainingConfig epochs, batch size, and LR."""
    from src.diffusion import SopeDiffuser

    device_str, _ = _resolve_training_device(cfg_training)
    if loader is None:
        from src.robomimic_interface.dataset import make_rollout_chunk_dataloader

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
        train_data_refs = train_paths
        eval_data_refs = eval_paths
    else:
        if stats is None:
            stats = _extract_dataset_stats(loader.dataset)
        if eval_loader is not None and getattr(cfg_dataset, "normalize", False):
            _assign_dataset_stats(eval_loader.dataset, stats)

    _validate_sope_training_inputs(cfg_dataset, cfg_diffusion)
    diffuser = SopeDiffuser(
        cfg=cfg_diffusion,
        normalization_stats=stats,
        device=device_str,
    )
    callbacks = _make_sope_train_callbacks(
        cfg_dataset=cfg_dataset,
        cfg_diffusion=cfg_diffusion,
        cfg_training=cfg_training,
        stats=stats,
        train_data_refs=train_data_refs,
        eval_data_refs=eval_data_refs,
    )
    train_general(
        model=diffuser,
        loader=loader,
        cfg_training=cfg_training,
        callbacks=callbacks,
        eval_loader=eval_loader,
    )


def _validate_rewardpred_training_inputs(
    cfg_dataset: Any,
    cfg_reward: Any,
) -> None:
    if getattr(cfg_dataset, "source", None) != "latents":
        raise ValueError(
            "Reward training currently supports source='latents' only so it can use raw low-dim state inputs."
        )
    if bool(getattr(cfg_dataset, "normalize", False)):
        raise ValueError("Reward training expects raw state/action inputs; set cfg_dataset.normalize=False.")

    dataset_state_dim = getattr(cfg_dataset, "latents_dim", None)
    if dataset_state_dim is None:
        dataset_state_dim = getattr(cfg_dataset, "state_dim", None)
    if dataset_state_dim is None:
        raise AttributeError(
            f"{type(cfg_dataset).__name__} must define latents_dim or state_dim."
        )
    dataset_action_dim = getattr(cfg_dataset, "action_dim", None)
    if dataset_action_dim is None:
        raise AttributeError(f"{type(cfg_dataset).__name__} must define action_dim.")
    if int(dataset_state_dim) != int(cfg_reward.state_dim):
        raise ValueError(
            "Reward config mismatch: dataset state_dim must equal RewardPredictorConfig.state_dim "
            f"({dataset_state_dim} != {cfg_reward.state_dim})."
        )
    if int(dataset_action_dim) != int(cfg_reward.action_dim):
        raise ValueError(
            "Reward config mismatch: dataset action_dim must equal RewardPredictorConfig.action_dim "
            f"({dataset_action_dim} != {cfg_reward.action_dim})."
        )


def _make_rewardpred_train_callbacks(
    *,
    cfg_dataset: Any,
    cfg_reward: Any,
    cfg_training: TrainingConfig,
    train_data_refs: Optional[Sequence[Any]] = None,
    eval_data_refs: Optional[Sequence[Any]] = None,
) -> TrainLoopCallbacks:
    checkpoint_dir = _resolve_checkpoint_dir(cfg_training)
    save_every = _resolve_save_every(cfg_training)
    eval_every = _resolve_eval_every(cfg_training)

    def _build_optimizer(
        reward_predictor: Any,
        training_cfg: TrainingConfig,
    ) -> torch.optim.Optimizer:
        del training_cfg
        return torch.optim.Adam(
            reward_predictor.parameters(),
            lr=cfg_reward.lr,
            weight_decay=cfg_reward.weight_decay,
        )

    def _compute_loss(
        reward_predictor: Any,
        batch_t: Dict[str, torch.Tensor],
        batch_idx: int,
        step: int,
        steps_per_epoch: int,
        max_steps: Optional[int],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del batch_idx, step, steps_per_epoch, max_steps
        return reward_predictor.loss(batch_t)

    def _format_batch_metrics(
        info: dict[str, Any],
        epoch: int,
        step: int,
        lr: float,
    ) -> dict[str, float]:
        metrics = {
            "reward_train/lr": lr,
            "reward_train/epoch": float(epoch),
            "reward_train/step": float(step),
        }
        metrics.update(_format_reward_info_metrics(info))
        return metrics

    def _evaluate(
        reward_predictor: Any,
        eval_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epoch: int,
        step: int,
        run: Optional[Any],
        epoch_iterator: Optional[Any],
        avg_epoch_loss: float,
        current_lr: float,
    ) -> Optional[dict[str, float]]:
        del epoch, step, run, epoch_iterator, avg_epoch_loss, current_lr
        from src.eval import _evaluate_reward_predictor

        return _evaluate_reward_predictor(reward_predictor, eval_loader, device)

    def _save_configs_to_dir(checkpoint_root: Path) -> None:
        _save_reward_configs(
            checkpoint_root / "reward_configs.json",
            cfg_reward,
            cfg_dataset,
            cfg_training,
        )

    def _save_reward_predictor_checkpoint(
        path: Path,
        reward_predictor: Any,
        epoch: int,
        step: int,
    ) -> None:
        _save_reward_checkpoint(
            path,
            reward_predictor,
            epoch,
            step,
            cfg_reward,
            cfg_dataset,
            cfg_training,
            None,
        )

    return TrainLoopCallbacks(
        name="train_rewardpred",
        build_optimizer=_build_optimizer,
        compute_loss=_compute_loss,
        format_batch_metrics=_format_batch_metrics,
        evaluate=_evaluate,
        save_configs=_save_configs_to_dir,
        save_checkpoint=_save_reward_predictor_checkpoint,
        checkpoint_stem="sope_reward_predictor",
        epoch_avg_metric_key="reward_train_epoch_avg/loss",
        wandb_config={
            "dataset": asdict(cfg_dataset),
            "reward": asdict(cfg_reward),
            "training": _build_wandb_training_payload(
                cfg_training,
                checkpoint_dir=checkpoint_dir,
                train_data_refs=train_data_refs,
                eval_data_refs=eval_data_refs,
                save_every=save_every,
                eval_every=eval_every,
            ),
        },
    )


def train_rewardpred(
    cfg_dataset: Any,
    cfg_reward: Any,
    cfg_training: TrainingConfig,
    *,
    loader: Optional[torch.utils.data.DataLoader] = None,
    eval_loader: Optional[torch.utils.data.DataLoader] = None,
    train_data_refs: Optional[Sequence[Any]] = None,
    eval_data_refs: Optional[Sequence[Any]] = None,
) -> None:
    """Train the reward predictor with phase-local epochs/batch size and RewardPredictorConfig.lr."""
    from src.diffusion import RewardPredictor

    device_str, _ = _resolve_training_device(cfg_training)
    reward_dataset_cfg = (
        replace(cfg_dataset, normalize=False)
        if bool(getattr(cfg_dataset, "normalize", False))
        else cfg_dataset
    )

    if loader is None:
        from src.robomimic_interface.dataset import make_rollout_chunk_dataloader

        train_paths, eval_paths = _split_rollout_paths(
            paths=cfg_training.data,
            seed=cfg_training.seed,
            train_fraction=cfg_training.train_fraction,
        )

        loader, _ = make_rollout_chunk_dataloader(
            paths=train_paths,
            config=reward_dataset_cfg,
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
                config=reward_dataset_cfg,
                batch_size=cfg_training.batch_size,
                num_workers=cfg_training.num_workers,
                shuffle=False,
                drop_last=False,
                encoder=None,
                obs_keys=None,
                encoder_device=device_str,
            )
        train_data_refs = train_paths
        eval_data_refs = eval_paths

    _validate_rewardpred_training_inputs(reward_dataset_cfg, cfg_reward)
    reward_predictor = RewardPredictor(cfg_reward, device=device_str)
    callbacks = _make_rewardpred_train_callbacks(
        cfg_dataset=reward_dataset_cfg,
        cfg_reward=cfg_reward,
        cfg_training=cfg_training,
        train_data_refs=train_data_refs,
        eval_data_refs=eval_data_refs,
    )
    train_general(
        model=reward_predictor,
        loader=loader,
        cfg_training=cfg_training,
        callbacks=callbacks,
        eval_loader=eval_loader,
    )


train_reward = train_rewardpred
train_sope_with_loaders = train_sope
train_rewardpred_with_loaders = train_rewardpred
train_reward_with_loaders = train_rewardpred


__all__ = [
    "derive_phase_training_config",
    "PathLike",
    "TrainLoopCallbacks",
    "TrainingConfig",
    "train",
    "train_general",
    "train_reward",
    "train_rewardpred",
    "train_rewardpred_with_loaders",
    "train_reward_with_loaders",
    "train_sope",
    "train_sope_with_loaders",
]
