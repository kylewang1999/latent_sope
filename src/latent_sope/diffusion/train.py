from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm

from src.latent_sope.diffusion.sope_diffuser import (
    NormalizationStats as SopeNormalizationStats,
    SopeDiffuser,
    SopeDiffusionConfig,
    cross_validate_configs,
)
from src.latent_sope.robomimic_interface.dataset import (
    NormalizationStats as DatasetNormalizationStats,
    RolloutChunkDatasetConfig,
    make_rollout_chunk_dataloader,
)
from src.latent_sope.utils.misc import resolve_device, set_global_seed

PathLike = Union[str, Path]


@dataclass(frozen=True)
class TrainingConfig:
    data: Sequence[PathLike]
    checkpoint_dir: Optional[PathLike] = None
    epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_steps: Optional[int] = None
    log_every: int = 50
    save_every: int = 1
    seed: int = 0
    device: Optional[str] = None
    prefer_cuda: bool = True


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
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


def _convert_stats(stats: Optional[DatasetNormalizationStats]) -> Optional[SopeNormalizationStats]:
    if stats is None:
        return None
    return SopeNormalizationStats(mean=stats.mean, std=stats.std)


def _save_checkpoint(
    path: Path,
    diffuser: SopeDiffuser,
    epoch: int,
    step: int,
    cfg_diffusion: SopeDiffusionConfig,
    cfg_dataset: RolloutChunkDatasetConfig,
    stats: Optional[DatasetNormalizationStats],
) -> None:
    payload = {
        "diffusion_state_dict": diffuser.diffusion.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "diffusion_config": asdict(cfg_diffusion),
        "dataset_config": asdict(cfg_dataset),
        "normalization_stats": None
        if stats is None
        else {"mean": stats.mean, "std": stats.std},
    }
    torch.save(payload, str(path))


def _save_configs(path: Path, cfg_diffusion: SopeDiffusionConfig, cfg_dataset: RolloutChunkDatasetConfig) -> None:
    payload = {
        "diffusion_config": asdict(cfg_diffusion),
        "dataset_config": asdict(cfg_dataset),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _as_paths(paths: Sequence[PathLike]) -> list[Path]:
    return [p if isinstance(p, Path) else Path(p) for p in paths]


def train(
    cfg_dataset: RolloutChunkDatasetConfig,
    cfg_diffusion: SopeDiffusionConfig,
    cfg_training: TrainingConfig,
) -> None:
    set_global_seed(cfg_training.seed)

    device_str = cfg_training.device or resolve_device(prefer_cuda=cfg_training.prefer_cuda)
    device = torch.device(device_str)

    if cfg_dataset.source == "obs":
        raise ValueError("source='obs' requires an encoder; train.py currently supports source='latents' only.")

    cross_validate_configs(cfg_dataset, cfg_diffusion)

    loader, stats = make_rollout_chunk_dataloader(
        paths=_as_paths(cfg_training.data),
        config=cfg_dataset,
        batch_size=cfg_training.batch_size,
        shuffle=True,
        drop_last=True,
        encoder=None,
        obs_keys=None,
        encoder_device=device_str,
    )

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

    checkpoint_dir = Path(cfg_training.checkpoint_dir) if cfg_training.checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _save_configs(checkpoint_dir / "configs.json", cfg_diffusion, cfg_dataset)

    step = 0
    max_steps = cfg_training.max_steps if cfg_training.max_steps and cfg_training.max_steps > 0 else None
    for epoch in range(1, cfg_training.epochs + 1):
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
                torch.nn.utils.clip_grad_norm_(diffuser.diffusion.parameters(), cfg_training.grad_clip)
            optimizer.step()

            if step % cfg_training.log_every == 0:
                info_str = ""
                if isinstance(info, dict) and info:
                    flat = {}
                    for k, v in info.items():
                        if torch.is_tensor(v):
                            if v.numel() == 1:
                                flat[k] = float(v.item())
                        elif np.isscalar(v):
                            flat[k] = float(v)
                    if flat:
                        info_str = " | " + " ".join(f"{k}={v:.4f}" for k, v in flat.items())
                print(
                    f"[epoch {epoch:03d} step {step:06d}] loss={loss.item():.6f}{info_str}",
                    flush=True,
                )

            if max_steps and step >= max_steps:
                break

        if checkpoint_dir is not None:
            if epoch % cfg_training.save_every == 0 or epoch == cfg_training.epochs:
                ckpt_path = checkpoint_dir / f"sope_diffuser_epoch_{epoch:04d}.pt"
                _save_checkpoint(ckpt_path, diffuser, epoch, step, cfg_diffusion, cfg_dataset, stats)
                latest_path = checkpoint_dir / "sope_diffuser_latest.pt"
                _save_checkpoint(latest_path, diffuser, epoch, step, cfg_diffusion, cfg_dataset, stats)

        if max_steps and step >= max_steps:
            break
