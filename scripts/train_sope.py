#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import PATHS, make_log_dir


def _resolve_default_checkpoint_dir() -> Path:
    return Path(make_log_dir("train_sope_film", verbose=False))


def _resolve_default_data_path() -> Path:
    rollout_root = (
        PATHS.repo_root / "data" / "rollout" / "rmimic-lift-ph-lowdim_diffusion_260130"
    )
    h5_dir = rollout_root / "h5_files"
    return (h5_dir if h5_dir.is_dir() else rollout_root).resolve()


def _resolve_rollout_reference(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.rglob("*.h5")) + sorted(path.rglob("*.hdf5")) + sorted(path.rglob("*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No rollout files found under {path}.")
        return candidates[0]
    raise FileNotFoundError(f"Rollout path not found: {path}")


def _infer_rollout_shapes(path: Path) -> tuple[int, int, int]:
    rollout_path = _resolve_rollout_reference(path)
    with h5py.File(rollout_path, "r") as handle:
        latents_shape = tuple(handle["latents"].shape)
        actions_shape = tuple(handle["actions"].shape)
        frame_stack = int(handle.attrs.get("frame_stack", latents_shape[1] if len(latents_shape) >= 3 else 1))

    latent_dim = int(latents_shape[-1])
    action_dim = int(actions_shape[-1])
    return latent_dim, action_dim, frame_stack


def _infer_eef_pos_slice(path: Path) -> tuple[int, int]:
    rollout_path = _resolve_rollout_reference(path)
    with h5py.File(rollout_path, "r") as handle:
        if "obs" not in handle or "robot0_eef_pos" not in handle["obs"]:
            raise ValueError(
                f"Rollout file {rollout_path} does not contain obs/robot0_eef_pos needed to infer the low-dim slice."
            )

        offset = 0
        for key in sorted(handle["obs"].keys()):
            dim = int(handle["obs"][key].shape[-1])
            next_offset = offset + dim
            if key == "robot0_eef_pos":
                return offset, next_offset
            offset = next_offset

    raise ValueError(f"Could not infer robot0_eef_pos slice from {rollout_path}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the canonical SOPE chunk diffusion model from saved rollout latents."
    )
    parser.add_argument("--data", type=Path, default=_resolve_default_data_path())
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--disable-lr-scheduler", action="store_true")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", choices=("cosine",))
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument(
        "--dim-mults",
        type=int,
        nargs="+",
        default=(1, 2),
        help="Robomimic ConditionalUnet1D width multipliers, e.g. --dim-mults 1 2 4.",
    )
    parser.add_argument("--diffusion-steps", type=int, default=512)
    parser.add_argument("--attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-weight", type=float, default=10.0)
    parser.add_argument("--predict-epsilon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--num-saves", type=int, default=10)
    parser.add_argument("--num-evals", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffuser-eef-pos-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="wkt_sope")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default="train_sope_film")
    parser.add_argument("--wandb-group", type=str, default="sope_diffusion_film")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=("online", "offline", "disabled"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from src.diffusion import SopeDiffusionConfig
    from src.robomimic_interface.dataset import RolloutChunkDatasetConfig
    from src.train import TrainingConfig, train_sope

    data_path = args.data.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else _resolve_default_checkpoint_dir()

    latent_dim, action_dim, frame_stack = _infer_rollout_shapes(data_path)

    cfg_dataset = RolloutChunkDatasetConfig(
        chunk_size=args.chunk_size,
        stride=args.stride,
        frame_stack=frame_stack,
        source="latents",
        latents_dim=latent_dim,
        action_dim=action_dim,
        normalize=bool(args.normalize),
        return_metadata=True,
    )
    cfg_diffusion = SopeDiffusionConfig(
        chunk_horizon=args.chunk_size,
        frame_stack=frame_stack,
        state_dim=latent_dim,
        action_dim=action_dim,
        diffusion_steps=args.diffusion_steps,
        dim_mults=tuple(args.dim_mults),
        attention=bool(args.attention),
        action_weight=args.action_weight,
        predict_epsilon=bool(args.predict_epsilon),
        diffuser_eef_pos_only=bool(args.diffuser_eef_pos_only),
    )
    if args.diffuser_eef_pos_only:
        eef_pos_slice = _infer_eef_pos_slice(data_path)
        if eef_pos_slice != (10, 13):
            raise ValueError(
                f"Expected robot0_eef_pos slice (10, 13) for robomimic low-dim rollouts, got {eef_pos_slice}."
            )

    cfg_training = TrainingConfig(
        data=[data_path],
        checkpoint_dir=checkpoint_dir,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        lr_scheduler_enabled=not args.disable_lr_scheduler,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_steps=args.max_steps,
        num_saves=args.num_saves,
        num_evals=args.num_evals,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        wandb_tags=("train_sope_film", data_path.stem),
    )

    train_sope(
        cfg_dataset=cfg_dataset,
        cfg_diffusion=cfg_diffusion,
        cfg_training=cfg_training,
    )


if __name__ == "__main__":
    main()
