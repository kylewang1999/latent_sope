#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sope_diffuser import SopeDiffusionConfig
from src.sope_interface.dataset import (
    SopeGymChunkDatasetConfig,
    load_sope_gym_dataset,
    make_sope_gym_chunk_dataloader,
    split_sope_gym_episodes,
    summarize_sope_gym_episodes,
    train_eval_split_sope_gym_episodes,
)
from src.train import TrainingConfig, _assign_dataset_stats, train_sope_with_loaders
from src.utils import PATHS, make_log_dir


def _resolve_default_checkpoint_dir() -> Path:
    return Path(make_log_dir("train_sope_gym", verbose=False))


def _resolve_default_data_path() -> Path:
    return (PATHS.repo_root / "data" / "sope_gym_data" / "IAcrobat").resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the SOPE chunk diffusion model on a SOPE Gym pdataset export."
    )
    parser.add_argument("--data", type=Path, default=_resolve_default_data_path())
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--disable-lr-scheduler", action="store_true")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", choices=("cosine",))
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--dim-mults",
        type=int,
        nargs="+",
        default=(1, 2),
        help="TemporalUnet channel multipliers, e.g. --dim-mults 1 2 4.",
    )
    parser.add_argument("--diffusion-steps", type=int, default=256)
    parser.add_argument("--attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--num-saves", type=int, default=10)
    parser.add_argument("--num-evals", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--normalization-source",
        type=str,
        default="asset",
        choices=("asset", "computed"),
        help="Use shipped normalization.json stats or recompute shared chunk stats from the train split.",
    )
    parser.add_argument("--diffuser-eef-pos-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="wkt_sope")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default="train_sope_gym")
    parser.add_argument("--wandb-group", type=str, default="sope_gym_diffusion")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=("online", "offline", "disabled"))
    return parser


def main() -> None:
    args = build_parser().parse_args()

    data_path = args.data.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else _resolve_default_checkpoint_dir()

    bundle = load_sope_gym_dataset(data_path)
    state_dim = int(bundle.observations.shape[-1])
    action_dim = int(bundle.actions.shape[-1])

    cfg_dataset = SopeGymChunkDatasetConfig(
        chunk_size=args.chunk_size,
        stride=args.stride,
        frame_stack=args.frame_stack,
        state_dim=state_dim,
        action_dim=action_dim,
        normalize=bool(args.normalize),
        normalization_source=args.normalization_source,
        return_metadata=True,
    )
    cfg_diffusion = SopeDiffusionConfig(
        chunk_horizon=args.chunk_size,
        frame_stack=args.frame_stack,
        state_dim=state_dim,
        action_dim=action_dim,
        diffusion_steps=args.diffusion_steps,
        dim_mults=tuple(args.dim_mults),
        attention=bool(args.attention),
        diffuser_eef_pos_only=bool(args.diffuser_eef_pos_only),
    )
    cfg_training = TrainingConfig(
        data=[data_path],
        data_kind="sope_gym",
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
        wandb_tags=("train_sope_gym", data_path.stem, args.normalization_source),
    )

    episodes = split_sope_gym_episodes(bundle)
    train_episodes, eval_episodes = train_eval_split_sope_gym_episodes(
        episodes,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )

    train_loader, train_stats = make_sope_gym_chunk_dataloader(
        episodes=train_episodes,
        config=cfg_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = None
    if eval_episodes:
        eval_loader, _ = make_sope_gym_chunk_dataloader(
            episodes=eval_episodes,
            config=cfg_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
        )
        _assign_dataset_stats(eval_loader.dataset, train_stats if cfg_dataset.normalize else None)

    episode_summary = summarize_sope_gym_episodes(episodes, max_episodes=3)
    print(f"data: {data_path}")
    print(
        "episodes:"
        f" total={episode_summary['num_episodes']}"
        f" train={len(train_episodes)}"
        f" eval={len(eval_episodes)}"
        f" length_mean={episode_summary['length_stats']['mean']:.1f}"
    )
    print(f"normalization_source: {cfg_dataset.normalization_source}")
    if bundle.normalization is not None:
        print("normalization_json: found")

    train_sope_with_loaders(
        cfg_dataset=cfg_dataset,
        cfg_diffusion=cfg_diffusion,
        cfg_training=cfg_training,
        loader=train_loader,
        eval_loader=eval_loader,
        stats=train_stats,
        train_data_refs=[episode.episode_id for episode in train_episodes],
        eval_data_refs=[episode.episode_id for episode in eval_episodes],
    )


if __name__ == "__main__":
    main()
