#!/usr/bin/env python3
"""Train SOPE models from saved robomimic latent trajectories.

This entrypoint trains the chunk diffusion model and reward predictor from a
directory of saved latent-trajectory files. The default no-arg path uses
`chunk_size=14` so the bundled robomimic defaults `To=2` and `Tp=16` satisfy
`Tp = To + H`, enabling exact whole-action-chunk scoring by default.

Example commands:

1. Train with the default postprocessed robomimic Lift MH visual-policy
   latents under `data/robomimic/lift/mh/postprocessed_for_ope/both`:

   ```bash
   python3 scripts/train_sope.py
   ```

   The default data uses the one-step `feat_type="both"` observation features
   for the visual diffusion policy
   `data/policy/rmimic-lift-mh-image-v15-diffusion_260123`, with width
   `147 = 19 low-dim + 128 visual` as described in

2. Train in the legacy state-only setting from the postprocessed robomimic
   low-dimensional demos under
   `data/robomimic/lift/ph/postprocessed_for_ope/low_dim`:

   ```bash
   python3 scripts/train_sope.py \
       --data data/robomimic/lift/ph/postprocessed_for_ope/low_dim
   ```

   This replaces the older state-only training path that read from
   `data/rollout/rmimic-lift-ph-lowdim_diffusion_260130`. With
   `--checkpoint-dir` omitted, the default log-directory prefix is
   `train-sope-feat:lowdim`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import PATHS, make_log_dir


DEFAULT_MATCHED_OBSERVATION_HORIZON = 2
DEFAULT_MATCHED_PREDICTION_HORIZON = 16
DEFAULT_CHUNK_SIZE = (
    DEFAULT_MATCHED_PREDICTION_HORIZON - DEFAULT_MATCHED_OBSERVATION_HORIZON
)
CHECKPOINT_FEAT_LABEL_ALIASES = {
    "lowdim": "lowdim",
    "low_dim": "lowdim",
    "low_dim_concat": "lowdim",
    "image": "image",
    "image_embedding": "image",
    "both": "both",
}


def _normalize_checkpoint_feat_label(feat_type: str | None) -> str | None:
    if feat_type is None:
        return None
    feat_type_token = str(feat_type).strip().lower()
    return CHECKPOINT_FEAT_LABEL_ALIASES.get(feat_type_token)


def _infer_checkpoint_feat_label_from_metadata(path: Path | str) -> str | None:
    try:
        latent_path = _resolve_latent_reference(path)
    except FileNotFoundError:
        return None

    if latent_path.suffix.lower() not in {".h5", ".hdf5"}:
        return None

    try:
        with h5py.File(latent_path, "r") as handle:
            feat_type = handle.attrs.get("feat_type")
    except OSError:
        return None

    if isinstance(feat_type, bytes):
        feat_type = feat_type.decode("utf-8")
    return _normalize_checkpoint_feat_label(feat_type)


def _infer_checkpoint_feat_label_from_path(path: Path | str) -> str | None:
    candidate_path = Path(path).expanduser()
    for part in reversed(candidate_path.parts):
        best_label = None
        best_start = -1
        lower_part = part.lower()
        for alias, label in CHECKPOINT_FEAT_LABEL_ALIASES.items():
            pattern = rf"(?:^|[^a-z0-9])({re.escape(alias)})(?=$|[^a-z0-9])"
            for match in re.finditer(pattern, lower_part):
                if match.start(1) >= best_start:
                    best_label = label
                    best_start = match.start(1)
        if best_label is not None:
            return best_label
    return None


def _resolve_default_checkpoint_description(data_path: Path | str) -> str:
    feat_label = _infer_checkpoint_feat_label_from_metadata(data_path)
    if feat_label is None:
        feat_label = _infer_checkpoint_feat_label_from_path(data_path)
    if feat_label is None:
        return "train-sope"
    return f"train-sope-feat:{feat_label}"


def _resolve_default_checkpoint_dir(data_path: Path | str) -> Path:
    return Path(make_log_dir(_resolve_default_checkpoint_description(data_path), verbose=False))


def _resolve_default_data_path() -> Path:
    return (
        PATHS.repo_root
        / "data"
        / "robomimic"
        / "lift"
        / "mh"
        / "postprocessed_for_ope"
        / "both"
    ).resolve()


def _resolve_latent_reference(path: Path | str) -> Path:
    path = Path(path).resolve()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.rglob("*.h5")) + sorted(path.rglob("*.hdf5")) + sorted(path.rglob("*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No latent trajectory files found under {path}.")
        return candidates[0]
    raise FileNotFoundError(f"Latent trajectory path not found: {path}")


def _infer_latent_shapes(path: Path | str) -> tuple[int, int, int]:
    latent_path = _resolve_latent_reference(path)
    with h5py.File(latent_path, "r") as handle:
        latents_shape = tuple(handle["latents"].shape)
        actions_shape = tuple(handle["actions"].shape)
        frame_stack = int(handle.attrs.get("frame_stack", latents_shape[1] if len(latents_shape) >= 3 else 1))

    latent_dim = int(latents_shape[-1])
    action_dim = int(actions_shape[-1])
    return latent_dim, action_dim, frame_stack


def _infer_eef_pos_slice(path: Path | str) -> tuple[int, int]:
    latent_path = _resolve_latent_reference(path)
    with h5py.File(latent_path, "r") as handle:
        if "obs" not in handle or "robot0_eef_pos" not in handle["obs"]:
            raise ValueError(
                f"Latent trajectory file {latent_path} does not contain obs/robot0_eef_pos "
                "needed to infer the low-dim slice."
            )

        offset = 0
        for key in sorted(handle["obs"].keys()):
            dim = int(handle["obs"][key].shape[-1])
            next_offset = offset + dim
            if key == "robot0_eef_pos":
                return offset, next_offset
            offset = next_offset

    raise ValueError(f"Could not infer robot0_eef_pos slice from {latent_path}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the canonical SOPE chunk diffusion model from saved latent trajectories."
    )
    parser.add_argument("--data", type=Path, default=_resolve_default_data_path())
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory. Defaults to a timestamped "
            "logs/train-sope-feat:<feature-space>_* path inferred from --data."
        ),
    )
    parser.add_argument("--reward-epochs", type=int, default=100)
    parser.add_argument("--reward-batch-size", type=int, default=1024)
    parser.add_argument("--reward-lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--disable-lr-scheduler", action="store_true")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", choices=("cosine",))
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Future chunk horizon H. Defaults to 14 so the bundled robomimic "
            "defaults To=2 and Tp=16 satisfy Tp = To + H."
        ),
    )
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

    from src.diffusion import RewardPredictorConfig, SopeDiffusionConfig
    from src.robomimic_interface.dataset import RolloutChunkDatasetConfig
    from src.train import (
        TrainingConfig,
        derive_phase_training_config,
        train_rewardpred,
        train_sope,
    )

    data_path = args.data.resolve()
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else _resolve_default_checkpoint_dir(data_path)
    )

    latent_dim, action_dim, frame_stack = _infer_latent_shapes(data_path)

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
    cfg_training_diffusion = derive_phase_training_config(
        cfg_training,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wandb_run_name_suffix="diffusion",
        wandb_tags=("diffusion",),
    )
    cfg_training_rewardpred = derive_phase_training_config(
        cfg_training,
        epochs=args.reward_epochs,
        batch_size=args.reward_batch_size,
        wandb_run_name_suffix="rewardpred",
        wandb_tags=("rewardpred",),
    )
    cfg_reward = RewardPredictorConfig(
        state_dim=latent_dim,
        action_dim=action_dim,
        lr=args.reward_lr,
    )

    train_sope(
        cfg_dataset=cfg_dataset,
        cfg_diffusion=cfg_diffusion,
        cfg_training=cfg_training_diffusion,
    )
    train_rewardpred(
        cfg_dataset=cfg_dataset,
        cfg_reward=cfg_reward,
        cfg_training=cfg_training_rewardpred,
    )


if __name__ == "__main__":
    main()
