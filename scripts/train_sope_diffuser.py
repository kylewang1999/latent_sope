"""
Train SOPE diffuser conditioned on image latent space.

Loads rollout latent trajectories collected by collect_rollout_latents.py and
trains a SopeDiffuser (TemporalUnet + GaussianDiffusion) over chunked (latent, action)
trajectories.

Usage:
    python train_sope_diffuser.py --task lift [--epochs 200] [--batch_size 256]
                                  [--data_dir /workspace/latent_sope/data/rollouts/lift]
                                  [--checkpoint_dir /workspace/latent_sope/data/sope_checkpoints/lift]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "robomimic"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "sope"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "clean_diffuser"))

import numpy as np

from src.latent_sope.diffusion.sope_diffuser import SopeDiffusionConfig
from src.latent_sope.diffusion.train import TrainingConfig, train
from src.latent_sope.robomimic_interface.dataset import RolloutChunkDatasetConfig

# latent_dim=64 (ResNet18 + SpatialSoftmax(32 kp) per camera, agentview only)
# action_dim=7  (6-DOF + gripper for lift/can)
LATENT_DIM = 64
ACTION_DIM = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["lift", "can"])
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="Directory with rollout .h5 files (recursively searched). "
                             "Defaults to data/rollouts/{task}")
    parser.add_argument("--checkpoint_dir", type=Path, default=None,
                        help="Where to save SOPE checkpoints. "
                             "Defaults to data/sope_checkpoints/{task}")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="Chunk horizon W (number of future steps per chunk)")
    parser.add_argument("--frame_stack", type=int, default=0,
                        help="Number of past frames to condition on")
    parser.add_argument("--stride", type=int, default=2,
                        help="Stride between chunk start indices")
    parser.add_argument("--diffusion_steps", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in checkpoint_dir")
    parser.add_argument("--lr_schedule", default=None, choices=[None, "cosine"],
                        help="LR scheduler: None or cosine")
    parser.add_argument("--predict_epsilon", type=lambda x: x.lower() != "false", default=True,
                        help="Whether to predict noise (True) or x0 (False)")
    args = parser.parse_args()

    data_dir = args.data_dir or (REPO_ROOT / "data" / "rollouts" / args.task)
    checkpoint_dir = args.checkpoint_dir or (REPO_ROOT / "data" / "sope_checkpoints" / args.task)

    # Collect all rollout .h5 files
    rollout_files = sorted(data_dir.rglob("*.h5"))
    if not rollout_files:
        raise FileNotFoundError(f"No rollout .h5 files found under {data_dir}")
    print(f"Found {len(rollout_files)} rollout files under {data_dir}")

    # Validate dim_mults: total_chunk_horizon = chunk_size + frame_stack must be divisible by 8
    total_horizon = args.chunk_size + args.frame_stack
    assert total_horizon % 8 == 0, (
        f"total_horizon={total_horizon} must be divisible by 8 (for dim_mults=(1,2,4,8)). "
        "Adjust --chunk_size or --frame_stack."
    )

    cfg_dataset = RolloutChunkDatasetConfig(
        chunk_size=args.chunk_size,
        stride=args.stride,
        frame_stack=args.frame_stack,
        source="latents",
        latents_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        normalize=True,
        return_metadata=False,
    )

    cfg_diffusion = SopeDiffusionConfig(
        chunk_horizon=args.chunk_size,
        frame_stack=args.frame_stack,
        state_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        diffusion_steps=args.diffusion_steps,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        loss_type="l2",
        action_weight=5.0,
        loss_discount=1.0,
        predict_epsilon=args.predict_epsilon,
        lr=args.lr,
        weight_decay=0.0,
        guided=False,
    )

    cfg_training = TrainingConfig(
        data=rollout_files,
        checkpoint_dir=checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=0.0,
        grad_clip=1.0,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
        lr_schedule=args.lr_schedule,
        prefer_cuda=True,
    )

    print(f"\nSopeDiffuser config:")
    print(f"  state_dim={LATENT_DIM}, action_dim={ACTION_DIM}, "
          f"chunk_horizon={args.chunk_size}, frame_stack={args.frame_stack}")
    print(f"  total_horizon={total_horizon}, diffusion_steps={args.diffusion_steps}, "
          f"predict_epsilon={args.predict_epsilon}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  checkpoint_dir={checkpoint_dir}\n")

    train(cfg_dataset, cfg_diffusion, cfg_training, resume=args.resume)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
