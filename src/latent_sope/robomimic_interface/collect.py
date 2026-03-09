"""Batch collection of latent rollout trajectories from a robomimic policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from src.latent_sope.robomimic_interface.checkpoints import (
    RobomimicCheckpoint,
    build_algo_from_checkpoint,
    build_rollout_policy_from_checkpoint,
    build_env_from_checkpoint,
)
from src.latent_sope.robomimic_interface.rollout import (
    rollout,
    PolicyFeatureHook,
    RolloutLatentRecorder,
    RolloutLatentTrajectory,
    save_rollout_latents,
)
from src.latent_sope.utils.common import CONSOLE_LOGGER


@dataclass(frozen=True)
class CollectionResult:
    """Summary of a batch rollout collection run."""

    output_dir: Path
    num_rollouts: int
    paths: list  # list of Path objects to saved .h5 files
    total_rewards: np.ndarray  # (num_rollouts,)
    success_rates: np.ndarray  # (num_rollouts,)


def discover_obs_keys(
    ckpt: RobomimicCheckpoint,
    low_dim_only: bool = True,
) -> list:
    """Discover the obs keys used by a policy checkpoint.

    Args:
        ckpt: loaded robomimic checkpoint
        low_dim_only: if True, filter to only low-dim modality keys
            (excludes rgb/depth/scan). Recommended for LowDimConcatEncoder.
    """
    algo = build_algo_from_checkpoint(ckpt, device="cpu")
    all_keys = sorted(list(algo.global_config.all_obs_keys))

    if low_dim_only:
        import robomimic.utils.obs_utils as ObsUtils
        all_keys = [
            k for k in all_keys
            if not ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb")
            and not ObsUtils.key_is_obs_modality(key=k, obs_modality="depth")
            and not ObsUtils.key_is_obs_modality(key=k, obs_modality="scan")
        ]

    return all_keys


def collect_rollouts(
    ckpt: RobomimicCheckpoint,
    output_dir: Path,
    num_rollouts: int = 100,
    horizon: int = 60,
    obs_keys: Optional[Sequence[str]] = None,
    feat_type: str = "low_dim_concat",
    store_obs: bool = True,
    device: str = "cuda",
    verbose: bool = True,
) -> CollectionResult:
    """Collect multiple latent rollout trajectories and save them to disk.

    Args:
        ckpt: loaded robomimic checkpoint
        output_dir: directory to save .h5 rollout files
        num_rollouts: number of episodes to collect
        horizon: max steps per episode
        obs_keys: obs keys to record (auto-discovered from policy if None)
        feat_type: "low_dim_concat" or "high_dim_encode"
        store_obs: whether to store raw observations alongside latents
        device: torch device for policy
        verbose: print progress

    Returns:
        CollectionResult with paths and summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = build_rollout_policy_from_checkpoint(ckpt, device=device, verbose=False)
    env = build_env_from_checkpoint(ckpt, render=False, render_offscreen=False, verbose=False)

    if obs_keys is None:
        obs_keys = discover_obs_keys(ckpt)
    obs_keys = sorted(obs_keys)

    paths = []
    total_rewards = []
    success_rates = []

    # Suppress per-rollout @timeit logging
    prev_level = CONSOLE_LOGGER.level
    CONSOLE_LOGGER.setLevel(logging.WARNING)
    try:
        for i in range(num_rollouts):
            feature_hook = PolicyFeatureHook(
                policy,
                obs_keys=obs_keys,
                feat_type=feat_type,
            )
            recorder = RolloutLatentRecorder(
                feature_hook,
                obs_keys=obs_keys,
                store_obs=store_obs,
                store_next_obs=False,
            )

            stats = rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=False,
                recorder=recorder,
            )
            traj = recorder.finalize(stats)

            save_path = output_dir / f"rollout_{i:04d}.h5"
            save_rollout_latents(save_path, traj)
            feature_hook.close()

            paths.append(save_path)
            total_rewards.append(stats.total_reward)
            success_rates.append(stats.success_rate)

            if verbose and (i + 1) % max(1, num_rollouts // 10) == 0:
                print(f"  [{i+1}/{num_rollouts}] reward={stats.total_reward:.1f}, "
                      f"success={stats.success_rate:.0%}, "
                      f"latents={traj.latents.shape}")
    finally:
        CONSOLE_LOGGER.setLevel(prev_level)

    return CollectionResult(
        output_dir=output_dir,
        num_rollouts=num_rollouts,
        paths=paths,
        total_rewards=np.array(total_rewards, dtype=np.float32),
        success_rates=np.array(success_rates, dtype=np.float32),
    )
