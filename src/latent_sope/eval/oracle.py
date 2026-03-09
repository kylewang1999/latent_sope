"""Oracle (on-policy) baseline: run a policy in the environment and compute ground truth value."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from src.latent_sope.robomimic_interface.checkpoints import (
    RobomimicCheckpoint,
    load_checkpoint,
    build_rollout_policy_from_checkpoint,
    build_env_from_checkpoint,
)
from src.latent_sope.robomimic_interface.rollout import rollout, RolloutStats
from src.latent_sope.utils.common import CONSOLE_LOGGER


@dataclass(frozen=True)
class OracleResult:
    """Ground-truth policy value computed by on-policy rollouts."""

    mean_return: float
    std_return: float
    returns: np.ndarray  # (K,) per-rollout returns
    gamma: float
    horizon: int
    num_rollouts: int


def oracle_value(
    ckpt: RobomimicCheckpoint,
    num_rollouts: int = 100,
    horizon: int = 60,
    gamma: float = 1.0,
    device: str = "cuda",
    verbose: bool = True,
) -> OracleResult:
    """Compute the ground-truth value of a policy by running it in the environment.

    Only supports gamma=1.0 (undiscounted returns) because rollout() only returns
    total reward. For discounted returns, use oracle_value_from_trajectories() with
    pre-collected RolloutLatentTrajectory objects that have per-step rewards.

    Args:
        ckpt: loaded robomimic checkpoint (from load_checkpoint)
        num_rollouts: number of episodes to average over
        horizon: max steps per episode
        gamma: discount factor (must be 1.0; see above)
        device: torch device for policy
        verbose: print progress

    Returns:
        OracleResult with mean/std return and per-rollout returns
    """
    if gamma < 1.0:
        raise ValueError(
            f"oracle_value() only supports gamma=1.0 (got {gamma}). "
            "For discounted returns, collect trajectories with collect_rollouts() "
            "and use oracle_value_from_trajectories() instead."
        )

    policy = build_rollout_policy_from_checkpoint(ckpt, device=device, verbose=False)
    env = build_env_from_checkpoint(ckpt, render=False, render_offscreen=False, verbose=False)

    returns = np.empty(num_rollouts, dtype=np.float64)

    # Suppress per-rollout @timeit logging (one line per rollout is too noisy)
    prev_level = CONSOLE_LOGGER.level
    CONSOLE_LOGGER.setLevel(logging.WARNING)
    try:
        for i in range(num_rollouts):
            stats = rollout(policy=policy, env=env, horizon=horizon, render=False)
            returns[i] = stats.total_reward

            if verbose and (i + 1) % max(1, num_rollouts // 10) == 0:
                print(f"  oracle rollout [{i+1}/{num_rollouts}] "
                      f"running mean={returns[:i+1].mean():.3f}")
    finally:
        CONSOLE_LOGGER.setLevel(prev_level)

    return OracleResult(
        mean_return=float(returns.mean()),
        std_return=float(returns.std()),
        returns=returns.astype(np.float32),
        gamma=gamma,
        horizon=horizon,
        num_rollouts=num_rollouts,
    )


def oracle_value_from_trajectories(
    trajectories: Sequence,
    gamma: float = 1.0,
) -> OracleResult:
    """Compute oracle value from pre-collected RolloutLatentTrajectory objects.

    Useful when gamma < 1.0, since we need per-step rewards.

    Args:
        trajectories: sequence of RolloutLatentTrajectory objects
        gamma: discount factor
    """
    returns = []
    horizon = 0
    for traj in trajectories:
        rewards = traj.rewards
        if gamma >= 1.0:
            returns.append(float(rewards.sum()))
        else:
            discounts = gamma ** np.arange(len(rewards))
            returns.append(float((discounts * rewards).sum()))
        horizon = max(horizon, len(rewards))

    returns = np.array(returns, dtype=np.float32)
    return OracleResult(
        mean_return=float(returns.mean()),
        std_return=float(returns.std()),
        returns=returns,
        gamma=gamma,
        horizon=horizon,
        num_rollouts=len(returns),
    )
