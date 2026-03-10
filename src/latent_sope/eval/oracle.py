"""Oracle (on-policy) baseline for off-policy evaluation.

Computes the ground-truth expected return V^pi of a policy by running it
on-policy in the environment. This value is the target that the OPE pipeline
(Steps 2-6) attempts to estimate without running the policy.

Typical workflow:
    1. Load a robomimic checkpoint with ``load_checkpoint()``.
    2. Call ``oracle_value(ckpt, num_rollouts=100)`` to get the true value.
    3. Call ``save_oracle_result()`` to persist the result as JSON.
    4. At evaluation time (Step 7), call ``load_oracle_result()`` and compare
       the OPE estimate against the saved ground truth.

The oracle only needs to be computed once per (policy, horizon) pair and can
be reused across many OPE experiments.

Functions:
    oracle_value              -- Run policy in env, return mean undiscounted return.
    oracle_value_from_trajectories -- Compute (optionally discounted) returns
                                     from pre-collected RolloutLatentTrajectory objects.
    save_oracle_result        -- Persist an OracleResult to JSON.
    load_oracle_result        -- Reload a saved OracleResult from JSON.

Example::

    ckpt = load_checkpoint("path/to/run_dir", ckpt_path="last.pth")
    result = oracle_value(ckpt, num_rollouts=100, horizon=60)
    save_oracle_result("data/oracle/my_policy.json", result,
                       policy_name="diffusion_policy_epoch_50")

    # Later, in a different session:
    result = load_oracle_result("data/oracle/my_policy.json")
    print(result.mean_return)
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
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


def _oracle_worker(
    ckpt_run_dir: str,
    ckpt_path: str,
    num_rollouts: int,
    horizon: int,
    worker_id: int,
) -> np.ndarray:
    """Worker function for parallel oracle rollouts.

    Each worker builds its own policy (CPU) and env, runs its assigned rollouts,
    and returns the per-rollout rewards. Runs on CPU to allow maximum parallelism.
    """
    import logging as _logging
    from src.latent_sope.robomimic_interface.checkpoints import (
        load_checkpoint as _load_checkpoint,
        build_rollout_policy_from_checkpoint as _build_policy,
        build_env_from_checkpoint as _build_env,
    )
    from src.latent_sope.robomimic_interface.rollout import rollout as _rollout
    from src.latent_sope.utils.common import CONSOLE_LOGGER as _LOGGER

    _prev = _LOGGER.level
    _LOGGER.setLevel(_logging.WARNING)
    try:
        ckpt = _load_checkpoint(ckpt_run_dir, ckpt_path=ckpt_path)
        policy = _build_policy(ckpt, device="cpu", verbose=False)
        env = _build_env(ckpt, render=False, render_offscreen=False, verbose=False)

        returns = np.empty(num_rollouts, dtype=np.float64)
        for i in range(num_rollouts):
            stats = _rollout(policy=policy, env=env, horizon=horizon, render=False)
            returns[i] = stats.total_reward
        return returns
    finally:
        _LOGGER.setLevel(_prev)


def oracle_value(
    ckpt: RobomimicCheckpoint,
    num_rollouts: int = 100,
    horizon: int = 60,
    gamma: float = 1.0,
    device: str = "cuda",
    verbose: bool = True,
    num_workers: int = 0,
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
        device: torch device for policy (only used when num_workers=0; parallel
            workers always use CPU to maximize parallelism)
        verbose: print progress
        num_workers: number of parallel worker processes. 0 = serial (original
            behavior). -1 = auto-detect (uses os.cpu_count()).

    Returns:
        OracleResult with mean/std return and per-rollout returns
    """
    if gamma < 1.0:
        raise ValueError(
            f"oracle_value() only supports gamma=1.0 (got {gamma}). "
            "For discounted returns, collect trajectories with collect_rollouts() "
            "and use oracle_value_from_trajectories() instead."
        )

    if num_workers == -1:
        num_workers = os.cpu_count() or 1
    num_workers = min(num_workers, num_rollouts)

    # ---------- serial path (original behavior) ----------
    if num_workers <= 0:
        policy = build_rollout_policy_from_checkpoint(ckpt, device=device, verbose=False)
        env = build_env_from_checkpoint(ckpt, render=False, render_offscreen=False, verbose=False)

        returns = np.empty(num_rollouts, dtype=np.float64)

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

    # ---------- parallel path ----------
    if verbose:
        print(f"  oracle: launching {num_workers} workers for {num_rollouts} rollouts")

    # Distribute rollouts evenly across workers
    base, remainder = divmod(num_rollouts, num_workers)
    worker_counts = [base + (1 if i < remainder else 0) for i in range(num_workers)]

    ckpt_run_dir = str(ckpt.run_dir)
    ckpt_rel_path = str(ckpt.ckpt_path.relative_to(ckpt.run_dir))

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(_oracle_worker, ckpt_run_dir, ckpt_rel_path, count, horizon, wid)
            for wid, count in enumerate(worker_counts)
            if count > 0
        ]
        all_returns = []
        for fut in futures:
            all_returns.append(fut.result())

    returns = np.concatenate(all_returns)
    if verbose:
        print(f"  oracle: {num_rollouts} rollouts done, "
              f"mean={returns.mean():.3f} ± {returns.std():.3f}")

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


def save_oracle_result(path: Path, result: OracleResult, policy_name: Optional[str] = None) -> Path:
    """Save an OracleResult to a JSON file.

    Args:
        path: output path (should end in .json)
        result: OracleResult to save
        policy_name: optional label for the policy (e.g. "diffusion_policy_epoch_50")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mean_return": result.mean_return,
        "std_return": result.std_return,
        "returns": result.returns.tolist(),
        "gamma": result.gamma,
        "horizon": result.horizon,
        "num_rollouts": result.num_rollouts,
    }
    if policy_name is not None:
        data["policy_name"] = policy_name
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_oracle_result(path: Path) -> OracleResult:
    """Load an OracleResult from a JSON file saved by save_oracle_result."""
    with open(path, "r") as f:
        data = json.load(f)
    return OracleResult(
        mean_return=data["mean_return"],
        std_return=data["std_return"],
        returns=np.array(data["returns"], dtype=np.float32),
        gamma=data["gamma"],
        horizon=data["horizon"],
        num_rollouts=data["num_rollouts"],
    )
