"""Batch collection of latent rollout trajectories from a robomimic policy.

Rolls out a behavior policy for many episodes, extracting latent features at
each timestep via ``PolicyFeatureHook``, and saves each trajectory as an HDF5
file. The resulting ``.h5`` files form the offline dataset consumed by Step 2
(``make_rollout_chunk_dataloader``).

Typical workflow:
    1. Load a robomimic checkpoint with ``load_checkpoint()``.
    2. Call ``collect_rollouts(ckpt, output_dir, num_rollouts=100)`` to generate
       the offline dataset. Each episode is saved as ``rollout_NNNN.h5``.
    3. Pass the directory (or file paths) to ``make_rollout_chunk_dataloader()``
       in Step 2 to chunk and batch the data for diffusion training.

The offline dataset only needs to be collected once per (policy, horizon) pair.
Subsequent experiments (different diffusion hyperparameters, guidance settings,
etc.) reuse the same data files.

Each ``.h5`` file contains:
    - ``latents``:  (T, frame_stack, Dz) encoder features per timestep
    - ``actions``:  (T, Da) actions taken by the behavior policy
    - ``rewards``:  (T,) per-step scalar rewards
    - ``dones``:    (T,) boolean termination flags
    - ``obs/``:     (optional) raw observations per key

Functions:
    collect_rollouts  -- Run policy N times, save trajectories to disk.
    discover_obs_keys -- Auto-discover low-dim observation keys from a checkpoint.

Example::

    ckpt = load_checkpoint("path/to/run_dir", ckpt_path="last.pth")
    collection = collect_rollouts(ckpt, output_dir="data/rollouts/lift/",
                                  num_rollouts=100, horizon=60)
    # collection.paths -> [Path('data/rollouts/lift/rollout_0000.h5'), ...]
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

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


def _collect_worker(
    ckpt_run_dir: str,
    ckpt_path: str,
    output_dir: str,
    rollout_indices: List[int],
    horizon: int,
    obs_keys: List[str],
    feat_type: str,
    store_obs: bool,
    worker_id: int,
) -> List[Tuple[str, float, float]]:
    """Worker function for parallel rollout collection.

    Each worker builds its own policy (CPU) and env, runs its assigned rollouts,
    saves .h5 files, and returns (path, reward, success) tuples.
    """
    import logging as _logging
    from src.latent_sope.robomimic_interface.checkpoints import (
        load_checkpoint as _load_checkpoint,
        build_rollout_policy_from_checkpoint as _build_policy,
        build_env_from_checkpoint as _build_env,
    )
    from src.latent_sope.robomimic_interface.rollout import (
        rollout as _rollout,
        PolicyFeatureHook as _PolicyFeatureHook,
        RolloutLatentRecorder as _RolloutLatentRecorder,
        save_rollout_latents as _save_rollout_latents,
    )
    from src.latent_sope.utils.common import CONSOLE_LOGGER as _LOGGER

    _prev = _LOGGER.level
    _LOGGER.setLevel(_logging.WARNING)
    try:
        ckpt = _load_checkpoint(ckpt_run_dir, ckpt_path=ckpt_path)
        policy = _build_policy(ckpt, device="cpu", verbose=False)
        env = _build_env(ckpt, render=False, render_offscreen=False, verbose=False)

        results = []
        for idx in rollout_indices:
            feature_hook = _PolicyFeatureHook(
                policy,
                obs_keys=obs_keys,
                feat_type=feat_type,
            )
            recorder = _RolloutLatentRecorder(
                feature_hook,
                obs_keys=obs_keys,
                store_obs=store_obs,
                store_next_obs=False,
            )

            stats = _rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=False,
                recorder=recorder,
            )
            traj = recorder.finalize(stats)

            save_path = Path(output_dir) / f"rollout_{idx:04d}.h5"
            _save_rollout_latents(save_path, traj)
            feature_hook.close()

            results.append((str(save_path), stats.total_reward, stats.success_rate))
        return results
    finally:
        _LOGGER.setLevel(_prev)


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
    num_workers: int = 0,
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
        device: torch device for policy (only used when num_workers=0; parallel
            workers always use CPU to maximize parallelism)
        verbose: print progress
        num_workers: number of parallel worker processes. 0 = serial (original
            behavior). -1 = auto-detect (uses os.cpu_count()).

    Returns:
        CollectionResult with paths and summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if obs_keys is None:
        obs_keys = discover_obs_keys(ckpt)
    obs_keys = sorted(obs_keys)

    if num_workers == -1:
        num_workers = os.cpu_count() or 1
    num_workers = min(num_workers, num_rollouts)

    # ---------- serial path (original behavior) ----------
    if num_workers <= 0:
        policy = build_rollout_policy_from_checkpoint(ckpt, device=device, verbose=False)
        env = build_env_from_checkpoint(ckpt, render=False, render_offscreen=False, verbose=False)

        paths = []
        total_rewards = []
        success_rates = []

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

    # ---------- parallel path ----------
    if verbose:
        print(f"  collect: launching {num_workers} workers for {num_rollouts} rollouts")

    # Distribute rollout indices evenly across workers
    all_indices = list(range(num_rollouts))
    chunks = [[] for _ in range(num_workers)]
    for i, idx in enumerate(all_indices):
        chunks[i % num_workers].append(idx)

    ckpt_run_dir = str(ckpt.run_dir)
    ckpt_rel_path = str(ckpt.ckpt_path.relative_to(ckpt.run_dir))

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _collect_worker,
                ckpt_run_dir, ckpt_rel_path, str(output_dir),
                chunk, horizon, list(obs_keys), feat_type, store_obs, wid,
            )
            for wid, chunk in enumerate(chunks)
            if chunk
        ]
        all_results = []
        for fut in futures:
            all_results.extend(fut.result())

    # Sort by rollout index (file name) to maintain deterministic ordering
    all_results.sort(key=lambda r: r[0])

    paths = [Path(r[0]) for r in all_results]
    total_rewards = np.array([r[1] for r in all_results], dtype=np.float32)
    success_rates_arr = np.array([r[2] for r in all_results], dtype=np.float32)

    if verbose:
        print(f"  collect: {num_rollouts} rollouts done, "
              f"mean reward={total_rewards.mean():.1f}, "
              f"success rate={success_rates_arr.mean():.0%}")

    return CollectionResult(
        output_dir=output_dir,
        num_rollouts=num_rollouts,
        paths=paths,
        total_rewards=total_rewards,
        success_rates=success_rates_arr,
    )
