from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import torch

from src.robomimic_interface.rollout import RolloutLatentTrajectory, load_rollout_latents
from src.utils import resolve_device

if TYPE_CHECKING:
    from src.sope_diffuser import SopeDiffuser


@dataclass(frozen=True)
class TrajectoryStateError:
    state_mse: float
    state_rmse: float
    state_mse_per_step: np.ndarray
    state_rmse_per_step: np.ndarray
    state_mse_per_dim: np.ndarray
    num_trajectories: int
    max_trajectory_length: int
    trajectory_lengths: np.ndarray


@dataclass(frozen=True)
class DiffusionTrajectoryStateErrorReport:
    unguided: TrajectoryStateError
    guided: Optional[TrajectoryStateError]


def load_diffusion_checkpoint(
    checkpoint_path: Path,
    *,
    device: Optional[str] = None,
    policy: Optional[Any] = None,
    behavior_policy: Optional[Any] = None,
) -> tuple[SopeDiffuser, dict[str, Any]]:
    from src.sope_diffuser import NormalizationStats, SopeDiffuser, SopeDiffusionConfig

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(str(checkpoint_path), map_location="cpu")

    cfg_diffusion = SopeDiffusionConfig(**payload["diffusion_config"])
    stats_payload = payload.get("normalization_stats")
    stats = None
    if stats_payload is not None:
        stats = NormalizationStats(
            mean=np.asarray(stats_payload["mean"], dtype=np.float32),
            std=np.asarray(stats_payload["std"], dtype=np.float32),
        )

    diffuser = SopeDiffuser(
        cfg=cfg_diffusion,
        normalization_stats=stats,
        device=device or resolve_device(prefer_cuda=True),
        policy=policy,
        behavior_policy=behavior_policy,
    )
    diffuser.diffusion.load_state_dict(payload["diffusion_state_dict"])
    diffuser.diffusion.eval()
    return diffuser, payload


def generate_full_trajectory(
    diffuser: SopeDiffuser,
    initial_states: torch.Tensor,
    *,
    max_length: int,
    guided: bool = False,
    verbose: bool = False,
    **guidance_kw: Any,
) -> tuple[np.ndarray, np.ndarray]:
    diffuser.diffusion.eval()

    batch_size = int(initial_states.shape[0])
    chunk_horizon = int(diffuser.cfg.chunk_horizon)
    frame_stack = int(diffuser.cfg.frame_stack)
    total_horizon = int(diffuser.cfg.total_chunk_horizon)

    all_states = np.zeros((batch_size, max_length, diffuser.state_dim), dtype=np.float32)
    all_actions = np.zeros((batch_size, max_length, diffuser.action_dim), dtype=np.float32)

    init_dev = initial_states.to(diffuser.device)
    dummy_actions = torch.zeros(
        batch_size,
        diffuser.action_dim,
        device=diffuser.device,
        dtype=init_dev.dtype,
    )
    init_padded = torch.cat([init_dev, dummy_actions], dim=-1)
    init_norm = diffuser.normalizer(init_padded)[:, : diffuser.state_dim]
    cond_states = init_norm.unsqueeze(1).expand(-1, frame_stack, -1).clone()

    total_generated = 0
    while total_generated < max_length:
        steps_to_add = min(chunk_horizon, max_length - total_generated)
        cond = {t: cond_states[:, t, :] for t in range(frame_stack)}
        sample = diffuser.diffusion.conditional_sample(
            shape=(batch_size, total_horizon, diffuser.transition_dim),
            cond=cond,
            guided=guided,
            verbose=verbose,
            **guidance_kw,
        )
        chunk = diffuser.unnormalizer(sample.trajectories)
        gen_states = chunk[:, frame_stack:, : diffuser.state_dim]
        gen_actions = chunk[:, frame_stack:, diffuser.state_dim :]

        t_end = total_generated + steps_to_add
        all_states[:, total_generated:t_end, :] = (
            gen_states[:, :steps_to_add, :].detach().cpu().numpy()
        )
        all_actions[:, total_generated:t_end, :] = (
            gen_actions[:, :steps_to_add, :].detach().cpu().numpy()
        )
        total_generated = t_end

        if total_generated >= max_length:
            break

        cond_states = sample.trajectories[:, -frame_stack:, : diffuser.state_dim].clone()

    return all_states, all_actions


def trajectory_state_error(
    real_states: np.ndarray,
    generated_states: np.ndarray,
    *,
    trajectory_lengths: Optional[np.ndarray] = None,
) -> TrajectoryStateError:
    real_states = np.asarray(real_states, dtype=np.float32)
    generated_states = np.asarray(generated_states, dtype=np.float32)
    if real_states.shape != generated_states.shape:
        raise ValueError(
            f"State shape mismatch: real={real_states.shape}, generated={generated_states.shape}"
        )

    batch_size, max_length, state_dim = real_states.shape
    if trajectory_lengths is None:
        lengths = np.full(batch_size, max_length, dtype=np.int64)
        mask = np.ones((batch_size, max_length), dtype=bool)
    else:
        lengths = np.asarray(trajectory_lengths, dtype=np.int64)
        if lengths.shape != (batch_size,):
            raise ValueError(
                f"trajectory_lengths must have shape {(batch_size,)}, got {lengths.shape}"
            )
        mask = np.arange(max_length)[None, :] < lengths[:, None]

    sq_err = (real_states - generated_states) ** 2 * mask[..., None]
    n_valid = max(int(mask.sum()), 1)
    n_valid_per_step = np.maximum(mask.sum(axis=0), 1)

    state_mse = float(sq_err.sum() / (n_valid * state_dim))
    state_mse_per_dim = sq_err.sum(axis=(0, 1)) / n_valid
    state_mse_per_step = sq_err.mean(axis=2).sum(axis=0) / n_valid_per_step

    return TrajectoryStateError(
        state_mse=state_mse,
        state_rmse=float(np.sqrt(state_mse)),
        state_mse_per_step=state_mse_per_step.astype(np.float32),
        state_rmse_per_step=np.sqrt(state_mse_per_step).astype(np.float32),
        state_mse_per_dim=state_mse_per_dim.astype(np.float32),
        num_trajectories=batch_size,
        max_trajectory_length=max_length,
        trajectory_lengths=lengths.astype(np.int64),
    )


def evaluate_diffusion_trajectory_state_error(
    diffuser: SopeDiffuser,
    trajectories: Sequence[RolloutLatentTrajectory],
    *,
    evaluate_guided: bool = True,
    guidance_kw: Optional[dict[str, Any]] = None,
) -> DiffusionTrajectoryStateErrorReport:
    if not trajectories:
        raise ValueError("At least one trajectory is required for evaluation.")

    max_length = max(int(traj.latents.shape[0]) for traj in trajectories)
    batch_size = len(trajectories)
    state_dim = diffuser.state_dim

    real_states = np.zeros((batch_size, max_length, state_dim), dtype=np.float32)
    initial_states = np.zeros((batch_size, state_dim), dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int64)

    for i, traj in enumerate(trajectories):
        traj_states = np.asarray(traj.latents, dtype=np.float32)
        if traj_states.ndim == 3:
            traj_states = traj_states[:, 0, :]
        if traj_states.shape[-1] != state_dim:
            raise ValueError(
                f"Trajectory {i} state_dim mismatch: expected {state_dim}, got {traj_states.shape[-1]}"
            )
        length = int(traj_states.shape[0])
        real_states[i, :length, :] = traj_states
        initial_states[i, :] = traj_states[0]
        lengths[i] = length

    initial_states_t = torch.from_numpy(initial_states).to(diffuser.device)
    unguided_states, _ = generate_full_trajectory(
        diffuser,
        initial_states_t,
        max_length=max_length,
        guided=False,
    )
    unguided_error = trajectory_state_error(
        real_states,
        unguided_states,
        trajectory_lengths=lengths,
    )

    guided_error = None
    if evaluate_guided and getattr(diffuser.diffusion, "policy", None) is not None:
        guided_states, _ = generate_full_trajectory(
            diffuser,
            initial_states_t,
            max_length=max_length,
            guided=True,
            **(guidance_kw or {}),
        )
        guided_error = trajectory_state_error(
            real_states,
            guided_states,
            trajectory_lengths=lengths,
        )

    return DiffusionTrajectoryStateErrorReport(
        unguided=unguided_error,
        guided=guided_error,
    )


def evaluate_saved_diffusion_trajectory_state_error(
    diffusion_checkpoint_path: Path,
    rollout_paths: Sequence[Path],
    *,
    policy_run_dir: Optional[Path] = None,
    policy_ckpt_path: Optional[Path] = None,
    max_trajectories: Optional[int] = None,
    device: Optional[str] = None,
    guidance_kw: Optional[dict[str, Any]] = None,
) -> DiffusionTrajectoryStateErrorReport:
    resolved_device = device or resolve_device(prefer_cuda=True)

    policy = None
    if policy_run_dir is not None:
        from src.robomimic_interface.checkpoints import (
            build_rollout_policy_from_checkpoint,
            load_checkpoint as load_policy_checkpoint,
        )

        ckpt = load_policy_checkpoint(policy_run_dir, ckpt_path=policy_ckpt_path)
        policy = build_rollout_policy_from_checkpoint(
            ckpt,
            device=resolved_device,
            verbose=False,
        )

    diffuser, _ = load_diffusion_checkpoint(
        diffusion_checkpoint_path,
        device=resolved_device,
        policy=policy,
    )

    trajectories = [load_rollout_latents(Path(path)) for path in rollout_paths]
    if max_trajectories is not None:
        trajectories = trajectories[:max_trajectories]

    return evaluate_diffusion_trajectory_state_error(
        diffuser,
        trajectories,
        evaluate_guided=policy is not None,
        guidance_kw=guidance_kw,
    )


__all__ = [
    "DiffusionTrajectoryStateErrorReport",
    "TrajectoryStateError",
    "evaluate_diffusion_trajectory_state_error",
    "evaluate_saved_diffusion_trajectory_state_error",
    "generate_full_trajectory",
    "load_diffusion_checkpoint",
    "trajectory_state_error",
]
