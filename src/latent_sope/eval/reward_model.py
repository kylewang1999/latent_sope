"""Reward functions for scoring synthetic trajectories.

Provides both ground-truth (analytical) reward functions for known environments
and a learned reward model fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.latent_sope.robomimic_interface.encoders import LowDimConcatEncoder
from src.latent_sope.robomimic_interface.rollout import (
    RolloutLatentTrajectory,
    load_rollout_latents,
)


# ─── Ground-Truth Reward Functions ───────────────────────────────────────────


class GroundTruthRewardFn(Protocol):
    """Protocol for analytical reward functions operating on decoded obs."""

    def __call__(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute per-step rewards from an obs dict.

        Args:
            obs_dict: mapping of obs key -> (T, *obs_shape) arrays.

        Returns:
            rewards: (T,) array of scalar rewards.
        """
        ...


class LiftRewardFn:
    """Ground-truth sparse reward for robosuite Lift task.

    Success = cube z-position > table_height + height_threshold.
    Reward = reward_scale when successful, 0.0 otherwise.

    The ``object`` obs key layout (10-dim) is:
        [0:3]  cube_pos (x, y, z)
        [3:7]  cube_quat (x, y, z, w)
        [7:10] gripper_to_cube_pos (dx, dy, dz)

    So ``cube_z = obs_dict["object"][:, 2]``.
    """

    def __init__(
        self,
        table_height: float = 0.8,
        height_threshold: float = 0.04,
        reward_scale: float = 1.0,
    ):
        self.table_height = table_height
        self.height_threshold = height_threshold
        self.reward_scale = reward_scale
        self.success_z = table_height + height_threshold

    def __call__(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        cube_z = np.asarray(obs_dict["object"])[:, 2]
        return np.where(cube_z > self.success_z, self.reward_scale, 0.0).astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"LiftRewardFn(table_height={self.table_height}, "
            f"height_threshold={self.height_threshold}, "
            f"success_z={self.success_z:.4f})"
        )


def make_lift_encoder(
    obs_keys: Optional[Sequence[str]] = None,
) -> LowDimConcatEncoder:
    """Build a LowDimConcatEncoder configured for Lift low-dim obs decoding.

    The obs_keys must be sorted alphabetically (matching how data was collected).
    Default: ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'].
    """
    if obs_keys is None:
        obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    encoder = LowDimConcatEncoder(obs_keys=list(obs_keys))
    # Per-key dims for single-frame latent (NOT frame_stack * dim)
    encoder.obs_dims = {
        "object": 10,
        "robot0_eef_pos": 3,
        "robot0_eef_quat": 4,
        "robot0_gripper_qpos": 2,
    }
    encoder.obs_shapes = {
        "object": (10,),
        "robot0_eef_pos": (3,),
        "robot0_eef_quat": (4,),
        "robot0_gripper_qpos": (2,),
    }
    return encoder


def score_trajectories_gt(
    reward_fn: GroundTruthRewardFn,
    encoder: LowDimConcatEncoder,
    states: np.ndarray,
    actions: np.ndarray,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score synthetic trajectories using a ground-truth reward function.

    Args:
        reward_fn: analytical reward function (e.g. LiftRewardFn).
        encoder: LowDimConcatEncoder with obs_keys/obs_dims set (for decoding).
        states: (B, T, state_dim) generated states (unnormalized latents).
        actions: (B, T, action_dim) generated actions.
        gamma: discount factor.

    Returns:
        returns: (B,) discounted return per trajectory.
        rewards: (B, T) per-step rewards.
    """
    states = np.asarray(states, dtype=np.float32)
    B, T, _ = states.shape
    all_rewards = np.zeros((B, T), dtype=np.float32)

    for i in range(B):
        obs_dict = encoder.decode_to_obs_dict(states[i])  # each value is (T, dim)
        all_rewards[i] = reward_fn(obs_dict)

    discounts = gamma ** np.arange(T, dtype=np.float64)
    returns = (all_rewards.astype(np.float64) * discounts[None, :]).sum(axis=1)

    return returns.astype(np.float32), all_rewards


class RewardMLP(nn.Module):
    """Small MLP that predicts scalar reward from (latent_state, action)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict rewards. states: (*, S), actions: (*, A) -> (*)."""
        x = torch.cat([states, actions], dim=-1)
        return self.net(x).squeeze(-1)


def _collect_tuples_from_trajectories(
    trajectories: Sequence[RolloutLatentTrajectory],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (z, a, r) tuples from rollout trajectories."""
    all_z, all_a, all_r = [], [], []
    for traj in trajectories:
        latents = np.asarray(traj.latents, dtype=np.float32)
        # Handle (T, frame_stack, D) latents — take first frame
        if latents.ndim == 3:
            latents = latents[:, 0, :]
        actions = np.asarray(traj.actions, dtype=np.float32)
        rewards = np.asarray(traj.rewards, dtype=np.float32)
        T = min(len(latents), len(actions), len(rewards))
        all_z.append(latents[:T])
        all_a.append(actions[:T])
        all_r.append(rewards[:T])
    return np.concatenate(all_z), np.concatenate(all_a), np.concatenate(all_r)


def train_reward_model(
    rollout_paths: Sequence[Path],
    state_dim: int,
    action_dim: int,
    *,
    hidden: int = 128,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True,
) -> RewardMLP:
    """Train a reward MLP on offline (z, a, r) data.

    Args:
        rollout_paths: paths to .h5/.npz rollout files.
        state_dim: latent state dimension.
        action_dim: action dimension.

    Returns:
        Trained RewardMLP on the specified device.
    """
    trajectories = [load_rollout_latents(p) for p in rollout_paths]
    z, a, r = _collect_tuples_from_trajectories(trajectories)

    if verbose:
        print(f"Reward model training data: {len(z)} samples")
        print(f"  reward stats: mean={r.mean():.4f}, std={r.std():.4f}, "
              f"min={r.min():.4f}, max={r.max():.4f}")

    dataset = TensorDataset(
        torch.from_numpy(z),
        torch.from_numpy(a),
        torch.from_numpy(r),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = RewardMLP(state_dim, action_dim, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n = 0
        for z_b, a_b, r_b in loader:
            z_b, a_b, r_b = z_b.to(device), a_b.to(device), r_b.to(device)
            r_hat = model(z_b, a_b)
            loss = loss_fn(r_hat, r_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(z_b)
            n += len(z_b)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}: MSE = {epoch_loss / n:.6f}")

    model.eval()
    return model


def score_trajectories(
    model: RewardMLP,
    states: np.ndarray,
    actions: np.ndarray,
    gamma: float = 1.0,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Score synthetic trajectories with a trained reward model.

    Args:
        model: trained RewardMLP.
        states: (B, T, state_dim) generated states.
        actions: (B, T, action_dim) generated actions.
        gamma: discount factor.

    Returns:
        returns: (B,) discounted return per trajectory.
        rewards: (B, T) per-step predicted rewards.
    """
    model.eval()
    B, T, _ = states.shape

    with torch.no_grad():
        s = torch.from_numpy(states).float().to(device)
        a = torch.from_numpy(actions).float().to(device)
        r = model(s, a)  # (B, T)
        rewards = r.cpu().numpy()

    # Discounted returns
    discounts = gamma ** np.arange(T)
    returns = (rewards * discounts[None, :]).sum(axis=1)

    return returns, rewards
