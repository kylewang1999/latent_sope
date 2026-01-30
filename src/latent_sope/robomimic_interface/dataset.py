from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.latent_sope.robomimic_interface.encoders import extract_embeddings_batched
from src.latent_sope.robomimic_interface.rollout import (
    RolloutLatentTrajectory,
    load_rollout_latents,
)


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray  # shape (D,)
    std: np.ndarray  # shape (D,)


def compute_normalization_stats(x: np.ndarray) -> NormalizationStats:
    """Compute (mean, std) over (N, W, D) arrays."""
    if x.ndim != 3:
        raise ValueError(f"Expected (N, W, D), got {x.shape}")
    mean = x.mean(axis=(0, 1)).astype(np.float32)
    std = (x.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    return NormalizationStats(mean=mean, std=std)


def _stack_frames(z: np.ndarray, frame_stack: int) -> np.ndarray:
    """Concatenate past frames to form stacked features per timestep."""
    z = np.asarray(z)
    if z.ndim == 3 and z.shape[1] == frame_stack:
        return z.reshape(z.shape[0], -1)
    if frame_stack <= 1:
        if z.ndim == 3:
            return z.reshape(z.shape[0], -1)
        return z

    if z.ndim == 3:
        z = z.reshape(z.shape[0], -1)
    if z.ndim != 2:
        raise ValueError(f"Expected z with shape (T, D) or (T, S, ...), got {z.shape}")

    T, D = z.shape
    stacked = np.zeros((T, D * frame_stack), dtype=z.dtype)
    for t in range(T):
        frames = []
        for i in range(frame_stack):
            t_i = t - (frame_stack - 1 - i)
            if t_i < 0:
                t_i = 0
            frames.append(z[t_i])
        stacked[t] = np.concatenate(frames, axis=0)
    return stacked


@dataclass(frozen=True)
class RolloutChunkDatasetConfig:
    chunk_size: int = 8  # length W of each (states, actions) chunk
    stride: int = 8  # step between chunk start indices
    frame_stack: int = 2  # number of past frames for chunk diffusion to condition on
    source: str = (
        "latents"  # "latents" (read from @RolloutLatentTrajectory.latents) or
                   # "obs" (computed from @RolloutLatentTrajectory.obs using @encoder)
    )
    include_actions: bool = True  # include actions in chunk targets
    normalize: bool = False  # apply dataset-level normalization stats
    return_meta: bool = False  # return index/demo_id/t0 metadata for debugging


class RolloutChunkDataset:
    """Torch Dataset over chunked rollouts with (states, actions) pairs."""

    def __init__(
        self,
        traj: RolloutLatentTrajectory,
        config: RolloutChunkDatasetConfig,
        *,
        encoder: Optional[Any] = None,
        obs_keys: Optional[Sequence[str]] = None,
        encoder_device: str = "cuda",
        demo_id: Optional[str] = None,
    ):
        self.traj = traj
        self.config = config
        self.demo_id = demo_id
        self.encoder = encoder
        self.obs_keys = obs_keys
        self.encoder_device = encoder_device

        if config.source == "latents":
            assert traj.latents is not None, "rollout has no latents; save with encoder/feature hook enabled"
            z = traj.latents
        elif config.source == "obs":
            if traj.obs is None:
                raise ValueError("rollout has no obs; save with store_obs=True")
            if encoder is None:
                raise ValueError("encoder must be provided when source='obs'")
            obs = traj.obs
            if obs_keys is not None:
                obs = {k: obs[k] for k in obs_keys}
            z = extract_embeddings_batched(encoder, obs, device=encoder_device)
        else:
            raise ValueError(f"Unknown source={config.source}. Use 'latents' or 'obs'.")

        z = _stack_frames(z, config.frame_stack)
        actions = traj.actions

        if not config.include_actions:
            actions = np.zeros((actions.shape[0], 0), dtype=np.float32)

        z = np.asarray(z, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if z.ndim != 2:
            raise ValueError(f"Expected z to have shape (T, Dz), got {z.shape}")
        if actions.ndim != 2:
            raise ValueError(
                f"Expected actions to have shape (T, Da), got {actions.shape}"
            )
        if z.shape[0] != actions.shape[0]:
            raise ValueError(
                f"time length mismatch: z has {z.shape[0]}, actions has {actions.shape[0]}"
            )

        T = z.shape[0]
        W = int(config.chunk_size)
        last_start = T - (W + 1)
        if last_start < 0:
            raise RuntimeError(
                "No chunks produced. Check chunk_size/stride and rollout length."
            )

        states_list: List[np.ndarray] = []
        actions_list: List[np.ndarray] = []
        t0_list: List[int] = []
        for t0 in range(0, last_start + 1, config.stride):
            # SOPE trajectory chunks: (s_t, a_t, s_{t+1}, ..., s_{t+W}) and actions (a_t..a_{t+W-1}).
            states_list.append(z[t0 : t0 + W + 1])
            actions_list.append(actions[t0 : t0 + W])
            t0_list.append(int(t0))

        self.states = np.stack(states_list, axis=0).astype(np.float32)
        self.actions = np.stack(actions_list, axis=0).astype(np.float32)
        self.t0 = np.asarray(t0_list, dtype=np.int64)
        if self.states.ndim != 3:
            raise ValueError(
                f"Expected states to have shape (N, W+1, Dz), got {self.states.shape}"
            )
        if self.actions.ndim != 3:
            raise ValueError(
                f"Expected actions to have shape (N, W, Da), got {self.actions.shape}"
            )

        self.obs_dim = int(self.states.shape[-1])
        self.action_dim = int(self.actions.shape[-1])
        self.normalize = bool(config.normalize)
        self.return_meta = bool(config.return_meta)
        self._stats = self._compute_transition_stats() if self.normalize else None

    def _compute_transition_stats(self) -> NormalizationStats:
        x = np.concatenate([self.states[:, :-1, :], self.actions], axis=-1)
        return compute_normalization_stats(x)

    @property
    def normalization_stats(self) -> Optional[NormalizationStats]:
        return self._stats

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, idx: int):
        states = self.states[idx]
        actions = self.actions[idx]
        if self._stats is not None:
            obs_mean = self._stats.mean[: self.obs_dim]
            obs_std = self._stats.std[: self.obs_dim]
            act_mean = self._stats.mean[self.obs_dim :]
            act_std = self._stats.std[self.obs_dim :]
            states = (states - obs_mean) / obs_std
            actions = (actions - act_mean) / act_std
        states_t = torch.from_numpy(np.asarray(states, dtype=np.float32))
        actions_t = torch.from_numpy(np.asarray(actions, dtype=np.float32))
        if self.return_meta:
            meta = {"index": int(idx)}
            if self.demo_id is not None:
                meta["demo_id"] = self.demo_id
            meta["t0"] = int(self.t0[idx])
            return (states_t, actions_t), meta
        return states_t, actions_t


def _resolve_rollout_paths(paths: Sequence[Path]) -> List[Path]:
    resolved: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for ext in ("*.npz", "*.h5", "*.hdf5"):
                resolved.extend(sorted(p.glob(ext)))
        else:
            resolved.append(p)
    resolved = [p for p in resolved if p.is_file()]
    if not resolved:
        raise FileNotFoundError("No rollout files found.")
    return resolved


def make_rollout_chunk_dataloader(
    paths: Sequence[Path],
    config: RolloutChunkDatasetConfig,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
    *,
    encoder: Optional[Any] = None,
    obs_keys: Optional[Sequence[str]] = None,
    encoder_device: str = "cuda",
) -> Tuple[DataLoader, Optional[NormalizationStats]]:
    """Prepare a DataLoader from saved rollouts for chunk diffusion training.
    This works with multiple rollout latent .h5 files and will automatically 
    create @RolloutChunkDataset objects for each file and concatenate 
    them into a single @ConcatDataset.
    """
    rollout_paths = _resolve_rollout_paths(paths)
    local_config = replace(config, normalize=False)

    datasets: List[RolloutChunkDataset] = []
    transitions_list: List[np.ndarray] = []
    for p in rollout_paths:
        traj = load_rollout_latents(p)
        dataset = RolloutChunkDataset(
            traj,
            local_config,
            encoder=encoder,
            obs_keys=obs_keys,
            encoder_device=encoder_device,
            demo_id=p.stem,
        )
        datasets.append(dataset)
        transitions_list.append(
            np.concatenate([dataset.states[:, :-1, :], dataset.actions], axis=-1)
        )

    if not datasets:
        raise RuntimeError("No chunks produced. Check W/stride and rollout lengths.")

    stats = None
    if config.normalize:
        x = np.concatenate(transitions_list, axis=0).astype(np.float32)
        stats = compute_normalization_stats(x)
        for ds in datasets:
            ds._stats = stats
            ds.normalize = True

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    return loader, stats
