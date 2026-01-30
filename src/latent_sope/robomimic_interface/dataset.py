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


def _collect_frame_stack(z: np.ndarray, t0: int, frame_stack: int) -> np.ndarray:
    """Collect past frames ending at t0, padding with earliest frame when needed."""
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError(f"Expected z with shape (T, D), got {z.shape}")
    frames = []
    for i in range(frame_stack):
        t_i = t0 - (frame_stack - 1 - i)
        if t_i < 0:
            t_i = 0
        frames.append(z[t_i])
    return np.stack(frames, axis=0)


@dataclass(frozen=True)
class RolloutChunkDatasetConfig:
    """Config for rollout chunk dataset.
    
    - chunk_size: length W of each (states, actions) chunk
    - stride: step between chunk start indices
    - frame_stack: number of past frames to condition on (overrides rollout/policy frame_stack)
    - source: "latents" (from RolloutLatentTrajectory.latents) or "obs" (from obs + encoder)
    - include_actions: include actions in chunk targets
    - normalize: apply dataset-level normalization stats
    - return_meta: return index/demo_id/t0 metadata for debugging
    """
    chunk_size: int = 8
    stride: int = 8
    frame_stack: int = 2
    source: str = "latents"
    include_actions: bool = True
    normalize: bool = False
    return_meta: bool = False


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
        self.traj: RolloutLatentTrajectory = traj
        self.config: RolloutChunkDatasetConfig = config
        self.encoder: Optional[Any] = encoder
        self.obs_keys: Optional[Sequence[str]] = obs_keys
        self.encoder_device: str = encoder_device
        self.demo_id: Optional[str] = demo_id

        if config.source == "latents":
            assert traj.latents is not None, "rollout has no latents; save with encoder/feature hook enabled"
            z = traj.latents
            if z.ndim == 3:
                z = z[:, 0, :]
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

        states_from_list: List[np.ndarray] = []
        states_to_list: List[np.ndarray] = []
        actions_to_list: List[np.ndarray] = []
        t0_list: List[int] = []
        for t0 in range(0, last_start + 1, config.stride):
            # SOPE trajectory chunks: (s_t, a_t, s_{t+1}, ..., s_{t+W}) and actions (a_t..a_{t+W-1}).
            states_to_list.append(z[t0 : t0 + W + 1])
            actions_to_list.append(actions[t0 : t0 + W])
            states_from_list.append(_collect_frame_stack(z, t0, config.frame_stack))
            t0_list.append(int(t0))

        self.states_from = np.stack(states_from_list, axis=0).astype(np.float32)
        self.states_to = np.stack(states_to_list, axis=0).astype(np.float32)
        self.actions_to = np.stack(actions_to_list, axis=0).astype(np.float32)
        self.t0 = np.asarray(t0_list, dtype=np.int64)

        N = int(self.states_to.shape[0])
        S = int(config.frame_stack)
        Dz = int(z.shape[-1])
        Da = int(actions.shape[-1])
        assert self.states_from.shape == (N, S, Dz), f"Expected states_from to have shape (N, S, Dz), got {self.states_from.shape}"
        assert self.states_to.shape == (N, W + 1, Dz), f"Expected states_to to have shape (N, W+1, Dz), got {self.states_to.shape}"
        assert self.actions_to.shape == (N, W, Da), f"Expected actions_to to have shape (N, W, Da), got {self.actions_to.shape}"

        self.latents_dim = Dz
        self.action_dim = Da
        self.normalize = bool(config.normalize)
        self.return_meta = bool(config.return_meta)
        self._stats = self._compute_transition_stats() if self.normalize else None

    def _compute_transition_stats(self) -> NormalizationStats:
        x = np.concatenate([self.states_to[:, :-1, :], self.actions_to], axis=-1)
        return compute_normalization_stats(x)

    @property
    def normalization_stats(self) -> Optional[NormalizationStats]:
        return self._stats

    def __len__(self) -> int:
        return int(self.states_to.shape[0])

    def __getitem__(self, idx: int):
        states_from = self.states_from[idx]
        states_to = self.states_to[idx]
        actions_to = self.actions_to[idx]
        if self._stats is not None:
            latents_mean = self._stats.mean[: self.latents_dim]
            latents_std = self._stats.std[: self.latents_dim]
            act_mean = self._stats.mean[self.latents_dim :]
            act_std = self._stats.std[self.latents_dim :]
            states_from = (states_from - latents_mean) / latents_std
            states_to = (states_to - latents_mean) / latents_std
            actions_to = (actions_to - act_mean) / act_std
        states_from_t = torch.from_numpy(np.asarray(states_from, dtype=np.float32))
        states_to_t = torch.from_numpy(np.asarray(states_to, dtype=np.float32))
        actions_to_t = torch.from_numpy(np.asarray(actions_to, dtype=np.float32))
        item = {
            "states_from": states_from_t,
            "states_to": states_to_t,
            "actions_to": actions_to_t,
        }
        if self.return_meta:
            meta = {"index": int(idx)}
            if self.demo_id is not None:
                meta["demo_id"] = self.demo_id
            meta["t0"] = int(self.t0[idx])
            item["meta"] = meta
        return item


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
            np.concatenate([dataset.states_to[:, :-1, :], dataset.actions_to], axis=-1)
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
