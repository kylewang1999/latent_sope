from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from src.robomimic_interface.encoders import extract_embeddings_batched
from src.robomimic_interface.rollout import (
    RolloutLatentTrajectory,
    load_rollout_latents,
)


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray  # shape (D,)
    std: np.ndarray  # shape (D,)


def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _collect_dataset_array(dataset: Any, attr: str) -> np.ndarray:
    if isinstance(dataset, ConcatDataset):
        return np.concatenate(
            [_collect_dataset_array(subdataset, attr) for subdataset in dataset.datasets],
            axis=0,
        ).astype(np.float32)
    if not hasattr(dataset, attr):
        raise AttributeError(f"Dataset {type(dataset)} has no attribute {attr!r}.")
    return np.asarray(getattr(dataset, attr), dtype=np.float32)


def summarize_dataset_feature_stats(dataset: Any) -> Dict[str, Dict[str, float]]:
    """Summarize raw and normalized feature scales for debugging."""
    states_from = _collect_dataset_array(dataset, "states_from")
    states_to = _collect_dataset_array(dataset, "states_to")
    actions_from = _collect_dataset_array(dataset, "actions_from")
    actions_to = _collect_dataset_array(dataset, "actions_to")
    stats = getattr(dataset, "stats", None)
    if isinstance(dataset, ConcatDataset) and stats is None and dataset.datasets:
        stats = getattr(dataset.datasets[0], "stats", None)

    summary = {
        "states_from_raw": _summary_stats(states_from),
        "states_to_raw": _summary_stats(states_to),
        "actions_from_raw": _summary_stats(actions_from),
        "actions_to_raw": _summary_stats(actions_to),
    }
    if stats is None:
        return summary

    latents_mean = stats.mean[: states_from.shape[-1]]
    latents_std = stats.std[: states_from.shape[-1]]
    act_mean = stats.mean[states_from.shape[-1] :]
    act_std = stats.std[states_from.shape[-1] :]
    summary.update(
        {
            "states_from_normalized": _summary_stats((states_from - latents_mean) / latents_std),
            "states_to_normalized": _summary_stats((states_to - latents_mean) / latents_std),
            "actions_from_normalized": _summary_stats((actions_from - act_mean) / act_std),
            "actions_to_normalized": _summary_stats((actions_to - act_mean) / act_std),
        }
    )
    return summary


def compute_normalization_stats(x: np.ndarray) -> NormalizationStats:
    """Compute (mean, std) over (N, W, D) arrays along the (N, W) axes."""
    assert x.ndim == 3, f"Expected (N, W, D), got {x.shape}"
    mean = x.mean(axis=(0, 1)).astype(np.float32)
    std = (x.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    return NormalizationStats(mean=mean, std=std)


def _collect_frame_stack(latents: np.ndarray, t0: int, frame_stack: int) -> np.ndarray:
    """Collect past frames ending (inclusive) at t0, padding by duplicating earliest frame when needed."""
    assert latents.ndim == 2, f"Expected latents with shape (T, D), got {latents.shape}"
    frames = []
    for i in range(frame_stack):
        t_i = max(0, t0 - (frame_stack - 1 - i))
        frames.append(latents[t_i])
    return np.stack(frames, axis=0)


def _resolve_eef_pos_slice(state_dim: int) -> slice:
    start = 10
    stop = 13
    if state_dim < stop:
        raise ValueError(
            f"robot0_eef_pos slice [10:13] requires state_dim >= 13, got {state_dim}."
        )
    return slice(start, stop)


@dataclass(frozen=True)
class RolloutChunkDatasetConfig:
    """Config for rollout chunk dataset.

    - chunk_size: length W of each (states, actions) chunk
    - stride: step between chunk start indices
    - frame_stack: number of past frames to condition on (overrides rollout/policy frame_stack)
    - source: "latents" (from RolloutLatentTrajectory.latents) or "obs" (from obs + encoder)
    - state_projection: use the full state or just robot0_eef_pos
    - disable_conditioning: zero action history/targets for unconditioned debug diffusion
    - normalize: apply dataset-level normalization stats
    - return_metadata: return index/demo_id/t0 metadatadata for debugging
    """
    chunk_size: int = 8
    stride: int = 2
    frame_stack: int = 2
    source: str = "latents"
    latents_dim: int = 19
    action_dim: int = 7
    state_projection: Literal["full", "eef_pos"] = "full"
    disable_conditioning: bool = False
    normalize: bool = False
    return_metadata: bool = True


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
        self._validate_config()

        self.latents = self._preprocess_latents()
        self.latents_dim = int(self.latents.shape[-1])
        self.actions = np.asarray(self.traj.actions, dtype=np.float32)
        self.action_dim = int(self.actions.shape[-1])

        T = self.latents.shape[0]
        W = int(config.chunk_size)
        S = int(config.frame_stack)
        last_start = T - (W + 1)
        assert last_start >= 0, "No chunks produced. Check chunk_size/stride and rollout length."

        states_from_list: List[np.ndarray] = []
        actions_from_list: List[np.ndarray] = []
        states_to_list: List[np.ndarray] = []
        actions_to_list: List[np.ndarray] = []
        t0_list: List[int] = []
        for t0 in range(0, last_start + 1, config.stride):
            # SOPE trajectory chunks: (s_t, a_t, s_{t+1}, ..., s_{t+W}) and actions (a_t..a_{t+W-1}).
            # incomplete chunks are discarded by the looping stride automatically
            states_to_list.append(self.latents[t0: t0+W+1])
            actions_to = self.actions[t0 : t0 + W]
            states_from = _collect_frame_stack(self.latents, t0 - 1, S)
            actions_from = _collect_frame_stack(self.actions, t0 - 1, S)
            if self.config.disable_conditioning:
                actions_to = np.zeros_like(actions_to, dtype=np.float32)
                actions_from = np.zeros_like(actions_from, dtype=np.float32)
            actions_to_list.append(actions_to)
            states_from_list.append(states_from)
            actions_from_list.append(actions_from)
            t0_list.append(int(t0))

        self.states_from = np.stack(states_from_list, axis=0).astype(np.float32)  # (N, frame_stack, Dz)
        self.states_to = np.stack(states_to_list, axis=0).astype(np.float32)  # (N, chunk_size + 1, Dz)
        self.actions_to = np.stack(actions_to_list, axis=0).astype(np.float32)  # (N, chunk_size, Da)
        self.actions_from = np.stack(actions_from_list, axis=0).astype(np.float32)  # (N, frame_stack, Da)
        self.t0 = np.asarray(t0_list, dtype=np.int32)
        self._validate_data_shapes()

        self.normalize = bool(config.normalize)
        self.return_metadata = bool(config.return_metadata)
        self.stats = self._compute_normalization_stats() if self.normalize else None

    def _validate_config(self) -> None:
        if self.config.source == "latents":
            assert self.traj.latents is not None, "rollout has no latents; save with encoder/feature hook enabled"
        elif self.config.source == "obs":
            assert (self.traj.obs is not None), "rollout has no obs; need to save with store_obs=True"
            assert self.encoder is not None, "encoder must be provided when source='obs'"

        else:
            raise ValueError(f"Unknown source={self.config.source}. Use 'latents' or 'obs'.")

    def _validate_data_shapes(self) -> None:
        N = int(self.states_to.shape[0])
        W = int(self.config.chunk_size)
        S = int(self.config.frame_stack)
        Dz = int(self.latents_dim)
        Da = int(self.action_dim)
        assert self.states_from.shape == (N, S, Dz), f"Expected states_from to have shape (N, S, Dz), got {self.states_from.shape}"
        assert self.actions_from.shape == (N, S, Da), f"Expected actions_from to have shape (N, S, Da), got {self.actions_from.shape}"
        assert self.states_to.shape == (N, W+1, Dz), f"Expected states_to to have shape (N, W+1, Dz), got {self.states_to.shape}"
        assert self.actions_to.shape == (N, W, Da), f"Expected actions_to to have shape (N, W, Da), got {self.actions_to.shape}"

    def _preprocess_latents(self) -> np.ndarray:
        if self.config.source == "latents":
            z = np.asarray(self.traj.latents)
            if z.ndim == 3:
                z = z[:, 0, :]
            if self.config.state_projection == "eef_pos":
                return np.asarray(
                    z[:, _resolve_eef_pos_slice(z.shape[-1])],
                    dtype=np.float32,
                )
            return np.asarray(z, dtype=np.float32)

        # otherwise, source == "obs"
        obs = self.traj.obs
        if self.obs_keys is not None:
            obs = {k: obs[k] for k in self.obs_keys}
        z = extract_embeddings_batched(self.encoder, obs, device=self.encoder_device)
        if self.config.state_projection == "eef_pos":
            raise ValueError(
                "state_projection='eef_pos' is only supported for source='latents'."
            )
        return z

    def _compute_normalization_stats(self) -> NormalizationStats:
        if self.config.disable_conditioning:
            zeros = np.zeros_like(self.actions_to, dtype=np.float32)
            x = np.concatenate([self.states_to[:, :-1, :], zeros], axis=-1)
        else:
            x = np.concatenate([self.states_to[:, :-1, :], self.actions_to], axis=-1)
        return compute_normalization_stats(x)

    @property
    def normalization_stats(self) -> Optional[NormalizationStats]:
        return self.stats

    def __len__(self) -> int:
        return int(self.states_to.shape[0])

    def __getitem__(self, idx: int):
        states_from = self.states_from[idx]
        actions_from = self.actions_from[idx]
        states_to = self.states_to[idx]
        actions_to = self.actions_to[idx]
        # NOTE: With the current construction, `states_from` ends at t0-1 and
        # `states_to` starts at t0. You can concatenate along time to form a
        # continuous, non-overlapping state sequence:
        #   full_states = np.concatenate([states_from, states_to], axis=0)
        # This yields state sequence length (frame_stack + chunk_size + 1), and
        # action sequence length (chunk_size).
        if self.normalize and self.stats is not None:
            latents_mean = self.stats.mean[: self.latents_dim]
            latents_std = self.stats.std[: self.latents_dim]
            act_mean = self.stats.mean[self.latents_dim :]
            act_std = self.stats.std[self.latents_dim :]
            states_from = (states_from - latents_mean) / latents_std
            states_to = (states_to - latents_mean) / latents_std
            actions_from = (actions_from - act_mean) / act_std
            actions_to = (actions_to - act_mean) / act_std
        states_from_t = torch.from_numpy(np.asarray(states_from, dtype=np.float32))
        actions_from_t = torch.from_numpy(np.asarray(actions_from, dtype=np.float32))
        states_to_t = torch.from_numpy(np.asarray(states_to, dtype=np.float32))
        actions_to_t = torch.from_numpy(np.asarray(actions_to, dtype=np.float32))

        return {
            "states_from": states_from_t,
            "actions_from": actions_from_t,
            "states_to": states_to_t,
            "actions_to": actions_to_t,
            "metadata": (
                {
                    "index": torch.tensor(int(idx), dtype=torch.int32),
                    "demo_id": self.demo_id,
                    "t0": torch.tensor(int(self.t0[idx]), dtype=torch.int32),  # start of chunk
                    "t1": torch.tensor(int(self.t0[idx] + self.config.chunk_size), dtype=torch.int32),  # end of chunk
                    "frame_stack": torch.tensor(int(self.config.frame_stack), dtype=torch.int32),
                }
                if self.return_metadata
                else None
            ),
        }


def _resolve_rollout_paths(paths: Sequence[Path]) -> List[Path]:
    resolved: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for ext in ("*.npz", "*.h5", "*.hdf5"):
                resolved.extend(sorted(p.rglob(ext)))
        else:
            resolved.append(p)
    resolved = [p for p in resolved if p.is_file()]
    if not resolved:
        raise FileNotFoundError("No rollout files found.")
    return resolved


def make_rollout_chunk_dataloader(
    paths: Sequence[Path],
    config: RolloutChunkDatasetConfig,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    *,
    encoder: Optional[Any] = None,
    obs_keys: Optional[Sequence[str]] = None,
    encoder_device: str = "cuda",
) -> Tuple[DataLoader, Optional[NormalizationStats]]:
    """Prepare a chunked rollout DataLoader from one or more rollout locations.

    Expected `paths` behavior:
    - If `paths` is a singleton list whose only element is a directory, that
      directory is scanned recursively for rollout files matching `*.npz`,
      `*.h5`, and `*.hdf5`, and every discovered file is loaded.
    - If `paths` is a list of rollout file paths, each listed file is loaded
      directly.
    - Mixed inputs are also allowed: directory entries are expanded, file
      entries are used as-is.

    Each resolved rollout file becomes its own `RolloutChunkDataset`. If more
    than one rollout file is resolved, the per-file datasets are combined into a
    single `ConcatDataset` before building the `DataLoader`.
    """
    rollout_paths = _resolve_rollout_paths(paths)
    pre_traj_config = replace(
        config, normalize=False
    )  # disable traj-leve normalization

    datasets: List[RolloutChunkDataset] = []
    transitions_list: List[np.ndarray] = []
    for p in rollout_paths:
        traj = load_rollout_latents(p)
        dataset = RolloutChunkDataset(
            traj,
            pre_traj_config,
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

    # normalize at super-trajectory level, not single-traj-level, not chunk level
    # stacking rule: [state_dims, action_dims] -> [state_dims + action_dims]
    stats: Optional[NormalizationStats] = None
    if config.normalize:
        x = np.concatenate(transitions_list, axis=0).astype(np.float32)
        stats = compute_normalization_stats(x)
        # update normalization stats for each sub-dataset
        for ds in datasets:
            ds.stats = stats
            ds.normalize = True

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        first = batch[0]
        out: Dict[str, Any] = {}

        for key in first.keys():
            values = [b[key] for b in batch]
            if any(v is None for v in values):
                out[key] = None
            else:
                out[key] = default_collate(values)
        return out

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        persistent_workers=bool(num_workers > 0),
        collate_fn=_collate_fn,
    )

    return loader, stats
