from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from src.robomimic_interface.dataset import (
    NormalizationStats,
    compute_normalization_stats,
)


@dataclass(frozen=True)
class SopeGymReferenceNormalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def as_transition_stats(self) -> NormalizationStats:
        return NormalizationStats(
            mean=np.concatenate([self.state_mean, self.action_mean], axis=0).astype(np.float32),
            std=np.concatenate([self.state_std, self.action_std], axis=0).astype(np.float32),
        )


@dataclass(frozen=True)
class SopeGymDataBundle:
    root: Path
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    normalization: Optional[SopeGymReferenceNormalization] = None


@dataclass(frozen=True)
class SopeGymEpisode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    episode_id: str
    start_index: int
    stop_index: int
    normalization: Optional[SopeGymReferenceNormalization] = None

    @property
    def length(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True)
class SopeGymEpisodeSummary:
    episode_id: str
    start_index: int
    stop_index: int
    num_steps: int
    total_reward: float
    terminal_at_end: bool


@dataclass(frozen=True)
class SopeGymChunkDatasetConfig:
    chunk_size: int = 8
    stride: int = 1
    frame_stack: int = 1
    state_dim: int = 3
    action_dim: int = 1
    normalize: bool = True
    normalization_source: str = "asset"
    return_metadata: bool = True


def _collect_frame_stack(sequence: np.ndarray, t0: int, frame_stack: int) -> np.ndarray:
    assert sequence.ndim == 2, f"Expected sequence with shape (T, D), got {sequence.shape}"
    frames = []
    for i in range(frame_stack):
        t_i = max(0, t0 - (frame_stack - 1 - i))
        frames.append(sequence[t_i])
    return np.stack(frames, axis=0)


def _load_reference_normalization(path: Path) -> Optional[SopeGymReferenceNormalization]:
    if not path.is_file():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    return SopeGymReferenceNormalization(
        state_mean=np.asarray(payload["state_mean"], dtype=np.float32),
        state_std=np.asarray(payload["state_std"], dtype=np.float32),
        action_mean=np.asarray(payload["action_mean"], dtype=np.float32),
        action_std=np.asarray(payload["action_std"], dtype=np.float32),
    )


def load_sope_gym_dataset(root: Path | str) -> SopeGymDataBundle:
    root = Path(root).resolve()
    observations = np.asarray(np.load(root / "observations.npy"), dtype=np.float32)
    actions = np.asarray(np.load(root / "actions.npy"), dtype=np.float32)
    rewards = np.asarray(np.load(root / "rewards.npy"), dtype=np.float32).reshape(-1)
    terminals = np.asarray(np.load(root / "terminals.npy"), dtype=np.bool_).reshape(-1)

    if observations.ndim == 1:
        observations = observations[:, None]
    if actions.ndim == 1:
        actions = actions[:, None]

    num_steps = int(observations.shape[0])
    if actions.shape[0] != num_steps or rewards.shape[0] != num_steps or terminals.shape[0] != num_steps:
        raise ValueError(
            "SOPE Gym arrays must share the same leading dimension. "
            f"Got observations={observations.shape}, actions={actions.shape}, "
            f"rewards={rewards.shape}, terminals={terminals.shape}."
        )

    return SopeGymDataBundle(
        root=root,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        normalization=_load_reference_normalization(root / "normalization.json"),
    )


def split_sope_gym_episodes(root: Path | str | SopeGymDataBundle) -> List[SopeGymEpisode]:
    bundle = root if isinstance(root, SopeGymDataBundle) else load_sope_gym_dataset(root)
    terminal_indices = np.flatnonzero(bundle.terminals)
    episode_stops = terminal_indices.tolist()
    if bundle.terminals.size and not bool(bundle.terminals[-1]):
        episode_stops.append(int(bundle.terminals.shape[0] - 1))

    episodes: List[SopeGymEpisode] = []
    start = 0
    for episode_idx, stop in enumerate(episode_stops):
        if stop < start:
            continue
        end = int(stop) + 1
        episodes.append(
            SopeGymEpisode(
                observations=bundle.observations[start:end],
                actions=bundle.actions[start:end],
                rewards=bundle.rewards[start:end],
                terminals=bundle.terminals[start:end],
                episode_id=f"episode_{episode_idx:05d}",
                start_index=int(start),
                stop_index=int(stop),
                normalization=bundle.normalization,
            )
        )
        start = end

    if not episodes:
        raise ValueError(f"No episodes were found under {bundle.root}.")

    if start < bundle.terminals.shape[0]:
        raise ValueError(
            "Episode segmentation left unconsumed samples. "
            f"Consumed up to {start}, total samples={bundle.terminals.shape[0]}."
        )

    return episodes


def summarize_sope_gym_episodes(
    episodes: Sequence[SopeGymEpisode],
    *,
    max_episodes: int = 5,
) -> Dict[str, Any]:
    if not episodes:
        return {
            "num_episodes": 0,
            "length_stats": None,
            "episodes": [],
        }

    lengths = np.asarray([episode.length for episode in episodes], dtype=np.int32)
    return {
        "num_episodes": int(len(episodes)),
        "length_stats": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "episodes": [
            SopeGymEpisodeSummary(
                episode_id=episode.episode_id,
                start_index=int(episode.start_index),
                stop_index=int(episode.stop_index),
                num_steps=int(episode.length),
                total_reward=float(episode.rewards.sum()),
                terminal_at_end=bool(episode.terminals[-1]),
            )
            for episode in episodes[:max_episodes]
        ],
    }


def train_eval_split_sope_gym_episodes(
    episodes: Sequence[SopeGymEpisode],
    *,
    seed: int,
    train_fraction: float = 0.8,
) -> tuple[list[SopeGymEpisode], list[SopeGymEpisode]]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(
            f"train_fraction must be strictly between 0 and 1, got {train_fraction}."
        )
    if len(episodes) < 2:
        return list(episodes), []

    shuffled = list(episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_fraction)
    split_idx = min(max(split_idx, 1), len(shuffled) - 1)
    return shuffled[:split_idx], shuffled[split_idx:]


class SopeGymChunkDataset:
    def __init__(
        self,
        episode: SopeGymEpisode,
        config: SopeGymChunkDatasetConfig,
    ):
        self.episode = episode
        self.config = config
        self.reference_normalization = episode.normalization
        self._validate_config()

        self.states = np.asarray(self.episode.observations, dtype=np.float32)
        self.actions = np.asarray(self.episode.actions, dtype=np.float32)
        self.rewards = np.asarray(self.episode.rewards, dtype=np.float32)
        self.terminals = np.asarray(self.episode.terminals, dtype=np.bool_)

        T = int(self.states.shape[0])
        W = int(self.config.chunk_size)
        S = int(self.config.frame_stack)
        last_start = T - (W + 1)
        if last_start < 0:
            raise ValueError(
                f"Episode {self.episode.episode_id} is too short for chunk_size={W}. "
                f"Need at least {W + 1} states, got {T}."
            )

        states_from_list: List[np.ndarray] = []
        actions_from_list: List[np.ndarray] = []
        states_to_list: List[np.ndarray] = []
        actions_to_list: List[np.ndarray] = []
        rewards_to_list: List[np.ndarray] = []
        terminals_to_list: List[np.ndarray] = []
        t0_list: List[int] = []
        for t0 in range(0, last_start + 1, self.config.stride):
            states_from_list.append(_collect_frame_stack(self.states, t0 - 1, S))
            actions_from_list.append(_collect_frame_stack(self.actions, t0 - 1, S))
            states_to_list.append(self.states[t0 : t0 + W + 1])
            actions_to_list.append(self.actions[t0 : t0 + W])
            rewards_to_list.append(self.rewards[t0 : t0 + W])
            terminals_to_list.append(self.terminals[t0 : t0 + W])
            t0_list.append(int(t0))

        self.states_from = np.stack(states_from_list, axis=0).astype(np.float32)
        self.actions_from = np.stack(actions_from_list, axis=0).astype(np.float32)
        self.states_to = np.stack(states_to_list, axis=0).astype(np.float32)
        self.actions_to = np.stack(actions_to_list, axis=0).astype(np.float32)
        self.rewards_to = np.stack(rewards_to_list, axis=0).astype(np.float32)
        self.terminals_to = np.stack(terminals_to_list, axis=0).astype(np.bool_)
        self.t0 = np.asarray(t0_list, dtype=np.int32)

        self.normalize = bool(self.config.normalize)
        self.return_metadata = bool(self.config.return_metadata)
        self.stats: Optional[NormalizationStats] = None
        self._validate_shapes()

    def _validate_config(self) -> None:
        if self.config.normalization_source not in {"asset", "computed"}:
            raise ValueError(
                "normalization_source must be one of {'asset', 'computed'}, "
                f"got {self.config.normalization_source!r}."
            )

        state_dim = int(self.episode.observations.shape[-1])
        action_dim = int(self.episode.actions.shape[-1])
        if state_dim != int(self.config.state_dim):
            raise ValueError(
                f"Expected state_dim={self.config.state_dim}, got {state_dim} "
                f"for {self.episode.episode_id}."
            )
        if action_dim != int(self.config.action_dim):
            raise ValueError(
                f"Expected action_dim={self.config.action_dim}, got {action_dim} "
                f"for {self.episode.episode_id}."
            )

    def _validate_shapes(self) -> None:
        N = int(self.states_to.shape[0])
        W = int(self.config.chunk_size)
        S = int(self.config.frame_stack)
        Dz = int(self.config.state_dim)
        Da = int(self.config.action_dim)
        assert self.states_from.shape == (N, S, Dz)
        assert self.actions_from.shape == (N, S, Da)
        assert self.states_to.shape == (N, W + 1, Dz)
        assert self.actions_to.shape == (N, W, Da)
        assert self.rewards_to.shape == (N, W)
        assert self.terminals_to.shape == (N, W)

    def __len__(self) -> int:
        return int(self.states_to.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        states_from = self.states_from[idx]
        actions_from = self.actions_from[idx]
        states_to = self.states_to[idx]
        actions_to = self.actions_to[idx]

        if self.normalize and self.stats is not None:
            state_mean = self.stats.mean[: self.config.state_dim]
            state_std = self.stats.std[: self.config.state_dim]
            action_mean = self.stats.mean[self.config.state_dim :]
            action_std = self.stats.std[self.config.state_dim :]
            states_from = (states_from - state_mean) / state_std
            states_to = (states_to - state_mean) / state_std
            actions_from = (actions_from - action_mean) / action_std
            actions_to = (actions_to - action_mean) / action_std

        return {
            "states_from": torch.from_numpy(np.asarray(states_from, dtype=np.float32)),
            "actions_from": torch.from_numpy(np.asarray(actions_from, dtype=np.float32)),
            "states_to": torch.from_numpy(np.asarray(states_to, dtype=np.float32)),
            "actions_to": torch.from_numpy(np.asarray(actions_to, dtype=np.float32)),
            "metadata": (
                {
                    "index": torch.tensor(int(idx), dtype=torch.int32),
                    "episode_id": self.episode.episode_id,
                    "t0": torch.tensor(int(self.t0[idx]), dtype=torch.int32),
                    "t1": torch.tensor(int(self.t0[idx] + self.config.chunk_size), dtype=torch.int32),
                    "frame_stack": torch.tensor(int(self.config.frame_stack), dtype=torch.int32),
                    "episode_start_index": torch.tensor(int(self.episode.start_index), dtype=torch.int32),
                    "episode_stop_index": torch.tensor(int(self.episode.stop_index), dtype=torch.int32),
                    "episode_length": torch.tensor(int(self.episode.length), dtype=torch.int32),
                }
                if self.return_metadata
                else None
            ),
        }


def _resolve_normalization_stats(
    datasets: Sequence[SopeGymChunkDataset],
    config: SopeGymChunkDatasetConfig,
) -> Optional[NormalizationStats]:
    if not config.normalize:
        return None

    if config.normalization_source == "asset":
        reference = next(
            (dataset.reference_normalization for dataset in datasets if dataset.reference_normalization is not None),
            None,
        )
        if reference is None:
            raise ValueError(
                "normalization_source='asset' requested, but normalization.json was not found."
            )
        return reference.as_transition_stats()

    transitions = [
        np.concatenate([dataset.states_to[:, :-1, :], dataset.actions_to], axis=-1)
        for dataset in datasets
    ]
    x = np.concatenate(transitions, axis=0).astype(np.float32)
    return compute_normalization_stats(x)


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}

    first = batch[0]
    out: Dict[str, Any] = {}
    for key in first.keys():
        values = [item[key] for item in batch]
        if any(value is None for value in values):
            out[key] = None
        else:
            out[key] = default_collate(values)
    return out


def make_sope_gym_chunk_dataloader(
    episodes: Sequence[SopeGymEpisode],
    config: SopeGymChunkDatasetConfig,
    *,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Tuple[DataLoader, Optional[NormalizationStats]]:
    datasets: List[SopeGymChunkDataset] = []
    for episode in episodes:
        if episode.length < config.chunk_size + 1:
            continue
        datasets.append(SopeGymChunkDataset(episode=episode, config=config))

    if not datasets:
        raise RuntimeError(
            "No SOPE Gym chunks were produced. Check chunk_size and episode lengths."
        )

    stats = _resolve_normalization_stats(datasets, config)
    for dataset in datasets:
        dataset.stats = stats
        dataset.normalize = stats is not None

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
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


__all__ = [
    "SopeGymChunkDataset",
    "SopeGymChunkDatasetConfig",
    "SopeGymDataBundle",
    "SopeGymEpisode",
    "SopeGymEpisodeSummary",
    "SopeGymReferenceNormalization",
    "load_sope_gym_dataset",
    "make_sope_gym_chunk_dataloader",
    "split_sope_gym_episodes",
    "summarize_sope_gym_episodes",
    "train_eval_split_sope_gym_episodes",
]
