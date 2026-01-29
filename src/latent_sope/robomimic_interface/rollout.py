"""Helpers to record and save latent rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from copy import deepcopy

import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
from robomimic.algo import RolloutPolicy, PolicyAlgo
from robomimic.utils import tensor_utils as TensorUtils
from robomimic.envs.env_base import EnvBase

from src.latent_sope.robomimic_interface.encoders import (
    resolve_module,
    LowDimConcatEncoder,
    HighDimObsEncoder,
    extract_embeddings_batched,
)
from src.latent_sope.data.chunking import make_chunks, pack_chunk_x
from src.latent_sope.data.chunk_dataset import compute_normalization_stats, NormalizationStats
from src.latent_sope.utils.common import timeit

@dataclass
class RolloutStats:
    total_reward: float
    horizon: int
    success_rate: float

@dataclass
class RolloutLatentTrajectory:
    z: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    success: bool
    total_reward: float
    horizon: int
    frame_stack: int
    obs: Optional[Dict[str, np.ndarray]] = None    #  key -> (T, frame_stack, d)
    next_obs: Optional[Dict[str, np.ndarray]] = None
    infos: Optional[List[Dict[str, Any]]] = None
    stats: Optional[RolloutStats] = None


class PolicyFeatureHook:
    """Capture features from a policy module via a forward hook."""

    def __init__(
        self,
        policy: Any,
        obs_keys: Optional[Sequence[str]] = None,
        feat_type: str = "low_dim_concat",
    ):
        assert feat_type in {"low_dim_concat", "high_dim_encode"}, f"Unknown feat_type: {feat_type}"
        self.policy = policy
        self.frame_stack = get_policy_frame_stack(policy)
        self.obs_keys = (
            list(_get_nested_attr(policy, "policy.obs_config.modalities.obs.low_dim"))
            if obs_keys is None
            else list(obs_keys)
        )

        self.feat_type = str(feat_type).lower()

        def _hook(_module, _inp, out):
            if torch.is_tensor(out):
                feat = out.detach()
            elif isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]):
                feat = out[0].detach()
            else:
                feat = out
            self._last_feature = feat
        
        self._last_feature = None
        self._policy_module = self._resolve_policy_module()
        self._hook_handle = self._policy_module.register_forward_hook(_hook)

    def _resolve_policy_module(self) -> Any:
        candidates = [
            "policy.nets.policy.obs_encoder", # for low_dim diffusion polciy
            "nets.policy",
            "policy",
        ]
        for path in candidates:
            try:
                mod = resolve_module(self.policy, path)
            except Exception:
                continue
            if hasattr(mod, "register_forward_hook"):
                return mod

        if hasattr(self.policy, "register_forward_hook"):
            return self.policy

        raise ValueError(
            "Could not resolve a torch policy module for low_dim_concat. "
            "Pass a policy with nets['policy'] or a policy module that supports hooks."
        )
    
    def clear(self) -> None:
        self._last_feature = None

    def _get_policy_algo(self) -> PolicyAlgo:
        return getattr(self.policy, "policy", self.policy)
    
    def _get_obs_shapes(self) -> Dict[str, Any]:
        algo = self._get_policy_algo()
        obs_shapes = getattr(algo, "obs_shapes", None)
        if obs_shapes is not None:
            return obs_shapes
        raise RuntimeError("Policy does not expose obs_shapes needed to encode observations.")

    def _prepare_obs_inputs(self, obs: Dict[str, Any], goal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prep: RolloutPolicy = self.policy if hasattr(self.policy, "_prepare_observation") else getattr(self.policy, "policy", None)
        assert prep is not None and hasattr(prep, "_prepare_observation"), "Policy does not expose _prepare_observation needed to encode observations."

        obs_t = prep._prepare_observation(obs, batched_ob=False)
        inputs = {"obs": obs_t}
        if goal is not None:
            inputs["goal"] = prep._prepare_observation(goal, batched_ob=False)

        obs_shapes = self._get_obs_shapes()
        if obs_shapes is not None:
            for k in obs_shapes:
                if inputs["obs"][k].ndim - 1 == len(obs_shapes[k]):
                    inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
        return inputs

    def update_from_obs(self, obs: Dict[str, Any], goal: Optional[Dict[str, Any]] = None) -> None:
        if self.feat_type != "low_dim_concat":
            return
        inputs = self._prepare_obs_inputs(obs, goal=goal)
        _ = TensorUtils.time_distributed(inputs, self._policy_module, inputs_as_kwargs=True)

    def ensure_feature(self, obs: Dict[str, Any], goal: Optional[Dict[str, Any]] = None) -> None:
        if self._last_feature is None:
            self.update_from_obs(obs, goal=goal)

    def close(self) -> None:
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            finally:
                self._hook_handle = None

    def __del__(self):
        self.close()

    def pull_feat(self, clear: bool = True) -> np.ndarray:
        """Return last captured feature as a float32 numpy array."""
        assert self._last_feature is not None, "Feature hook has no cached output. Ensure the module runs during policy forward before calling pull_feat."

        feat = self._last_feature
        if hasattr(feat, "detach"):
            feat = feat.detach().cpu().numpy()
        else:
            feat = np.asarray(feat)

        if clear: self.clear()

        return np.asarray(feat, dtype=np.float32)


class RolloutLatentRecorder:
    """Record a rollout trajectory and optional latent embeddings."""

    def __init__(
        self,
        obs_keys: Optional[Sequence[str]] = None,
        store_obs: bool = True,
        store_next_obs: bool = False,
        encoder: Optional[Any] = None,
        encoder_device: str = "cuda",
        feature_hook: Optional[PolicyFeatureHook] = None,
        frame_stack: Optional[int] = None,
    ):
        self.obs_keys = None if obs_keys is None else list(obs_keys)
        self.store_obs = bool(store_obs)
        self.store_next_obs = bool(store_next_obs)
        self.encoder = encoder
        self.encoder_device = encoder_device
        self.feature_hook = feature_hook
        self.frame_stack = 1 if frame_stack is None else int(frame_stack)
        
        self._obs: Dict[str, List[np.ndarray]] = {}
        self._next_obs: Dict[str, List[np.ndarray]] = {}
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._infos: List[Dict[str, Any]] = []
        self._z: List[np.ndarray] = []

    def start_episode(self, obs: Dict[str, Any]) -> None:
        keys = list(obs.keys()) if self.obs_keys is None else self.obs_keys
        self.obs_keys = list(keys)
        if self.feature_hook is not None:
            self.feature_hook.clear()
        if self.store_obs:
            for k in self.obs_keys:
                self._obs.setdefault(k, [])
        if self.store_next_obs:
            for k in self.obs_keys:
                self._next_obs.setdefault(k, [])

    def record_step(
        self,
        obs: Dict[str, Any],
        action: Any,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        next_obs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.obs_keys is None:
            self.start_episode(obs)

        if self.store_obs:
            for k in self.obs_keys or []:
                self._obs[k].append(np.asarray(obs[k]))

        if self.store_next_obs and next_obs is not None:
            for k in self.obs_keys or []:
                self._next_obs[k].append(np.asarray(next_obs[k]))

        if self.feature_hook is not None:
            self.feature_hook.ensure_feature(obs)
            z = self.feature_hook.pull_feat(clear=True)
            self._z.append(np.asarray(z))

        self._actions.append(np.asarray(action))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._infos.append({} if info is None else dict(info))

    def _stack_obs(self, obs_list: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k, items in obs_list.items():
            if not items: continue
            out[k] = np.stack(items, axis=0)
        return out

    def _encode_batch(self, obs_batch: Dict[str, np.ndarray]) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("No encoder provided to compute latents")

        if hasattr(self.encoder, "encode_obs_batch"):
            out = self.encoder.encode_obs_batch(obs_batch, device=self.encoder_device)
            z = out.z if hasattr(out, "z") else out
        else:
            out = self.encoder(obs_batch)
            z = out.z if hasattr(out, "z") else out

        return np.asarray(z, dtype=np.float32)

    def finalize(
        self,
        stats: RolloutStats,
    ) -> RolloutLatentTrajectory:
        obs = self._stack_obs(self._obs) if self.store_obs else None
        next_obs = self._stack_obs(self._next_obs) if self.store_next_obs else None

        if self._z:
            z = np.stack(self._z, axis=0).astype(np.float32)
        else:
            if obs is None:
                raise RuntimeError("No stored obs to compute latents")
            z = self._encode_batch(obs)

        actions = np.stack(self._actions, axis=0).astype(np.float32)
        rewards = np.asarray(self._rewards, dtype=np.float32)
        dones = np.asarray(self._dones, dtype=np.bool_)

        return RolloutLatentTrajectory(
            z=z,
            actions=actions,
            rewards=rewards,
            dones=dones,
            success=bool(stats.success_rate > 0.0),
            total_reward=float(stats.total_reward),
            horizon=int(stats.horizon),
            frame_stack=self.frame_stack,
            obs=obs,
            next_obs=next_obs,
            infos=self._infos,
            stats=stats,
        )


def _get_nested_attr(obj: Any, path: str) -> Any:
    cur = obj
    for tok in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(tok, None)
        else:
            cur = getattr(cur, tok, None)
    return cur


def get_policy_frame_stack(policy: PolicyAlgo, default: int = 1) -> int:
    """Best-effort retrieval of frame_stack from a robomimic policy."""
    candidates = [
        "policy.algo_config.horizon.observation_horizon", # default
        "frame_stack",
        "policy.frame_stack",
        "algo_config.train.frame_stack",
        "policy.algo_config.train.frame_stack",
        "config.train.frame_stack",
        "policy.config.train.frame_stack",
        "algo_config.horizon.observation_horizon",
    ]
    for path in candidates:
        val = _get_nested_attr(policy, path)
        if val is None: continue
        if (val := int(val)) >=1 : return val
    return int(default)


def save_rollout_latents(path: Path, traj: RolloutLatentTrajectory) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert path.suffix in {".npz", ".h5", ".hdf5"}, f"Unsupported output format: {path.suffix}"

    if path.suffix == ".npz":
        np.savez_compressed(
            path,
            z=traj.z,
            actions=traj.actions,
            rewards=traj.rewards,
            dones=traj.dones,
            success=np.asarray([traj.success], dtype=np.bool_),
            total_reward=np.asarray([traj.total_reward], dtype=np.float32),
            horizon=np.asarray([traj.horizon], dtype=np.int64),
            frame_stack=np.asarray([traj.frame_stack], dtype=np.int64),
        )
        return path

    with h5py.File(path, "w") as f:
        f.create_dataset("z", data=traj.z, compression="gzip")
        f.create_dataset("actions", data=traj.actions, compression="gzip")
        f.create_dataset("rewards", data=traj.rewards, compression="gzip")
        f.create_dataset("dones", data=traj.dones, compression="gzip")
        f.create_dataset("t", data=np.arange(traj.z.shape[0], dtype=np.int64))

        f.attrs["success"] = int(traj.success)
        f.attrs["total_reward"] = float(traj.total_reward)
        f.attrs["horizon"] = int(traj.horizon)
        f.attrs["frame_stack"] = int(traj.frame_stack)

        if traj.obs is not None:
            obs_group = f.create_group("obs")
            for k, v in traj.obs.items():
                obs_group.create_dataset(k, data=v, compression="gzip")

        if traj.next_obs is not None:
            next_group = f.create_group("next_obs")
            for k, v in traj.next_obs.items():
                next_group.create_dataset(k, data=v, compression="gzip")

    return path


def load_rollout_latents(path: Path) -> Dict[str, Any]:
    """Load rollout latents saved by save_rollout_latents.

    Returns a dict with keys: z, actions, rewards, dones, obs (optional), next_obs (optional).
    """
    path = Path(path)
    assert path.is_file(), f"rollout file not found: {path}"
    assert path.suffix in {".npz", ".h5", ".hdf5"}, f"Unsupported rollout format: {path.suffix}"

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {
            "z": np.asarray(data["z"], dtype=np.float32),
            "actions": np.asarray(data["actions"], dtype=np.float32),
            "rewards": np.asarray(data["rewards"], dtype=np.float32),
            "dones": np.asarray(data["dones"], dtype=np.bool_),
            "frame_stack": int(data["frame_stack"][0]) if "frame_stack" in data else 1,
            "obs": None,
            "next_obs": None,
        }

    else:
        with h5py.File(path, "r") as f:
            z = np.asarray(f["z"], dtype=np.float32)
            actions = np.asarray(f["actions"], dtype=np.float32)
            rewards = np.asarray(f["rewards"], dtype=np.float32)
            dones = np.asarray(f["dones"], dtype=np.bool_)
            frame_stack = int(f.attrs.get("frame_stack", 1))

            obs = None
            if "obs" in f:
                obs = {k: np.asarray(v) for k, v in f["obs"].items()}

            next_obs = None
            if "next_obs" in f:
                next_obs = {k: np.asarray(v) for k, v in f["next_obs"].items()}

        return {
            "z": z,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "frame_stack": frame_stack,
            "obs": obs,
            "next_obs": next_obs,
        }


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


class RolloutChunkDataset:
    """Torch Dataset over chunked rollouts (N, W, D)."""

    def __init__(self, x: np.ndarray, normalize: bool = False, return_meta: bool = False):
        if x.ndim != 3:
            raise ValueError(f"Expected (N, W, D), got {x.shape}")
        self.x = np.asarray(x, dtype=np.float32)
        self.normalize = bool(normalize)
        self.return_meta = bool(return_meta)
        self._stats = compute_normalization_stats(self.x) if self.normalize else None

    @property
    def normalization_stats(self) -> Optional[NormalizationStats]:
        return self._stats

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        import torch

        x = self.x[idx]
        if self._stats is not None:
            x = (x - self._stats.mean) / self._stats.std
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if self.return_meta:
            return x_t, {"index": int(idx)}
        return x_t


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
    W: int,
    stride: int = 1,
    batch_size: int = 64,
    frame_stack: int = 1,
    source: str = "z",
    encoder: Optional[Any] = None,
    obs_keys: Optional[Sequence[str]] = None,
    include_actions: bool = True,
    normalize: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    encoder_device: str = "cuda",
    return_meta: bool = False,
) -> Tuple[DataLoader, Optional[NormalizationStats]]:
    """Prepare a DataLoader from saved rollouts for chunk diffusion training."""
    import torch
    from torch.utils.data import DataLoader

    rollout_paths = _resolve_rollout_paths(paths)
    chunks = []

    for p in rollout_paths:
        data = load_rollout_latents(p)
        if source == "z":
            z = data["z"]
        elif source == "obs":
            if data["obs"] is None:
                raise ValueError(f"rollout {p} has no obs; save with store_obs=True")
            if encoder is None:
                raise ValueError("encoder must be provided when source='obs'")
            obs = data["obs"]
            if obs_keys is not None:
                obs = {k: obs[k] for k in obs_keys}
            z = extract_embeddings_batched(encoder, obs, device=encoder_device)
        else:
            raise ValueError(f"Unknown source={source}. Use 'z' or 'obs'.")

        z = _stack_frames(z, frame_stack)
        actions = data["actions"]
        rewards = data["rewards"]

        if not include_actions:
            actions = np.zeros((actions.shape[0], 0), dtype=np.float32)

        demo_id = p.stem
        chunks.extend(make_chunks(demo_id, z, actions, rewards, W=W, stride=stride))

    if not chunks:
        raise RuntimeError("No chunks produced. Check W/stride and rollout lengths.")

    x = np.stack([pack_chunk_x(c) for c in chunks], axis=0).astype(np.float32)
    dataset = RolloutChunkDataset(x, normalize=normalize, return_meta=return_meta)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return loader, dataset.normalization_stats


@timeit
def rollout(
    policy: RolloutPolicy,
    env: EnvBase,
    horizon: int,
    render: bool = False,
    video_writer=None,
    video_skip: int = 5,
    camera_names: Optional[List[str]] = None,
    recorder: Optional[RolloutLatentRecorder] = None,
) -> RolloutStats:
    assert not (render and (video_writer is not None))
    camera_names = camera_names or ["agentview"]

    if recorder is not None:
        recorder.frame_stack = get_policy_frame_stack(policy, default=recorder.frame_stack)

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)
    if recorder is not None:
        recorder.start_episode(obs)

    total_reward = 0.0
    success = False
    video_count = 0

    try:
        for step_i in range(horizon):
            act = policy(ob=obs)
            next_obs, reward, done, info = env.step(act)
            total_reward += reward
            success = env.is_success()["task"]

            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    frames = []
                    for cam_name in camera_names:
                        frames.append(
                            env.render(
                                mode="rgb_array",
                                height=512,
                                width=512,
                                camera_name=cam_name,
                            )
                        )
                    video_writer.append_data(np.concatenate(frames, axis=1))
                video_count += 1

            if recorder is not None:
                recorder.record_step(
                    obs=obs,
                    action=act,
                    reward=reward,
                    done=done,
                    info=info,
                    next_obs=next_obs,
                )

            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print(f"WARNING: got rollout exception {e}")

    return RolloutStats(total_reward=total_reward, horizon=(step_i + 1), success_rate=float(success))
