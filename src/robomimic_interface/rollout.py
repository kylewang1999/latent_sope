"""Helpers to record and save latent rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set
from copy import deepcopy
from collections import OrderedDict
import h5py
import json

import torch
import numpy as np

from third_party.robomimic.robomimic.utils import tensor_utils as TensorUtils
from third_party.robomimic.robomimic.algo import RolloutPolicy, PolicyAlgo
from third_party.robomimic.robomimic.envs.env_base import EnvBase

from src.utils import timeit


def resolve_module(root: Any, dotted_path: str) -> Any:
    """Resolve a dotted attribute / dict path inside a robomimic object tree."""
    cur = root
    for tok in dotted_path.split("."):
        if isinstance(cur, dict):
            cur = cur[tok]
        else:
            # torch.nn.ModuleDict behaves like a dict for __getitem__.
            if hasattr(cur, "__getitem__") and not hasattr(cur, tok):
                try:
                    cur = cur[tok]
                    continue
                except Exception:
                    pass
            cur = getattr(cur, tok)
    return cur


@dataclass
class RolloutStats:
    total_reward: float
    horizon: int
    success_rate: float

@dataclass
class RolloutLatentTrajectory:
    latents: np.ndarray
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
    feature_type: Optional[str] = None
    feature_keys: Optional[List[str]] = None


class PolicyFeatureHook:
    """Extract per-step rollout features from a robomimic policy.

    Supported feature modes:
    - ``low_dim_concat``: current-frame prepared low-dimensional observations.
    - ``image_embedding``: current-frame RGB VisualCore embeddings for all RGB keys.
    - ``both``: current-frame full observation-encoder output consumed by the policy.
    """

    VALID_FEAT_TYPES = {"low_dim_concat", "image_embedding", "both"}

    def __init__(
        self,
        policy: Any,
        obs_keys: Optional[Sequence[str]] = None,
        feat_type: str = "low_dim_concat",
    ):
        feat_type = str(feat_type).lower()
        assert feat_type in self.VALID_FEAT_TYPES, f"Unknown feat_type: {feat_type}"
        self.policy = policy
        self.frame_stack = get_policy_frame_stack(policy)
        self.feat_type = feat_type
        self.low_dim_keys = list(obs_keys) if obs_keys is not None else self._resolve_obs_modality_keys("low_dim")
        self.rgb_keys = self._resolve_obs_modality_keys("rgb")
        # Raw rollout obs storage remains compact by default, independent of feature mode.
        self.obs_keys = list(self.low_dim_keys)
        self._last_feature = None
        self._policy_module = self._resolve_policy_module()
        self._obs_encoder = self._resolve_obs_encoder()
        self._hook_handle = None

        if self.feat_type == "low_dim_concat" and not self.low_dim_keys:
            raise ValueError("feat_type='low_dim_concat' requires low-dimensional observation keys.")
        if self.feat_type == "image_embedding" and not self.rgb_keys:
            raise ValueError("feat_type='image_embedding' requires RGB observation keys.")

    @property
    def feature_type(self) -> str:
        return self.feat_type

    @property
    def feature_keys(self) -> List[str]:
        if self.feat_type == "low_dim_concat":
            return list(self.low_dim_keys)
        if self.feat_type == "image_embedding":
            return list(self.rgb_keys)
        obs_shapes = getattr(self._obs_encoder, "obs_shapes", None)
        if obs_shapes is not None:
            return list(obs_shapes.keys())
        return list(self.low_dim_keys) + list(self.rgb_keys)

    def _resolve_obs_modality_keys(self, modality: str) -> List[str]:
        candidates = [
            f"policy.obs_config.modalities.obs.{modality}",
            f"obs_config.modalities.obs.{modality}",
            f"policy.global_config.observation.modalities.obs.{modality}",
            f"global_config.observation.modalities.obs.{modality}",
        ]
        for path in candidates:
            keys = _get_nested_attr(self.policy, path)
            if keys is not None:
                return list(keys)
        return []

    def _resolve_policy_module(self) -> Any:
        algo = self._get_policy_algo()
        ema = getattr(algo, "ema", None)
        averaged_model = getattr(ema, "averaged_model", None)
        if averaged_model is not None:
            try:
                mod = resolve_module(averaged_model, "policy.obs_encoder")
                if callable(mod):
                    return mod
            except Exception:
                pass

        candidates = [
            "policy.nets.policy.obs_encoder",
            "nets.policy.obs_encoder",
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
            "Could not resolve a torch policy observation encoder. "
            "Pass a policy with nets['policy'] or a policy module that supports hooks."
        )

    def _resolve_obs_encoder(self) -> Any:
        try:
            obs_encoder = resolve_module(self._policy_module, "nets.obs")
            if self._is_observation_encoder(obs_encoder):
                return obs_encoder
        except Exception:
            pass
        if self._is_observation_encoder(self._policy_module):
            return self._policy_module
        for _name, module in getattr(self._policy_module, "named_modules", lambda: [])():
            if self._is_observation_encoder(module):
                return module
        raise RuntimeError(
            "Could not resolve the per-observation-group encoder from the policy "
            "obs_encoder. Expected a module with obs_shapes, obs_nets, and "
            "obs_randomizers."
        )

    @staticmethod
    def _is_observation_encoder(module: Any) -> bool:
        return all(
            hasattr(module, attr)
            for attr in ("obs_shapes", "obs_nets", "obs_randomizers")
        )

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
        assert prep is not None and hasattr(prep, "_prepare_observation"), (
            "Policy does not expose _prepare_observation needed to encode observations."
        )

        obs_t = prep._prepare_observation(obs, batched_ob=False)
        inputs = {"obs": obs_t}
        if goal is not None:
            inputs["goal"] = prep._prepare_observation(goal, batched_ob=False)

        obs_shapes = self._get_obs_shapes()
        for k in obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
        return inputs

    @staticmethod
    def _current_frame(x: torch.Tensor, obs_shape: Sequence[int]) -> torch.Tensor:
        if x.ndim - 2 == len(obs_shape):
            return x[:, -1]
        if x.ndim - 1 == len(obs_shape):
            return x
        raise ValueError(
            f"Observation tensor shape {tuple(x.shape)} does not match expected "
            f"obs shape {tuple(obs_shape)} with or without a time dimension."
        )

    @staticmethod
    def _flatten_feature(x: torch.Tensor) -> torch.Tensor:
        return TensorUtils.flatten(x, begin_axis=1)

    @staticmethod
    def _squeeze_single_batch(x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2, f"Expected batched feature tensor, got shape {tuple(x.shape)}."
        assert x.shape[0] == 1, (
            "Rollout feature extraction expects one environment observation at a time; "
            f"got batch size {x.shape[0]}."
        )
        return x[0].detach()

    def _extract_low_dim_feature(self, obs_t: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs_shapes = self._get_obs_shapes()
        parts = []
        for key in self.low_dim_keys:
            if key not in obs_t:
                raise KeyError(f"Prepared observation is missing low-dimensional key {key!r}.")
            current = self._current_frame(obs_t[key], obs_shapes[key])
            parts.append(self._flatten_feature(current))
        return torch.cat(parts, dim=-1)

    def _extract_image_feature(self, obs_t: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs_shapes = self._get_obs_shapes()
        feats = []
        for key in self.rgb_keys:
            if key not in obs_t:
                raise KeyError(f"Prepared observation is missing RGB key {key!r}.")
            x = self._current_frame(obs_t[key], obs_shapes[key])
            for randomizer in self._obs_randomizers_for_key(key):
                if randomizer is not None:
                    x = randomizer.forward_in(x)
            visual_net = self._obs_net_for_key(key)
            if visual_net is None:
                raise RuntimeError(f"RGB key {key!r} does not have an observation network.")
            x = visual_net(x)
            activation = getattr(self._obs_encoder, "activation", None)
            if activation is not None:
                x = activation(x)
            for randomizer in reversed(self._obs_randomizers_for_key(key)):
                if randomizer is not None:
                    x = randomizer.forward_out(x)
            feats.append(self._flatten_feature(x))
        return torch.cat(feats, dim=-1)

    def _obs_net_for_key(self, key: str) -> Any:
        obs_nets = getattr(self._obs_encoder, "obs_nets")
        if hasattr(obs_nets, "__getitem__"):
            return obs_nets[key]
        return getattr(obs_nets, key)

    def _obs_randomizers_for_key(self, key: str) -> List[Any]:
        obs_randomizers = getattr(self._obs_encoder, "obs_randomizers")
        randomizers = obs_randomizers[key] if hasattr(obs_randomizers, "__getitem__") else getattr(obs_randomizers, key)
        return list(randomizers)

    def _extract_full_obs_encoder_feature(self, inputs: Dict[str, Any]) -> torch.Tensor:
        # time_distributed mirrors robomimic's diffusion-policy path:
        # [B, T, ...] -> [B * T, ...] -> obs_encoder -> [B, T, Dobs].
        obs_features = TensorUtils.time_distributed(
            inputs,
            self._policy_module,
            inputs_as_kwargs=True,
        )
        if obs_features.ndim == 3:
            return obs_features[:, -1, :]
        if obs_features.ndim == 2:
            return obs_features
        raise ValueError(f"Unexpected obs_encoder output shape: {tuple(obs_features.shape)}.")

    def update_latent_from_obs(self, obs: Dict[str, Any], goal: Optional[Dict[str, Any]] = None) -> None:
        """Refresh the cached feature for the current environment observation."""
        inputs = self._prepare_obs_inputs(obs, goal=goal)
        with torch.no_grad():
            if self.feat_type == "low_dim_concat":
                feat = self._extract_low_dim_feature(inputs["obs"])
            elif self.feat_type == "image_embedding":
                feat = self._extract_image_feature(inputs["obs"])
            elif self.feat_type == "both":
                feat = self._extract_full_obs_encoder_feature(inputs)
            else:
                raise AssertionError(f"Unhandled feat_type={self.feat_type!r}.")
        self._last_feature = self._squeeze_single_batch(feat)

    def ensure_feature(self, obs: Dict[str, Any], goal: Optional[Dict[str, Any]] = None) -> None:
        if self._last_feature is None:
            self.update_latent_from_obs(obs, goal=goal)

    def clear(self) -> None:
        self._last_feature = None

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
    """Record rollout trajectories and optional latent embeddings for offline storage.

    Use this during policy rollout/evaluation. The works flow is:
    1. call start_episode() with the initial obs,
    2. call step() each environment step, and then
    3. call finalize_episode() to get a RolloutLatentTrajectory (and optional stats) that can be saved
    """

    def __init__(
        self,
        feature_hook: PolicyFeatureHook,
        obs_keys: Optional[Sequence[str]] = None,
        store_obs: bool = True,
        store_next_obs: bool = False,
        encoder: Optional[Any] = None,
        encoder_device: str = "cuda",
        frame_stack: Optional[int] = None,
    ):
        self.feature_hook = feature_hook
        self.obs_keys = list(obs_keys) if obs_keys is not None else list(self.feature_hook.obs_keys)
        self.store_obs = bool(store_obs)
        self.store_next_obs = bool(store_next_obs)
        self.encoder = encoder
        self.encoder_device = encoder_device
        self.frame_stack = self.feature_hook.frame_stack

        # containers to hold rollout data over T rollout timesteps
        self._obs: OrderedDict[str, List[np.ndarray]] = OrderedDict()
        self._next_obs: OrderedDict[str, List[np.ndarray]] = OrderedDict()
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._infos: List[Dict[str, Any]] = []
        self._z: List[np.ndarray] = []

    def start_episode(self, obs: Dict[str, Any]) -> None:
        keys = sorted(obs.keys()) if self.obs_keys is None else self.obs_keys
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
        self._infos.append(OrderedDict() if info is None else \
                           OrderedDict(sorted(info.items(), key=lambda kv: kv[0])))

    def finalize(
        self,
        stats: RolloutStats,
    ) -> RolloutLatentTrajectory:
        """ Add final statistics object @RolloutStats to the trajectory """
        
        def _stack_obs_along_time(obs_list: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
            out: OrderedDict[str, np.ndarray] = OrderedDict()
            for k, items in obs_list.items():
                if not items: continue
                out[k] = np.stack(items, axis=0)
            return out

        return RolloutLatentTrajectory(
            latents=np.stack(self._z, axis=0).astype(np.float32) if self._z else None,
            actions= np.stack(self._actions, axis=0).astype(np.float32),
            rewards=np.asarray(self._rewards, dtype=np.float32),
            dones=np.asarray(self._dones, dtype=np.bool_),
            success=bool(stats.success_rate > 0.0),
            total_reward=float(stats.total_reward),
            horizon=int(stats.horizon),
            frame_stack=self.frame_stack,
            obs=_stack_obs_along_time(self._obs) if self.store_obs else None,
            next_obs=_stack_obs_along_time(self._next_obs) if self.store_next_obs else None,
            infos=self._infos,
            stats=stats,
            feature_type=self.feature_hook.feature_type if self.feature_hook is not None else None,
            feature_keys=self.feature_hook.feature_keys if self.feature_hook is not None else None,
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
            latents=traj.latents,
            actions=traj.actions,
            rewards=traj.rewards,
            dones=traj.dones,
            success=np.asarray([traj.success], dtype=np.bool_),
            total_reward=np.asarray([traj.total_reward], dtype=np.float32),
            horizon=np.asarray([traj.horizon], dtype=np.int64),
            frame_stack=np.asarray([traj.frame_stack], dtype=np.int64),
            feature_type=np.asarray([traj.feature_type or ""], dtype=np.str_),
            feature_keys=np.asarray(traj.feature_keys or [], dtype=np.str_),
        )
        return path

    with h5py.File(path, "w") as f:
        f.create_dataset("latents", data=traj.latents, compression="gzip")
        f.create_dataset("actions", data=traj.actions, compression="gzip")
        f.create_dataset("rewards", data=traj.rewards, compression="gzip")
        f.create_dataset("dones", data=traj.dones, compression="gzip")
        f.create_dataset("t", data=np.arange(traj.latents.shape[0], dtype=np.int64))

        f.attrs["success"] = int(traj.success)
        f.attrs["total_reward"] = float(traj.total_reward)
        f.attrs["horizon"] = int(traj.horizon)
        f.attrs["frame_stack"] = int(traj.frame_stack)
        if traj.feature_type is not None:
            f.attrs["feature_type"] = str(traj.feature_type)
        if traj.feature_keys is not None:
            f.attrs["feature_keys"] = json.dumps(list(traj.feature_keys))

        if traj.obs is not None:
            obs_group = f.create_group("obs")
            for k, v in traj.obs.items():
                obs_group.create_dataset(k, data=v, compression="gzip")

        if traj.next_obs is not None:
            next_group = f.create_group("next_obs")
            for k, v in traj.next_obs.items():
                next_group.create_dataset(k, data=v, compression="gzip")

    return path


def load_rollout_latents(path: Path) -> RolloutLatentTrajectory:
    """Load rollout latents saved by save_rollout_latents into a RolloutLatentTrajectory."""
    path = Path(path)
    assert path.is_file(), f"rollout file not found: {path}"
    assert path.suffix in {".npz", ".h5", ".hdf5"}, f"Unsupported rollout format: {path.suffix}"

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        latents = np.asarray(data["latents"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        rewards = np.asarray(data["rewards"], dtype=np.float32)
        dones = np.asarray(data["dones"], dtype=np.bool_)
        frame_stack = int(data["frame_stack"][0]) if "frame_stack" in data else 1
        success = bool(np.asarray(data["success"])[0]) if "success" in data else False
        total_reward = float(np.asarray(data["total_reward"])[0]) if "total_reward" in data else float(rewards.sum())
        horizon = int(np.asarray(data["horizon"])[0]) if "horizon" in data else int(rewards.shape[0])
        feature_type = str(np.asarray(data["feature_type"])[0]) if "feature_type" in data else None
        if feature_type == "":
            feature_type = None
        feature_keys = [str(k) for k in np.asarray(data["feature_keys"]).tolist()] if "feature_keys" in data else None
        return RolloutLatentTrajectory(
            latents=latents,
            actions=actions,
            rewards=rewards,
            dones=dones,
            success=success,
            total_reward=total_reward,
            horizon=horizon,
            frame_stack=frame_stack,
            obs=None,
            next_obs=None,
            infos=None,
            stats=None,
            feature_type=feature_type,
            feature_keys=feature_keys,
        )

    else:
        with h5py.File(path, "r") as f:
            latents = np.asarray(f["latents"], dtype=np.float32)
            actions = np.asarray(f["actions"], dtype=np.float32)
            rewards = np.asarray(f["rewards"], dtype=np.float32)
            dones = np.asarray(f["dones"], dtype=np.bool_)
            frame_stack = int(f.attrs.get("frame_stack", 1))
            success = bool(f.attrs.get("success", 0))
            total_reward = float(f.attrs.get("total_reward", rewards.sum()))
            horizon = int(f.attrs.get("horizon", rewards.shape[0]))

            obs = None
            if "obs" in f:
                obs = OrderedDict((k, np.asarray(v)) for k, v in f["obs"].items())

            next_obs = None
            if "next_obs" in f:
                next_obs = OrderedDict((k, np.asarray(v)) for k, v in f["next_obs"].items())
            feature_type = f.attrs.get("feature_type", None)
            feature_keys_raw = f.attrs.get("feature_keys", None)
            feature_keys = None
            if feature_keys_raw is not None:
                try:
                    feature_keys = list(json.loads(feature_keys_raw))
                except Exception:
                    feature_keys = [str(feature_keys_raw)]

        return RolloutLatentTrajectory(
            latents=latents,
            actions=actions,
            rewards=rewards,
            dones=dones,
            success=success,
            total_reward=total_reward,
            horizon=horizon,
            frame_stack=frame_stack,
            obs=obs,
            next_obs=next_obs,
            infos=None,
            stats=None,
            feature_type=None if feature_type is None else str(feature_type),
            feature_keys=feature_keys,
        )



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
