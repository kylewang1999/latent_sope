"""Helpers to record and save latent rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set
from copy import deepcopy
from collections import OrderedDict
import h5py

import torch
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.obs_nets import ObservationEncoder
from robomimic.algo import RolloutPolicy, PolicyAlgo
from robomimic.utils import tensor_utils as TensorUtils
from robomimic.envs.env_base import EnvBase

from src.latent_sope.robomimic_interface.encoders import resolve_module
from src.latent_sope.utils.common import timeit

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


class PolicyFeatureHook:
    """Attach forward hooks to cache features.

    Supported feat_type:
      - low_dim_concat: (original) hooks policy obs encoder output (often concat of low-dim)
      - visual_latent : (new) hooks VisualCore output for an rgb key and mirrors
                        activation -> rand.forward_out -> flatten to produce [B, D]
    """

    def __init__(
        self,
        policy: Any,
        obs_keys: Optional[Sequence[str]] = None,
        feat_type: str = "low_dim_concat",
        rgb_key: Optional[str] = None,   # optional override for visual_latent
    ):
        feat_type = str(feat_type).lower()
        assert feat_type in {"low_dim_concat", "high_dim_encode", "visual_latent"}, f"Unknown feat_type: {feat_type}"

        self.policy = policy
        self.frame_stack = get_policy_frame_stack(policy)
        self.feat_type = feat_type

        # NOTE: historically this was low-dim keys; keep for compatibility
        self.obs_keys = sorted(
            list(_get_nested_attr(policy, "policy.obs_config.modalities.obs.low_dim"))
            if obs_keys is None
            else list(obs_keys)
        )

        self._last_feature = None

        # This module is what update_latent_from_obs force-runs.
        self._policy_module = self._resolve_policy_module()

        # Hook handles
        self._hook_handle = None
        self._vis_hook_handle = None

        # Visual-latent state
        self._obs_encoder = None
        self._rgb_key = None
        self._visual_net = None
        self._randomizers = None
        self._activation = None

        if self.feat_type == "visual_latent":
            # ---- find ObservationEncoder inside the SAME object used at rollout time ----
            obs_encoder = self._resolve_obs_encoder()

            # choose rgb key
            self._rgb_key = rgb_key or self._pick_rgb_key(obs_encoder)

            if self._rgb_key not in obs_encoder.obs_nets:
                raise RuntimeError(
                    f"rgb_key '{self._rgb_key}' not found in obs_encoder.obs_nets keys: {list(obs_encoder.obs_nets.keys())}"
                )

            self._obs_encoder = obs_encoder
            self._visual_net = obs_encoder.obs_nets[self._rgb_key]              # VisualCore / VisualCoreLanguageConditioned
            self._randomizers = list(obs_encoder.obs_randomizers[self._rgb_key])
            self._activation = obs_encoder.activation

            def _vis_hook(_module, _inp, out):
                x = out
                # mirror ObservationEncoder.forward AFTER obs_nets[k](x)
                if self._activation is not None:
                    x = self._activation(x)
                for rand in reversed(self._randomizers):
                    if rand is not None:
                        x = rand.forward_out(x)
                x = TensorUtils.flatten(x, begin_axis=1).detach()
                self._last_feature = x

            self._vis_hook_handle = self._visual_net.register_forward_hook(_vis_hook)

        else:
            # ---- original behavior: hook policy module output ----
            def _hook(_module, _inp, out):
                if torch.is_tensor(out):
                    feat = out.detach()
                elif isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]):
                    feat = out[0].detach()
                else:
                    feat = out
                self._last_feature = feat

            self._hook_handle = self._policy_module.register_forward_hook(_hook)

    # ----------------------------
    # module resolution helpers
    # ----------------------------
    def _resolve_policy_module(self) -> Any:
        candidates = [
            "policy.nets.policy.obs_encoder",  # diffusion policy (often ObservationGroupEncoder)
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
            "Could not resolve a torch policy module for hooks. "
            "Pass a policy with nets['policy'] or a policy module that supports hooks."
        )

    def _resolve_obs_encoder(self) -> ObservationEncoder:
        """
        Find the first ObservationEncoder inside the rollout-time policy object.
        We try a few roots (policy wrapper, nested policy net) to avoid hooking
        the wrong module instance.
        """
        roots = []

        # Try the rollout wrapper first
        roots.append(self.policy)

        # Try nested modules commonly present in RolloutPolicy wrapper
        for p in ["policy", "policy.nets.policy", "nets.policy"]:
            try:
                roots.append(resolve_module(self.policy, p))
            except Exception:
                pass

        # Deduplicate while preserving order
        seen = set()
        unique_roots = []
        for r in roots:
            if r is None:
                continue
            rid = id(r)
            if rid in seen:
                continue
            seen.add(rid)
            unique_roots.append(r)

        for r in unique_roots:
            for name, m in getattr(r, "named_modules", lambda: [])():
                if isinstance(m, ObservationEncoder):
                    return m

        raise RuntimeError(
            "Could not find ObservationEncoder inside the rollout policy modules. "
            "This checkpoint may not be vision-enabled (no rgb modality in obs encoder)."
        )

    def _pick_rgb_key(self, obs_encoder: ObservationEncoder) -> str:
        keys = list(obs_encoder.obs_shapes.keys())
        rgb_keys = [k for k in keys if ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb")]
        if len(rgb_keys) == 0:
            raise RuntimeError(f"No rgb keys in ObservationEncoder.obs_shapes. Keys were: {keys}")
        return rgb_keys[0]

    # ----------------------------
    # policy / shape helpers
    # ----------------------------
    def _get_policy_algo(self) -> PolicyAlgo:
        return getattr(self.policy, "policy", self.policy)

    def _get_obs_shapes(self) -> Dict[str, Any]:
        algo = self._get_policy_algo()
        obs_shapes = getattr(algo, "obs_shapes", None)
        if obs_shapes is not None:
            return obs_shapes
        raise RuntimeError("Policy does not expose obs_shapes needed to encode observations.")

    # ----------------------------
    # feature refresh
    # ----------------------------
    def update_latent_from_obs(self, obs: Dict[str, Any], goal: Optional[Dict[str, Any]] = None) -> None:
        """Force-run model forward so hooks fire (needed for diffusion action buffer)."""

        def _prepare_obs_inputs(obs: Dict[str, Any], goal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            prep: RolloutPolicy = self.policy if hasattr(self.policy, "_prepare_observation") else getattr(self.policy, "policy", None)
            assert prep is not None and hasattr(prep, "_prepare_observation"), \
                "Policy does not expose _prepare_observation needed to encode observations."

            obs_t = prep._prepare_observation(obs, batched_ob=False)
            inputs = {"obs": obs_t}
            if goal is not None:
                inputs["goal"] = prep._prepare_observation(goal, batched_ob=False)

            obs_shapes = self._get_obs_shapes()
            if obs_shapes is not None:
                for k in obs_shapes:
                    # add time dimension if missing
                    if inputs["obs"][k].ndim - 1 == len(obs_shapes[k]):
                        inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            return inputs

        # IMPORTANT: allow visual_latent to use the same "force-run" pathway
        if self.feat_type not in {"low_dim_concat", "visual_latent"}:
            return

        inputs = _prepare_obs_inputs(obs, goal=goal)
        _ = TensorUtils.time_distributed(inputs, self._policy_module, inputs_as_kwargs=True)

    def ensure_feature(self, obs: Dict[str, Any], goal: Optional[Dict[str, Any]] = None) -> None:
        if self._last_feature is None:
            self.update_latent_from_obs(obs, goal=goal)

    def clear(self) -> None:
        self._last_feature = None

    def close(self) -> None:
        for h in [self._hook_handle, self._vis_hook_handle]:
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
        self._hook_handle = None
        self._vis_hook_handle = None

    def __del__(self):
        self.close()

    def pull_feat(self, clear: bool = True) -> np.ndarray:
        """Return last captured feature as float32 numpy array."""
        assert self._last_feature is not None, (
            "Feature hook has no cached output. Ensure the module runs during policy forward "
            "before calling pull_feat."
        )

        feat = self._last_feature
        if hasattr(feat, "detach"):
            feat = feat.detach().cpu().numpy()
        else:
            feat = np.asarray(feat)

        if clear:
            self.clear()

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
        self.obs_keys = None if obs_keys is None else sorted(obs_keys)
        self.store_obs = bool(store_obs)
        self.store_next_obs = bool(store_next_obs)
        self.encoder = encoder
        self.encoder_device = encoder_device
        self.frame_stack = self.feature_hook.frame_stack
        self.obs_keys = sorted(self.feature_hook.obs_keys)

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
