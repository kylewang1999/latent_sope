"""Utilities for working with robomimic experiment outputs.

Why this exists:
- robomimic stores checkpoints + config under an experiment folder.
- we want robust discovery of "which checkpoint to load" and a thin wrapper
  around robomimic's checkpoint loading APIs.

Robomimic output directory structure (per docs):
- <train.output_dir>/<experiment.name>/<date>/
    config.json
    models/
    logs/
    videos/

See robomimic documentation for details.

NOTE: This module requires robomimic at runtime for building policies. It is
written to fail with clear error messages if robomimic is missing.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import h5py
from copy import deepcopy

import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import algo_factory
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from src.latent_sope.utils.common import CONSOLE_LOGGER


def print_h5_tree(
    group: h5py.Group,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 2,
    max_children: int = 8,
) -> None:
    if depth > max_depth:
        return
    keys = sorted(list(group.keys()))
    for key in keys[:max_children]:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            shape = tuple(item.shape)
            print(f"{prefix}{key}  [dataset] shape={shape} dtype={item.dtype}")
        else:
            print(f"{prefix}{key}/")
            print_h5_tree(
                item,
                prefix=prefix + "  ",
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
            )
    if len(keys) > max_children:
        print(f"{prefix}... ({len(keys) - max_children} more)")


def load_demo(
    h5: h5py.File, demo_key: str, obs_keys: List[str], num_steps: int
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    demo = h5["data"][demo_key]
    obs_group = demo["obs"]
    obs = {k: obs_group[k][:num_steps] for k in obs_keys}
    actions = demo["actions"][:num_steps]
    return obs, actions


def prepare_obs(
    obs: Dict[str, np.ndarray],
    device: torch.device,
    obs_stats: Dict[str, Dict[str, np.ndarray]] | None,
) -> Dict[str, torch.Tensor]:
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.tensor_utils as TensorUtils

    obs_t = TensorUtils.to_tensor(obs)
    obs_t = TensorUtils.to_device(obs_t, device)
    obs_t = TensorUtils.to_float(obs_t)

    if obs_stats is not None:
        stats_t = TensorUtils.to_float(
            TensorUtils.to_device(TensorUtils.to_tensor(obs_stats), device)
        )
        obs_t = ObsUtils.normalize_dict(obs_t, normalization_stats=stats_t)

    for k in obs_t:
        if ObsUtils.key_is_obs_modality(
            key=k, obs_modality="rgb"
        ) or ObsUtils.key_is_obs_modality(key=k, obs_modality="depth"):
            obs_t[k] = ObsUtils.process_obs(obs=obs_t[k], obs_key=k)
    return obs_t


@dataclass
class RobomimicCheckpoint:
    """Container for a robomimic checkpoint plus some convenient metadata."""

    run_dir: Path  # Root directory of the robomimic experiment (contains config.json, models/, logs/, videos/)
    ckpt_path: Path  # Full path to the checkpoint file (.pth)
    epoch: int  # Training epoch number extracted from the checkpoint filename
    algo_name: Optional[str]  # Algorithm name from the config (e.g., "bc", "diffusion_policy"), or None if unavailable
    config_json: Optional[Dict[str, Any]]  # Full parsed config.json from the experiment, or None if not found
    ckpt_dict: Dict[str, Any]  # Loaded checkpoint dictionary from the .pth file (contains model weights and training state)


_EPOCH_PATTERNS = [
    re.compile(r"model_epoch_(\d+)", re.IGNORECASE),
    re.compile(r"model_epoch(\d+)", re.IGNORECASE),
    re.compile(r"epoch_(\d+)", re.IGNORECASE),
]


def _parse_epoch_from_filename(path: Path) -> int:
    stem = path.stem
    for pat in _EPOCH_PATTERNS:
        m = pat.search(stem)
        if m is not None:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return -1


def _find_latest_checkpoint(run_dir: Path, models_subdir: str = "models") -> Path:
    """Return the most recent (highest-epoch) checkpoint under run_dir/models.

    We primarily look for files named like `model_epoch_XX.pth`, but fall back
    to modification time if the epoch can't be parsed.

    Args:
        run_dir: robomimic experiment run directory (contains config.json + models/)
        models_subdir: name of the checkpoint folder (default: "models")

    Returns:
        Path to a checkpoint file.

    Raises:
        FileNotFoundError: if no checkpoint is found.
    """

    models_dir = run_dir / models_subdir
    if not models_dir.is_dir():
        raise FileNotFoundError(f"models directory not found: {models_dir}")

    candidates = sorted({*models_dir.glob("*.pth"), *models_dir.glob("*.pt")})
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found under: {models_dir}")

    scored = []
    for p in candidates:
        epoch = _parse_epoch_from_filename(p)
        scored.append((epoch, p.stat().st_mtime, p))

    # Prefer highest epoch; if epoch=-1 for all, falls back to mtime.
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[-1][2]


def _load_ckpt_dict_torch(ckpt_path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    try:
        import torch

        obj = torch.load(str(ckpt_path), map_location=map_location)
    except Exception as e:
        raise RuntimeError(
            f"Failed to torch.load checkpoint at {ckpt_path}: {e}"
        ) from e

    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected checkpoint to be a dict-like object, got type={type(obj)}"
        )
    return obj


def _load_ckpt_dict_robomimic(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    """Load checkpoint using robomimic's helper if available."""

    try:
        import robomimic.utils.file_utils as FileUtils  # type: ignore

        # maybe_dict_from_checkpoint handles path strings and legacy formats
        if hasattr(FileUtils, "maybe_dict_from_checkpoint"):
            return FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(ckpt_path))
    except Exception:
        return None

    return None


def _load_config_json(run_dir: Path) -> Optional[Dict[str, Any]]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        # config might be malformed or non-json; keep it optional
        pass
    return None


def _convert_normalization_stats(
    stats: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Convert nested lists in normalization stats to numpy arrays."""

    if stats is None:
        return None

    def _convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return np.array(obj)
        return obj

    return _convert(stats)


def load_checkpoint(
    run_dir: Path,
    ckpt_path: Optional[Path] = None,
    map_location: str = "cpu",
) -> RobomimicCheckpoint:
    """Load a robomimic checkpoint dict + minimal metadata.

    Robomimic checkpoints (per robomimic Algo docs) include keys like:
    - algo_name
    - shape_metadata (including ac_dim and obs shapes)
    - model (serialized weights)

    and can be used to reconstruct the Algo via algo_factory.

    Args:
        run_dir: robomimic run directory (contains config.json, models/)
        ckpt_path: optional explicit checkpoint path relative to run_dir. If None, auto-select.
        map_location: torch map_location for loading.

    Returns:
        RobomimicCheckpoint
    """

    run_dir = Path(run_dir)
    if ckpt_path is None:
        ckpt_path = _find_latest_checkpoint(run_dir)
    ckpt_path = run_dir / ckpt_path

    ckpt_dict = _load_ckpt_dict_robomimic(ckpt_path)
    if ckpt_dict is None:
        ckpt_dict = _load_ckpt_dict_torch(ckpt_path, map_location=map_location)

    algo_name = ckpt_dict.get("algo_name", None)
    epoch = _parse_epoch_from_filename(ckpt_path)

    return RobomimicCheckpoint(
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        epoch=epoch,
        algo_name=algo_name,
        config_json=_load_config_json(run_dir),
        ckpt_dict=ckpt_dict,
    )


def build_algo_from_checkpoint(
    ckpt: RobomimicCheckpoint,
    device: str = "cuda",
) -> Any:
    """Reconstruct a robomimic Algo instance from a checkpoint.

    This mirrors the logic shown in robomimic's Algo documentation:
    - load ckpt_dict
    - recover algo_name + config
    - call algo_factory with obs shapes + action dimension
    - deserialize weights



    Returns:
        robomimic Algo instance.

    Raises:
        ImportError if robomimic is not importable.
        KeyError if the checkpoint is missing required keys.
    """

    ckpt_dict = ckpt.ckpt_dict

    algo_name = ckpt_dict.get("algo_name", None) or ckpt.algo_name
    if algo_name is None:
        raise KeyError("Checkpoint missing 'algo_name' key")

    if not hasattr(FileUtils, "config_from_checkpoint"):
        raise AttributeError(
            "robomimic.utils.file_utils.config_from_checkpoint not found. "
            "Your robomimic version may be too old."
        )

    config, _ = FileUtils.config_from_checkpoint(
        algo_name=algo_name, ckpt_dict=ckpt_dict
    )
    ObsUtils.initialize_obs_utils_with_config(config)

    shape_md = ckpt_dict.get("shape_metadata", None)
    if shape_md is None:
        raise KeyError("Checkpoint missing 'shape_metadata' key")

    obs_shapes = shape_md.get("all_shapes", None)
    ac_dim = shape_md.get("ac_dim", None)
    if obs_shapes is None or ac_dim is None:
        raise KeyError("shape_metadata missing 'all_shapes' and/or 'ac_dim'")

    # Create Algo instance and load weights
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=obs_shapes,
        ac_dim=ac_dim,
        device=device,
    )

    model.deserialize(ckpt_dict["model"])
    model.set_eval()
    return model


def build_rollout_policy_from_checkpoint(
    ckpt: RobomimicCheckpoint,
    device: str = "cuda",
    verbose: bool = True,
) -> Any:
    """Build a robomimic rollout policy (callable) from checkpoint.

    Preferred path: call robomimic's FileUtils.policy_from_checkpoint, which
    handles many version quirks and returns a policy wrapper.

    If that isn't available, we fall back to build_algo_from_checkpoint and
    wrap it with RolloutPolicy.
    """

    try:
        import robomimic.utils.file_utils as FileUtils  # type: ignore

        if hasattr(FileUtils, "policy_from_checkpoint"):
            out = FileUtils.policy_from_checkpoint(
                ckpt_path=str(ckpt.ckpt_path),
                device=device,
                verbose=verbose,
            )
            # Some versions return (policy, ckpt_dict). Others return just policy.
            if isinstance(out, tuple) and len(out) >= 1:
                return out[0]
            return out
    except Exception:
        pass

    # Fallback route (no policy_from_checkpoint)
    algo = build_algo_from_checkpoint(ckpt, device=device)
    try:
        from robomimic.algo.algo import RolloutPolicy  # type: ignore

        obs_stats = _convert_normalization_stats(
            ckpt.ckpt_dict.get("obs_normalization_stats", None)
        )
        action_stats = _convert_normalization_stats(
            ckpt.ckpt_dict.get("action_normalization_stats", None)
        )
        return RolloutPolicy(
            algo,
            obs_normalization_stats=obs_stats,
            action_normalization_stats=action_stats,
        )
    except Exception as e:
        raise ImportError(
            "Could not import RolloutPolicy from robomimic. "
            "Your robomimic installation may be incomplete."
        ) from e


def build_env_from_checkpoint(
    ckpt: RobomimicCheckpoint,
    render: bool = False,
    render_offscreen: bool = False,
    verbose: bool = True,
    env_name: Optional[str] = None,
) -> Any:
    """Reconstruct a robomimic environment from a checkpoint."""

    try:
        import robomimic.utils.file_utils as FileUtils  # type: ignore
    except Exception as e:
        raise ImportError(
            "robomimic is required to build an environment from checkpoint. "
            "Make sure third_party/robomimic is installed in your environment."
        ) from e

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt.ckpt_dict,
        env_name=env_name,
        render=render,
        render_offscreen=render_offscreen,
        verbose=verbose,
    )
    return env


def rollout(
    policy: RolloutPolicy,
    env: EnvBase,
    horizon: int,
    render: bool = False,
    video_writer=None,
    video_skip: int = 5,
    camera_names: List[str] | None = None,
) -> Dict[str, float]:
    assert not (render and (video_writer is not None))
    camera_names = camera_names or ["agentview"]

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    total_reward = 0.0
    success = False
    video_count = 0

    try:
        for step_i in range(horizon):
            act = policy(ob=obs)
            next_obs, reward, done, _ = env.step(act)
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

            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()
            
    except env.rollout_exceptions as e:
        print(f"WARNING: got rollout exception {e}")

    return dict(
        Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success)
    )
