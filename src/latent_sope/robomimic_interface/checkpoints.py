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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class RobomimicCheckpoint:
    """Container for a robomimic checkpoint plus some convenient metadata."""

    run_dir: Path
    ckpt_path: Path
    epoch: int
    algo_name: Optional[str]
    config_json: Optional[Dict[str, Any]]
    ckpt_dict: Dict[str, Any]


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


def find_latest_checkpoint(run_dir: Path, models_subdir: str = "models") -> Path:
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
        raise RuntimeError(f"Failed to torch.load checkpoint at {ckpt_path}: {e}") from e

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
        return None


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
        ckpt_path: optional explicit checkpoint path. If None, auto-select.
        map_location: torch map_location for loading.

    Returns:
        RobomimicCheckpoint
    """

    run_dir = Path(run_dir)
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(run_dir)
    ckpt_path = Path(ckpt_path)

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

    try:
        import robomimic.utils.file_utils as FileUtils  # type: ignore
        from robomimic.algo import algo_factory  # type: ignore
    except Exception as e:
        raise ImportError(
            "robomimic is required to build an Algo from checkpoint. "
            "Make sure third_party/robomimic is installed in your environment."
        ) from e

    ckpt_dict = ckpt.ckpt_dict

    algo_name = ckpt_dict.get("algo_name", None) or ckpt.algo_name
    if algo_name is None:
        raise KeyError("Checkpoint missing 'algo_name' key")

    if not hasattr(FileUtils, "config_from_checkpoint"):
        raise AttributeError(
            "robomimic.utils.file_utils.config_from_checkpoint not found. "
            "Your robomimic version may be too old."
        )

    config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict)

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

        return RolloutPolicy(algo)
    except Exception as e:
        raise ImportError(
            "Could not import RolloutPolicy from robomimic. "
            "Your robomimic installation may be incomplete."
        ) from e
