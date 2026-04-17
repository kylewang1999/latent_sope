#!/usr/bin/env python3
"""Convert robomimic dataset demonstrations into policy observation embeddings.

This is the rollout-free counterpart to rollout latent collection: it reads
observations from a robomimic HDF5 dataset, runs the checkpoint policy encoder,
and writes one RolloutLatentTrajectory file per source demo.

Example commands:

1. Run with all defaults, which converts
   `data/robomimic/lift/mh/image_v15.hdf5` using policy
   `rmimic-lift-mh-image-v15-diffusion_260123`:

   ```bash
   python3 src/robomimic_interface/post_process_robomimic_h5.py
   ```

2. Convert `data/robomimic/lift/ph/low_dim_v15.hdf5` using checkpoint
   `data/policy/rmimic-lift-ph-lowdim_diffusion_260130/last.pth`:

   ```bash
   python3 src/robomimic_interface/post_process_robomimic_h5.py \
       --policy-name rmimic-lift-ph-lowdim_diffusion_260130 \
       --checkpoint-name data/policy/rmimic-lift-ph-lowdim_diffusion_260130/last.pth \
       --dataset data/robomimic/lift/ph/low_dim_v15.hdf5 \
       --feat-type low_dim
   ```
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Sequence

import h5py
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.robomimic_interface.rollout import RolloutLatentTrajectory, RolloutStats
from src.utils import PATHS

DEFAULT_POLICY_NAME = "rmimic-lift-mh-image-v15-diffusion_260123"
DEFAULT_CHECKPOINT_NAME = REPO_ROOT / "data" / "policy" / "rmimic-lift-mh-image-v15-diffusion_260123" / "last.pth"
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "robomimic" / "lift" / "mh" / "image_v15.hdf5"
FEAT_TYPE_ALIASES = {
    "low_dim": "low_dim_concat",
    "low_dim_concat": "low_dim_concat",
    "image_embedding": "image_embedding",
    "both": "both",
}
VALID_FEAT_TYPES = tuple(sorted(FEAT_TYPE_ALIASES.keys()))


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_path(dataset: Path) -> Path:
    dataset_path = Path(dataset).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Robomimic dataset not found: {dataset_path}. "
            "Pass --dataset with the robomimic .hdf5 file you want to convert."
        )
    return dataset_path


def _normalize_feat_type(feat_type: str) -> str:
    feat_type_token = str(feat_type).lower()
    if feat_type_token not in FEAT_TYPE_ALIASES:
        raise ValueError(
            f"Unknown feat_type={feat_type!r}. Expected one of {sorted(FEAT_TYPE_ALIASES)}."
        )
    return FEAT_TYPE_ALIASES[feat_type_token]


def _resolve_default_output_dir(dataset_path: Path, feat_type_token: str) -> Path:
    parts = dataset_path.parts
    robomimic_index = None
    for idx in range(len(parts) - 4, -1, -1):
        if parts[idx] == "robomimic":
            robomimic_index = idx
            break

    if robomimic_index is None:
        raise ValueError(
            "Could not infer the default output directory from dataset path "
            f"{dataset_path}. Expected a path of the form "
            "<prefix>/robomimic/<task>/<quality>/<observation-description>_v15.h5 "
            "or .hdf5. Pass --output-dir explicitly."
        )

    if len(parts) <= robomimic_index + 3:
        raise ValueError(
            "Could not infer the default output directory from dataset path "
            f"{dataset_path}. Expected robomimic/<task>/<quality>/<file>. "
            "Pass --output-dir explicitly."
        )

    dataset_filename = parts[robomimic_index + 3]
    if Path(dataset_filename).suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(
            "Could not infer the default output directory from dataset path "
            f"{dataset_path}. Expected the robomimic dataset file to end in "
            ".h5 or .hdf5. Pass --output-dir explicitly."
        )

    robomimic_quality_dir = Path(*parts[: robomimic_index + 3])
    return robomimic_quality_dir / "postprocessed_for_ope" / feat_type_token


def _resolve_output_dir(
    *,
    dataset_path: Path,
    output_dir: Optional[Path],
    feat_type_token: str,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return _resolve_default_output_dir(dataset_path, feat_type_token)


def _config_obs_modality_keys(config_json: dict[str, Any], modality: str) -> List[str]:
    observation = config_json.get("observation", {})
    modalities = observation.get("modalities", {})
    obs_modalities = modalities.get("obs", {})
    values = obs_modalities.get(modality, [])
    return [str(value) for value in values]


def _allowed_feat_type_tokens_for_modalities(has_low_dim: bool, has_rgb: bool) -> List[str]:
    if has_low_dim and has_rgb:
        return ["both"]
    if has_low_dim:
        return ["both", "low_dim", "low_dim_concat"]
    if has_rgb:
        return ["both", "image_embedding"]
    return []


def _validate_feat_type_against_checkpoint_config(
    *,
    config_json: Optional[dict[str, Any]],
    policy_name: str,
    checkpoint_path: Path,
    requested_feat_type: str,
    canonical_feat_type: str,
) -> None:
    if config_json is None:
        raise RuntimeError(
            f"Policy {policy_name!r} checkpoint {checkpoint_path} does not expose config.json, "
            "so --feat-type cannot be validated."
        )

    low_dim_keys = _config_obs_modality_keys(config_json, "low_dim")
    rgb_keys = _config_obs_modality_keys(config_json, "rgb")
    has_low_dim = bool(low_dim_keys)
    has_rgb = bool(rgb_keys)

    if has_low_dim and has_rgb:
        allowed_canonical = {"both"}
    elif has_low_dim:
        allowed_canonical = {"both", "low_dim_concat"}
    elif has_rgb:
        allowed_canonical = {"both", "image_embedding"}
    else:
        raise ValueError(
            f"Policy {policy_name!r} checkpoint {checkpoint_path} has no supported obs modalities "
            f"in config.json. low_dim={low_dim_keys}, rgb={rgb_keys}."
        )

    if canonical_feat_type not in allowed_canonical:
        allowed_display = ", ".join(
            repr(token) for token in _allowed_feat_type_tokens_for_modalities(has_low_dim, has_rgb)
        )
        raise ValueError(
            f"--feat-type {requested_feat_type!r} is inconsistent with policy {policy_name!r} "
            f"checkpoint {checkpoint_path}. config.json modalities are "
            f"low_dim={low_dim_keys}, rgb={rgb_keys}. Allowed feat types: {allowed_display}."
        )


def _expected_current_step_feature_dim(feature_hook: Any) -> int:
    output_shape = getattr(feature_hook._obs_encoder, "output_shape", None)
    if not callable(output_shape):
        raise RuntimeError("Resolved robomimic obs_encoder does not expose output_shape().")
    shape = tuple(output_shape())
    if len(shape) != 1:
        raise RuntimeError(
            f"Expected obs_encoder.output_shape() to describe one current-step feature width, got {shape}."
        )
    return int(shape[0])


def _sorted_demo_keys_from_h5(
    h5: h5py.File,
    *,
    filter_key: Optional[str] = None,
    demo_limit: Optional[int] = None,
) -> List[str]:
    if filter_key is not None:
        mask_path = f"mask/{filter_key}"
        if mask_path not in h5:
            raise KeyError(f"Dataset is missing filter mask {mask_path!r}.")
        demos = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in np.asarray(h5[mask_path])
        ]
    else:
        demos = list(h5["data"].keys())

    def _demo_sort_key(name: str) -> tuple[int, str]:
        try:
            return (int(name.split("_")[-1]), name)
        except Exception:
            return (10**12, name)

    demos = sorted(demos, key=_demo_sort_key)
    if demo_limit is not None:
        demos = demos[: int(demo_limit)]
    return demos


def _read_demo_array(
    demo_group: h5py.Group,
    key: str,
    horizon: int,
    *,
    required: bool = False,
) -> Optional[np.ndarray]:
    if key not in demo_group:
        if required:
            raise KeyError(f"Demo {demo_group.name} is missing required dataset {key!r}.")
        return None
    arr = np.asarray(demo_group[key])
    return arr[:horizon]


def _resolve_demo_horizon(demo_group: h5py.Group) -> int:
    if "num_samples" in demo_group.attrs:
        return int(demo_group.attrs["num_samples"])
    if "actions" in demo_group:
        return int(demo_group["actions"].shape[0])
    if "obs" in demo_group:
        obs_group = demo_group["obs"]
        first_key = next(iter(obs_group.keys()))
        return int(obs_group[first_key].shape[0])
    raise ValueError(f"Could not resolve horizon for demo {demo_group.name}.")


def _read_rewards_and_dones(demo_group: h5py.Group, horizon: int) -> tuple[np.ndarray, np.ndarray, bool]:
    rewards_raw = _read_demo_array(demo_group, "rewards", horizon)
    rewards_exist = rewards_raw is not None
    rewards = (
        np.asarray(rewards_raw, dtype=np.float32).reshape(horizon)
        if rewards_raw is not None
        else np.zeros((horizon,), dtype=np.float32)
    )

    dones_raw = _read_demo_array(demo_group, "dones", horizon)
    if dones_raw is None:
        dones = np.zeros((horizon,), dtype=np.bool_)
        if horizon > 0:
            dones[-1] = True
    else:
        dones = np.asarray(dones_raw, dtype=np.bool_).reshape(horizon)
    return rewards, dones, rewards_exist


def _success_from_dataset(rewards: np.ndarray, dones: np.ndarray, rewards_exist: bool) -> bool:
    if rewards_exist:
        return bool(np.any(np.asarray(rewards) > 0.0))
    return bool(np.any(np.asarray(dones, dtype=np.bool_)))


def _feature_obs_keys(feature_hook: Any) -> List[str]:
    obs_shapes = feature_hook._get_obs_shapes()
    return list(obs_shapes.keys())


def _stored_obs_keys(
    feature_hook: Any,
    *,
    store_rgb_obs: bool,
) -> List[str]:
    keys = list(feature_hook.low_dim_keys)
    if store_rgb_obs:
        keys.extend(k for k in feature_hook.rgb_keys if k not in keys)
    return keys


def build_sequence_dataset(
    dataset_path: Path,
    obs_keys: Sequence[str],
    *,
    frame_stack: int,
    filter_key: Optional[str] = None,
    demo_limit: Optional[int] = None,
) -> Any:
    # Import after checkpoint setup has placed the vendored robomimic package on sys.path.
    from robomimic.utils.dataset import SequenceDataset

    return SequenceDataset(
        hdf5_path=str(dataset_path),
        obs_keys=tuple(obs_keys),
        action_keys=(),
        dataset_keys=(),
        action_config={},
        frame_stack=int(frame_stack),
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=filter_key,
        load_next_obs=False,
        demo_limit=demo_limit,
    )


def convert_demo_to_trajectory(
    h5: h5py.File,
    sequence_dataset: Any,
    demo_key: str,
    feature_hook: Any,
    *,
    store_obs: bool = True,
    store_rgb_obs: bool = False,
) -> RolloutLatentTrajectory:

    demo_group = h5[f"data/{demo_key}"]
    horizon = _resolve_demo_horizon(demo_group)
    if horizon <= 0:
        raise ValueError(f"Demo {demo_key!r} has non-positive horizon {horizon}.")

    actions = np.asarray(
        _read_demo_array(demo_group, "actions", horizon, required=True),
        dtype=np.float32,
    )
    rewards, dones, rewards_exist = _read_rewards_and_dones(demo_group, horizon)
    frame_stack = int(feature_hook.frame_stack)
    feature_obs_keys = tuple(sequence_dataset.obs_keys)

    latents: List[np.ndarray] = []
    for t in range(horizon):
        obs_window = sequence_dataset.get_obs_sequence_from_demo(
            demo_key,
            index_in_demo=t,
            keys=feature_obs_keys,
            num_frames_to_stack=frame_stack - 1,
            seq_length=1,
            prefix="obs",
        )
        feature_hook.update_latent_from_obs(obs_window)
        feat = np.asarray(feature_hook.pull_feat(clear=True), dtype=np.float32)
        if feat.ndim != 1:
            raise ValueError(
                f"Expected one current-step feature vector per dataset step, got shape {tuple(feat.shape)} "
                f"for demo {demo_key!r} at t={t}."
            )
        latents.append(feat)

    obs: Optional[OrderedDict[str, np.ndarray]] = None
    if store_obs:
        obs = OrderedDict()
        for key in _stored_obs_keys(feature_hook, store_rgb_obs=store_rgb_obs):
            if f"obs/{key}" in demo_group:
                obs[key] = np.asarray(demo_group[f"obs/{key}"][:horizon])

    success = _success_from_dataset(rewards, dones, rewards_exist)
    stats = RolloutStats(
        total_reward=float(rewards.sum()),
        horizon=int(horizon),
        success_rate=float(success),
    )
    # This is one converted source demo, so arrays stay time-major [T, ...]
    # for a single trajectory rather than batched [B, T, ...].
    latent_array = np.stack(latents, axis=0).astype(np.float32)
    if feature_hook.feature_type == "both":
        expected_dim = _expected_current_step_feature_dim(feature_hook)
        assert latent_array.ndim == 2 and latent_array.shape[1] == expected_dim, (
            f"feat_type='both' must store one-step D_obs features. Got latents with shape "
            f"{tuple(latent_array.shape)} for demo {demo_key!r}, expected second dimension "
            f"{expected_dim} from obs_encoder.output_shape()[0]."
        )

    return RolloutLatentTrajectory(
        latents=latent_array,
        actions=actions,
        rewards=rewards,
        dones=dones,
        success=success,
        total_reward=float(rewards.sum()),
        horizon=int(horizon),
        frame_stack=frame_stack,
        obs=obs,
        next_obs=None,
        infos=None,
        stats=stats,
        feature_type=feature_hook.feature_type,
        feature_keys=feature_hook.feature_keys,
    )


def save_converted_demo(
    output_path: Path,
    traj: RolloutLatentTrajectory,
    *,
    overwrite: bool,
    source_dataset_path: Path,
    demo_key: str,
    policy_train_dir: Path,
    checkpoint_path: Path,
    feat_type: str,
) -> Path:
    from src.robomimic_interface.rollout import save_rollout_latents

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing converted demo file {output_path}. "
            "Pass --overwrite to replace files."
        )

    save_rollout_latents(output_path, traj)
    with h5py.File(output_path, "a") as handle:
        handle.attrs["source_dataset_path"] = str(Path(source_dataset_path))
        handle.attrs["demo_key"] = str(demo_key)
        handle.attrs["policy_train_dir"] = str(Path(policy_train_dir))
        handle.attrs["checkpoint_path"] = str(Path(checkpoint_path))
        handle.attrs["source_format"] = "robomimic_dataset"
        handle.attrs["feat_type"] = str(feat_type)
    return output_path


def convert_dataset(
    *,
    policy_name: str,
    checkpoint_name: Path,
    dataset_path: Optional[Path],
    output_dir: Optional[Path],
    feat_type: str,
    device: str,
    demo_limit: Optional[int],
    filter_key: Optional[str],
    overwrite: bool,
    store_obs: bool,
    store_rgb_obs: bool,
) -> List[Path]:
    from src.robomimic_interface.checkpoints import (
        build_rollout_policy_from_checkpoint,
        load_checkpoint,
    )
    from src.robomimic_interface.rollout import PolicyFeatureHook, get_policy_frame_stack

    policy_train_dir = PATHS.robomimic_policy_dir(policy_name).resolve()
    if not policy_train_dir.is_dir():
        raise FileNotFoundError(
            f"Policy directory not found: {policy_train_dir}. "
            "Expected a prepared policy under data/policy/<policy-name>."
        )
    if dataset_path is None:
        raise ValueError("convert_dataset requires an explicit dataset_path.")
    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    requested_feat_type = str(feat_type).lower()
    canonical_feat_type = _normalize_feat_type(requested_feat_type)
    resolved_output_dir = _resolve_output_dir(
        dataset_path=resolved_dataset_path,
        output_dir=output_dir,
        feat_type_token=requested_feat_type,
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(policy_train_dir, ckpt_path=Path(checkpoint_name))
    _validate_feat_type_against_checkpoint_config(
        config_json=checkpoint.config_json,
        policy_name=policy_name,
        checkpoint_path=checkpoint.ckpt_path,
        requested_feat_type=requested_feat_type,
        canonical_feat_type=canonical_feat_type,
    )
    policy = build_rollout_policy_from_checkpoint(
        checkpoint,
        device=device,
        verbose=False,
    )
    feature_hook = PolicyFeatureHook(policy, feat_type=canonical_feat_type)
    feature_hook.frame_stack = get_policy_frame_stack(policy, default=feature_hook.frame_stack)
    sequence_dataset = build_sequence_dataset(
        resolved_dataset_path,
        _feature_obs_keys(feature_hook),
        frame_stack=feature_hook.frame_stack,
        filter_key=filter_key,
        demo_limit=demo_limit,
    )

    saved_paths: List[Path] = []
    try:
        with sequence_dataset.hdf5_file_opened() as h5:
            demo_keys = _sorted_demo_keys_from_h5(
                h5,
                filter_key=filter_key,
                demo_limit=demo_limit,
            )
            for demo_key in tqdm(demo_keys, desc="Dataset embeddings", unit="demo"):
                traj = convert_demo_to_trajectory(
                    h5,
                    sequence_dataset,
                    demo_key,
                    feature_hook,
                    store_obs=store_obs,
                    store_rgb_obs=store_rgb_obs,
                )
                output_path = resolved_output_dir / f"{demo_key}.h5"
                saved_paths.append(
                    save_converted_demo(
                        output_path,
                        traj,
                        overwrite=overwrite,
                        source_dataset_path=resolved_dataset_path,
                        demo_key=demo_key,
                        policy_train_dir=policy_train_dir,
                        checkpoint_path=checkpoint.ckpt_path,
                        feat_type=canonical_feat_type,
                    )
                )
    finally:
        feature_hook.close()
        close_dataset = getattr(sequence_dataset, "close_and_delete_hdf5_handle", None)
        if callable(close_dataset):
            close_dataset()

    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert robomimic image_v15.hdf5 demonstrations into checkpoint "
            "policy observation embeddings without environment rollout."
        )
    )
    parser.add_argument("--policy-name", type=str, default=DEFAULT_POLICY_NAME)
    parser.add_argument("--checkpoint-name", type=Path, default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Path to the robomimic .hdf5 file to convert. "
            "Defaults to data/robomimic/lift/mh/image_v15.hdf5."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for converted demo files. When omitted, defaults to "
            "<prefix>/robomimic/<task>/<quality>/postprocessed_for_ope/<feat-type>."
        ),
    )
    parser.add_argument(
        "--feat-type",
        choices=tuple(sorted(VALID_FEAT_TYPES)),
        default="both",
        help=(
            "Feature representation to store per timestep. "
            "'low_dim' is accepted as an alias for the canonical 'low_dim_concat'."
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--demo-limit", type=int, default=None)
    parser.add_argument("--filter-key", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--store-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store compact raw observations for metadata/debugging.",
    )
    parser.add_argument(
        "--store-rgb-obs",
        action="store_true",
        default=False,
        help="Also store raw RGB observations. Disabled by default to keep files compact.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.demo_limit is not None and args.demo_limit <= 0:
        raise ValueError(f"--demo-limit must be positive when set, got {args.demo_limit}.")
    if args.store_rgb_obs and not args.store_obs:
        raise ValueError("--store-rgb-obs requires --store-obs.")

    device = args.device or _default_device()
    saved_paths = convert_dataset(
        policy_name=args.policy_name,
        checkpoint_name=args.checkpoint_name,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        feat_type=args.feat_type,
        device=device,
        demo_limit=args.demo_limit,
        filter_key=args.filter_key,
        overwrite=bool(args.overwrite),
        store_obs=bool(args.store_obs),
        store_rgb_obs=bool(args.store_rgb_obs),
    )

    if saved_paths:
        output_parent = saved_paths[0].parent
    else:
        output_parent = _resolve_output_dir(
            dataset_path=_resolve_dataset_path(args.dataset),
            output_dir=args.output_dir,
            feat_type_token=str(args.feat_type).lower(),
        )
    print(f"Saved {len(saved_paths)} converted demo files to {output_parent}")
    if saved_paths:
        print(f"First file: {saved_paths[0].name}")
        print(f"Last file: {saved_paths[-1].name}")


if __name__ == "__main__":
    main()
