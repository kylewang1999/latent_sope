#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _default_run_dir() -> Path:
    return (REPO_ROOT / "logs" / "train_sope_film_0406_102649").resolve()


def _default_behavior_policy_checkpoint() -> Path:
    return (
        REPO_ROOT
        / "data"
        / "policy"
        / "rmimic-lift-ph-lowdim_diffusion_260130"
        / "last.pth"
    ).resolve()


def _default_target_policy_checkpoint() -> Path:
    return (
        REPO_ROOT
        / "data"
        / "policy"
        / "rmimic-lift-ph-lowdim_diffusion_260130"
        / "models"
        / "model_epoch_50_low_dim_v15_success_0.92.pth"
    ).resolve()


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def _report_path(diffusion_checkpoint: Path) -> Path:
    diffusion_checkpoint = diffusion_checkpoint.resolve()
    return diffusion_checkpoint.parent / f"{diffusion_checkpoint.stem}_ope_guided_report.json"


def _resolve_data_path(
    *,
    data_arg: Path | None,
    diffusion_payload: dict[str, Any],
) -> Path:
    if data_arg is not None:
        return data_arg.resolve()

    training_payload = diffusion_payload.get("training_config") or {}
    data_entries = training_payload.get("data") or []
    if not data_entries:
        raise ValueError(
            "Could not infer rollout data path from diffusion checkpoint training_config['data']; "
            "pass --data explicitly."
        )
    return Path(data_entries[0]).resolve()


def _select_rollout_paths(
    *,
    data_path: Path,
    split: str,
    diffusion_payload: dict[str, Any],
) -> tuple[list[Path], int, float]:
    from src.train import _split_rollout_paths

    training_payload = diffusion_payload.get("training_config") or {}
    split_seed = int(training_payload.get("seed", 0))
    train_fraction = float(training_payload.get("train_fraction", 0.8))
    train_paths, eval_paths = _split_rollout_paths(
        [data_path],
        seed=split_seed,
        train_fraction=train_fraction,
    )

    if split == "train":
        selected_paths = train_paths
    elif split == "eval":
        if not eval_paths:
            raise ValueError(
                "Requested eval split, but the rollout corpus does not produce a held-out split."
            )
        selected_paths = eval_paths
    elif split == "all":
        selected_paths = train_paths + eval_paths
    else:
        raise ValueError(f"Unknown split={split!r}.")

    return selected_paths, split_seed, train_fraction


def _build_initial_states(
    rollout_paths: list[Path],
    *,
    dataset_config_payload: dict[str, Any],
) -> np.ndarray:
    from src.robomimic_interface.dataset import RolloutChunkDataset, RolloutChunkDatasetConfig
    from src.robomimic_interface.rollout import load_rollout_latents

    cfg_dataset = RolloutChunkDatasetConfig(**dataset_config_payload)
    initial_states: list[np.ndarray] = []
    for rollout_path in rollout_paths:
        traj = load_rollout_latents(rollout_path)
        dataset = RolloutChunkDataset(
            traj=traj,
            config=cfg_dataset,
            encoder=None,
            obs_keys=None,
            encoder_device="cpu",
            demo_id=rollout_path.stem,
        )
        if dataset.latents.shape[0] <= 0:
            raise ValueError(f"Rollout {rollout_path} has no latent states.")
        initial_states.append(np.asarray(dataset.latents[0], dtype=np.float32))

    if not initial_states:
        raise ValueError("No rollout initial states were collected for OPE evaluation.")
    return np.stack(initial_states, axis=0).astype(np.float32)


def _load_true_rollout_targets(
    rollout_paths: list[Path],
    *,
    dataset_config_payload: dict[str, Any],
    ope_gamma: float,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    from src.robomimic_interface.dataset import (
        RolloutChunkDataset,
        RolloutChunkDatasetConfig,
        resolve_reward_transform,
    )
    from src.robomimic_interface.rollout import load_rollout_latents

    cfg_dataset = RolloutChunkDatasetConfig(**dataset_config_payload)
    reward_transform = resolve_reward_transform(cfg_dataset.reward_transform)

    true_states: list[np.ndarray] = []
    true_actions: list[np.ndarray] = []
    true_rewards: list[np.ndarray] = []
    effective_horizon: int | None = None

    for rollout_path in rollout_paths:
        traj = load_rollout_latents(rollout_path)
        dataset = RolloutChunkDataset(
            traj=traj,
            config=cfg_dataset,
            encoder=None,
            obs_keys=None,
            encoder_device="cpu",
            demo_id=rollout_path.stem,
        )
        states = np.asarray(dataset.latents, dtype=np.float32)
        actions = np.asarray(dataset.actions, dtype=np.float32)
        rewards_raw = np.asarray(dataset.rewards, dtype=np.float32)
        rewards = np.asarray(reward_transform(states, rewards_raw), dtype=np.float32)

        rollout_horizon = min(
            int(max_length),
            int(states.shape[0]),
            int(actions.shape[0]),
            int(rewards.shape[0]),
        )
        if rollout_horizon <= 0:
            raise ValueError(f"Rollout {rollout_path} has no valid steps for OPE evaluation.")

        if effective_horizon is None:
            effective_horizon = rollout_horizon
        else:
            effective_horizon = min(effective_horizon, rollout_horizon)

        true_states.append(states[:rollout_horizon])
        true_actions.append(actions[:rollout_horizon])
        true_rewards.append(rewards[:rollout_horizon])

    if effective_horizon is None:
        raise ValueError("No true rollout targets were collected for OPE evaluation.")

    aligned_states = np.stack([states[:effective_horizon] for states in true_states], axis=0).astype(
        np.float32
    )
    aligned_actions = np.stack(
        [actions[:effective_horizon] for actions in true_actions],
        axis=0,
    ).astype(np.float32)
    discounts = np.power(np.float32(ope_gamma), np.arange(effective_horizon, dtype=np.float32))
    aligned_returns = np.asarray(
        [float(np.sum(rewards[:effective_horizon] * discounts)) for rewards in true_rewards],
        dtype=np.float32,
    )
    return (aligned_states, aligned_actions, aligned_returns, int(effective_horizon))


def _resolve_policy_checkpoint(
    policy_checkpoint: Path,
) -> tuple[Path, Path, Path]:
    resolved_checkpoint = policy_checkpoint.expanduser().resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {resolved_checkpoint}")

    for run_dir in resolved_checkpoint.parents:
        if (run_dir / "config.json").is_file():
            return resolved_checkpoint, run_dir, resolved_checkpoint.relative_to(run_dir)

    raise ValueError(
        "Could not locate a robomimic run directory for checkpoint "
        f"{resolved_checkpoint}. Expected an ancestor containing config.json."
    )


def _load_diffusion_policy(
    *,
    policy_checkpoint: Path,
    device: str | None,
    score_timestep: int,
):
    from src.robomimic_interface.checkpoints import build_algo_from_checkpoint, load_checkpoint
    from src.robomimic_interface.policy import DiffusionPolicy, DiffusionPolicyScoreConfig
    from src.utils import resolve_device

    resolved_checkpoint, run_dir, relative_checkpoint = _resolve_policy_checkpoint(policy_checkpoint)
    checkpoint = load_checkpoint(run_dir, ckpt_path=relative_checkpoint)
    resolved_device = device or resolve_device(prefer_cuda=True)
    policy = DiffusionPolicy(
        policy=build_algo_from_checkpoint(checkpoint, device=resolved_device),
        obs_normalization_stats=checkpoint.ckpt_dict.get("obs_normalization_stats"),
        action_normalization_stats=checkpoint.ckpt_dict.get("action_normalization_stats"),
        config=DiffusionPolicyScoreConfig(score_timestep=score_timestep),
    )
    return policy, resolved_checkpoint


def _validate_guidance_policy_horizon(
    *,
    diffuser: Any,
    policy: Any,
    role: str,
) -> None:
    assert int(policy.observation_horizon) == int(diffuser.cfg.frame_stack), (
        "SOPE frame_stack must match the robomimic observation_horizon when "
        f"using {role} guidance: got frame_stack={diffuser.cfg.frame_stack}, "
        f"{role}.observation_horizon={policy.observation_horizon}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a guided checkpoint-backed SOPE OPE estimate from held-out rollout initial states."
    )
    parser.add_argument("--run-dir", type=Path, default=_default_run_dir())
    parser.add_argument("--diffusion-checkpoint", type=Path, default=None)
    parser.add_argument("--reward-checkpoint", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--split", type=str, default="eval", choices=("train", "eval", "all"))
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--target-policy-checkpoint",
        type=Path,
        default=_default_target_policy_checkpoint(),
    )
    parser.add_argument("--behavior-policy-checkpoint",
        type=Path,
        default=_default_behavior_policy_checkpoint(),
    )
    parser.add_argument("--action-score-scale", type=float, default=0.2)
    parser.add_argument("--num-guidance-iters", type=int, default=2)
    parser.add_argument("--action-score-postprocess",
        type=str,
        default="l2",
        choices=("none", "l2", "clamp"),
    )
    parser.add_argument("--action-neg-score-weight", type=float, default=1.0)
    parser.add_argument("--clamp-linf", type=float, default=1.0)
    parser.add_argument("--target-score-timestep", type=int, default=1)
    parser.add_argument("--behavior-score-timestep", type=int, default=1)
    parser.add_argument("--use-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use-neg-grad",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from src.utils import resolve_device

    resolved_device = args.device or resolve_device(prefer_cuda=True)
    if args.rollout_batch_size <= 0:
        raise ValueError(
            f"--rollout-batch-size must be positive, got {args.rollout_batch_size}."
        )
    if args.max_trajectories is not None and args.max_trajectories <= 0:
        raise ValueError(
            f"--max-trajectories must be positive when provided, got {args.max_trajectories}."
        )
    if args.action_score_scale <= 0:
        raise ValueError(f"--action-score-scale must be positive, got {args.action_score_scale}.")
    if args.num_guidance_iters <= 0:
        raise ValueError(
            f"--num-guidance-iters must be positive, got {args.num_guidance_iters}."
        )
    if args.action_neg_score_weight < 0:
        raise ValueError(
            "--action-neg-score-weight must be non-negative, got "
            f"{args.action_neg_score_weight}."
        )
    if args.clamp_linf < 0:
        raise ValueError(f"--clamp-linf must be non-negative, got {args.clamp_linf}.")

    run_dir = args.run_dir.resolve()
    diffusion_checkpoint = (
        args.diffusion_checkpoint.resolve()
        if args.diffusion_checkpoint is not None
        else (run_dir / "sope_diffuser_latest.pt").resolve()
    )
    reward_checkpoint = (
        args.reward_checkpoint.resolve()
        if args.reward_checkpoint is not None
        else (run_dir / "sope_reward_predictor_latest.pt").resolve()
    )
    target_policy_checkpoint = args.target_policy_checkpoint.expanduser().resolve()
    behavior_policy_checkpoint = args.behavior_policy_checkpoint.expanduser().resolve()

    quiet_stdout = io.StringIO()
    with redirect_stdout(quiet_stdout):
        from src.eval import load_diffusion_checkpoint, load_reward_checkpoint

        target_policy, target_policy_checkpoint = _load_diffusion_policy(
            policy_checkpoint=target_policy_checkpoint,
            device=resolved_device,
            score_timestep=args.target_score_timestep,
        )
        behavior_policy = None
        if args.use_neg_grad:
            behavior_policy, behavior_policy_checkpoint = _load_diffusion_policy(
                policy_checkpoint=behavior_policy_checkpoint,
                device=resolved_device,
                score_timestep=args.behavior_score_timestep,
            )

        diffuser, diffusion_payload = load_diffusion_checkpoint(
            diffusion_checkpoint,
            device=resolved_device,
            policy=target_policy,
            behavior_policy=behavior_policy,
        )
        reward_predictor, _ = load_reward_checkpoint(
            reward_checkpoint,
            device=resolved_device,
        )

    _validate_guidance_policy_horizon(
        diffuser=diffuser,
        policy=target_policy,
        role="target policy",
    )
    if behavior_policy is not None:
        _validate_guidance_policy_horizon(
            diffuser=diffuser,
            policy=behavior_policy,
            role="behavior policy",
        )

    data_path = _resolve_data_path(
        data_arg=args.data,
        diffusion_payload=diffusion_payload,
    )

    with redirect_stdout(quiet_stdout):
        selected_paths, split_seed, train_fraction = _select_rollout_paths(
            data_path=data_path,
            split=args.split,
            diffusion_payload=diffusion_payload,
        )
    if args.max_trajectories is not None:
        selected_paths = selected_paths[: args.max_trajectories]
    if not selected_paths:
        raise ValueError("No rollout files selected for OPE evaluation.")

    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)

    guidance_config = {
        "action_score_scale": float(args.action_score_scale),
        "use_adaptive": bool(args.use_adaptive),
        "use_neg_grad": bool(args.use_neg_grad),
        "action_score_postprocess": args.action_score_postprocess,
        "num_guidance_iters": int(args.num_guidance_iters),
        "clamp_linf": float(args.clamp_linf),
        "action_neg_score_weight": float(args.action_neg_score_weight),
    }

    rollout_horizon = int(diffuser.cfg.trajectory_horizon)
    with redirect_stdout(quiet_stdout):
        initial_states = _build_initial_states(
            selected_paths,
            dataset_config_payload=diffusion_payload["dataset_config"],
        )
        true_states, true_actions, true_returns, effective_horizon = _load_true_rollout_targets(
            selected_paths,
            dataset_config_payload=diffusion_payload["dataset_config"],
            ope_gamma=float(diffuser.cfg.ope_gamma),
            max_length=rollout_horizon,
        )

    total_weighted_estimate = 0.0
    total_count = 0
    total_squared_error = 0.0
    total_transition_elements = 0
    batch_starts = range(0, len(initial_states), args.rollout_batch_size)
    batch_iterator = tqdm(batch_starts, desc="Guided OPE", unit="batch")
    for start in batch_iterator:
        end = min(start + args.rollout_batch_size, len(initial_states))
        batch_np = initial_states[start:end]
        batch_initial_states = torch.from_numpy(batch_np).to(
            diffuser.device,
            dtype=torch.float32,
        )
        with redirect_stdout(quiet_stdout):
            batch_states, batch_actions = diffuser.generate_full_trajectory(
                batch_initial_states,
                max_length=effective_horizon,
                guided=True,
                **guidance_config,
            )
        batch_size = end - start
        batch_true_states = true_states[start:end]
        batch_true_actions = true_actions[start:end]
        batch_pred_transition = np.concatenate([batch_states, batch_actions], axis=-1)
        batch_true_transition = np.concatenate([batch_true_states, batch_true_actions], axis=-1)
        total_squared_error += float(
            np.square(batch_pred_transition - batch_true_transition).sum(dtype=np.float64)
        )
        total_transition_elements += int(batch_pred_transition.size)
        with redirect_stdout(quiet_stdout):
            batch_reward_preds = reward_predictor.predict(batch_states, batch_actions)
        batch_reward_preds_t = (
            batch_reward_preds
            if torch.is_tensor(batch_reward_preds)
            else torch.as_tensor(batch_reward_preds, dtype=torch.float32)
        ).to(dtype=torch.float32)
        if batch_reward_preds_t.ndim == 3 and batch_reward_preds_t.shape[-1] == 1:
            batch_reward_preds_t = batch_reward_preds_t.squeeze(-1)
        discounts_t = torch.pow(
            torch.full(
                (effective_horizon,),
                float(diffuser.cfg.ope_gamma),
                dtype=batch_reward_preds_t.dtype,
                device=batch_reward_preds_t.device,
            ),
            torch.arange(effective_horizon, device=batch_reward_preds_t.device),
        )
        batch_returns = torch.sum(batch_reward_preds_t * discounts_t.unsqueeze(0), dim=1)
        total_weighted_estimate += float(batch_returns.sum().item())
        total_count += batch_size
        batch_iterator.set_postfix(
            ope_return=f"{(total_weighted_estimate / max(total_count, 1)):.4f}",
            rollout_mse=f"{(total_squared_error / max(total_transition_elements, 1)):.4e}",
        )

    mean_ope_return_estimate = total_weighted_estimate / max(total_count, 1)
    mean_true_return = float(np.mean(true_returns))
    autoregressive_rollout_mse = total_squared_error / max(total_transition_elements, 1)

    report = {
        "run_dir": run_dir,
        "diffusion_checkpoint": diffusion_checkpoint,
        "reward_checkpoint": reward_checkpoint,
        "json_report": _report_path(diffusion_checkpoint),
        "data": data_path,
        "split": args.split,
        "training_seed": split_seed,
        "train_fraction": train_fraction,
        "num_rollout_files": len(selected_paths),
        "rollout_batch_size": args.rollout_batch_size,
        "max_trajectories": args.max_trajectories,
        "rollout_horizon": effective_horizon,
        "ope_gamma": float(diffuser.cfg.ope_gamma),
        "autoregressive_rollout_mse": float(autoregressive_rollout_mse),
        "return_gt_transformed": float(mean_true_return),
        "return_ope_estimate": float(mean_ope_return_estimate),
        "device": str(diffuser.device),
        "guided": True,
        "target_policy_checkpoint": target_policy_checkpoint,
        "behavior_policy_checkpoint": behavior_policy_checkpoint,
        "target_score_timestep": int(args.target_score_timestep),
        "behavior_score_timestep": int(args.behavior_score_timestep),
        "guidance_config": guidance_config,
    }

    report_path = Path(report["json_report"])
    report_path.write_text(
        json.dumps(_to_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(_to_jsonable(report), indent=2, sort_keys=True))
        return

    print(f"run_dir: {run_dir}")
    print(f"diffusion_checkpoint: {diffusion_checkpoint}")
    print(f"reward_checkpoint: {reward_checkpoint}")
    print(f"json_report: {report_path}")
    print(f"data: {data_path}")
    print(f"split: {args.split}")
    print("guided: True")
    print(f"target_policy_checkpoint: {target_policy_checkpoint}")
    print(f"behavior_policy_checkpoint: {behavior_policy_checkpoint}")
    print(f"rollout_horizon: {effective_horizon}")
    print(f"ope_gamma: {float(diffuser.cfg.ope_gamma):.6f}")
    print(f"autoregressive_rollout_mse: {float(autoregressive_rollout_mse):.6f}")
    print(f"return_gt_transformed: {float(mean_true_return):.6f}")
    print(f"return_ope_estimate: {float(mean_ope_return_estimate):.6f}")
    print(f"guidance_config: {json.dumps(guidance_config, sort_keys=True)}")


if __name__ == "__main__":
    main()
