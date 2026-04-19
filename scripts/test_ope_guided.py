#!/usr/bin/env python3
"""Compute a guided checkpoint-backed SOPE OPE estimate from rollout initial states.

This script loads a trained SOPE diffuser and reward predictor, selects rollout
trajectories from a saved corpus, and estimates a guided OPE value using
checkpointed robomimic behavior and target policies.

Example commands:

1. Run with defaults, which reads from `logs/train_sope_film_0406_102649`,
   uses that run's latest SOPE checkpoints, evaluates the held-out split, and
   prints a text report:

   ```bash
   python3 scripts/test_ope_guided.py
   ```

2. Evaluate the `train-sope-feat:both_0417_163242` run with the visual
   robomimic behavior / target checkpoints:

   ```bash
   python scripts/test_ope_guided.py \
       --run-dir logs/train-sope-feat:both_0417_163242 \
       --behavior-policy-checkpoint data/policy/rmimic-lift-mh-image-v15-diffusion_260123/last.pth \
       --target-policy-checkpoint data/policy/rmimic-lift-mh-image-v15-diffusion_260123/models/model_epoch_40_image_v15_success_0.75.pth
   ```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


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


def _report_path(diffusion_checkpoint: Path) -> Path:
    diffusion_checkpoint = diffusion_checkpoint.resolve()
    return diffusion_checkpoint.parent / f"{diffusion_checkpoint.stem}_ope_guided_report.json"


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
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of selected rollout trajectories to evaluate. "
            "When omitted or set to None, the script evaluates every trajectory in the requested split."
        ),
    )
    parser.add_argument("--target-policy-checkpoint", type=Path, default=_default_target_policy_checkpoint())
    parser.add_argument("--behavior-policy-checkpoint", type=Path, default=_default_behavior_policy_checkpoint())
    parser.add_argument("--action-score-scale", type=float, default=0.2)
    parser.add_argument("--num-guidance-iters", type=int, default=2)
    parser.add_argument(
        "--action-score-postprocess",
        type=str,
        default="l2",
        choices=("none", "l2", "clamp"),
    )
    parser.add_argument("--action-neg-score-weight", type=float, default=1.0)
    parser.add_argument("--clamp-linf", type=float, default=1.0)
    parser.add_argument("--target-score-timestep", type=int, default=1)
    parser.add_argument("--behavior-score-timestep", type=int, default=1)
    parser.add_argument(
        "--use-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-neg-grad",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

    from src.eval import GuidanceConfig, evaluate_saved_rollout_ope, serialize_report

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
    report_path = _report_path(diffusion_checkpoint)
    guidance_config = GuidanceConfig(
        action_score_scale=float(args.action_score_scale),
        use_adaptive=bool(args.use_adaptive),
        use_neg_grad=bool(args.use_neg_grad),
        action_score_postprocess=args.action_score_postprocess,
        num_guidance_iters=int(args.num_guidance_iters),
        clamp_linf=float(args.clamp_linf),
        action_neg_score_weight=float(args.action_neg_score_weight),
    )

    report = evaluate_saved_rollout_ope(
        diffusion_checkpoint_path=diffusion_checkpoint,
        reward_checkpoint_path=reward_checkpoint,
        data_path=args.data.resolve() if args.data is not None else None,
        split=args.split,
        rollout_batch_size=args.rollout_batch_size,
        max_trajectories=args.max_trajectories,
        target_policy_checkpoint=args.target_policy_checkpoint,
        behavior_policy_checkpoint=args.behavior_policy_checkpoint,
        target_score_timestep=args.target_score_timestep,
        behavior_score_timestep=args.behavior_score_timestep,
        guidance_config=guidance_config,
        device=args.device,
        run_dir=run_dir,
    )
    payload = serialize_report(report, json_report=report_path)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"run_dir: {report.run_dir}")
    print(f"diffusion_checkpoint: {report.diffusion_checkpoint}")
    print(f"reward_checkpoint: {report.reward_checkpoint}")
    print(f"json_report: {report_path}")
    print(f"data: {report.data}")
    print(f"split: {report.split}")
    print("guided: True")
    print(f"target_policy_checkpoint: {report.target_policy_checkpoint}")
    print(f"behavior_policy_checkpoint: {report.behavior_policy_checkpoint}")
    print(f"rollout_horizon: {report.rollout_horizon}")
    print(f"ope_gamma: {report.ope_gamma:.6f}")
    print(f"reward_transform: {report.reward_transform}")
    print(f"autoregressive_rollout_mse: {report.autoregressive_rollout_mse:.6f}")
    print(f"return_gt_transformed: {report.return_gt_transformed:.6f}")
    print(f"return_ope_estimate: {report.return_ope_estimate:.6f}")
    print(f"guidance_config: {json.dumps(guidance_config.to_sampling_kwargs(), sort_keys=True)}")


if __name__ == "__main__":
    main()
