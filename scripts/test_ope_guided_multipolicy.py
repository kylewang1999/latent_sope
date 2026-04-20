#!/usr/bin/env python3
"""Compute guided SOPE OPE metrics across a directory of target policies.

This script evaluates a trained SOPE diffuser and reward predictor against
multiple robomimic target-policy checkpoints. It generates guided SOPE
trajectories from held-out initial states, rolls each target policy out from the
corresponding source-demo initial conditions, and compares ranking / value
metrics across target policies.

Example commands:

1. Run with defaults, which evaluates all `model_epoch*.pth` checkpoints under
   `data/policy/rmimic-lift-mh-image-v15-diffusion_260123/models` against the
   latest SOPE checkpoints in `logs/train-sope-feat:both_0417_163242` and
   writes the timestamped JSON report under
   `logs/train-sope-feat:both_0417_163242/eval_ope/`:

   ```bash
   python scripts/test_ope_guided_multipolicy.py \
        --max-trajectories 10
   ```

2. Smoke-test on a small slice and emit JSON:

   ```bash
   python scripts/test_ope_guided_multipolicy.py \
       --max-trajectories 2 \
       --max-target-policies 2 \
       --json
   ```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
if os.environ.get("DISPLAY", "") == "" and os.environ.get("MUJOCO_GL") is None:
    os.environ["MUJOCO_GL"] = "osmesa"
if os.environ.get("MUJOCO_GL", "").lower() == "osmesa":
    os.environ.setdefault("MUJOCO_PY_FORCE_CPU", "1")


DEFAULT_ROLLOUT_HORIZON = 200


def _default_run_dir() -> Path:
    return (REPO_ROOT / "logs" / "train-sope-feat:both_0417_163242").resolve()


def _default_report_dir(run_dir: Path) -> Path:
    return run_dir.resolve() / "eval_ope"


def _default_behavior_policy_checkpoint() -> Path:
    return (
        REPO_ROOT
        / "data"
        / "policy"
        / "rmimic-lift-mh-image-v15-diffusion_260123"
        / "last.pth"
    ).resolve()


def _default_target_policy_dir() -> Path:
    return (
        REPO_ROOT
        / "data"
        / "policy"
        / "rmimic-lift-mh-image-v15-diffusion_260123"
        / "models"
    ).resolve()


def _report_path(*, run_dir: Path, diffusion_checkpoint: Path) -> Path:
    diffusion_checkpoint = diffusion_checkpoint.resolve()
    timestamp = datetime.now().strftime("%m%d_%M%S")
    return (
        _default_report_dir(run_dir)
        / f"{diffusion_checkpoint.stem}_ope_guided_multipolicy_report_{timestamp}.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute guided SOPE OPE metrics across a directory of robomimic target-policy checkpoints."
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
        default=10,
        help=(
            "Optional cap on the number of selected rollout trajectories to evaluate. "
            "When omitted or set to None, the script evaluates every trajectory in the requested split."
        ),
    )
    parser.add_argument(
        "--target-policy-dir",
        type=Path,
        default=_default_target_policy_dir(),
        help="Directory containing target-policy checkpoints matching model_epoch*.pth.",
    )
    parser.add_argument(
        "--behavior-policy-checkpoint",
        type=Path,
        default=_default_behavior_policy_checkpoint(),
        help="Behavior-policy checkpoint used for negative guidance and target-rollout re-encoding.",
    )
    parser.add_argument(
        "--max-target-policies",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of target-policy checkpoints discovered under --target-policy-dir. "
            "When omitted or set to None, every discovered checkpoint is evaluated."
        ),
    )
    parser.add_argument(
        "--rollout-horizon",
        "--true-rollout-horizon",
        dest="rollout_horizon",
        type=int,
        default=DEFAULT_ROLLOUT_HORIZON,
        help=(
            "Shared horizon for both guided SOPE rollouts and true robomimic target-policy "
            "rollouts. Defaults to 80 steps."
        ),
    )
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
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print the full report as JSON to stdout instead of the compact text summary. "
            "The JSON report file is still written to disk regardless of this flag."
        ),
    )
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
    if args.max_target_policies is not None and args.max_target_policies <= 0:
        raise ValueError(
            f"--max-target-policies must be positive when provided, got {args.max_target_policies}."
        )
    if args.rollout_horizon <= 0:
        raise ValueError(f"--rollout-horizon must be positive, got {args.rollout_horizon}.")
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

    from src.eval import GuidanceConfig, evaluate_guided_multipolicy_ope, serialize_report

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
    report_path = _report_path(run_dir=run_dir, diffusion_checkpoint=diffusion_checkpoint)
    guidance_config = GuidanceConfig(
        action_score_scale=float(args.action_score_scale),
        use_adaptive=bool(args.use_adaptive),
        use_neg_grad=bool(args.use_neg_grad),
        action_score_postprocess=args.action_score_postprocess,
        num_guidance_iters=int(args.num_guidance_iters),
        clamp_linf=float(args.clamp_linf),
        action_neg_score_weight=float(args.action_neg_score_weight),
    )

    report = evaluate_guided_multipolicy_ope(
        diffusion_checkpoint_path=diffusion_checkpoint,
        reward_checkpoint_path=reward_checkpoint,
        target_policy_dir=args.target_policy_dir,
        behavior_policy_checkpoint=args.behavior_policy_checkpoint,
        data_path=args.data.resolve() if args.data is not None else None,
        split=args.split,
        rollout_batch_size=args.rollout_batch_size,
        max_trajectories=args.max_trajectories,
        max_target_policies=args.max_target_policies,
        rollout_horizon=args.rollout_horizon,
        target_score_timestep=args.target_score_timestep,
        behavior_score_timestep=args.behavior_score_timestep,
        guidance_config=guidance_config,
        device=args.device,
        run_dir=run_dir,
        json_report=report_path,
    )
    payload = serialize_report(report, json_report=report_path)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"run_dir: {report.run_dir}")
    print(f"diffusion_checkpoint: {report.diffusion_checkpoint}")
    print(f"reward_checkpoint: {report.reward_checkpoint}")
    print(f"json_report: {report_path}")
    print(f"data: {report.data}")
    print(f"source_dataset_path: {report.source_dataset_path}")
    print(f"split: {report.split}")
    print(f"num_rollout_files: {report.num_rollout_files}")
    print(f"num_target_policies: {report.num_target_policies}")
    print(f"rollout_horizon: {report.rollout_horizon}")
    print(f"ope_gamma: {report.ope_gamma:.6f}")
    print(f"reward_transform: {report.reward_transform}")
    print(f"spearman_correlation_transformed: {report.spearman_correlation_transformed}")
    print(f"policy_value_rmse_transformed: {report.policy_value_rmse_transformed:.6f}")
    print(f"guidance_config: {json.dumps(guidance_config.to_sampling_kwargs(), sort_keys=True)}")
    for policy_report in report.policies:
        print(
            "policy_epoch={epoch} pred={pred:.6f} true_transformed={true_transformed:.6f} "
            "true_raw={true_raw:.6f} success_true={success_true:.3f} success_guided={success_guided:.3f} "
            "rollout_rmse={rmse:.6f}".format(
                epoch=policy_report.target_policy_epoch,
                pred=policy_report.mean_predicted_transformed_return,
                true_transformed=policy_report.mean_true_transformed_return,
                true_raw=policy_report.mean_true_raw_return,
                success_true=policy_report.true_rollout_success_rate,
                success_guided=policy_report.guided_rollout_success_rate,
                rmse=policy_report.guided_target_rollout_rmse,
            )
        )


if __name__ == "__main__":
    main()
