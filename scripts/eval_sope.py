#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval import evaluate_saved_diffusion_chunk_mse


def _default_checkpoint_path() -> Path:
    return (REPO_ROOT / "logs" / "train_sope_0326_154155" / "sope_diffuser_latest.pt").resolve()


def _default_data_path() -> Path:
    return (REPO_ROOT / "data" / "rollout" / "rmimic-lift-ph-lowdim_diffusion_260130").resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained SOPE diffusion checkpoint on one-chunk held-out RMSE.")
    parser.add_argument("--checkpoint", type=Path, default=_default_checkpoint_path())
    parser.add_argument("--data", type=Path, default=_default_data_path())
    parser.add_argument("--split", type=str, default="eval", choices=("train", "eval", "all"))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--policy-run-dir", type=Path, default=None)
    parser.add_argument("--policy-ckpt-path", type=Path, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--guidance-action-scale", type=float, default=0.2)
    parser.add_argument("--guidance-k", type=int, default=2)
    parser.add_argument("--guidance-normalize-grad", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def _report_path(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    return checkpoint_path.parent / f"{checkpoint_path.stem}_eval_report.json"


def main() -> None:
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    guidance_kw = {
        "action_scale": args.guidance_action_scale,
        "k_guide": args.guidance_k,
        "normalize_grad": args.guidance_normalize_grad,
        "use_neg_grad": False,
    }
    report = evaluate_saved_diffusion_chunk_mse(
        diffusion_checkpoint_path=args.checkpoint.resolve(),
        data_path=args.data.resolve(),
        split=args.split,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        policy_run_dir=args.policy_run_dir.resolve() if args.policy_run_dir is not None else None,
        policy_ckpt_path=args.policy_ckpt_path,
        device=args.device,
        guidance_kw=guidance_kw,
        max_chunks=args.max_chunks,
    )

    payload = _to_jsonable(asdict(report))
    report_path = _report_path(args.checkpoint)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"json_report: {report_path}")
    print(f"data: {args.data.resolve()}")
    print(f"split: {report.split} ({report.num_rollout_files} rollout files)")
    unguided_gen_raw = report.unguided["gen_unnormalized"]
    unguided_gen_norm = report.unguided["gen_normalized"]
    unguided_base_raw = report.unguided["baseline_unnormalized"]
    print(
        "unguided:"
        f" raw_transition_rmse={unguided_gen_raw.transition_rmse:.6f}"
        f" raw_state_rmse={unguided_gen_raw.state_rmse:.6f}"
        f" raw_action_rmse={unguided_gen_raw.action_rmse:.6f}"
        f" norm_transition_rmse={unguided_gen_norm.transition_rmse:.6f}"
        f" baseline_raw_transition_rmse={unguided_base_raw.transition_rmse:.6f}"
        f" chunks={unguided_gen_raw.num_samples}"
    )
    if report.guided is not None:
        guided_gen_raw = report.guided["gen_unnormalized"]
        guided_gen_norm = report.guided["gen_normalized"]
        guided_base_raw = report.guided["baseline_unnormalized"]
        print(
            "guided:"
            f" raw_transition_rmse={guided_gen_raw.transition_rmse:.6f}"
            f" raw_state_rmse={guided_gen_raw.state_rmse:.6f}"
            f" raw_action_rmse={guided_gen_raw.action_rmse:.6f}"
            f" norm_transition_rmse={guided_gen_norm.transition_rmse:.6f}"
            f" baseline_raw_transition_rmse={guided_base_raw.transition_rmse:.6f}"
            f" chunks={guided_gen_raw.num_samples}"
        )
    if report.dataset_stats is not None:
        for name, stats in report.dataset_stats.items():
            print(
                f"dataset_stats[{name}]"
                f" mean={stats['mean']:.6f}"
                f" std={stats['std']:.6f}"
                f" min={stats['min']:.6f}"
                f" max={stats['max']:.6f}"
            )


if __name__ == "__main__":
    main()
