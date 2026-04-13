"""
Collect image-latent rollout trajectories from trained diffusion policy checkpoints.

For each task and each checkpoint epoch, runs N rollouts using the trained policy,
captures image latents via PolicyFeatureHook (feat_type='visual_latent'), and saves
each trajectory as an HDF5 file for downstream SOPE diffuser training.

Usage:
    python collect_rollout_latents.py [--tasks lift can] [--epochs 50 100 200 300 400 500 600]
                                      [--n_rollouts 20] [--horizon 500] [--device cuda]
                                      [--output_dir /workspace/latent_sope/data/rollouts]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from copy import deepcopy

# Set up paths and env vars before any robomimic/mujoco imports
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", "/root/.mujoco/mujoco210")
_mj_bin = "/root/.mujoco/mujoco210/bin"
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _mj_bin not in _ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = f"{_mj_bin}:{_ld}".strip(":")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "robomimic"))

import numpy as np
from src.latent_sope.robomimic_interface.checkpoints import (
    load_checkpoint,
    build_rollout_policy_from_checkpoint,
    build_env_from_checkpoint,
)
from src.latent_sope.robomimic_interface.rollout import (
    PolicyFeatureHook,
    RolloutLatentRecorder,
    RolloutStats,
    save_rollout_latents,
    rollout,
)

CHECKPOINT_BASE = REPO_ROOT / "third_party" / "robomimic" / "diffusion_policy_trained_models"

TASK_CONFIG = {
    "lift": "lift_mh/lift_mh_diffusion",
    "can":  "can_mh/can_mh_diffusion",
}


def find_run_dir(task: str) -> Path:
    """Find the timestamped run directory (there should be exactly one)."""
    base = CHECKPOINT_BASE / TASK_CONFIG[task]
    candidates = sorted(base.iterdir())
    candidates = [c for c in candidates if c.is_dir() and (c / "models").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directory found under {base}")
    return candidates[-1]  # use the latest if multiple


def collect_for_task_epoch(
    task: str,
    epoch: int,
    n_rollouts: int,
    horizon: int,
    output_dir: Path,
    device: str,
) -> None:
    run_dir = find_run_dir(task)
    ckpt_path = run_dir / "models" / f"model_epoch_{epoch}.pth"
    if not ckpt_path.is_file():
        print(f"  [SKIP] checkpoint not found: {ckpt_path}")
        return

    out_dir = output_dir / task / f"epoch_{epoch:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check how many already done
    existing = sorted(out_dir.glob("rollout_*.h5"))
    start_i = len(existing)
    if start_i >= n_rollouts:
        print(f"  [SKIP] {task}/epoch_{epoch}: already have {start_i}/{n_rollouts} rollouts")
        return

    print(f"  Loading checkpoint: {ckpt_path.name}")
    ckpt = load_checkpoint(run_dir, Path("models") / f"model_epoch_{epoch}.pth")
    policy = build_rollout_policy_from_checkpoint(ckpt, device=device, verbose=False)
    env = build_env_from_checkpoint(ckpt, render_offscreen=False, verbose=False)

    hook = PolicyFeatureHook(policy, feat_type="visual_latent")
    recorder = RolloutLatentRecorder(feature_hook=hook, store_obs=False)

    successes = 0
    for i in range(start_i, n_rollouts):
        out_path = out_dir / f"rollout_{i:04d}.h5"
        recorder._obs.clear()
        recorder._next_obs.clear()
        recorder._actions.clear()
        recorder._rewards.clear()
        recorder._dones.clear()
        recorder._infos.clear()
        recorder._z.clear()

        stats: RolloutStats = rollout(
            policy=policy,
            env=env,
            horizon=horizon,
            recorder=recorder,
        )

        traj = recorder.finalize(stats)
        save_rollout_latents(out_path, traj)
        successes += int(traj.success)
        print(
            f"    rollout {i+1}/{n_rollouts} → {traj.horizon} steps, "
            f"success={traj.success}, latents={traj.latents.shape}, "
            f"saved → {out_path.name}"
        )

    hook.close()
    print(f"  Done {task}/epoch_{epoch}: {successes}/{n_rollouts} successes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["lift", "can"])
    parser.add_argument("--epochs", nargs="+", type=int,
                        default=[50, 100, 200, 300, 400, 500, 600])
    parser.add_argument("--n_rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", type=Path,
                        default=REPO_ROOT / "data" / "rollouts")
    args = parser.parse_args()

    print(f"Collecting rollouts: tasks={args.tasks}, epochs={args.epochs}")
    print(f"  n_rollouts={args.n_rollouts}, horizon={args.horizon}, device={args.device}")
    print(f"  output_dir={args.output_dir}")

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        for epoch in args.epochs:
            print(f"\n--- {task} / epoch {epoch} ---")
            collect_for_task_epoch(
                task=task,
                epoch=epoch,
                n_rollouts=args.n_rollouts,
                horizon=args.horizon,
                output_dir=args.output_dir,
                device=args.device,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
