#!/usr/bin/env python
"""Full OPE pipeline at 200-rollout scale.

Usage:
    python scripts/run_200.py
    python scripts/run_200.py --force          # recompute everything
    python scripts/run_200.py --epochs 200     # more training epochs
    python scripts/run_200.py --rollouts 300   # even larger run

Estimated runtime on 2x P100 (first run): ~2 hours
  Step 0 (oracle):     ~50 min  (200 rollouts × ~15s each)
  Step 1 (collection): ~50 min  (200 rollouts × ~15s each)
  Step 3 (training):   ~15 min  (100 epochs × ~1250 chunks)
  Steps 5-7:           ~5 min

Cached runs skip Steps 0-3 and only run Steps 5-7 (~5 min).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.latent_sope.robomimic_interface.checkpoints import (
    load_checkpoint,
    build_rollout_policy_from_checkpoint,
    build_env_from_checkpoint,
)
from src.latent_sope.robomimic_interface.rollout import (
    rollout,
    save_rollout_latents,
    load_rollout_latents,
    get_policy_frame_stack,
    PolicyFeatureHook,
    RolloutLatentRecorder,
)
from src.latent_sope.robomimic_interface.dataset import (
    RolloutChunkDatasetConfig,
    make_rollout_chunk_dataloader,
)
from src.latent_sope.diffusion.sope_diffuser import (
    SopeDiffusionConfig,
    SopeDiffuser,
    NormalizationStats as DiffusionNormStats,
    cross_validate_configs,
)
from src.latent_sope.eval.reward_model import (
    LiftRewardFn,
    score_trajectories_gt,
    make_lift_encoder,
)
from src.latent_sope.eval.metrics import ope_eval


# ─── Defaults ────────────────────────────────────────────────────────────────

POLICY_DIR = REPO_ROOT / "third_party/robomimic/diffusion_policy_trained_models/test"
OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
HORIZON = 60
GAMMA = 1.0


def parse_args():
    p = argparse.ArgumentParser(description="Full OPE pipeline at scale")
    p.add_argument("--rollouts", type=int, default=200, help="Number of offline rollouts (Step 1)")
    p.add_argument("--oracle-rollouts", type=int, default=None,
                    help="Number of oracle rollouts (Step 0). Defaults to --rollouts.")
    p.add_argument("--epochs", type=int, default=100, help="Diffusion training epochs")
    p.add_argument("--num-trajs", type=int, default=100, help="Synthetic trajectories to generate")
    p.add_argument("--horizon", type=int, default=HORIZON, help="Max steps per episode")
    p.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor")
    p.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    p.add_argument("--diffusion-steps", type=int, default=256, help="Diffusion denoising steps")
    p.add_argument("--device", type=str, default="cuda", help="Torch device")
    p.add_argument("--force", action="store_true", help="Recompute all steps from scratch")
    p.add_argument("--policy-dir", type=str, default=None,
                    help="Override policy directory (defaults to latest in test/)")
    p.add_argument("--reuse-dir", type=str, default=None,
                    help="Reuse rollouts from this directory (e.g. rollout_latents_50/)")
    return p.parse_args()


def find_policy_dir(override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    dirs = sorted([d for d in POLICY_DIR.glob("*") if d.is_dir()])
    assert dirs, f"No trained policies found in {POLICY_DIR}"
    return dirs[-1]


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ─── Step 0: Oracle ─────────────────────────────────────────────────────────

def step0_oracle(policy_dir: Path, args) -> float:
    K = args.oracle_rollouts or args.rollouts
    oracle_path = policy_dir / f"oracle_{K}.json"

    if oracle_path.exists() and not args.force:
        with open(oracle_path) as f:
            data = json.load(f)
        print(f"  Loaded cached oracle from {oracle_path}")
        print(f"  V^pi = {data['oracle_value']:.3f} (std={np.std(data['returns']):.3f}, K={len(data['returns'])})")
        return data["oracle_value"]

    print(f"  Running {K} oracle rollouts (horizon={args.horizon})...")
    t0 = time.time()

    ckpt = load_checkpoint(policy_dir.resolve(), ckpt_path="last.pth")
    policy = build_rollout_policy_from_checkpoint(ckpt, device=torch.device(args.device), verbose=False)
    env = build_env_from_checkpoint(ckpt, render=False, render_offscreen=False, verbose=False)

    returns = []
    for i in range(K):
        stats = rollout(policy=policy, env=env, horizon=args.horizon, render=False)
        returns.append(stats.total_reward)
        if (i + 1) % max(1, K // 10) == 0:
            print(f"    [{i+1}/{K}] running mean={np.mean(returns):.3f}")

    oracle_value = float(np.mean(returns))

    with open(oracle_path, "w") as f:
        json.dump({"oracle_value": oracle_value, "returns": returns,
                    "K": K, "horizon": args.horizon, "gamma": args.gamma}, f, indent=2)

    elapsed = time.time() - t0
    print(f"  Oracle V^pi = {oracle_value:.3f} (std={np.std(returns):.3f})")
    print(f"  Saved to {oracle_path} ({fmt_time(elapsed)})")
    return oracle_value


# ─── Step 1: Collect Offline Data ────────────────────────────────────────────

def step1_collect(policy_dir: Path, args) -> list[Path]:
    N = args.rollouts
    output_dir = policy_dir / f"rollout_latents_{N}"
    output_dir.mkdir(exist_ok=True)

    # Check for reusable rollouts from a smaller run
    reuse_paths = []
    if args.reuse_dir:
        reuse_dir = policy_dir / args.reuse_dir
        if reuse_dir.exists():
            reuse_paths = sorted(reuse_dir.glob("*.h5"))
            print(f"  Found {len(reuse_paths)} reusable rollouts in {reuse_dir}")

    existing = sorted(output_dir.glob("*.h5"))
    if len(existing) >= N and not args.force:
        print(f"  Found {len(existing)} existing rollouts in {output_dir} — skipping collection.")
        return sorted(existing)[:N]

    # Copy reusable rollouts
    n_copied = 0
    if reuse_paths:
        import shutil
        for src_path in reuse_paths:
            dst_path = output_dir / f"rollout_{n_copied:04d}.h5"
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
            n_copied += 1
            if n_copied >= N:
                break
        print(f"  Copied {n_copied} rollouts from {args.reuse_dir}")

    remaining = N - n_copied
    if remaining <= 0:
        return sorted(output_dir.glob("*.h5"))[:N]

    print(f"  Collecting {remaining} new rollouts (horizon={args.horizon})...")
    t0 = time.time()

    ckpt = load_checkpoint(policy_dir.resolve(), ckpt_path="last.pth")
    policy = build_rollout_policy_from_checkpoint(ckpt, device=torch.device(args.device), verbose=False)
    env = build_env_from_checkpoint(ckpt, render=False, render_offscreen=False, verbose=False)

    for i in range(remaining):
        idx = n_copied + i
        save_path = output_dir / f"rollout_{idx:04d}.h5"
        if save_path.exists() and not args.force:
            continue

        feature_hook = PolicyFeatureHook(policy, feat_type="low_dim_concat")
        recorder = RolloutLatentRecorder(
            feature_hook, obs_keys=OBS_KEYS, store_obs=True, store_next_obs=False,
        )
        stats = rollout(policy=policy, env=env, horizon=args.horizon, render=False, recorder=recorder)
        traj = recorder.finalize(stats)
        save_rollout_latents(save_path, traj)
        feature_hook.close()

        if (i + 1) % max(1, remaining // 10) == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (remaining - i - 1)
            print(f"    [{i+1}/{remaining}] reward={stats.total_reward:.1f}, "
                  f"latents={traj.latents.shape} (ETA: {fmt_time(eta)})")

    paths = sorted(output_dir.glob("*.h5"))[:N]
    print(f"  Collection complete: {len(paths)} rollouts ({fmt_time(time.time() - t0)})")
    return paths


# ─── Steps 2+3: Chunk & Train Diffusion ─────────────────────────────────────

def step2_3_train(policy_dir: Path, rollout_paths: list[Path], args):
    N = args.rollouts
    ckpt_dir = policy_dir / f"diffusion_ckpts_{N}"
    ckpt_path = ckpt_dir / "sope_diffuser_latest.pt"

    if ckpt_path.exists() and not args.force:
        print(f"  Loading diffusion checkpoint from {ckpt_path}")
        payload = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)

        diffusion_config = SopeDiffusionConfig(**payload["diffusion_config"])
        diff_norm_stats = None
        if payload["normalization_stats"] is not None:
            ns = payload["normalization_stats"]
            diff_norm_stats = DiffusionNormStats(mean=ns["mean"], std=ns["std"])

        diffuser = SopeDiffuser(cfg=diffusion_config, normalization_stats=diff_norm_stats, device=args.device)
        diffuser.diffusion.load_state_dict(payload["diffusion_state_dict"])
        diffuser.diffusion.eval()

        print(f"  Loaded: epoch={payload['epoch']}, step={payload['step']}, "
              f"state_dim={diffusion_config.state_dim}, action_dim={diffusion_config.action_dim}")
        return diffuser

    # Step 2: Chunk
    sample_traj = load_rollout_latents(rollout_paths[0])
    latents_dim = sample_traj.latents.shape[-1]
    action_dim = sample_traj.actions.shape[-1]

    dataset_config = RolloutChunkDatasetConfig(
        chunk_size=8, stride=2, frame_stack=2, source="latents",
        latents_dim=latents_dim, action_dim=action_dim,
        normalize=True, return_metadata=True,
    )

    dataloader, norm_stats = make_rollout_chunk_dataloader(
        paths=rollout_paths, config=dataset_config,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    print(f"  DataLoader: {len(dataloader)} batches of {args.batch_size}")

    # Step 3: Train
    diffusion_config = SopeDiffusionConfig(
        chunk_horizon=dataset_config.chunk_size,
        frame_stack=dataset_config.frame_stack,
        state_dim=latents_dim, action_dim=action_dim,
        diffusion_steps=args.diffusion_steps,
        dim_mults=(1, 2), attention=False,
        loss_type="l2", action_weight=5.0, predict_epsilon=True,
        lr=3e-4, guided=False,
    )
    cross_validate_configs(dataset_config, diffusion_config)

    diff_norm_stats = None
    if norm_stats is not None:
        diff_norm_stats = DiffusionNormStats(mean=norm_stats.mean, std=norm_stats.std)

    diffuser = SopeDiffuser(cfg=diffusion_config, normalization_stats=diff_norm_stats, device=args.device)
    optimizer = diffuser.make_optimizer()

    n_params = sum(p.numel() for p in diffuser.diffusion.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Training for {args.epochs} epochs...")

    t0 = time.time()
    diffuser.diffusion.train()
    step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for batch in dataloader:
            step += 1
            batch_dev = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            loss, info = diffuser.loss(batch_dev)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffuser.diffusion.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = time.time() - t0
            eta = elapsed / epoch * (args.epochs - epoch)
            print(f"    Epoch {epoch:3d}/{args.epochs}: loss={np.mean(epoch_losses):.4f} "
                  f"({fmt_time(elapsed)} elapsed, ETA: {fmt_time(eta)})")

    # Save checkpoint
    ckpt_dir.mkdir(exist_ok=True)
    payload = {
        "diffusion_state_dict": diffuser.diffusion.state_dict(),
        "epoch": args.epochs, "step": step,
        "diffusion_config": asdict(diffusion_config),
        "dataset_config": asdict(dataset_config),
        "normalization_stats": {"mean": norm_stats.mean, "std": norm_stats.std}
        if norm_stats is not None else None,
    }
    torch.save(payload, str(ckpt_path))
    diffuser.diffusion.eval()
    print(f"  Saved checkpoint to {ckpt_path} ({fmt_time(time.time() - t0)})")

    return diffuser


# ─── Step 5: Generate Synthetic Trajectories ────────────────────────────────

def step5_generate(diffuser: SopeDiffuser, rollout_paths: list[Path], args):
    print(f"  Generating {args.num_trajs} stitched trajectories (max {args.horizon} steps)...")
    t0 = time.time()

    init_states = []
    for i in range(args.num_trajs):
        traj = load_rollout_latents(rollout_paths[i % len(rollout_paths)])
        latents = traj.latents
        if latents.ndim == 3:
            latents = latents[:, 0, :]
        init_states.append(latents[0])

    init_states_t = torch.tensor(np.stack(init_states), dtype=torch.float32)

    syn_states, syn_actions, end_indices = diffuser.generate_full_trajectory(
        initial_states=init_states_t,
        max_length=args.horizon,
        guided=False, verbose=False,
    )

    print(f"  Generated: states={syn_states.shape}, actions={syn_actions.shape}")
    print(f"  State range: [{syn_states.min():.2f}, {syn_states.max():.2f}]")
    print(f"  ({fmt_time(time.time() - t0)})")

    return syn_states, syn_actions, end_indices


# ─── Step 6: Score Trajectories ──────────────────────────────────────────────

def step6_score(syn_states, syn_actions, args):
    encoder = make_lift_encoder(obs_keys=OBS_KEYS)
    reward_fn = LiftRewardFn(table_height=0.8, height_threshold=0.04)

    returns, rewards = score_trajectories_gt(
        reward_fn=reward_fn, encoder=encoder,
        states=syn_states, actions=syn_actions, gamma=args.gamma,
    )

    print(f"  Returns: mean={returns.mean():.3f}, std={returns.std():.3f}")
    print(f"  Per-step reward rate: {(rewards > 0).mean():.2%}")

    # Diagnostic: cube z range
    obs_decoded = encoder.decode_to_obs_dict(syn_states[0])
    cube_z = obs_decoded["object"][:, 2]
    print(f"  Traj 0 cube z: [{cube_z.min():.4f}, {cube_z.max():.4f}] (threshold: {reward_fn.success_z:.4f})")

    return returns, rewards


# ─── Step 7: OPE Evaluation ─────────────────────────────────────────────────

def step7_evaluate(oracle_value: float, synthetic_returns, args):
    result = ope_eval(oracle_value, synthetic_returns)

    print()
    print("=" * 60)
    print(f"  OPE EVALUATION ({args.rollouts} rollouts, {args.epochs} epochs)")
    print("=" * 60)
    print(f"  Oracle V^pi:      {result.oracle_value:.3f}")
    print(f"  OPE estimate:     {result.ope_estimate:.3f} (std={result.ope_std:.3f})")
    print(f"  MSE:              {result.mse:.6f}")
    print(f"  Relative error:   {result.relative_error:.2%}")
    print("=" * 60)

    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    policy_dir = find_policy_dir(args.policy_dir)

    print(f"\n{'='*60}")
    print(f"  Latent SOPE — {args.rollouts}-rollout pipeline")
    print(f"  Policy: {policy_dir.name}")
    print(f"  Config: {args.rollouts} rollouts, {args.epochs} epochs, "
          f"{args.num_trajs} synth trajs, gamma={args.gamma}")
    print(f"  Device: {args.device}")
    print(f"  Force rerun: {args.force}")
    print(f"{'='*60}\n")

    t_total = time.time()

    # Step 0
    print("[Step 0] Oracle Ground Truth")
    oracle_value = step0_oracle(policy_dir, args)
    print()

    # Step 1
    print("[Step 1] Collect Offline Data")
    rollout_paths = step1_collect(policy_dir, args)
    print()

    # Steps 2+3
    print("[Steps 2+3] Chunk & Train Diffusion")
    diffuser = step2_3_train(policy_dir, rollout_paths, args)
    print()

    # Step 5
    print("[Step 5] Generate Synthetic Trajectories")
    syn_states, syn_actions, end_indices = step5_generate(diffuser, rollout_paths, args)
    print()

    # Step 6
    print("[Step 6] Score Trajectories (Ground-Truth Reward)")
    synthetic_returns, synthetic_rewards = step6_score(syn_states, syn_actions, args)
    print()

    # Step 7
    print("[Step 7] OPE Evaluation")
    result = step7_evaluate(oracle_value, synthetic_returns, args)

    total_elapsed = time.time() - t_total
    print(f"\nTotal pipeline time: {fmt_time(total_elapsed)}")


if __name__ == "__main__":
    main()
