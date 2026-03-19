# Hopper D4RL Diagnostics: Guided vs Unguided (v0.2.5.13)

**Date:** 2026-03-18
**Notebook:** `experiments/2026-03-17/hopper_d4rl_diagnostics_guided.ipynb`
**SLURM job:** 7439619 (completed ~2:13am)

## Goal

Compare guided vs unguided trajectory generation using SOPE's diffusion model on Hopper D4RL. Evaluate across four axes: per-dim RMSE, action NLL (two scoring functions), trajectory plots, and trajectory survival.

## Setup

- 11 SAC target policies from SOPE's Hopper policy set
- Behavior policy: `D4RLPolicy("hopper-medium-v2")`
- Dataset: 1,000,000 transitions from hopper-medium-v2
- 100 real trajectories extracted (avg length 470)
- Diffusion: TemporalUnet, T_chunk=8, D_steps=256
- Guidance hyperparams from `hopper_stitch.json`: action_scale=0.5, normalize_grad=True, k_guide=1, use_neg_grad=True, neg_grad_scale=0.1, use_action_grad_only=True, ratio=0.5
- N_TRAJS=10, T_GEN=200

## Results

### Per-Dimension RMSE (Guided vs Unguided)

| Condition | Total RMSE | Mean Traj Length |
|-----------|-----------|-----------------|
| Unguided | 1.8481 | 200 |
| Guided P0 | **1.6966** | 200 |
| Guided P1 | 1.8151 | 200 |
| Guided P2 | 1.8074 | 200 |
| Guided P3 | 1.9822 | 200 |
| Guided P4 | 1.9328 | 200 |
| Guided P5 | 2.0013 | 200 |
| Guided P6 | 2.1357 | 200 |
| Guided P7 | 2.1666 | 198 |
| Guided P8 | 2.0338 | 200 |
| Guided P9 | 2.0096 | 200 |
| Guided P10 | 2.0474 | 200 |

Only Policy 0 (closest to behavior) improves RMSE vs unguided. All others increase RMSE.

### Action NLL — `log_prob` (standard)

| Policy | Unguided NLL | Guided NLL | Delta | Better? |
|--------|-------------|-----------|-------|---------|
| 0 | 4.03 | 2.94 | +1.09 | YES |
| 1 | 41.23 | 282.40 | -241.17 | no |
| 2 | 38.65 | 241.22 | -202.57 | no |
| 3 | 38.67 | 325.80 | -287.13 | no |
| 4 | 33.54 | 301.97 | -268.43 | no |
| 5 | 36.32 | 261.46 | -225.15 | no |
| 6 | 41.52 | 267.88 | -226.36 | no |
| 7 | 28.19 | 207.81 | -179.62 | no |
| 8 | 32.07 | 203.96 | -171.89 | no |
| 9 | 31.30 | 264.67 | -233.37 | no |
| 10 | 35.29 | 262.79 | -227.51 | no |

Guidance massively increases NLL for off-behavior policies — pushes trajectories OOD.

### Action NLL — `log_prob_extended` (SOPE's actual scoring)

| Policy | Unguided NLL | Guided NLL | Delta | Better? |
|--------|-------------|-----------|-------|---------|
| 0 | 3.24 | 3.18 | +0.06 | YES |
| 1 | 3.89 | 3.73 | +0.17 | YES |
| 2 | 3.70 | 3.56 | +0.15 | YES |
| 3 | 3.85 | 3.72 | +0.12 | YES |
| 4 | 3.81 | 3.58 | +0.22 | YES |
| 5 | 3.76 | 3.53 | +0.23 | YES |
| 6 | 3.85 | 3.65 | +0.20 | YES |
| 7 | 3.73 | 3.51 | +0.21 | YES |
| 8 | 3.68 | 3.54 | +0.15 | YES |
| 9 | 3.77 | 3.62 | +0.15 | YES |
| 10 | 3.91 | 3.60 | +0.32 | YES |

With `log_prob_extended`: guidance helps ALL policies, but the effect is tiny (deltas 0.06–0.32 on a scale of ~3.5). This scoring function is less sensitive to guidance artifacts.

### Trajectory Survival

No survival issues — all policies generate full-length trajectories (200 steps), except P7 (198). Guidance does NOT cause early termination.

## Key Takeaways

1. **Guidance only helps for near-behavior policies** under standard `log_prob` scoring. For far-from-behavior policies it's catastrophic.
2. **`log_prob_extended` is robust to guidance artifacts** — shows universally positive but tiny improvements.
3. **The divergence between the two scoring functions is a red flag**: `log_prob` says guidance destroys far-policy trajectories, `log_prob_extended` says it helps slightly. This means the OPE result depends heavily on which scoring function is used.
4. No trajectory survival issues from guidance — it doesn't destabilize the dynamics.

## Comparison to Prior Work

- Extends the unguided diagnostics from `results/2026-03-17/hopper_d4rl_diagnostics.md`
- Confirms the hypothesis from action diversity analysis: guidance works when target ≈ behavior, degrades when they diverge
