# Hopper D4RL Diagnostics — Reference Baseline for v0.2.5.X Metrics

**Date:** 2026-03-17
**Script:** `experiments/2026-03-17/hopper_diagnostics.py`
**Purpose:** Extract v0.2.5.X custom diagnostics on SOPE's reference D4RL Hopper implementation to establish ground-truth values for comparison against our robomimic Lift implementation.

## Setup

- **Diffusion model:** `opelab/examples/d4rl/models/hopper.pth` (T=8, D=256, UNet [1,2,4,8])
- **Target policies:** 11 SAC policies (`policy/hopper/dope/{0..10}.pkl`)
- **Behavior policy:** `D4RLPolicy("hopper-medium-v2")`
- **Dataset:** `hopper-medium-v2` (~1M transitions, 11d state, 3d action)
- **Device:** CPU (gradient diagnostics fast; trajectory generation feasible at small scale)

## Diagnostic 1: Pairwise Cosine Similarity of Scorer Gradients

Computed `grad log_prob(s, a)` for each of the 11 SAC policies at 500 random (s, a) pairs from the dataset, then measured pairwise cosine similarity of the gradient vectors across policies.

| Metric | Hopper D4RL | Our Lift (v0.2.5.X) |
|--------|-------------|----------------------|
| **Mean cosine** | **0.4737** | 0.72–0.95 |
| Std | 0.1852 | — |
| Min | 0.2023 | — |
| Max | 0.7734 | — |
| Median | 0.4847 | — |

**Per-policy gradient norms** (mean over 500 samples):

| Policy | Grad Mean Norm |
|--------|---------------|
| 0 | 159.4 |
| 1 | 1597.5 |
| 2 | 766.8 |
| 3 | 941.0 |
| 4 | 544.1 |
| 5 | 923.0 |
| 6 | 1327.8 |
| 7 | 862.0 |
| 8 | 1328.0 |
| 9 | 1658.0 |
| 10 | 1460.7 |

**Key finding:** Hopper SAC policies produce **much more distinguishable** gradient directions (mean cosine 0.47) than our robomimic policies (0.72–0.95). This is a primary reason guidance can rank policies on Hopper but fails on Lift.

## Diagnostic 2: |grad| / |action| Ratio

| Policy | |grad|/|action| (mean) | |grad| mean | |action| mean |
|--------|----------------------|-------------|--------------|
| 0 | 93.3 | 115.9 | 1.07 |
| 1 | 774.9 | 950.6 | 1.07 |
| 2 | 445.9 | 571.3 | 1.07 |
| 3 | 359.5 | 466.9 | 1.07 |
| 4 | 376.8 | 473.6 | 1.07 |
| 5 | 627.8 | 813.1 | 1.07 |
| 6 | 805.7 | 1036.1 | 1.07 |
| 7 | 456.8 | 591.2 | 1.07 |
| 8 | 665.4 | 866.7 | 1.07 |
| 9 | 764.6 | 995.0 | 1.07 |
| 10 | 929.3 | 1194.4 | 1.07 |
| **Mean** | **572.7** | — | — |

**Comparison:** Our v0.2.5.3 found |grad|/|action| = 12.86. Hopper's raw gradients are even larger (573x), but SOPE handles this by using `normalize_grad=true` (L2-normalizes per timestep to unit norm) combined with `action_scale=0.5`. The raw ratio is not directly comparable because normalization is always applied.

## Diagnostic 3: Gradient Direction Test

Can gradient ascent on `log_prob(s, a)` converge from random action toward real action? (lr=0.1, 50 steps, 200 samples)

| Policy | Init Dist | Final Dist | Improvement |
|--------|-----------|------------|-------------|
| 0 | 1.308 | 1.652 | **-26.3%** |
| 1 | 1.307 | 1.730 | **-32.4%** |
| 2 | 1.283 | 1.562 | **-21.7%** |
| 3 | 1.324 | 1.646 | **-24.4%** |
| 4 | 1.321 | 1.659 | **-25.6%** |
| 5 | 1.286 | 1.746 | **-35.8%** |
| 6 | 1.344 | 1.639 | **-22.0%** |
| 7 | 1.289 | 1.621 | **-25.8%** |
| 8 | 1.296 | 1.616 | **-24.7%** |
| 9 | 1.325 | 1.617 | **-22.0%** |
| 10 | 1.301 | 1.672 | **-28.5%** |
| **Mean** | — | — | **-26.3%** |

**Comparison:** Our v0.2.5.4 showed +72% convergence. Hopper **diverges** (-26%) because the SAC policies use tanh-squashed actions, making the `log_prob` landscape non-concave in action space.

**Important caveat:** SOPE's guidance actually uses `log_prob_extended` (not `log_prob`) for computing gradients. `log_prob_extended` uses a simpler Gaussian around `tanh(mean)` with fixed `std=1`, which has a well-behaved gradient landscape. This means the GD convergence test on raw `log_prob` is not representative of what SOPE actually does during guidance.

## Diagnostic 4: Per-Dimension Trajectory RMSE (Unguided)

Generated 20 unguided trajectories (T_gen=200) via autoregressive chunk stitching. Mean trajectory length: 200 (no early termination).

| Dimension | RMSE |
|-----------|------|
| z_pos | 0.372 |
| angle | 0.048 |
| thigh_angle | 0.207 |
| leg_angle | 0.044 |
| foot_angle | 0.593 |
| z_vel | 0.608 |
| angle_vel | 1.319 |
| thigh_vel | 0.752 |
| leg_vel | 1.513 |
| foot_vel | 1.076 |
| x_vel | **5.857** |
| **Total RMSE** | **1.930** |

**Pattern:** Velocities are hardest to reconstruct (especially x_vel at 5.86), positions are easiest. This is expected — velocities have higher variance and are harder to stitch coherently across chunks.

## Diagnostic 5: Action NLL Under Target Policies

NLL of unguided synthetic trajectory actions evaluated under each target policy's `log_prob`:

| Policy | Mean NLL | Std |
|--------|----------|-----|
| **0** | **3.94** | 0.39 |
| 1 | 39.27 | 3.56 |
| 2 | 43.84 | 13.78 |
| 3 | 37.35 | 4.71 |
| 4 | 33.25 | 3.45 |
| 5 | 36.24 | 6.84 |
| 6 | 41.32 | 11.02 |
| 7 | 28.89 | 4.59 |
| 8 | 32.13 | 5.85 |
| 9 | 29.43 | 3.41 |
| 10 | 34.43 | 4.08 |

**Key finding:** Policy 0 has dramatically lower NLL (3.94) than all others (28–44), suggesting the behavior data (medium-quality) is closest to policy 0. The wide spread across policies indicates the NLL signal **is informative** for distinguishing policies on Hopper — unlike our Lift setting where NLL differences were marginal.

## Diagnostic 6: Trajectory Plots

Saved to `experiments/2026-03-17/diag6_trajectory_plots.png`. Shows real (solid) vs synthetic (dashed) trajectories for z_pos, angle, z_vel, and x_vel.

## Summary & Comparison to v0.2.5.X

| Diagnostic | Hopper D4RL (reference) | Our Lift (v0.2.5.X) | Implication |
|------------|------------------------|---------------------|-------------|
| Cosine sim | **0.47** | 0.72–0.95 | Hopper policies are 2x more distinguishable |
| |grad|/|action| | 573 (before normalization) | 12.86 | Both need normalization; raw ratio less important |
| GD convergence | **-26% (diverges)** | +72% | SOPE uses `log_prob_extended`, not `log_prob` |
| NLL spread | 3.9 to 43.8 (10x range) | Marginal differences | Hopper has much more NLL signal |

**Primary bottleneck for Lift:** Scorer indistinguishability (cosine 0.72–0.95). The robomimic policies produce nearly parallel gradient directions, making it impossible for guidance to steer differently across policies.

**Secondary insight:** SOPE uses `log_prob_extended` for guidance (simpler Gaussian with `std=1` around `tanh(mean)`), not the full `log_prob` with tanh-Jacobian correction. This may be important for our implementation — the scoring function used for guidance matters as much as the policy architecture.
