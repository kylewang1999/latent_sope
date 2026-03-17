# Hopper D4RL Diagnostics — Reference Baseline for v0.2.5.X Metrics

**Date:** 2026-03-17
**Notebook:** `experiments/2026-03-17/hopper_d4rl_diagnostics.ipynb`
**Purpose:** Extract v0.2.5.X custom diagnostics on SOPE's reference D4RL Hopper implementation to establish ground-truth values for comparison against our robomimic Lift implementation.

## Setup

- **Diffusion model:** `opelab/examples/d4rl/models/hopper.pth` (T=8, D=256, UNet [1,2,4,8])
- **Target policies:** 11 SAC policies (`policy/hopper/dope/{0..10}.pkl`)
- **Behavior policy:** `D4RLPolicy("hopper-medium-v2")`
- **Dataset:** `hopper-medium-v2` (~1M transitions, 11d state, 3d action)
- **Device:** CPU (gradient diagnostics fast; trajectory generation feasible at small scale)

## Diagnostic Methodology

Each diagnostic measures a different aspect of whether guidance-based multi-policy OPE is feasible:

### Diagnostic 1: Pairwise Cosine Similarity of Scorer Gradients
**Calculation:** Sample 500 random (s, a) pairs from the D4RL dataset. For each of the 11 target policies, compute `grad log_prob_extended(s, a)` w.r.t. the action (a 3-dim gradient vector per sample). For every pair of policies (i, j), compute cosine similarity between their gradient vectors at each sample, then average. Report mean of upper triangle of the 11×11 matrix.

**What it tells us:** If two policies produce gradients pointing in the same direction, guidance pushes actions the same way regardless of target. Cosine sim ≈ 1.0 means guidance can't distinguish policies. Lower = more distinguishable. This is the most direct test of whether guidance-based multi-policy ranking is feasible.

### Diagnostic 2: |grad| / |action| Ratio
**Calculation:** Same 500 (s, a) pairs. Compute L2 norm of gradient and L2 norm of action. Ratio = `||grad|| / ||action||` per sample, averaged.

**What it tells us:** How large the guidance perturbation is relative to the action. Ratio of 573 (`log_prob`) means a single unscaled gradient step moves the action 573x its magnitude — catastrophic. Ratio of 1.5 (`log_prob_extended`) means guidance naturally produces action-scale perturbations. Determines how much `action_scale` dampening is needed.

### Diagnostic 3: Gradient Direction Test
**Calculation:** Sample 200 (s, a) pairs. For each policy, start from random actions (Gaussian × 0.5), run 50 steps of gradient ascent (`a += lr × grad log_prob_extended(s, a)`, clamped to [-1, 1]). Measure L2 distance to real dataset action before and after. Improvement = `(init_dist - final_dist) / init_dist × 100%`.

**What it tells us:** Whether the scoring function's gradient points toward "correct" actions. Convergence (+%) = well-behaved landscape. Divergence (-%) for non-behavior policies is expected — the gradient correctly pushes toward *that policy's* preferred actions, which differ from dataset actions (which come from the behavior policy).

### Diagnostic 4: Per-Dimension Trajectory RMSE
**Calculation:** Load pre-trained diffusion model. Take initial states from 20 real trajectories. Generate 20 synthetic trajectories via unguided autoregressive chunk stitching (condition chunk k+1 on last state of chunk k, repeat until T_gen=200). Compute MSE per state dimension across matched timesteps, average across trajectories, sqrt for RMSE.

**What it tells us:** How well the diffusion model reconstructs trajectory dynamics without guidance. This is a baseline for diffusion model quality. High RMSE on certain dimensions (e.g., x_vel) means the model struggles with those features — guidance can't fix what the base model can't generate.

### Diagnostic 5: Action NLL Under Target Policies
**Calculation:** Take the 20 unguided synthetic trajectories from Diagnostic 4. For each target policy, evaluate `log_prob(s, a)` on every (s, a) pair. Report mean negative log-likelihood per trajectory, averaged. Also computed with `log_prob_extended` to compare.

**What it tells us:** How "likely" synthetic actions are under each target policy. Wide NLL spread = policies are distinguishable from action preferences alone. Flat spread = scoring function can't tell which policy generated the actions. The spread difference between `log_prob` (22x range) and `log_prob_extended` (1.2x range) reveals that SOPE's guidance relies purely on gradient *direction*, not magnitude.

### Diagnostic 6: Trajectory Plots
**Calculation:** Plot real (solid) vs synthetic (dashed) trajectories for key state dimensions (z_pos, angle, z_vel, x_vel) for 5 trajectory pairs sharing the same initial state.

**What it tells us:** Visual sanity check — where synthetic trajectories diverge from real, whether they stay physically plausible, and whether autoregressive stitching introduces visible chunk-boundary discontinuities.

---

## Scoring Function Discovery

SOPE's `gradlog()` function uses `log_prob_extended` by default (not `log_prob`). This is a critical difference:

| | `log_prob` | `log_prob_extended` (SOPE default) |
|---|---|---|
| Mean | Network mean | `tanh(network_mean)` |
| Std | Learned per-dim | **Fixed = 1** |
| Tanh correction | Yes (Jacobian) | **No** |
| Gradient | `-(a - mean) / std² + tanh terms` | `-(a - tanh(mean))` |

We ran diagnostics with both to understand the impact. The `log_prob_extended` results are the ones that matter for comparison since that's what SOPE actually uses for guidance.

## Diagnostic 1: Pairwise Cosine Similarity of Scorer Gradients

Computed gradients for each of the 11 SAC policies at 500 random (s, a) pairs from the dataset.

| Metric | `log_prob` | `log_prob_extended` | Our Lift (v0.2.5.X) |
|--------|-----------|-------------------|----------------------|
| **Mean cosine** | **0.47** | **0.64** | 0.72–0.95 |
| Std | 0.19 | 0.12 | — |
| Min | 0.20 | 0.40 | — |
| Max | 0.77 | 0.84 | — |

**Per-policy gradient norms:**

| Policy | `log_prob` grad norm | `log_prob_extended` grad norm |
|--------|---------------------|------------------------------|
| 0 | 159.4 | 1.0 |
| 1 | 1597.5 | 1.5 |
| 2 | 766.8 | 1.3 |
| 3 | 941.0 | 1.5 |
| 4 | 544.1 | 1.4 |
| 5 | 923.0 | 1.5 |
| 6 | 1327.8 | 1.6 |
| 7 | 862.0 | 1.4 |
| 8 | 1328.0 | 1.4 |
| 9 | 1658.0 | 1.5 |
| 10 | 1460.7 | 1.5 |

**Why cosine sim went UP with `log_prob_extended`:** `log_prob` gradients depend on both the policy mean AND its learned per-dimension std, which effectively rotates/scales gradients differently per policy — more degrees of freedom to differ. `log_prob_extended` collapses std to 1 for all policies, so the only differentiator is `tanh(mean)`. If two policies have means pointing in similar directions relative to the action, their gradients become nearly parallel. The learned std was actually helping distinguish policies, but SOPE discards it for numerical stability (gradient norms drop from 100-1600 to ~1.5).

**Key finding:** Even with the scoring function SOPE actually uses (`log_prob_extended`), Hopper policies are more distinguishable (0.64) than our Lift policies (0.72–0.95). The remaining gap likely comes from SAC policies being trained to different convergence points (diverse means) vs our robomimic BC checkpoints producing very similar mean predictions.

## Diagnostic 2: |grad| / |action| Ratio

| | `log_prob` | `log_prob_extended` | Our Lift (v0.2.5.X) |
|---|---|---|---|
| **Mean ratio** | **572.7** | **1.5** | 12.86 |

With `log_prob_extended`, per-policy ratios are all in the range 1.0–1.7. This is the big practical win — gradients are naturally action-scale, no aggressive dampening needed. Our Lift ratio of 12.86 (with `log_prob`) is ~8x larger than SOPE's reference.

## Diagnostic 3: Gradient Direction Test

Can gradient ascent converge from random action toward real action? (lr=0.1, 50 steps, 200 samples)

| | `log_prob` | `log_prob_extended` | Our Lift (v0.2.5.X) |
|---|---|---|---|
| **Mean improvement** | **-26.3%** (all diverge) | **-5.9%** (policy 0: +20.5%) | +72% |

`log_prob_extended` results per policy:

| Policy | Init Dist | Final Dist | Improvement |
|--------|-----------|------------|-------------|
| 0 | 1.308 | 1.040 | **+20.5%** |
| 1 | 1.307 | 1.491 | -14.1% |
| 2 | 1.283 | 1.381 | -7.6% |
| 3 | 1.324 | 1.456 | -10.0% |
| 4 | 1.321 | 1.409 | -6.7% |
| 5 | 1.286 | 1.434 | -11.6% |
| 6 | 1.344 | 1.451 | -8.0% |
| 7 | 1.289 | 1.360 | -5.5% |
| 8 | 1.296 | 1.353 | -4.4% |
| 9 | 1.325 | 1.410 | -6.4% |
| 10 | 1.301 | 1.441 | -10.8% |

**Interpretation:** The GD test asks: "if I optimize an action toward policy X's preferred action, does it get closer to the dataset action?" The dataset actions come from the behavior policy (medium-quality SAC on Hopper, or BC on Lift).

**Hopper diverges (-5.9%)** — this is actually a **good sign**. It means the target policies genuinely want different actions than the behavior policy. When you ascend policy 10's log-prob, you move *away* from the behavior policy's actions, which is correct — policy 10 is a different policy. Policy 0 converges (+20.5%) precisely because it's closest to the behavior policy.

**Our Lift converged (+72%)** — this looks good but is actually **evidence of the problem**. All our BC policies agree on what actions to take (they were trained on the same demonstrations), so ascending any policy's log-prob moves toward the same place — which also happens to be the dataset action. The +72% convergence proves all our policies have nearly identical action preferences, which is exactly why guidance can't rank them.

**In short: divergence = policies are different (guidance can distinguish). Convergence = policies agree (guidance is blind).**

## Diagnostic 4: Per-Dimension Trajectory RMSE (Unguided)

Generated 20 unguided trajectories (T_gen=200) via autoregressive chunk stitching. Mean trajectory length: 200 (no early termination). *(Unchanged from first run — does not depend on scoring function.)*

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

**Pattern:** Velocities are hardest to reconstruct (especially x_vel at 5.86), positions are easiest.

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

**Key finding (`log_prob`):** Policy 0 has dramatically lower NLL (3.94) than all others (28–44), suggesting the behavior data (medium-quality) is closest to policy 0. The wide spread across policies indicates the NLL signal **is informative** for distinguishing policies on Hopper — unlike our Lift setting where NLL differences were marginal.

### NLL comparison: `log_prob` vs `log_prob_extended`

Evaluated on 2000 dataset (s,a) pairs to see how scoring function affects NLL discrimination:

| Policy | `log_prob` NLL | `log_prob_extended` NLL |
|--------|---------------|------------------------|
| 0 | **2.29** | 3.33 |
| 1 | 50.25 | 4.00 |
| 2 | 41.44 | 3.84 |
| 3 | 34.67 | 4.00 |
| 4 | 26.02 | 3.99 |
| 5 | 39.73 | 4.04 |
| 6 | 39.29 | 4.10 |
| 7 | 23.85 | 3.91 |
| 8 | 26.26 | 3.81 |
| 9 | 28.50 | 3.93 |
| 10 | 32.32 | 4.03 |
| **Range** | **2.3 – 50.2 (22x)** | **3.3 – 4.1 (1.2x)** |

**Critical finding:** `log_prob_extended` **collapses the NLL spread** from 22x to 1.2x — essentially flat. The fixed `std=1` washes out differences because all policies have similar means, and the learned std was the main source of NLL discrimination. This means SOPE's guidance is **not relying on NLL discrimination** — it relies purely on gradient *direction* being different enough (cosine sim 0.64) even though gradient magnitude (and thus NLL) is nearly identical across policies. The guidance signal is purely directional, amplified by `action_scale` after per-timestep normalization.

For our Lift policies, both the direction (cosine 0.72–0.95) and the magnitude are too similar. There is no signal on either axis.

## Diagnostic 6: Trajectory Plots

See notebook for inline figures. Shows real (solid) vs synthetic (dashed) trajectories for z_pos, angle, z_vel, and x_vel.

## Summary & Comparison to v0.2.5.X

Using `log_prob_extended` (what SOPE actually uses for guidance):

| Diagnostic | Hopper D4RL (reference) | Our Lift (v0.2.5.X) | Implication |
|------------|------------------------|---------------------|-------------|
| Cosine sim | **0.64** | 0.72–0.95 | Still more distinguishable, but gap is smaller than log_prob suggested |
| \|grad\|/\|action\| | **1.5** | 12.86 | SOPE's gradients are naturally action-scale; ours are 8x too large |
| GD convergence | **-5.9%** | +72% | Broad basin (std=1) doesn't strongly pull; our test converges because learned std is tighter |
| NLL spread (`log_prob`) | 2.3–50.2 (22x) | Marginal | `log_prob` NLL is informative on Hopper but not used by guidance |
| NLL spread (`log_prob_extended`) | 3.3–4.1 (1.2x) | Marginal | **Both flat** — guidance relies on direction, not magnitude |

**Primary bottleneck for Lift:** Scorer indistinguishability (cosine 0.72–0.95 vs Hopper's 0.64). The robomimic BC policies produce nearly parallel gradient directions. The gap narrows when using the same scoring function SOPE uses, but remains significant. This is not an implementation bug — it's a domain mismatch. SOPE's guidance assumes target policies produce distinguishable action distributions. BC policies trained on the same demonstrations violate that assumption.

**Key insight on scoring function:** SOPE deliberately uses a simplified scoring function (`log_prob_extended`) that trades gradient diversity (higher cosine sim) for numerical stability (|grad|/|action| ≈ 1.5 vs 573). This is a design choice — the learned std provides more distinguishable gradients but with catastrophically large magnitudes. The NLL spread also collapses with `log_prob_extended` (1.2x vs 22x), confirming guidance works purely through directional signal, not magnitude.
