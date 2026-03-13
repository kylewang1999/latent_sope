# MVP v0.2.5.3: Target Scorer Gradient Debugging

**Date:** 2026-03-13
**Builds on:** MVP v0.2.5.2 (trajectory MSE analysis)
**Notebook:** `experiments/2026-03-13/MVP_v0.2.5.3_scorer_debug.ipynb`

## Goal

Debug `RobomimicDiffusionScorer.grad_log_prob()` to determine why target policy
guidance in v0.2.5.2 suppressed success rate (60% → 24%) instead of improving it.

## Scorer Configuration

| Parameter | Value |
|-----------|-------|
| score_timestep | 1 |
| sigma at t=1 | 0.0418 |
| observation_horizon (To) | 2 |
| prediction_horizon (Tp) | 16 |
| action_start | 1 |
| Scoreable chunk positions | 4/4 (no padding) |

## Test Results

### Test 1: Sigma Check — SMALL BUT NOT EXTREME

sigma=0.0418 at t=1. The score formula is `-noise_pred / sigma`, so gradients
are amplified by ~24x. Not near-zero, but a significant multiplier.

### Test 2: Gradient Direction — FAILS AT LR=0.1

- Only **20%** of gradient steps moved closer to real action (worse than random 50%)
- Mean relative distance change: **-52%** (moved 52% farther away)
- |grad| at real actions: mean=15.0, at random actions: mean=58.0
- The gradient magnitude ordering is correct (|grad_real| < |grad_rand|), but
  a single step at lr=0.1 overshoots so badly it ends up farther away.

### Test 3: Gradient Magnitude — **WAY TOO LARGE**

- **|grad|/|action| ratio = 12.86** (gradients are 13x larger than actions)
- Per-dimension ratios range from 5x (gripper) to **257x** (act_3)

**Impact on guidance in v0.2.5.2:**

| action_scale | Guidance magnitude | % of action magnitude |
|-------------|-------------------|----------------------|
| 0.05 | 0.72 | **64%** |
| 0.10 | 1.44 | **129%** |
| 0.20 | 2.88 | **258%** |
| 0.50 | 7.21 | **645%** |

Even the smallest guidance scale (0.05) adds perturbations that are 64% of the
action magnitude. At scale=0.5, the guidance is 6.5x larger than the actions
themselves — completely overwhelming the diffusion model.

### Test 4: Gradient Field — (visual only, see notebook)

### Test 5: Gradient Descent — WORKS AT LR=0.01

- **90% converged** toward real action (18/20)
- Mean distance improvement: **66.3%**
- Final action std ≈ real action std (no mode collapse)

This is the key finding: at a small enough learning rate, following the gradient
**does** recover the real actions.

### Test 5b: LR Sweep — (visual only, see notebook)

## Key Finding: The Gradient Direction Is Correct, But the Magnitude Is Catastrophic

The apparent contradiction between Test 2 (20% success) and Test 5 (90% success)
resolves cleanly:

- **Test 2** uses lr=0.1 → step size = 0.1 × 58 = **5.8** (for random actions).
  With |a_real| ≈ 1.1, this overshoots by 5x and lands farther away.
- **Test 5** uses lr=0.01 → step size = 0.01 × 58 = **0.58**, small enough to
  converge without overshooting.

**The scorer's gradient direction is valid. The problem is purely one of scale.**

## Root Cause of v0.2.5.2 Guidance Failure

The guidance loop in `generate_trajectories_full_guidance` applies:
```
model_mean += action_scale * grad
```
where `|grad| ≈ 15` at real actions and `|action| ≈ 1.1`. Even `action_scale=0.05`
produces a perturbation of 0.75, which is 64% of the action magnitude — far too
aggressive for a single denoising step.

The guidance doesn't just nudge the trajectory — it **overwrites** the diffusion
model's prediction with the scorer's gradient. This explains:
- pos_only configs reducing SR: the oversized gradient destabilizes the trajectory
- Stronger guidance = worse MSE: larger perturbations = more destabilization
- full_0.2_r0.5 having 68% SR: the negative behavior term partially cancels the
  oversized positive term, accidentally producing a net guidance closer to zero

## Fix

Reduce `action_scale` by 10–100x. Based on the gradient magnitudes:
- Target step size should be ~1–5% of action magnitude
- With |grad| ≈ 15, need action_scale ≈ 0.001–0.005
- Alternatively, normalize gradients to unit norm before applying action_scale
  (the notebook already has `normalize_grad=True`, but this normalizes per-dim,
  not the full gradient vector — need to verify this is working correctly)

## Resolved: Why normalize_grad=True didn't fix this

Verified the v0.2.5.2 code. The normalization IS correct:
```python
target_grad = target_grad / (target_grad.norm(dim=-1, keepdim=True) + eps)
```
This normalizes each (B, T) position's gradient to unit norm across action_dim.
After normalization, `|target_grad| = 1.0` per timestep, and `action_scale`
directly controls step size per denoising step.

**The real problem is accumulation across denoising steps.**

Guidance is applied at **every one of the 256 denoising steps**. The cumulative
perturbation is approximately:

```
total_perturbation ≈ action_scale × N_DIFFUSION_STEPS
  action_scale=0.05 → 0.05 × 256 = 12.8  (1164% of |action|)
  action_scale=0.10 → 0.10 × 256 = 25.6  (2327% of |action|)
  action_scale=0.20 → 0.20 × 256 = 51.2  (4655% of |action|)
```

Even the smallest guidance scale accumulates a perturbation that is **11.6x the
action magnitude** — completely overwhelming the diffusion model's prediction.

This also explains the SOPE reference implementation: SOPE uses much fewer
diffusion steps (e.g., 20–100) and smaller guidance scales on simpler environments.
With 256 steps, the guidance scale needs to be proportionally smaller.

### Corrected action_scale estimates

To achieve ~5–10% total perturbation relative to action magnitude:
```
target = 0.05 * |action| = 0.055
action_scale = target / N_DIFFUSION_STEPS = 0.055 / 256 ≈ 0.0002
```

Recommended sweep: `action_scale ∈ [0.0001, 0.0002, 0.0005, 0.001, 0.002]`

## Next Steps

1. **Re-run guidance sweep with corrected action_scale** in [0.0001 – 0.002]
2. **Consider reducing N_DIFFUSION_STEPS** (e.g., 64 or 100) to reduce
   accumulation and speed up generation (currently 256 steps × 20 chunks =
   ~120s per config)
3. **Fix the rollout recorder bug** before drawing OPE conclusions
