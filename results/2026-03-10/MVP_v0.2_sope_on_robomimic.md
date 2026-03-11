# MVP v0.2: Stitch-OPE on Robomimic — With Policy Guidance

**Date:** 2026-03-10
**Notebook:** `experiments/2026-03-10/MVP_v0.2_sope_on_robomimic.ipynb`
**Status:** Complete
**Builds on:** MVP v0.1 (unguided pipeline smoke test)

---

## Objective

**Add policy guidance** to the SOPE pipeline validated in v0.1. Without guidance, v0.1's OPE estimate reflected the behavior policy (human demos, 96% success) rather than the target policy (diffusion policy, 54% success). Guidance should steer synthetic trajectory generation toward the target policy, bringing the OPE estimate closer to the oracle value.

## What Changed from v0.1

| Aspect | v0.1 | v0.2 |
|--------|------|------|
| Guidance | None (unguided) | BC_Gaussian proxy + `gradlog()` |
| Diffusion model | Trained from scratch (10 epochs) | **Reuse v0.1 checkpoint** |
| Target policy proxy | N/A | BC_Gaussian trained on 50 target policy rollouts |
| Data source | 200 human demos | Same |
| Oracle | 50 pre-collected rollouts (V^π = 0.54) | Same |

## Guidance Implementation

**Approach chosen: Option A — BC_Gaussian proxy (simplest)**

The target policy (robomimic DiffusionPolicyUNet) doesn't expose `log_prob` or `grad_log_prob`.
Instead of wrapping the diffusion policy directly (which has frame stacking complications), we:

1. **Loaded existing target policy rollouts** from `rollout_latents_50/` (50 rollouts, 2,675 (state, action) pairs)
2. **Trained a BC_Gaussian** MLP (11→64→64→4 + learnable log_std) on those pairs
3. **Used `log_prob_extended()`** for SOPE-style guidance: ∇_a log π(a|s) via autograd

### How Guidance Works (implemented in `generate_trajectories()`)

At each denoising step of the stitching loop:
1. Get model prediction (mean + variance) — standard diffusion `p_mean_variance()`
2. **Unnormalize** the mean to real space
3. Compute `∇_a log π(a|s)` using BC_Gaussian's analytic Gaussian log-prob + autograd
4. **Normalize gradient** (unit norm per-timestep, SOPE default)
5. Add `action_scale * gradient` to the mean
6. **Re-normalize** and re-apply conditioning

This follows SOPE's `default_sample_fn()` in `diffusion.py:152-250`.

## Configuration

Inherits all v0.1 config, plus:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Guidance scales swept | 0.1, 1.0, 10.0 | Find sweet spot |
| BC_Gaussian hidden_dim | 64 | Small MLP sufficient |
| BC_Gaussian training | 200 epochs, lr=1e-3 | Quick convergence |
| BC training data | 2,675 pairs from 50 target policy rollouts | Existing data |
| Gradient normalization | True | SOPE default |

## Success Criteria

1. **OPE estimate closer to oracle**: guided estimate should be nearer to 0.54 than unguided (14.96) — **PASS** (0.68 at scale=0.1)
2. **Synthetic success rate decreases**: should move from 96% toward ~54% — **PARTIAL** (dropped to 10%, overshooting)
3. **Relative error < 100%**: a major improvement over v0.1's 2670% — **PASS** (25.9% at scale=0.1)
4. **Pipeline completes**: guidance doesn't break stitching loop — **PASS**

## Results

### Oracle Value (from v0.1)
| Metric | Value |
|--------|-------|
| V^π (mean return) | 0.5400 |
| Success rate | 54.0% |

### Guidance Scale Sweep
| Scale | OPE Estimate | OPE Std | Success Rate | Rel Error |
|-------|-------------|---------|-------------|-----------|
| Unguided | 16.08 | 6.02 | 98.0% | 2878% |
| **0.1** | **0.68** | **2.38** | **10.0%** | **25.9%** |
| 1.0 | 0.00 | 0.00 | 0.0% | 100% |
| 10.0 | 0.00 | 0.00 | 0.0% | 100% |

### Best Result (scale=0.1)
| Metric | Value |
|--------|-------|
| OPE estimate | 0.6800 |
| OPE std | 2.3786 |
| Synthetic success rate | 10.0% |
| Relative error vs oracle | 25.93% |

### Comparison: v0.1 (Unguided) vs v0.2 (Guided)
| Metric | v0.1 (Unguided) | v0.2 (Guided, scale=0.1) |
|--------|-----------------|--------------------------|
| OPE estimate | 16.08 | 0.68 |
| Relative error | 2878% | 25.9% |
| Synthetic success rate | 98.0% | 10.0% |

### BC_Gaussian Training
| Metric | Value |
|--------|-------|
| Training pairs | 2,675 |
| Final NLL | 3.40 |
| Learned std | [0.82, 0.82, 0.82, 0.83] |

### Figures
- `ope_summary_mvp_v02.png` — OPE vs guidance scale, success rate, trajectory comparison
- `synthetic_trajectories_mvp_v02.png` — Per-dimension guided vs unguided trajectories
- `bc_gaussian_loss.png` — BC_Gaussian training loss curve

### Saved Artifacts
- `results/2026-03-10/mvp_v02_results.json` — Full results JSON

## Observations

### Guidance dramatically reduces OPE estimate — from 16 to 0.68
At scale=0.1, guidance brings the OPE estimate from 16.08 (reflecting expert demos) down to 0.68, which is close to the oracle value of 0.54. The relative error drops from 2878% to 25.9% — a 100x improvement.

### Higher guidance scales overshoot — scale ≥ 1.0 kills all trajectories
At scale=1.0 and 10.0, guidance is too strong: it pushes actions so far toward the target policy's distribution that the diffusion model's learned trajectory dynamics are destroyed. All trajectories have 0% success, meaning the cube is never lifted. The guidance overwhelms the diffusion prior.

### The sweet spot is near scale=0.1
A finer sweep around scale=0.1 (e.g., 0.01, 0.03, 0.05, 0.1, 0.2, 0.5) would likely find a better value. The success rate at scale=0.1 is 10%, which is below the oracle's 54% — this suggests the guidance is slightly too strong even at 0.1, or the BC_Gaussian proxy is imperfect.

### BC_Gaussian proxy is crude but sufficient
The BC_Gaussian was trained on only 2,675 (state, action) pairs with a simple 2-layer MLP. The learned std is ~0.82 for all action dims, which is high — the proxy doesn't capture the target policy's state-conditional action variance well. A better proxy (more data, larger model, or GMM) could improve guidance quality.

### Guidance gradient normalization is critical
SOPE normalizes the guidance gradient to unit norm per-timestep. Without this, the raw gradient magnitude would vary wildly across denoising steps, making the `action_scale` hyperparameter impossible to tune consistently.

## Critical Analysis: Does This Actually Prove Guidance Works?

**No — this single-policy result does not prove guidance works.** The OPE estimate landing near the oracle (0.68 vs 0.54) could be a coincidence rather than evidence of correct policy-conditioned trajectory generation. Key concerns:

### The result pattern looks like trajectory destruction, not proper guidance
- The relationship is monotonic: more guidance = less success. There is no "sweet spot" where guided trajectories match the target policy's actual behavior pattern (54% success).
- Scale=0.1 gives 10% success (undershooting the oracle's 54%), scales ≥ 1.0 give 0%. Guidance is progressively destroying trajectory quality, and scale=0.1 just happens to destroy it to a degree that produces an estimate near the oracle.
- If we had swept finer scales, we'd likely find *some* scale that hits any desired success rate between 0% and 98% — this doesn't mean the pipeline is doing OPE correctly.

### The BC_Gaussian proxy is too crude to provide meaningful guidance
- Learned std ~0.82 across all action dims means the proxy is barely state-conditional. The gradient signal is mostly pushing actions toward a fixed mean rather than capturing the target policy's actual state-dependent behavior.
- A good guidance signal should make trajectories *look like* what the target policy actually does, not just degrade them.

### What would constitute real evidence
1. **Multi-policy ranking (strongest test):** Evaluate multiple target policies of varying quality. If guided OPE correctly *ranks* policies (Spearman ρ ≈ 1), that's much harder to achieve by coincidence than one estimate being close.
2. **Guided success rate ≈ oracle success rate:** Guided synthetic trajectories should match the target policy's 54% success, not just be "lower than unguided."
3. **Qualitative trajectory match:** Guided trajectories should resemble the target policy's actual rollout patterns, not just look like degraded expert demos.

### Implication for v0.3
This analysis motivates v0.3's multi-policy evaluation as the critical next experiment. One close estimate proves nothing; correct ranking across policies of varying quality would be strong evidence.

## Next Steps

- **v0.3 (critical)**: Multiple target policies of varying quality → test ranking (Spearman ρ, regret@1). This is the real test of whether guidance works.
- **Finer guidance scale sweep**: Try 0.01, 0.03, 0.05, 0.1, 0.2, 0.5 to find optimal scale
- **Better BC proxy**: More training epochs, larger model, or GMM to better capture target policy
- **Adaptive guidance scheduling**: SOPE supports cosine/linear schedules for guidance strength over denoising steps (see `get_schedule_multiplier()`)
