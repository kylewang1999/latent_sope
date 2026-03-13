# Guidance Findings: SOPE vs Our Setting

**Date:** 2026-03-12

## How SOPE guidance works

The guided sampling modifies the diffusion denoising step:

```
g(τ) = α · ∇log π_target(a|s) − λ · ∇log β(a|s)
```

- **α (action_scale):** strength of positive guidance toward target policy
- **λ (ratio × action_scale):** strength of negative guidance pushing away from behavior policy
- In code: `guide = grad_target - ratio * grad_behavior`, then `model_mean += action_scale * guide`

## SOPE paper recommendations

- λ < α works best (i.e., ratio < 1.0)
- λ=0 (no negative guidance): collapses to behavior policy high-density regions
- λ=α (ratio=1.0): can push too far from behavior distribution
- Hyperparameters are **tuned per task** against oracle — the paper doesn't provide a task-agnostic selection method

### Reference values from SOPE codebase

| Environment | action_scale | ratio | k_guide |
|-------------|-------------|-------|---------|
| D4RL (Walker, Hopper, HalfCheetah) | 0.5 | 0.5 | 1 |
| Gym Pendulum | 0.1 | 1.0 | 1 |
| Gym Acrobat | 0.5 | 1.0 | 1 |
| Diffusion Policy Walker | 0.5 | 0.5 | 1 |

## Our setting is fundamentally different

In D4RL, behavior data is a single mediocre policy. Pushing away from it makes sense — it encourages the diffuser to explore beyond what the behavior policy would do.

In our setting, **71% of behavior data is expert demos (100% SR)**. Pushing away from the behavior policy = pushing away from the lifting pattern = destroying success.

## Evidence from our experiments

### v0.2.5 sweep (single policy, 54% SR target)

| Config | Scale | Ratio | SR | Rel Error |
|--------|-------|-------|----|-----------|
| unguided | 0.0 | 0.0 | 76% | 40.7% |
| **pos_only_0.05** | **0.05** | **0.0** | **48%** | **11.1%** |
| pos_only_0.1 | 0.10 | 0.0 | 44% | 18.5% |
| pos_only_0.2 | 0.20 | 0.0 | 34% | 37.0% |
| full_0.05_r0.25 | 0.05 | 0.25 | 74% | 37.0% |
| full_0.2_r0.25 | 0.20 | 0.25 | 64% | 18.5% |
| full_0.5_r0.5 | 0.50 | 0.50 | 0% | 100.0% |

**Key patterns:**
1. Positive-only guidance with small scale (0.05) gives best results
2. Increasing negative guidance monotonically reduces accuracy when BC is expert-heavy
3. High ratio + high scale (0.5/0.5) kills SR entirely — same failure mode as v0.3.2.1

### v0.3.2.1 (multi-policy, shared diffuser)

- Base SR (unguided): 60%
- With guidance (scale=0.2, ratio=0.25): **0% SR for all 4 policies**
- Guidance is too strong — same pattern as v0.2.5's `full_0.5_r0.5`

## Prediction for v0.3.2.2 sweep

Based on v0.2.5, expect:
- **ratio=0.0 (positive only) with small scale (0.05)** to be the best config
- Negative guidance to be counterproductive across all settings
- Higher scales to progressively reduce SR

## Open question: real-world applicability

In a real OPE scenario, there are no oracle values to sweep against. SOPE tunes hyperparameters per-task on benchmarks with known ground truth. This is a fundamental limitation — for deployment you'd need either:
1. Fixed hyperparameters that generalize across policies (not demonstrated)
2. A hyperparameter-free guidance method
3. A proxy metric (e.g., trajectory plausibility) for selection without oracle
