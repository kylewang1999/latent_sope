# SOPE D4RL Hopper — Reference Baseline Results

**Date:** 2026-03-19 (job 7451190, submitted 2026-03-18)
**Notebook:** `experiments/2026-03-17/sope_d4rl_hopper.ipynb`
**Status:** SUCCESS

## Summary

Ran SOPE's reference implementation end-to-end on D4RL Hopper with 10 policies (varying quality levels). This establishes the baseline performance that our latent SOPE implementation should match.

## Key Results

| Metric | Diffuser | Naive |
|--------|----------|-------|
| **Spearman rho** | **0.804** | NaN (constant) |
| **log RMSE** | **-2.17** | -1.22 |
| **Mean MSE** | **0.013** | 0.086 |
| **Mean regret** | **0.0** | 0.22 |
| **Regret@1** | **0.0** | 1.0 |

## Policy Values (normalized)

| Policy | Oracle (normalized) | Diffuser Estimate | Naive |
|--------|-------------------|-------------------|-------|
| 0 (worst) | 0.000 | 0.158 | 0.701 |
| 1 | 0.777 | 0.779 | 0.701 |
| 2 | 0.810 | 0.693 | 0.701 |
| 3 | 0.717 | 0.783 | 0.701 |
| 4 | 0.818 | 1.059 | 0.701 |
| 5 | 0.879 | 0.969 | 0.701 |
| 6 | 0.989 | 0.976 | 0.701 |
| 7 | 0.950 | 0.835 | 0.701 |
| 8 | 0.975 | 1.000 | 0.701 |
| 9 (best) | 1.000 | 1.082 | 0.701 |

Raw oracle values range: [130.4, 260.8] (Hopper episodic returns).

## Analysis

- **Diffuser correctly identifies the best policy** (regret@1 = 0.0) — critical for policy selection.
- **Strong rank correlation** (rho = 0.804) — ordering of policies is largely preserved.
- **Estimate variance is low** — std across 5 trials per policy is ~0.01 normalized units.
- **Naive baseline fails completely** — returns constant estimates (dataset mean), so it can't rank policies at all.
- **Worst policy (policy 0) is well-separated** — Diffuser estimates 0.158 vs oracle 0.0, clearly identified as worst.
- The Naive baseline's Spearman is NaN because all estimates are identical (constant 0.701).

## Significance for Latent SOPE

This confirms SOPE's diffuser-based OPE works well on D4RL Hopper:
- rho=0.804 is a strong baseline to aim for
- The method successfully distinguishes 10 policies spanning a wide quality range
- Our latent SOPE on Lift should target similar rank correlation and regret metrics
