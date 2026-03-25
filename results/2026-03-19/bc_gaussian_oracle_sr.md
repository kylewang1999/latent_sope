# BC_Gaussian Oracle Success Rates

**Date:** 2026-03-19 (job 7473300, rerun of failed 7451187)
**Notebook:** `experiments/2026-03-17/bc_gaussian_oracle_sr.ipynb`
**Status:** SUCCESS (~10 min)

## Summary

Trained 6 BC_Gaussian policies from scratch on varying amounts of Lift demo data, then collected 50 oracle rollouts each to get ground-truth success rates. Goal: establish oracle baselines for evaluating guidance-based OPE (v0.2.5.9).

## Policy Training

Architecture: MLP [256, 256], softplus std, ReLU (74,510 params). Total training: 18s.

| Policy | Demos | Epochs | Final Loss |
|--------|-------|--------|------------|
| bc_10demos_e50 | 10 | 50 | 3.18 |
| bc_10demos_e200 | 10 | 200 | -4.54 |
| bc_25demos_e100 | 25 | 100 | -4.33 |
| bc_50demos_e100 | 50 | 100 | -4.75 |
| bc_100demos_e100 | 100 | 100 | -5.82 |
| bc_200demos_e100 | 200 (all) | 100 | -7.58 |

## Oracle Results (50 Rollouts Each)

| Policy | Oracle SR | Mean Reward |
|--------|-----------|-------------|
| bc_10demos_e50 | **0%** | 0.00 |
| bc_10demos_e200 | **0%** | 0.00 |
| bc_25demos_e100 | **0%** | 0.00 |
| bc_50demos_e100 | **0%** | 0.00 |
| bc_100demos_e100 | **0%** | 0.00 |
| bc_200demos_e100 | **16%** | 0.16 |

Only the policy trained on all 200 demos achieved any successes (8/50 rollouts). SR range: 0%–16%.

## Comparison with v0.2.5.9 Guidance Estimates

| Policy | Oracle SR | v0.2.5.9 Guided SR | Gap |
|--------|-----------|-------------------|-----|
| bc_10demos_e50 | 0% | 52% | +52% |
| bc_10demos_e200 | 0% | 60% | +60% |
| bc_25demos_e100 | 0% | 58% | +58% |
| bc_50demos_e100 | 0% | 76% | +76% |
| bc_100demos_e100 | 0% | 52% | +52% |
| bc_200demos_e100 | 16% | 52% | +36% |

**Spearman rho = -0.42** (p=0.41, not significant). Guidance estimates are severely inflated and negatively correlated with oracle SR.

## Key Findings

1. **BC_Gaussian policies are weak on Lift**: 5/6 policies achieve 0% SR. Only training on all 200 demos produces a barely functional policy (16%).
2. **Guidance-based OPE is broken**: v0.2.5.9 estimated 52–76% SR for policies that actually achieve 0%. The negative Spearman rho means guidance ranks policies inversely.
3. **Insufficient policy diversity**: The SR spread (0%–16%) is extremely narrow, making ranking inherently difficult.

## Artifacts

- Policies: `diffusion_ckpts/bc_gaussians/` (6 `.pt` files)
- Oracle JSONs: `diffusion_ckpts/bc_gaussians/oracle_results/`
- Rollout trajectories: `rollouts/bc_gaussians/` (300 total trajectories)
