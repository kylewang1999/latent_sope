# MVP v0.2.5.11 — D4RL SOPE Calibration

**Date:** 2026-03-19/20 (job 7473301, ~6.5h runtime)
**Notebook:** `experiments/2026-03-17/MVP_v0.2.5.11_d4rl_sope_calibration.ipynb`
**Status:** SUCCESS

## Summary

Tests whether SOPE guidance works in the within-dataset regime (limited policy diversity) vs cross-dataset regime (high diversity). Uses BC_Gaussian policies on D4RL Hopper with guided diffusion trajectory generation and learned reward model scoring.

## Setup

- **Cross-dataset (6 policies):** BC_Gaussian on D4RL Hopper: random, medium, medium-replay, medium-expert, expert, full-replay
- **Within-dataset (6 policies):** BC_Gaussian on subsets (500, 1k, 5k, 10k, 50k, 100k transitions) from hopper-medium-v2
- **Oracle:** 100 rollouts per policy (gamma=0.99, horizon=768)
- **OPE:** Guided generation (50 trajectories, T=768), scored with learned reward model
- **Reward model:** MLP, MSE loss decreased 14.30 → 0.0056 over 1000 iterations

## Cross-Dataset OPE Results

| Metric | Value |
|--------|-------|
| **Spearman rho** | **+0.657** (p=0.156) |
| **Pearson r** | **+0.883** (p=0.020) |
| Regret@1 | 0.000 |
| Regret@2 | 0.000 |
| logRMSE | -0.222 |

Oracle range: [17.8, 257.5]. OPE estimates collapse to narrow range ~6.1–7.2 but preserve ranking reasonably well. Perfect top-1 and top-2 identification.

## Within-Dataset OPE Results

| Metric | Value |
|--------|-------|
| **Spearman rho** | **+0.314** (p=0.544) |
| **Pearson r** | **+0.728** (p=0.101) |
| Regret@1 | 0.069 |
| Regret@2 | 0.069 |
| logRMSE | 0.191 |

Oracle range: [72.4, 217.1]. OPE estimates near-constant ~6.5–6.8 — fails to differentiate policies.

## Distribution Diagnostics

| Diagnostic | Cross | Within | SAC Ref |
|-----------|-------|--------|---------|
| Mean cosine sim | 0.252 | 0.145 | 0.474 |
| \|grad\|/\|action\| | 23.5 | 6.5 | 572.7 |
| GD convergence | +289% | +167% | -26% |
| NLL spread | 562.9 | 74.5 | 39.9 |

## Key Findings

1. **Cross-dataset works moderately**: Spearman=0.657, Pearson=0.883 (significant). Perfect regret@1. Guidance preserves ranking despite compressed OPE scale.
2. **Within-dataset fails**: Spearman=0.314 (not significant). OPE estimates collapse to near-constant values across policies with oracle values ranging 72–217.
3. **Guidance signal too weak for BC_Gaussian**: Gradient magnitudes are 25–90x smaller than SAC reference. Gradient ascent diverges (+167% to +289%) instead of converging like SAC (-26%).
4. **Confirms policy diversity requirement**: SOPE guidance needs cross-dataset-level diversity to produce meaningful rankings. Within-dataset (same data, varying amounts) is fundamentally beyond current capability.

## Verdict

**MIXED.** Cross-dataset shows promise (rho=0.66, perfect regret). Within-dataset is a clear failure. The core issue is BC_Gaussian gradients being too weak and directionally wrong for effective guidance.
