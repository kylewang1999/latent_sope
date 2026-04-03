# Oracle Sum-of-Rewards (12 Policies, Full Horizon)

**Date:** 2026-03-26
**SLURM Job:** 7612897
**Notebook:** `experiments/2026-03-26/oracle_discounted_returns.ipynb`
**Builds on:** Previous oracle SR values (binary 0/1 per episode)

## Goal

Compute oracle values as sum of per-step rewards (not binary success rate) for 12 policies on Lift. This matches the quantity that `compute_ope_hard()` measures on synthetic trajectories, giving a fairer oracle--OPE comparison.

## Setup

- **12 target policies** spanning oracle SR 0%--92%
- **50 rollouts per policy**, horizon=60, gamma=1.0
- **Reward:** hard threshold, `cube_z > 0.84` gives reward 1.0 per step
- **No early termination** (full 60-step horizon always run)
- **Pre-step observation recording** (matches synthetic trajectory semantics)

## Results

| Policy | Old SR | New SR | Sum Reward (mean) | Std | Min | Max |
|--------|--------|--------|-------------------|-----|-----|-----|
| 50demos_epoch10 | 0% | 4% | 0.56 | 2.81 | 0 | 17 |
| 10demos_epoch10 | 8% | 18% | 1.14 | 3.34 | 0 | 17 |
| 200demos_epoch10 | 18% | 12% | 1.40 | 4.55 | 0 | 18 |
| 200demos_epoch20 | 24% | 16% | 2.10 | 5.23 | 0 | 23 |
| 100demos_epoch20 | 42% | 28% | 4.08 | 6.83 | 0 | 21 |
| test_checkpoint | 54% | 54% | 6.78 | 7.87 | 0 | 24 |
| 50demos_epoch20 | 60% | 52% | 7.90 | 9.22 | 0 | 26 |
| 10demos_epoch30 | 62% | 70% | 9.08 | 7.72 | 0 | 24 |
| 100demos_epoch30 | 72% | 62% | 9.82 | 8.84 | 0 | 23 |
| 200demos_epoch40 | 90% | 82% | 11.94 | 7.05 | 0 | 23 |
| 50demos_epoch40 | 88% | 82% | 13.10 | 7.53 | 0 | 27 |
| 50demos_epoch30 | 82% | 92% | 16.32 | 6.60 | 0 | 26 |

**Spearman rho (old SR vs new sum-reward): +0.972 (p < 0.0001)**

## Key Findings

1. **Sum-of-rewards preserves policy rankings** (rho=0.972 with old SR), while adding more granularity (continuous values instead of binary).

2. **Captures "quality of success"** -- how long the cube stays lifted, not just whether it was ever lifted. 50demos_epoch30 has lower SR (82% old) than 200demos_epoch40 (90% old) but higher sum reward (16.32 vs 11.94), indicating earlier/more stable lifts.

3. **High variance across rollouts** -- all policies have min return of 0.0 (every policy sometimes fails completely). CVs are often > 0.5.

4. **Ranking swap at the top:** Top-3 by old SR = (200demos_epoch40, 50demos_epoch40, 50demos_epoch30). Top-3 by sum reward = (50demos_epoch30, 50demos_epoch40, 200demos_epoch40). Ranks 4--12 are unchanged.

5. **Rollouts saved** to `rollouts/oracle_full_horizon/{policy_name}/` and results JSON to `results/2026-03-26/oracle_sum_rewards_12policies.json`.

## Implications

Sum-of-rewards is a better oracle metric for evaluating OPE methods that estimate cumulative reward rather than binary success. Future OPE evaluations should use these oracle values for fairer comparison.
