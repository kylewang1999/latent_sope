# MVP v0.3.2: Multi-Policy Positive and Negative Test

**Date:** 2026-03-12
**Notebook:** `experiments/2026-03-12/MVP_v0.3.2_multi-policy_positive_and_negative_test.ipynb`
**SLURM Job:** 7358173 (d13-11, V100-32GB)
**Status:** COMPLETED — 1.6 hours wall time
**Result:** MIXED — correct top-1 pick but poor ranking and false positive on 0% policy

## What was tried

Evaluate OPE pipeline across 4 target policies spanning 0–90% SR, using positive+negative guidance with BC_Gaussian policy. This is the first multi-policy evaluation.

**Positive test:** OPE estimates should rank policies correctly (Spearman ρ > 0.7).
**Negative test:** A 0% SR policy should get OPE ≈ 0 (< 0.1).

### Setup
- 4 target policies: 50demos_epoch10 (0%), 200demos_epoch10 (18%), 10demos_epoch20 (52%), 200demos_epoch40 (90%)
- Oracle values from `oracle_eval_all_checkpoints.json` (50 rollouts each)
- 20 target rollouts per policy + 200 expert demos = 220 episodes combined
- Training: chunk diffuser ~50 epochs, ~4900 chunks, ~76 batches/epoch
- Guidance: BC_Gaussian positive+negative
- Reward: episode-level binary (cube_z > 0.84)

## Key Metrics

- **Spearman rank correlation:** ρ = 0.40 (p = 0.60) — **FAIL** (target: ρ > 0.7)
- **Regret@1:** 0.00 — **PASS** (correctly identifies 200demos_epoch40 as best)
- **Mean relative error:** 74.57% (excluding 0% oracle policies)
- **MSE:** 0.0958
- **MAE:** 0.2875

### Per-policy results

| Checkpoint | Oracle SR | Oracle V^pi | OPE Estimate | Synth SR | Rel Error | Diffuser Loss | BC NLL |
|-----------|-----------|-------------|--------------|----------|-----------|---------------|--------|
| 50demos_epoch10 | 0% | 0.0000 | 0.2000 | 20.0% | huge | 0.1756 | -6.22 |
| 200demos_epoch10 | 18% | 0.1800 | 0.0000 | 0.0% | 100.0% | 0.1756 | -6.36 |
| 10demos_epoch20 | 52% | 0.5200 | 0.0500 | 5.0% | 90.4% | 0.1613 | -8.91 |
| 200demos_epoch40 | 90% | 0.9000 | 0.6000 | 60.0% | 33.3% | 0.1594 | -8.76 |

### Key observations from rollout data

- **All 4 target policies had 0% SR in collected rollouts** (20 rollouts each). This is a critical issue — the rollout collection got unlucky or 20 rollouts is too few for policies with moderate SR.
- The diffuser is trained on 200 expert demos (100% SR) + 20 target rollouts (0% SR in all cases), so the training data is essentially the same across all 4 policies.
- Despite identical training data, guidance differentiates somewhat: the 90% SR policy gets OPE=0.60, while the 52% policy only gets 0.05. This suggests guidance is doing *something* but working from a very weak signal.

## Comparison to prior experiments

| Version | Target | Oracle V^pi | Best OPE | Rel Error | Notes |
|---------|--------|-------------|----------|-----------|-------|
| v0.2.4.2 | single (54% SR) | 0.54 | 0.64 | 18.5% | Best single-policy result, 50 target rollouts |
| v0.2.5 | single (54% SR) | 0.54 | 0.54 | ~0% | Fixed reward to episode-level binary |
| **v0.3.2** | **multi (0–90%)** | **varies** | **0.00–0.60** | **74.6% avg** | **This experiment** |

## Analysis

### What went wrong

1. **Insufficient target rollouts (20):** With only 20 rollouts per policy, even the 52% and 90% SR policies got 0% SR in the collected data. This means the diffuser training data contains zero successful target trajectories — the only successes come from expert demos. The diffuser can't learn policy-specific behavior from data that doesn't reflect the policy.

2. **Negative test failure:** The 0% policy got OPE=0.20 (synthetic SR=20%). Since training data is essentially the same for all policies (200 expert + 20 failures), the guidance from the BC policy must be hallucinating success trajectories for the 0% policy.

3. **Ranking inversion:** 200demos_epoch10 (oracle=0.18) got OPE=0.00, while 50demos_epoch10 (oracle=0.00) got OPE=0.20. This is a complete inversion for the bottom two policies.

### What went right

- **Regret@1 = 0:** The best policy (200demos_epoch40, 90% SR) was correctly identified as the best by OPE (0.60), despite significant underestimation.
- **Monotonicity at the top:** The 90% policy got the highest OPE, suggesting guidance has some discriminative power for strong policies.

### Root cause: observation recording bug (NOT sample size)

Initial hypothesis was that 20 rollouts was too few, but investigation revealed the actual problem: **the rollout recorder drops the success state.** The recorder stores pre-step observations (`obs`), not post-step (`next_obs`). When an episode terminates on success, the loop breaks before `obs = next_obs`, so the final state where `cube_z > 0.84` is never recorded. The env's reward signal (1.0) is recorded correctly, but the observation data maxes out at ~0.838.

This means:
1. Target rollout SR is always 0% (cube_z never exceeds 0.84 in recorded data), despite env reporting 5–90% SR via rewards
2. The diffuser trains on data where "success" looks like cube_z ≈ 0.835, so generated trajectories also stay below 0.84
3. Synthetic trajectory scoring (cube_z > 0.84) never finds successes

Expert demos in the HDF5 don't have this issue — they reach cube_z = 0.864–0.886 — because they were recorded with a different pipeline.

See full bug report: `results/2026-03-12/bugs_v032_observation_recording.md`

### Next steps

1. **Fix the rollout recorder** to store the final `next_obs` when the episode terminates on success
2. **Re-collect all rollouts** with the fixed recorder
3. **Re-run v0.3.2** with corrected data — the OPE results should improve dramatically since the diffuser will learn actual success states
