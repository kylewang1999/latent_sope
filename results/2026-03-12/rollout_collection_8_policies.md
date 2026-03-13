# Rollout Collection: 8 Target Policies x 100 Rollouts

**Date:** 2026-03-12 (completed overnight)
**SLURM Job:** 7368312 on d13-07 (V100)
**Runtime:** 91.3 min

## What was done

Collected 100 rollouts per policy for 8 target policies (topping up where prior rollouts existed). This provides the behavior data for multi-policy OPE experiments.

## Results

| Policy | New Rollouts | Total | Observed SR (from rollout data) | Notes |
|--------|-------------|-------|--------------------------------|-------|
| 50demos_epoch10 | 80 | 100 | **0%** | All reward=0.0 |
| 10demos_epoch10 | 100 | 100 | **0%** | All reward=0.0 |
| 200demos_epoch10 | 80 | 100 | **0%** | All reward=0.0 |
| 100demos_epoch20 | 100 | 100 | **~3%** | 3 successes (latents 46, 43, 55 steps) |
| 10demos_epoch20 | 80 | 100 | **~25%** | Scattered successes |
| 10demos_epoch30 | 100 | 100 | **~90%** | Most succeed (latents 39–54 steps) |
| 100demos_epoch40 | 100 | 100 | **~90%** | Most succeed (latents 32–53 steps) |
| 200demos_epoch40 | 80 | 100 | **~90%** | Most succeed (latents 37–52 steps) |

Total: 720 rollouts across 8 policies. ~8 seconds per rollout.

## Observations

1. **Epoch10 policies show 0% SR**: All three epoch10 variants (10/50/200 demos) report `reward=0.0` for every single rollout. Their oracle SRs are 0%, 8%, and 18% respectively. The 0% rollout SR is consistent with the rollout recorder bug for the low-SR policies (few successes to miss), but the 8% and 18% oracle SRs should produce *some* successes in 100 rollouts. This suggests the `reward` field in the log may also be affected by the recorder bug — need to verify whether `reward=0.0` means "no success happened" or "success happened but wasn't recorded."

2. **Successful rollouts terminate early**: Success trajectories have 32–55 steps (vs 60 horizon), confirming early termination on success. These shorter trajectories have the success state *just before* the final one — the rollout recorder bug means the actual `cube_z > 0.84` observation is dropped.

3. **Higher-epoch policies succeed more**: epoch30 and epoch40 policies show ~90% SR, consistent with oracle values (62%, 76%, 90%). The discrepancy (observed ~90% vs oracle 62–90%) may be due to the `reward` field in the log being a running indicator (100% if any of the last batch succeeded) rather than per-rollout.
