# MVP v0.3.2.3: Medium-Quality Behavior Data (Option B)

**Date:** 2026-03-13
**Status:** COMPLETE — failed, 0% SR across all configs

## Motivation

v0.3.2–v0.3.2.2 failed because behavior data was expert-dominated (200 expert demos at 100% SR + 80 target rollouts). This biases the diffuser's prior too high, leaving guidance no room to steer.

SOPE trains exclusively on **medium-quality behavior data** (~50% SR). The medium anchor gives guidance bidirectional range. This experiment tests whether matching SOPE's assumption fixes guidance.

## Setup (Option B)

- **Behavior data**: 80 rollouts from `10demos_epoch20` (52% oracle SR) — **no expert demos**
- **Chunk config**: chunk_size=4, stride=1 → 3,600+ training chunks
- **Diffusion training**: 50 epochs, batch=64, lr=3e-4, x0-prediction
- **BC behavior policy**: BCGaussian trained on same medium-quality data (500 epochs)
- **Normalization**: From behavior data only
- **Target policies**: 200demos_epoch10 (18% SR) and 200demos_epoch40 (90% SR)
- **Guidance sweep**: 13 configs — scales [0, 0.05, 0.1, 0.2, 0.5] × ratios [0, 0.25, 0.5]
- **OPE**: 50 synthetic trajectories per config, T=60, episode-level binary reward (cube_z > 0.84)

## Results

**All configs produced 0% synthetic SR. No guidance setting had any effect.**

| scale | ratio | 200demos_epoch10 (18%) | 200demos_epoch40 (90%) | rho | MAE |
|-------|-------|----------------------|----------------------|-----|-----|
| 0.00 | 0.00 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.05 | 0.00 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.05 | 0.25 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.05 | 0.50 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.10 | 0.00 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.10 | 0.25 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.10 | 0.50 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.20 | 0.00 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.20 | 0.25 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.20 | 0.50 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.50 | 0.00 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.50 | 0.25 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.50 | 0.50 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| **ORACLE** | | **0.18 (18%)** | **0.90 (90%)** | | |

## Root Cause Analysis

The failure is caused by **Known Bug 1: observation recording drops success state**.

The rollout recorder (`rollout.py:462–498`) stores pre-step `obs`, not post-step `next_obs`. On success termination, the final observation where `cube_z > 0.84` is never recorded. This means:

1. **Behavior rollout data** (from 10demos_epoch20, 52% SR) has max cube_z ≈ 0.835 — never reaching the 0.84 threshold
2. **The diffuser is trained exclusively on this data** — it has never seen cube_z > 0.84 in any training example
3. **The diffuser cannot generate success states** because they are entirely out-of-distribution
4. **Guidance cannot help** — it steers actions, but the diffuser's learned state dynamics have a hard ceiling below the success threshold
5. **All synthetic trajectories score 0** regardless of guidance config

### Why v0.3.2.2 had nonzero SR

v0.3.2.2 mixed 200 expert demos into the training data. Expert demos (from the HDF5 file, not from rollout recorder) reach cube_z = 0.864–0.886, giving the diffuser examples of successful lifts. The unguided baseline in v0.3.2.2 achieved 10-30% SR precisely because of these expert examples.

### Why this was predicted but not heeded

The planned experiment (key question 3) hypothesized that "the diffuser doesn't need to see cube_z > 0.84 to generate useful trajectories — guidance does the steering." This is wrong. Guidance steers actions within the learned distribution — it cannot push state dynamics beyond what the diffuser has learned. The diffuser needs to have success states in its training data.

## Comparison to Prior Experiments

| Experiment | Behavior data | Unguided SR | Best guided SR | Best rho |
|---|---|---|---|---|
| v0.3.2.2 | 200 expert + 80 target | 10-30% | 5% (guidance destroys) | 0.775 |
| **v0.3.2.3** | **80 medium only** | **0%** | **0%** | **NaN** |

## Key Finding

**The observation recording bug is a hard blocker for medium-only behavior data.** Without expert demos providing success state examples, the diffuser cannot generate trajectories that cross the success threshold. This bug must be fixed before the SOPE framework can be properly tested on robomimic.

## Next Steps

1. **Fix rollout recorder** (`rollout.py`) to store `next_obs` on success termination — this is now the #1 priority
2. **Re-collect all behavior rollouts** with the fixed recorder
3. **Re-run v0.3.2.3** with corrected data — the experiment design is sound, only the data is broken
4. **Alternative**: Mix a small amount of expert demos into diffuser training data (not BC) to provide success state coverage while keeping behavior policy medium-quality

## Notebook

`experiments/2026-03-13/MVP_v0.3.2.3_medium_behavior_data.ipynb`
