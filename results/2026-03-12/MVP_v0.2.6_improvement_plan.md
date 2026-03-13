# MVP v0.2.6: Diagnosing and Improving OPE Relative Error

**Date:** 2026-03-12
**Builds on:** MVP v0.2.5 (reward model fix, best rel_error=11.11%)

## Current State

Best configs from v0.2.5 (`pos_only_0.05` and `full_0.5_r0.25`) achieve 11.11% relative error — OPE estimates of 0.48 and 0.60 vs oracle 0.54. The unguided diffuser overestimates at 76% SR due to training data imbalance.

## Root Causes Identified

### 1. Training Data Imbalance (biggest lever)

Diffuser trains on 200 expert demos (100% SR) + 50 target rollouts (0% SR in stored data) — 80% expert data. The unguided diffuser generates 76% SR trajectories, heavily biased toward expert behavior. Guidance fights this prior to bring the estimate down.

Potential fixes:
- **Upsample target rollouts** — duplicate or oversample target data to get closer to 50/50 mix, or use weighted loss giving target chunks higher weight
- **Train on target data only** — expert demos may hurt more than help if the goal is modeling the target policy's trajectory distribution (needs more epochs since fewer chunks)
- **Collect more target rollouts** — 200 target + 200 expert for a balanced mix

### 2. 0% Target Rollout SR Bug

Oracle says 54% SR, but stored `.h5` rollouts show 0% SR when evaluated with `cube_z > 0.84`. The diffuser has never seen a successful target trajectory — it only knows success from expert demos. This fundamentally limits guidance effectiveness because the model can't interpolate between target success and target failure modes.

Fix: Investigate why `.h5` rollout files don't contain the lift event. Likely stored trajectories terminate before `cube_z > 0.84` is reached, or there's a frame indexing issue.

### 3. Low Synthetic Trajectory Count

`NUM_SYNTHETIC_TRAJS = 50` with binary reward (0 or 1) means OPE estimate = `mean(successes)`, quantized to multiples of 1/50 = 0.02. Closest possible value to oracle 0.54 is exactly 0.54 (27/50), but variance is high.

Fix: Increase to 200–500 synthetic trajectories to reduce variance and quantization noise.

### 4. Chunk Size and Stitching Drift

`CHUNK_SIZE = 4` requires ~15 stitch iterations for a 60-step trajectory. Each stitch introduces autoregressive drift via conditioning on the previous chunk's last state.

Fix: Try `CHUNK_SIZE = 8` or `16` — fewer stitching steps, less error accumulation. Requires more training data per chunk, pairs well with collecting more rollouts.

### 5. Light Training

50 epochs on ~500 chunks (batch size 64 → ~8 batches/epoch) is very light. The model may not have fully converged.

Fix: Train longer (200+ epochs) and/or increase data. Monitor training loss to confirm convergence.

## Recommended Priority Order

1. **Fix the 0% target SR bug** — likely corrupting the entire data pipeline
2. **Rebalance training data** (or train on target-only)
3. **More synthetic trajectories** (cheap, reduces variance)
4. **More training epochs** (cheap if data exists)
5. **Larger chunk size** (requires retraining)

## Note on v0.3.2 Multi-Policy Test

The pending v0.3.2 experiment will be informative — if Spearman rank correlation is high even with moderate per-policy error, the pipeline is working directionally and the task is tightening per-policy estimates.

## Key Metrics Reference (v0.2.5)

| Config | OPE | Rel Error | Notes |
|--------|-----|-----------|-------|
| unguided | 0.76 | 40.74% | Expert-biased prior |
| pos_only_0.05 | 0.48 | **11.11%** | Best (underestimates) |
| full_0.5_r0.25 | 0.60 | **11.11%** | Best (overestimates) |
| full_0.5_r0.5 | 0.00 | 100.00% | Guidance too strong |
