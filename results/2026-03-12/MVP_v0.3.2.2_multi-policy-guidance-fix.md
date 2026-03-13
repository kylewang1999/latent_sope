# MVP v0.3.2.2: Multi-Policy Guidance Fix (Sweep)

**Date:** 2026-03-12
**Notebook:** `experiments/2026-03-12/MVP_v0.3.2.2_multi-policy-guidance-fix.ipynb`
**Status:** COMPLETED — 26.4 min sweep + 3 min training = ~30 min total (ran locally on login node GPU)
**Result:** FAIL — guidance kills SR across all configs; best ranking (ρ=0.775) is driven by a single trajectory

## What was tried

Hyperparameter sweep over guidance strength to find a config that works for multi-policy OPE. Same shared diffuser/BC architecture as v0.3.2.1, with 10 guidance configs × 4 policies = 40 evaluations.

### Sweep grid
- **action_scale:** [0.0, 0.05, 0.1, 0.2]
- **ratio:** [0.0, 0.1, 0.25]
- 10 configs (excluding scale=0.0 with nonzero ratio)

### Setup
- Same 4 target policies: 50demos_epoch10 (0%), 200demos_epoch10 (18%), 10demos_epoch20 (52%), 200demos_epoch40 (90%)
- Shared diffuser trained on 280 episodes (200 expert + 80 target), 25 epochs, final loss=0.162
- Shared BC_Gaussian on same data, final NLL=-6.25
- 20 synthetic trajectories per policy per config, T=60
- 4 target scorers pre-loaded and reused across configs

## Key Metrics

### Full sweep results

| Scale | Ratio | 0% SR | 18% SR | 52% SR | 90% SR | Spearman ρ | MAE |
|-------|-------|-------|--------|--------|--------|-----------|-----|
| **0.00** | **0.00** | **30%** | **25%** | **25%** | **10%** | **-0.95** | **0.360** |
| 0.05 | 0.00 | 0% | 10% | 0% | 0% | -0.26 | 0.375 |
| 0.05 | 0.10 | 0% | 0% | 0% | 0% | NaN | 0.400 |
| **0.05** | **0.25** | **0%** | **0%** | **0%** | **5%** | **0.775** | **0.387** |
| 0.10 | 0.00 | 0% | 5% | 0% | 0% | -0.26 | 0.387 |
| 0.10 | 0.10 | 5% | 0% | 0% | 0% | -0.78 | 0.413 |
| 0.10 | 0.25 | 5% | 0% | 0% | 5% | 0.00 | 0.400 |
| 0.20 | 0.00 | 0% | 0% | 0% | 0% | NaN | 0.400 |
| 0.20 | 0.10 | 0% | 0% | 5% | 0% | 0.26 | 0.387 |
| 0.20 | 0.25 | 10% | 0% | 5% | 0% | -0.63 | 0.413 |

### Best config: scale=0.05, ratio=0.25
- Spearman ρ = 0.775 (only because 90% policy got 1/20 success, all others got 0/20)
- MAE = 0.387
- Per-policy: OPE = [0.00, 0.00, 0.00, 0.05] vs Oracle = [0.00, 0.18, 0.52, 0.90]

## Comparison to prior experiments

| Version | Key change | Best ρ | Best MAE | Unguided SR | Guided SR range |
|---------|-----------|--------|----------|-------------|-----------------|
| v0.2.5 | Single policy (54%), sweep | N/A (1 policy) | 0.06 | 76% | 0–76% |
| v0.3.2 | Multi-policy, per-policy retrain | 0.40 | 0.288 | N/A | 0–60% |
| v0.3.2.1 | Shared diffuser, fixed guidance | NaN | 0.400 | 60% | 0% (all) |
| **v0.3.2.2** | **Shared diffuser, guidance sweep** | **0.775** | **0.360** | **30%** | **0–10%** |

## Analysis

### Why guidance destroys SR regardless of hyperparameters

1. **Any positive guidance kills SR.** Even the gentlest setting (scale=0.05, ratio=0.0) drops the unguided 30%/25%/25%/10% to 0%/10%/0%/0%. Positive guidance steers toward the target policy's score function, but these policies range from 0% to 90% SR — steering toward a mediocre policy pulls the diffuser away from the expert-like base distribution.

2. **Negative guidance makes it worse.** Adding ratio>0 pushes away from the BC trained on 71% expert data, further destroying the lifting pattern.

3. **The unguided diffuser has the wrong ranking.** scale=0.0 gives the lowest MAE (0.360) but ρ=-0.95 (perfectly inverted). The 0% oracle policy gets 30% synthetic SR while the 90% policy gets 10%. Without guidance, the diffuser generates from its prior which is dominated by expert demos — all policies look similar.

### Why is the unguided base SR so much lower than v0.3.2.1?

v0.3.2.1 unguided SR was 60% (12/20), but v0.3.2.2 unguided averages ~22% (30%/25%/25%/10%). Both trained on the same data (200 expert + 80 target) with the same architecture and similar final loss (0.160 vs 0.162).

The cause: **v0.3.2.2 retrained the diffuser from scratch with a different random seed.** Even with similar training loss, diffusion models can converge to different local minima with very different generation quality. The [30%, 25%, 25%, 10%] per-policy spread is just noise from 20 binary trials (standard error ±10%), but the overall average being 22% vs 60% reflects a worse-quality diffuser.

This reinforces the importance of **saving and reusing checkpoints** (now fixed — shared diffuser saved at `diffusion_ckpts/mvp_v032_shared/`). Future experiments should load the saved diffuser rather than retraining, unless the training data changes.

### Root cause: binary reward + few trajectories + recording bug

The results are dominated by noise. With 20 trajectories and binary reward (cube_z > 0.84):
- OPE estimates are quantized to multiples of 5%
- A single lucky/unlucky trajectory flips the ranking
- The observation recording bug means target rollouts never reach 0.84, so the diffuser doesn't learn policy-specific success patterns
- The "best" ρ=0.775 is an artifact of exactly 1 trajectory succeeding for the 90% policy

### What we learned about guidance hyperparameters

Consistent with v0.2.5 findings and the SOPE paper:
- Positive guidance >> negative guidance works best (ratio < 1.0)
- But even minimal guidance (scale=0.05) is too strong for our expert-heavy data mix
- The fundamental issue isn't the hyperparameters — it's the combination of expert-dominated behavior data, binary sparse reward, observation recording bug, and insufficient trajectory count

### Possible next steps

1. **Lower reward threshold** (e.g., 0.835 instead of 0.84) — analysis shows 0.835 already discriminates policies in the recorded data (0/20 vs 11/20 for 0% vs 90% policy)
2. **Continuous reward** (e.g., max cube_z) — gives richer signal than binary
3. **Fix observation recording bug** — store next_obs so cube_z > 0.84 is reachable in recorded data
4. **More synthetic trajectories** (100+) — reduce quantization noise from binary reward
5. **Train BC only on target rollouts** — so negative guidance pushes away from target behavior, not expert behavior
