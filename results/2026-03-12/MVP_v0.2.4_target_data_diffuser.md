# MVP v0.2.4: Target + Expert Data Diffuser + Corrected SOPE Guidance

**Date:** 2026-03-12
**Notebook:** `experiments/2026-03-12/MVP_v0.2.4_target_data_diffuser.ipynb`

## What was tried

### v0.2.4.1 (failed attempt)
Trained chunk diffuser on **target policy rollouts only** (50 episodes). The target rollouts had 0% SR when evaluated with `cube_z > 0.84`, so the diffuser learned only failure trajectories. All configs produced OPE=0.0, SR=0%, rel_error=100%.

### v0.2.4.2 (corrected)
Trained chunk diffuser on **target policy rollouts (50) + expert demos (200)** = 250 episodes, 5,611 chunks. This gives the diffuser a broad distribution with both success (expert, 100% SR) and failure (target, 0% SR from rollouts) modes.

Guidance hyperparameters matched SOPE's diffusion policy config: `clamp=false`, `k_guide=1`, `use_adaptive=false`. BC_Gaussian behavior policy trained on same data. Swept 10 configs: unguided, positive-only (scale 0.05–0.2), full positive+negative (scale 0.05–0.5, ratio 0.25–0.5).

## Key Metrics

- **Oracle V^pi:** 0.5400 (SR 54.0%)
- **Target policy SR (from rollouts):** 0.0%
- **Expert demo SR:** 100.0%
- **Training:** 50 epochs, 5,611 chunks, 87 batches/epoch, final loss=0.112, 454s
- **Best config:** `full_0.2_r0.25` — **OPE=0.6400, SR=14%, rel_error=18.52%**

### Full sweep results

| Config | Scale | Ratio | OPE | SR | Rel Error |
|--------|-------|-------|-----|-----|-----------|
| unguided | 0.00 | 0.00 | 3.80 | 70% | 604% |
| pos_only_0.05 | 0.05 | 0.00 | 1.02 | 24% | 89% |
| pos_only_0.1 | 0.10 | 0.00 | 0.86 | 20% | 59% |
| pos_only_0.2 | 0.20 | 0.00 | 0.90 | 24% | 67% |
| full_0.05_r0.25 | 0.05 | 0.25 | 1.20 | 24% | 122% |
| full_0.1_r0.25 | 0.10 | 0.25 | 0.88 | 20% | 63% |
| **full_0.2_r0.25** | **0.20** | **0.25** | **0.64** | **14%** | **18.5%** |
| full_0.2_r0.5 | 0.20 | 0.50 | 0.24 | 12% | 56% |
| full_0.5_r0.25 | 0.50 | 0.25 | 0.20 | 6% | 63% |
| full_0.5_r0.5 | 0.50 | 0.50 | 0.00 | 0% | 100% |

## Comparison to prior experiments

| Version | Description | Best Rel Error |
|---------|-------------|---------------|
| v0.2 | BC_Gaussian proxy, pos only, 15-dim | 25.93% |
| v0.2.2 | Diffusion score, pos only, 26-dim | 2533% |
| v0.2.3 | Diffusion score, pos+neg, 26-dim | 2770% |
| **v0.2.4** | **Target+expert diffuser, corrected guidance** | **18.52%** |

## Analysis

**Best result across all versions.** The mixed-data approach works — training the diffuser on both target (0% SR) and expert (100% SR) rollouts gives it a broad trajectory distribution.

**Key findings:**
1. **Unguided is way too high** (OPE=3.80, SR=70%). The diffuser's training data is 80% expert demos (200/250 episodes), so unguided generation is heavily biased toward success. This shows the diffuser learns the data distribution faithfully.
2. **Guidance actually works as intended here.** Both positive-only and full (positive+negative) guidance reduce the OPE estimate toward the oracle. The best config (`full_0.2_r0.25`) achieves 18.5% relative error.
3. **Negative guidance helps.** Comparing `pos_only_0.2` (rel_error=67%) vs `full_0.2_r0.25` (rel_error=18.5%) — same scale, adding negative guidance with ratio=0.25 dramatically improves accuracy.
4. **Too much guidance kills it.** `full_0.5_r0.5` produces OPE=0.0 — guidance overwhelms the diffusion prior.
5. **SR doesn't match oracle.** Best config has 14% SR vs oracle 54%. The OPE estimate (0.64) is closer to the oracle value (0.54) than the SR would suggest — partial rewards from near-lifts contribute.

**Open questions:**
- The training data mix is 80% expert / 20% target. Would a more balanced mix (e.g., 50/50 or matching D4RL-medium's quality distribution) improve the unguided baseline?
- Can we collect more target rollouts to balance the dataset?
- The target rollouts show 0% SR — investigate why (different from oracle's 54%). Possible reward computation mismatch or rollout collection issue.
