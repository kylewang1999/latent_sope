# MVP v0.2.5.1: OPE with 90% Target Policy (200demos_epoch40)

**Date:** 2026-03-13
**Builds on:** MVP v0.2.5

## Setup

- **Target policy**: 200demos_epoch40 (~90% SR)
- **Oracle**: V^pi = 0.90 (from oracle_eval_all_checkpoints.json, 50 rollouts)
- **Chunk diffuser**: pre-trained from v0.2.5 (trained on 54% SR target + 200 expert demos)
- **Behavior policy**: BC_Gaussian trained on 54% SR rollout data (same as v0.2.5)
- **Data**: 50 target rollouts (54% SR) + 200 expert demos — identical to v0.2.5
- **Only change**: target policy scorer swapped to 200demos_epoch40

## Question

Can positive guidance steer a 76% unguided synthetic SR toward the 90% target policy SR?

## Results

| Config | Scale | Ratio | OPE | SR | Rel Error |
|--------|-------|-------|-----|-----|-----------|
| **unguided** | **0.00** | **0.00** | **0.76** | **76%** | **15.56%** |
| full_0.5_r0.25 | 0.50 | 0.25 | 0.70 | 70% | 22.22% |
| full_0.2_r0.25 | 0.20 | 0.25 | 0.66 | 66% | 26.67% |
| pos_only_0.05 | 0.05 | 0.00 | 0.64 | 64% | 28.89% |
| pos_only_0.1 | 0.10 | 0.00 | 0.48 | 48% | 46.67% |
| full_0.05_r0.25 | 0.05 | 0.25 | 0.46 | 46% | 48.89% |
| full_0.1_r0.25 | 0.10 | 0.25 | 0.46 | 46% | 48.89% |
| pos_only_0.2 | 0.20 | 0.00 | 0.30 | 30% | 66.67% |
| full_0.2_r0.5 | 0.20 | 0.50 | 0.08 | 8% | 91.11% |
| full_0.5_r0.5 | 0.50 | 0.50 | 0.00 | 0% | 100.00% |

**Best: unguided at 15.56% relative error.**

## Analysis

- **Guidance does not help.** Every guided config performs worse than unguided (76% SR).
- Positive guidance (target scorer) pushes SR *down*, not up — opposite of intended effect.
- Higher guidance scale → worse performance. The stronger the guidance, the worse the estimate.
- High negative guidance ratio (0.5) is catastrophic (0–8% SR), consistent with v0.3.2 findings.
- The unguided diffuser naturally produces 76% SR, biased upward by the expert demos in the training mix.

## Why guidance fails

1. **Score magnitude mismatch**: the 200demos_epoch40 diffusion scorer has different score magnitudes than the 54% SR policy the BC was trained on, so the guidance ratio is miscalibrated.
2. **Distribution gap**: the diffuser was trained on 54% SR + expert data — may be too far from the 90% policy's distribution for gradient guidance to bridge.
3. **Observation recording bug**: rollout data never records the final success state (cube_z > 0.84), so the diffuser's learned distribution is truncated near the threshold.

## Comparison to prior experiments

- v0.2.5 (54% SR target, same diffuser): guidance worked, best rel_error improved over unguided
- v0.2.5.1 (90% SR target, same diffuser): guidance fails, unguided is best
- v0.3.2 series: guidance hyperparameters don't generalize across policies

## Conclusion

Positive guidance cannot steer the diffuser from 76% to 90% SR with these hyperparameters. The guidance mechanism appears sensitive to the match between the diffuser's training distribution and the target policy. When the target policy is significantly different from the data the diffuser was trained on, guidance degrades rather than improves the OPE estimate.
