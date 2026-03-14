# How SOPE Actually Does Diffusion Policy Guidance (Section 4.3)

**Date:** 2026-03-13
**Source:** SOPE paper (https://stitch-ope.github.io/paper/paper.pdf), Section 4.3 + Appendix K

## Key Finding

SOPE does NOT use the original D4RL target policies directly as diffusion scorers.
Instead, they train purpose-built small diffusion MLPs as proxies:

1. Take pre-trained D4RL target policies (Gaussian/GMM — analytical)
2. Roll out each policy to generate **1M transitions**
3. Train a **brand new small diffusion MLP** per policy via behavior cloning on those rollouts
4. Use those trained MLPs as the "diffusion policy" targets for guidance

## Their Diffusion Policy Architecture

From Appendix K, Table 12:

| Parameter | Value |
|-----------|-------|
| Architecture | PearceMlp (single-step MLP) |
| Embedding dim | 64 |
| Hidden dim | 256 |
| Diffusion steps | 32 |
| Schedule | Linear |
| EMA rate | 0.9999 |
| Training steps | 10,000 |
| Training data | 1M transitions per policy |
| Batch size | 256 |

Score function: `-ε_pred(a, k=1, s) / σ[1]` where σ[1] = 0.113

## Why Their Gradients Differentiate Policies (and Ours Don't)

Their 9 diffusion policies are small MLPs (256 hidden), each trained on a different
policy's rollouts (1M transitions). Each MLP has genuinely different weights because
it learned from different action distributions. The score function naturally differs.

Our 6 robomimic policies are all ConditionalUnet1D (65M params), same architecture,
same task (Lift), trained on similar data (10-200 demos). At score_timestep=5, the
noise prediction is dominated by the shared architecture's prior, producing cosine
similarity 0.86-0.95 across policies.

| | SOPE | Ours |
|---|---|---|
| Scorer architecture | PearceMlp (small MLP) | ConditionalUnet1D (65M params) |
| Scorer type | Single-step (To=1) | Temporal sequence (Tp=16) |
| Diffusion steps | 32 | 100 |
| Schedule | Linear | Cosine |
| sigma[1] | 0.113 | 0.025 |
| Training data per policy | 1M transitions | N/A (use checkpoint directly) |
| Trained independently? | Yes (BC on each policy's rollouts) | No (same arch, similar data) |
| Cross-policy cosine sim | Presumably low (different data) | 0.86-0.95 (nearly identical) |

## Their Section 4.3 Results

Table 3 from the paper:

| Metric | Env | FQE | DRE | MBR | PGD | SOPE |
|--------|-----|-----|-----|-----|-----|------|
| Rank Corr. | Hopper | 0.35 | 0.35 | 0.68 | 0.45 | **0.81** |
| Rank Corr. | Walker2d | 0.03 | 0.45 | 0.47 | 0.52 | 0.46 |
| Rank Corr. | HalfCheetah | 0.59 | 0.80 | 0.75 | 0.46 | **0.81** |
| Regret@1 | Hopper | 0.06 | 0.41 | 0.18 | <0.01 | **<0.01** |
| Regret@1 | Walker2d | 0.24 | 0.59 | 0.17 | 0.23 | **0.03** |
| Regret@1 | HalfCheetah | <0.01 | <0.01 | 0.03 | 0.02 | 0.02 |

SOPE gets Spearman rho=0.81 on Hopper and HalfCheetah with diffusion policies.
Our cross-policy test (v0.2.5.6) got rho=-0.10.

## Implication for Our Work

We cannot directly use robomimic's UNet score function for cross-policy guidance.
To follow SOPE's actual approach, we would need to either:

1. **Train a small diffusion MLP per target policy** on that policy's rollouts
   (matching what SOPE actually does)
2. **Train a BC-Gaussian per target policy** on that policy's rollouts
   (even simpler, already have the code from v0.2.5.5)
3. **Extract analytical action distributions** from robomimic policies directly

All options require collecting rollouts from each target policy first.
