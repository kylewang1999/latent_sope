# MVP v0.2.5.8: Small Diffusion MLP Scorers for Cross-Policy Guidance

**Date:** 2026-03-13
**Status:** DONE (ran locally on P100, ~10 min total)
**Notebook:** `experiments/2026-03-13/MVP_v0.2.5.8_small_mlp_scorers.ipynb`

## Setup

- 6 target policies (oracle SR range: 8%–90%)
- Loaded 80–100 existing rollouts per policy from disk (~3600–5900 transitions each)
- Trained small diffusion MLP per policy (256 hidden, 64 emb, 32 diff steps, 5000 train steps)
- Guided generation: action_scale=0.01, normalize_grad=True, 50 synthetic trajectories
- Swapped `50demos_epoch30` (no rollouts) for `100demos_epoch40` (76% oracle SR)

## Pairwise Cosine Similarity (Key Diagnostic)

|                    | 10d_e10 | 100d_e20 | test   | 10d_e30 | 100d_e40 | 200d_e40 |
|--------------------|---------|----------|--------|---------|----------|----------|
| 10demos_epoch10    | 1.0000  | 0.6573   | 0.6623 | 0.4746  | 0.6954   | 0.7805   |
| 100demos_epoch20   | 0.6573  | 1.0000   | 0.8192 | 0.6791  | 0.8659   | 0.7527   |
| test_checkpoint    | 0.6623  | 0.8192   | 1.0000 | 0.6854  | 0.8148   | 0.7633   |
| 10demos_epoch30    | 0.4746  | 0.6791   | 0.6854 | 1.0000  | 0.6696   | 0.5762   |
| 100demos_epoch40   | 0.6954  | 0.8659   | 0.8148 | 0.6696  | 1.0000   | 0.9239   |
| 200demos_epoch40   | 0.7805  | 0.7527   | 0.7633 | 0.5762  | 0.9239   | 1.0000   |

**Mean pairwise cosine: 0.7214** (v0.2.5.6 UNet scorers: 0.7260)

No improvement — small MLPs produce essentially the same gradient similarity as the 65M-param UNet scorers.

## OPE Results

| Policy             | Oracle | Unguided | Guided | Delta (G-U) |
|--------------------|--------|----------|--------|-------------|
| 10demos_epoch10    |     8% |      60% |    58% |         -2% |
| 100demos_epoch20   |    42% |      60% |    66% |         +6% |
| test_checkpoint    |    54% |      60% |    52% |         -8% |
| 10demos_epoch30    |    62% |      60% |    72% |        +12% |
| 100demos_epoch40   |    76% |      60% |    64% |         +4% |
| 200demos_epoch40   |    90% |      60% |    60% |         +0% |

**Spearman rho = +0.20 (p=0.704)** — not significant.
Guided OPE range: [0.520, 0.720] vs unguided baseline 0.600.

## Verdict: FAIL

Small MLP scorers do not differentiate policies. Guidance does not rank policies.

## What Pairwise Cosine Similarity Shows

For guidance to rank policies, each policy's scorer must push actions in a **different direction**. The diagnostic works as follows:

1. Take a batch of test (state, action) chunks from unguided generation
2. For each of the 6 small MLP scorers, compute `grad_log_prob(state, action)` — the direction guidance would push actions to look more like that policy
3. For every pair of scorers, compute cosine similarity between their gradient vectors

- **1.0** = two scorers push in the exact same direction → guidance can't tell the policies apart
- **0.0** = orthogonal, independent directions
- **Negative** = opposite directions

We wanted low cosine sim (<0.3), meaning each scorer gives a distinct "shift actions this way to match my policy" signal. We got **0.72** — all scorers say roughly the same thing, so guided trajectories are nearly identical regardless of target policy.

The key finding: UNet scorers (v0.2.5.6) had mean cosine 0.7260, small MLPs got 0.7214. No change. The problem is not model architecture — the **underlying action distributions across policies are genuinely similar** on the Lift task. Any model trained to approximate them learns similar score functions. In SOPE's D4RL experiments, their 9 policies range from random to expert on locomotion (17+ dim actions) — those policies behave very differently, so trained MLPs naturally learn distinct score functions.

## Analysis

The core problem is **not** the scorer architecture (UNet vs MLP). The pairwise cosine similarity is nearly identical (0.7214 vs 0.7260), which means the score functions are similar regardless of model size.

Likely reasons:
1. **Lift task is too simple** — all policies produce similar (state, action) distributions, so MLPs trained on different policies' data converge to similar functions
2. **Not enough data differentiation** — ~5000 transitions per policy vs SOPE's 1M. But this alone wouldn't explain identical cosine sim to UNets
3. **7-dim action space** — limited room for distributional differences compared to D4RL locomotion (17-dim+)
4. **All policies are diffusion policies trained on the same task** — unlike D4RL where policies range from random to expert with fundamentally different behaviors

## MLP Training Details

| Policy             | Transitions | Final Loss | Time |
|--------------------|-------------|------------|------|
| 10demos_epoch10    |       5,921 |   0.192001 |  37s |
| 100demos_epoch20   |       5,407 |   0.237327 |  37s |
| test_checkpoint    |       2,675 |   0.186153 |  37s |
| 10demos_epoch30    |       5,213 |   0.148311 |  37s |
| 100demos_epoch40   |       4,595 |   0.205771 |  37s |
| 200demos_epoch40   |       3,649 |   0.191321 |  37s |

Total training: 220s. Total generation: 320s.
