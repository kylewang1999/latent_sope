# MVP v0.2.5.12 — D4RL Diagnostics (Quick)

**Date:** 2026-03-19 (job 7473302, ~2.5 min runtime)
**Notebook:** `experiments/2026-03-17/MVP_v0.2.5.12_d4rl_diagnostics_quick.ipynb`
**Status:** SUCCESS

## Summary

Quick diagnostic comparing 6 cross-dataset and 6 within-dataset BC_Gaussian policies on D4RL Hopper. No guided generation or OPE — pure policy diagnostics to understand why guidance fails. Companion to v0.2.5.11 (which does the actual OPE).

## Setup

- **Cross-dataset (6):** random, medium, medium-replay, medium-expert, expert, full-replay
- **Within-dataset (6):** med_500, med_1k, med_5k, med_10k, med_50k, med_100k (subsets of medium)
- **Reference:** SOPE's 11 SAC policies on Hopper

## Diagnostic Results

### 1. Pairwise Cosine Similarity of Scorer Gradients

| | Cross | Within | SAC Ref |
|---|-------|--------|---------|
| Mean | 0.469 | **0.710** | 0.474 |
| Min | 0.277 | 0.515 | 0.202 |
| Max | 0.778 | 0.940 | 0.773 |

Within-dataset policies have much more aligned gradients (0.71 vs 0.47), confirming they learn nearly identical scoring functions.

### 2. |grad|/|action| Ratio

| | Cross | Within | SAC Ref |
|---|-------|--------|---------|
| Mean | 18.4 | 6.6 | **572.7** |

BC_Gaussian gradients are 30–90x weaker than SAC. Expert policy is the strongest BC scorer (68.0) but still far below SAC.

### 3. Gradient Direction Test (50 steps, lr=0.1)

| Policy | Cross | | Policy | Within |
|--------|-------|-|--------|--------|
| random | **-40.6%** (converges) | | med_500 | **-54.1%** (converges) |
| medium | +515.8% | | med_1k | **-57.7%** (converges) |
| medium-replay | +64.5% | | med_5k | +91.7% |
| medium-expert | +504.4% | | med_10k | +150.6% |
| expert | +539.5% | | med_50k | +326.5% |
| full-replay | +74.7% | | med_100k | +472.3% |
| **Mean** | **+276.4%** | | **Mean** | **+154.9%** |
| SAC ref | **-26.3%** | | | |

Critical finding: gradient ascent on BC log-prob **diverges** from real actions for most policies. Only the weakest policies (random, med_500, med_1k) converge. Better-trained policies diverge more severely.

### 4. Unguided Trajectory RMSE

Total RMSE: **1.722** (SAC ref: 1.930 — slightly better). Position dims reconstruct well; velocity dims drift, especially x_vel (RMSE 5.21).

### 5. Action NLL Spread

| | Cross | Within | SAC Ref |
|---|-------|--------|---------|
| NLL spread | 10.3 | **1.7** | 39.9 |

Within-dataset NLL spread is 6x smaller than cross and 23x smaller than SAC, confirming policies are too similar for guidance to differentiate.

## Key Findings

1. **BC_Gaussian gradients diverge**: Gradient ascent pushes actions *away* from real actions for most policies (+155% to +276%), opposite of SAC (-26%). This is the root cause of guidance failure.
2. **Better policies diverge more**: Counterintuitively, stronger policies (more data, more training) produce worse gradient directions. Only the weakest policies converge.
3. **Within-dataset policies are near-identical**: Cosine similarity 0.71, NLL spread 1.7 — guidance has almost no signal to work with.
4. **Gradient magnitude too weak**: 30–90x smaller than SAC reference, compounding the direction problem.
5. **Unguided diffusion is fine**: RMSE 1.72 is comparable to SAC reference, so the diffusion model itself isn't the bottleneck.
