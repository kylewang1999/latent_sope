# MVP v0.2.5.10: Action Distribution Diversity Diagnostics

**Date:** 2026-03-17
**Status:** DONE (ran locally, ~30 min — mostly D4RL training)
**Notebook:** `experiments/2026-03-13/MVP_v0.2.5.10_action_distribution_diagnostics.ipynb`

## Purpose

Before trying more guidance fixes, quantify **how diverse Lift policy action distributions actually are** compared to D4RL (where SOPE guidance works). Four diagnostics, each calibrated against D4RL hopper-{random, medium, expert}.

## Summary Table

| Metric | D4RL cross | D4RL within | Lift |
|--------|-----------|-------------|------|
| Mean KL divergence | **366.97** | 26.73 | 48.46 |
| Mean cosine similarity | **0.336** | 0.661 | 0.575 |
| Normalized W1 | **0.344** | 0.131 | 0.321 |
| Classifier accuracy | **100.0%** | 100.0% | 77.7% |
| *(chance level)* | *33.3%* | *33.3%* | *16.7%* |

**Lift KL / D4RL-cross KL = 0.132x** — Lift policies are ~7.6x less diverse than D4RL cross-dataset.

## Diagnostic 1: Per-State KL Divergence

Analytic KL between BC_Gaussian pairs on 500 test states:

|                  | 10d_e50 | 10d_e200 | 25d_e100 | 50d_e100 | 100d_e100 | 200d_e100 |
|------------------|---------|----------|----------|----------|-----------|-----------|
| bc_10demos_e50   |     0.0 |     64.9 |     98.7 |    124.9 |     143.6 |     201.3 |
| bc_10demos_e200  |     6.8 |      0.0 |      0.9 |      2.0 |       3.5 |      30.1 |
| bc_25demos_e100  |     7.1 |      1.0 |      0.0 |      0.6 |       1.9 |      27.1 |
| bc_50demos_e100  |     7.2 |      1.7 |      0.5 |      0.0 |       0.9 |      19.4 |
| bc_100demos_e100 |     7.5 |      2.1 |      1.1 |      0.6 |       0.0 |       7.1 |
| bc_200demos_e100 |     8.8 |      3.9 |      3.0 |      2.2 |       1.4 |       0.0 |

**Mean pairwise KL: 48.5** — driven by `bc_10demos_e50` outlier (KL 65–201 with all others). Excluding it, the remaining 5 policies have KL 0.6–30 — moderate but asymmetric (KL is much larger toward 200-demo policy due to its tight std ~0.12 vs ~0.26 for others).

## Diagnostic 2: D4RL Calibration

Trained BC_Gaussian on hopper-{random, medium, expert} (1M transitions each) and medium subsets (1k, 10k, 100k).

- **D4RL cross-dataset** (random/medium/expert): mean KL = **366.97**, cosine = **0.336**
- **D4RL within-dataset** (medium subsets): mean KL = 26.73, cosine = 0.661
- **Lift**: mean KL = 48.46, cosine = 0.575

**Key finding:** Lift diversity is between D4RL within-dataset and D4RL cross-dataset. It's closer to D4RL-within (same behavior distribution, different data subsets) than to D4RL-cross (genuinely different policies). This is exactly the regime where guidance struggles — policies are somewhat different but not enough to produce distinct score functions.

## Diagnostic 3: Action Marginal Wasserstein-1

| Setting | Raw W1 | Normalized W1 |
|---------|--------|---------------|
| D4RL cross | 0.2051 | **0.344** |
| D4RL within | 0.0784 | 0.132 |
| Lift | 0.0862 | **0.321** |

**Surprise:** After normalizing by action std, Lift's W1 (0.321) is comparable to D4RL cross (0.344). This suggests the *marginal* action distributions have meaningful spread — the problem isn't marginal overlap but that the policies only differ in a few action dimensions, and those differences don't consistently affect trajectory outcomes.

Lift W1 is driven by `bc_10demos_e50` (W1 ~0.17 with all others). Between well-trained policies: W1 = 0.02–0.08.

## Diagnostic 4: Policy Identity Classifier

MLP classifier (128-128 hidden, 200 epochs) predicting which policy generated (state, action):

- **D4RL cross:** 100.0% accuracy (chance: 33.3%) — trivially separable
- **D4RL within:** 100.0% accuracy (chance: 33.3%) — also separable even within medium subsets
- **Lift:** 77.7% accuracy (chance: 16.7%) — well above chance, policies are distinguishable

Lift classifier accuracy = 73.2% of max above chance. Policies ARE distinguishable in (s,a) space.

*Note: per-class accuracy display has a rendering issue — shows 0% for some classes despite high overall accuracy. Confusion matrices in notebook show the actual class-level breakdown.*

## Verdict: LIKELY INSUFFICIENT

Lift policies are **significantly less diverse than D4RL cross-dataset** (KL 7.6x lower, cosine 1.7x higher) but **not completely indistinguishable** (classifier at 78%, W1 comparable after normalization).

### What this means for guidance-based OPE

1. **The problem is real but not absolute.** Lift policies have measurable differences — a classifier can tell them apart 78% of the time. But these differences are concentrated in the undertrained outlier (`bc_10demos_e50`) and don't scale smoothly with policy quality.

2. **Lift resembles D4RL within-dataset, not cross-dataset.** Our setup (varying demo count on the same task) produces diversity similar to training on different subsets of the same D4RL dataset — not like comparing random/medium/expert policies. SOPE works on cross-dataset diversity.

3. **Normalized W1 is misleadingly optimistic.** Marginal action overlap looks comparable to D4RL, but the *conditional* differences (KL, cosine) are much smaller. The policy differences are in low-variance dimensions that don't drive trajectory outcomes.

4. **Guidance needs ~7x more policy diversity** to match the D4RL regime where it works. Options:
   - Use fundamentally different policy classes (not just more/fewer demos)
   - Target a task with higher-dimensional actions and more diverse policy behavior
   - Abandon guidance-based differentiation for Lift

## D4RL Training Details

| Policy | Transitions | Final Loss | Time |
|--------|-------------|------------|------|
| random | 999,999 | 2.606 | 563s |
| medium | 999,998 | -1.157 | 551s |
| expert | 999,061 | -1.055 | 555s |
| medium_1k | 1,000 | 1.703 | 0s |
| medium_10k | 10,000 | 0.403 | 6s |
| medium_100k | 100,000 | -0.557 | 55s |
