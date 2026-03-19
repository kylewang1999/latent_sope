# Action Diversity Comparison: Hopper D4RL vs Lift Robomimic

**Date:** 2026-03-18
**Notebook:** `experiments/2026-03-17/action_diversity_comparison.ipynb`
**SLURM job:** 7440113 (completed ~3:59am)

## Goal

Determine whether cross-policy action diversity is sufficient for guidance-based multi-policy ranking. If all policies predict similar actions, no scoring function can distinguish them.

## Setup

- **Hopper:** 11 SAC target policies, 500 sampled states from hopper-medium-v2, action_dim=3
- **Lift:** 6 DiffusionPolicyUNet policies (10/50/100/200 demos at e40, plus 200demos at e10/e20), 200 sampled states from demo dataset, action_dim=7

## Results

### Cross-Policy Diversity

| Metric | Hopper D4RL | Lift Robomimic | Ratio (H/L) |
|--------|-------------|----------------|-------------|
| Num policies | 11 | 6 | — |
| Action dim | 3 | 7 | — |
| **Mean normalized cross-policy std** | **0.7453** | **0.4870** | **1.53x** |
| **Mean pairwise L2 (normalized)** | **1.7710** | **2.1517** | **0.82x** |
| Agreement rate (within 0.1σ) | 0.0% | 0.0% | — |

Hopper has 1.53x higher per-state action diversity. Lift has higher pairwise L2 due to higher action dimensionality (7 vs 3).

### Pairwise Distances

- **Hopper:** min 1.076 (P7–P9), max 2.581 (P1–P5), mean 1.771
- **Lift:** min 1.688 (100demos–50demos), max 2.731 (200demos_e10–200demos_e20), mean 2.152

### Within-Policy Stochasticity (Lift, 200demos_e40)

| Metric | Value |
|--------|-------|
| Mean normalized within-policy std | 0.3813 |
| Mean normalized cross-policy std | 0.4771 |
| **Cross/within ratio** | **1.25x** |

Cross-policy variance barely exceeds within-policy stochasticity for Lift. The diversity signal exists but is weak.

### Per-Dimension Notes (Lift)

- Dimension 6 (gripper) has 57% agreement rate — policies mostly agree on gripper actions
- Most diversity is in position-control dimensions

## Key Takeaways

1. **Hopper has substantially more cross-policy diversity than Lift** (1.53x normalized std), explaining why SOPE works well on D4RL Hopper.
2. **Lift's cross-policy signal is weak** — only 1.25x the within-policy noise from DiffusionPolicyUNet stochasticity. Guidance-based ranking will struggle.
3. **Agreement rate is 0% for both** at the 0.1σ threshold, so there is _some_ diversity in both domains.
4. **Implication for our pipeline:** Guidance-based OPE may work for Hopper but faces fundamental challenges for Lift due to (a) lower normalized diversity and (b) stochastic policies whose noise nearly drowns out the cross-policy signal.
5. **Next steps:** Consider alternative scoring approaches for Lift that don't rely on action-level guidance (e.g., state-level value functions, trajectory-level features).
