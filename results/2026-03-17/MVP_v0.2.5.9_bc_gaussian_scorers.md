# MVP v0.2.5.9: BC_Gaussian Analytic Scorers for Cross-Policy Guidance

**Date:** 2026-03-13
**Status:** DONE (ran locally, ~6 min total)
**Notebook:** `experiments/2026-03-13/MVP_v0.2.5.9_bc_gaussian_scorers.ipynb`

## Hypothesis

Diffusion policies all produce similar action distributions on Lift, so any scorer trained on their data learns similar score functions (cosine ~0.72). **BC_Gaussian policies** should produce more diverse gradients because:
1. Simpler model class — bad BC_Gaussian (10 demos) has wide shifted Gaussians, good one (200 demos) has tight accurate ones
2. Analytic `grad_log_prob = -(a - mean) / std^2` — zero approximation error
3. Different model class = different failure modes than diffusion policies

## Setup

- 6 BC_Gaussian target policies trained on Lift demo subsets (10–200 demos, 50–200 epochs)
- Architecture: MLP [256, 256], softplus std, ReLU backbone (74K params each)
- Trained on demo HDF5 filter keys (not rollout data)
- Guidance: action_scale=0.01, normalize_grad=True, 50 synthetic trajectories
- No oracle SR available (these are BC_Gaussian, not the diffusion policy checkpoints)

## Phase 1: Cosine Similarity Diagnostic

|                    | 10d_e50 | 10d_e200 | 25d_e100 | 50d_e100 | 100d_e100 | 200d_e100 |
|--------------------|---------|----------|----------|----------|-----------|-----------|
| bc_10demos_e50     | 1.0000  | 0.2397   | 0.2004   | 0.1736   | 0.1891    | 0.1956    |
| bc_10demos_e200    | 0.2397  | 1.0000   | 0.8751   | 0.7333   | 0.7562    | 0.5784    |
| bc_25demos_e100    | 0.2004  | 0.8751   | 1.0000   | 0.7752   | 0.8805    | 0.6731    |
| bc_50demos_e100    | 0.1736  | 0.7333   | 0.7752   | 1.0000   | 0.6295    | 0.4411    |
| bc_100demos_e100   | 0.1891  | 0.7562   | 0.8805   | 0.6295   | 1.0000    | 0.8053    |
| bc_200demos_e100   | 0.1956  | 0.5784   | 0.6731   | 0.4411   | 0.8053    | 1.0000    |

**Mean pairwise cosine: 0.5431** (v0.2.5.6 UNet: 0.7260, v0.2.5.8 MLP: 0.7214)

Improvement over prior methods — but driven almost entirely by `bc_10demos_e50` being an outlier (cos ~0.19 with everything else). That policy barely trained (50 epochs on 511 samples, NLL=2.67 vs -7.93 for 200-demo). Excluding it, the remaining 5 policies still have high cosine (0.44–0.88).

## Phase 2: OPE Results

| Policy             | NLL    | Unguided | Guided | Delta (G-U) |
|--------------------|--------|----------|--------|-------------|
| bc_10demos_e50     |  2.67  |      60% |    52% |         -8% |
| bc_10demos_e200    | -3.43  |      60% |    60% |         +0% |
| bc_25demos_e100    | -3.97  |      60% |    58% |         -2% |
| bc_50demos_e100    | -4.64  |      60% |    76% |        +16% |
| bc_100demos_e100   | -5.76  |      60% |    52% |         -8% |
| bc_200demos_e100   | -7.93  |      60% |    52% |         -8% |

**Spearman rho (NLL vs guided SR) = +0.21 (p=0.686)** — not significant.

Guided SR range: [0.520, 0.760] (spread=0.240), slightly wider than v0.2.5.8 (0.200) but no monotonic relationship with policy quality. The best-guided result (76%, bc_50demos) is in the middle of the NLL ranking, not at the top.

## Verdict: FAIL

BC_Gaussian analytic scorers reduce cosine similarity (0.54 vs 0.72) but do **not** produce meaningful policy ranking. Spearman rho = 0.21 (non-significant). The guidance effect is noisy and non-monotonic.

## Analysis

### What improved
- **Cosine similarity dropped** from 0.72 → 0.54. BC_Gaussian policies are more distinguishable than diffusion policies, especially the undertrained `bc_10demos_e50` (cos ~0.19 with all others).
- **Wider SR spread** (0.24 vs 0.20), suggesting guidance has slightly more power to differentiate.
- **Zero approximation error** — analytic grad_log_prob eliminates scorer training noise as a confound.

### Why it still fails
1. **Lower cosine doesn't mean useful cosine.** The `bc_10demos_e50` outlier (barely trained, high NLL) drives the low mean cosine. The well-trained policies (10d_e200 through 200d_e100) still have cosine 0.44–0.88 — too similar for ranking.
2. **Guidance direction ≠ policy quality signal.** Even with distinct gradients, pushing actions toward a BC_Gaussian's mode doesn't necessarily change trajectory outcomes in a way that correlates with that policy's actual performance. The guidance modifies local action choices, but Lift success depends on a precise sequence of reach-grasp-lift that small action perturbations don't reliably produce.
3. **No oracle SR for BC_Gaussian policies.** We used NLL as a proxy for policy quality, but NLL may not correlate with actual rollout performance (a well-fit Gaussian on 10 demos could still fail at the task).
4. **The bc_50demos_e100 spike (76%) is suspicious.** It's the only policy where guidance helps substantially, and it has middle-of-the-road NLL. This looks like noise, not a real signal.

### Implications for the guidance approach
This experiment eliminates two more hypotheses:
- ~~Scorer architecture matters~~ (v0.2.5.8 showed UNet ≈ MLP)
- ~~Approximation error matters~~ (v0.2.5.9 shows analytic ≈ trained)

The remaining problem is fundamental: **guidance-based OPE requires policies with substantially different action distributions**, and Lift policies (whether diffusion or BC_Gaussian) produce too-similar behavior for 7-dim actions. SOPE works on D4RL locomotion where policies range from random to expert with 17+ dim actions.

## BC_Gaussian Training Details

| Policy             | Transitions | Final Loss | Mean Std | Time |
|--------------------|-------------|------------|----------|------|
| bc_10demos_e50     |         511 |     2.6199 |   0.2882 |   1s |
| bc_10demos_e200    |         511 |    -4.3485 |   0.2627 |   2s |
| bc_25demos_e100    |       1,218 |    -4.2854 |   0.2621 |   4s |
| bc_50demos_e100    |       2,447 |    -4.7689 |   0.2384 |   8s |
| bc_100demos_e100   |       4,898 |    -5.7756 |   0.1975 |  17s |
| bc_200demos_e100   |       9,666 |    -7.7070 |   0.1215 |  33s |

Total training: 64s. Total generation: 320s (5.3 min).

## Per-Action-Dimension Gradient Analysis

Most distinguishable dims: a[2] (min cos=0.20), a[3] (min cos=0.24), a[4] (min cos=0.34).
Least distinguishable dims: a[5] (min cos=0.68), a[6] (min cos=0.70).
The 10demos_e50 policy is always involved in the minimum-cosine pair, confirming it's the outlier driving low mean cosine.
