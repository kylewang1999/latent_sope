# Debug: Why Cross-Policy Guidance Failed (v0.2.5.6)

**Date:** 2026-03-13
**Script:** `experiments/2026-03-13/debug_cross_policy_gradients.py`
**Builds on:** v0.2.5.6 (cross-policy guidance, Spearman rho=-0.10)

## Goal

Determine whether cross-policy guidance failed because:
- **A)** The scorer gives the same gradients for all policies (scorer can't distinguish them)
- **B)** The gradients are different but too weak to shift trajectories

## Method

Took 50 (state, action) chunks from unguided trajectories at t=20 and computed
`grad_log_prob_chunk` from all 6 target policy scorers. Compared gradient direction
(cosine similarity), magnitude, and correlation with oracle SR.

## Results

### Cosine similarity between policy gradients

|  | 8% SR | 42% SR | 54% SR | 62% SR | 82% SR | 90% SR |
|---|---|---|---|---|---|---|
| **8% SR** | 1.00 | 0.59 | 0.61 | 0.59 | 0.62 | 0.59 |
| **42% SR** | | 1.00 | **0.92** | 0.63 | **0.86** | **0.93** |
| **54% SR** | | | 1.00 | 0.61 | **0.86** | **0.95** |
| **62% SR** | | | | 1.00 | 0.61 | 0.62 |
| **82% SR** | | | | | 1.00 | **0.89** |
| **90% SR** | | | | | | 1.00 |

Mean pairwise cosine: 0.726 ± 0.277

### Gradient magnitudes

| Policy | Oracle SR | |grad| mean | |grad| std |
|--------|----------|-------------|------------|
| 10demos_epoch10 | 8% | 8.56 | 3.61 |
| 100demos_epoch20 | 42% | 8.36 | 4.08 |
| test_checkpoint | 54% | 7.38 | 3.57 |
| 10demos_epoch30 | 62% | 13.30 | 4.62 |
| 50demos_epoch30 | 82% | 8.99 | 4.07 |
| 200demos_epoch40 | 90% | 8.64 | 4.78 |

Spearman(oracle_sr, |grad|) = +0.54, p=0.27 (not significant)

### Gradient similarity at different trajectory timepoints

| Time | Cosine(test_ckpt vs 8%SR) |
|------|--------------------------|
| t=5 | 0.787 |
| t=10 | 0.701 |
| t=20 | 0.613 |
| t=30 | 0.699 |
| t=40 | 0.683 |
| t=50 | 0.741 |

### 8% vs 90% gradient difference

- |grad_best - grad_worst| = 7.26
- avg |grad| = 8.60
- Relative difference = 90.3%

## Diagnosis: Partial A — Scorer Can't Distinguish Good Policies

The 42%, 54%, 82%, and 90% SR policies produce nearly identical gradient directions
(cosine 0.86-0.95). The scorer literally points in the same direction for these four
policies. Guidance cannot rank them.

The 8% and 62% SR policies are outliers (cosine ~0.6 with the rest), but this doesn't
correlate with quality — the 62% policy is MORE different from the 90% policy than
the 8% is.

Gradient magnitudes don't correlate with oracle SR either (Spearman=0.54, p=0.27).

## Root Cause

All 6 policies use the same UNet architecture (ConditionalUnet1D, 65M params) trained
on the same task (Lift). At score_timestep=5 (sigma=0.105, near-clean), the noise
prediction is dominated by the shared architectural prior about "what clean Lift
actions look like," not the policy-specific differences in which actions each
policy prefers.

The diffusion score function `-noise_pred / sigma` was designed for denoising, not for
distinguishing between similar policies. SOPE's reference implementation uses
`torch.autograd.grad` on an explicit `log_prob()` from analytical policy distributions
(Gaussian, GMM) — a fundamentally different scorer.

## Why normalize_grad makes it worse

With `normalize_grad=True`, guidance discards gradient magnitude and keeps only
direction. Since direction is 86-95% identical across the good policies, normalization
throws away the only remaining signal (magnitude, which at least has rho=0.54).

## Potential Fixes

1. **Analytical log-prob scorer** — extract mean/std from the diffusion policy
   and compute Gaussian log-prob directly (matches what SOPE does for non-diffusion
   policies)
2. **Train separate BC-Gaussian per target policy** — lightweight MLP scorer that
   captures each policy's action distribution explicitly. Code already exists in
   v0.2.5.5 (`BCGaussian` class).
3. **Higher score_timestep** — at noisier levels (t=20-50) the policies might diverge
   more, but v0.2.5.4 showed gradient quality degrades (50% GD improvement at t=50)
4. **Drop normalize_grad** — let gradient magnitude contribute. Risk: magnitude
   varies per policy causing scale instability.
