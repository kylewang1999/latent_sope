# MVP v0.2.5.3: Target Scorer Gradient Debugging

**Date:** 2026-03-13
**Builds on:** MVP v0.2.5.2 (trajectory MSE analysis)
**Notebook:** `experiments/2026-03-13/MVP_v0.2.5.3_scorer_debug.ipynb`

## Goal

Debug `RobomimicDiffusionScorer.grad_log_prob()` to determine why target policy
guidance in v0.2.5.2 suppressed success rate (60% → 24%) instead of improving it.

## TL;DR

**The scorer's gradient direction is correct, but the magnitude is catastrophic.**

- Gradient descent at lr=0.01 converges 90% of the time (Test 5) — the scorer
  knows where the target policy's actions are
- But raw gradients are 13x larger than actions (|grad|=15 vs |action|=1.1)
- v0.2.5.2's `action_scale=0.05–0.5` was way too large, even with normalized
  gradients, because guidance accumulates over 256 denoising steps
- Root cause: our sigma at t=1 is 4.5x smaller than SOPE's (0.025 vs 0.113),
  amplifying gradients 39.8x instead of 8.85x

## Test Results

| Test | Result | Interpretation |
|------|--------|---------------|
| Sigma check | sigma=0.0418, amplification=24x | Significant but not extreme |
| Direction (lr=0.1) | 20% closer (worse than random) | Overshooting — step too large |
| Direction (lr=0.01) | Works (see GD test) | Direction is correct at small scale |
| |grad|/|action| | 12.86x | Gradients massively larger than actions |
| GD convergence (lr=0.01) | **90% converged, 66% improvement** | **Scorer works at small lr** |

## Why v0.2.5.2 Guidance Failed

Two compounding problems:

### 1. Gradient accumulation over 256 denoising steps

With `normalize_grad=True`, each gradient is unit-norm. But guidance is applied
at every denoising step, so total perturbation ≈ `action_scale × 256`:

| action_scale | Total perturbation | % of |action|≈1.1 |
|---|---|---|
| 0.05 | 12.8 | **1164%** |
| 0.10 | 25.6 | **2327%** |
| 0.50 | 128.0 | **11636%** |

Every config in v0.2.5.2 was massively overpowered.

### 2. Smaller sigma than SOPE → larger gradient amplification

The score formula is `-noise_pred / sigma`. Our sigma is much smaller:

| System | Steps | Schedule | sigma[1] | Amplification |
|--------|-------|----------|----------|--------------|
| SOPE DiffusionPolicy | 32 | linear | 0.113 | 8.85x |
| Our RobomimicScorer | 100 | cosine | 0.025 | 39.8x |

The cosine schedule has much less noise at t=1, and more total steps means t=1
is a smaller fraction of the schedule (closer to clean data where sigma → 0).

## Comparison to SOPE Reference Implementation

SOPE also applies guidance at every denoising step (`default_sample_fn`, line 226:
`guide = action_scale * guide`), so the accumulation issue exists there too. But
SOPE works because the numbers are very different:

| | SOPE | Ours |
|---|---|---|
| Policy scorer | `gradlog()` — autograd on log_prob, no sigma division | `-noise_pred / sigma` |
| Diffusion policy steps | 32 | 100 |
| Noise schedule | linear (beta0=0.1, beta1=20) | squaredcos_cap_v2 |
| sigma[1] | 0.113 | 0.025 |
| Gradient amplification | 8.85x | 39.8x |
| Chunk diffusion steps | 20–200 | 256 |
| Default action_scale | 0.2 | 0.05–0.5 (v0.2.5.2) |

Key differences:
1. **SOPE's `gradlog()` uses `torch.autograd.grad` on the actual log probability**
   (policy.py line 71). No sigma division, so gradients are naturally smaller.
   Our scorer uses the score function `-noise_pred / sigma`, which amplifies by
   1/sigma = 39.8x at t=1.
2. **SOPE's diffusion policy uses only 32 steps** with a linear schedule.
   At t=1/32, sigma is already 0.113 — much noisier than our t=1/100 with cosine.
3. **Fewer chunk diffusion steps** means less accumulation per chunk.

The net effect: SOPE's effective guidance strength per chunk is roughly
`0.2 × 100_steps × 8.85x_amp = 177`, while ours was
`0.05 × 256_steps × 39.8x_amp = 509` — about 3x stronger even at the smallest
scale, and that's *before* accounting for gradient normalization differences.

## Fix (tested in v0.2.5.4)

Two options:
1. **Use `score_timestep=5`** → sigma=0.089, comparable to SOPE's 0.113
2. **Reduce `action_scale` to 0.0003–0.001** → total perturbation = 8–26% of |action|
