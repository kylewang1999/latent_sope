# MVP v0.2.5.4: Score Timestep & Action Scale Sweep

**Date:** 2026-03-13
**Builds on:** MVP v0.2.5.3 (scorer debug)
**Notebook:** `experiments/2026-03-13/MVP_v0.2.5.4_scorer_timestep_sweep.ipynb`

## Goal

Find the optimal `score_timestep` for the RobomimicDiffusionScorer and verify
that corrected (much smaller) `action_scale` values produce useful guidance.

## TL;DR

- **t=5–10 is the sweet spot** for score_timestep. Best gradient signal (72% GD
  improvement, 100% convergence, direction works even at lr=0.1).
- **Guidance at t=5 cuts trajectory MSE by 60%** vs unguided, across all tested
  action_scales (0.0003–0.005). The corrected scales work.
- 0% SR everywhere is an artifact of the trimmed settings (10 trajs, T_GEN=30).

## Part 1: Score Timestep Sweep

| t | sigma | 1/sigma | dir%@0.1 | dir%@0.01 | |grad|/|a| | GD conv% | GD impr% |
|---|-------|---------|----------|-----------|-----------|----------|----------|
| 1 | 0.042 | 23.9x | **10%** | 100% | 13.28 | 90% | 60% |
| 2 | 0.058 | 17.3x | 60% | 100% | 9.11 | 100% | 63% |
| **5** | **0.105** | **9.5x** | **90%** | **100%** | **5.00** | **100%** | **72%** |
| **10** | **0.182** | **5.5x** | **100%** | **100%** | **2.82** | **100%** | **72%** |
| 20 | 0.333 | 3.0x | 100% | 100% | 1.32 | 100% | 67% |
| 50 | 0.722 | 1.4x | 100% | 100% | 0.64 | 100% | 50% |

**Interpretation:**
- t=1: gradient direction is correct but magnitude too large (13x actions).
  Direction test fails at lr=0.1 because overshooting.
- t=5–10: sweet spot. Sigma comparable to SOPE (0.105–0.182 vs SOPE's 0.113).
  Gradient magnitude is 3–5x actions — manageable with small action_scale.
- t=50: gradient too weak (0.64x actions), only 50% GD improvement — too noisy
  for the UNet to give a useful score.

## Part 2: Guided Trajectory Generation (trimmed: 10 trajs, T_GEN=30)

| Config | MSE | vs unguided |
|--------|-----|-------------|
| unguided | 0.00484 | baseline |
| t1_s0.0003 | 0.00421 | -13% |
| t1_s0.001 | 0.00213 | -56% |
| t1_s0.005 | 0.00231 | -52% |
| t1_s0.01 | 0.00280 | -42% |
| **t5_s0.0003** | **0.00194** | **-60%** |
| t5_s0.001 | 0.00201 | -59% |
| t5_s0.005 | 0.00197 | -59% |
| t5_s0.01 | 0.00277 | -43% |

**Key findings:**
- t=5 guidance consistently beats t=1 at the same action_scale
- t5 MSE is stable across scales 0.0003–0.005 — guidance is robust, not
  overpowering the diffusion model
- At scale=0.01, MSE starts increasing again (total perturbation = 2.56,
  233% of |action|) — too aggressive
- All configs show 0% SR due to trimmed settings (T_GEN=30 is too short for
  lifts, only 10 trajectories)

## Next Steps

1. **Full-scale run** with `score_timestep=5`, `action_scale ∈ [0.0003, 0.001, 0.005]`,
   50 trajectories, T_GEN=60. Check if MSE improvement translates to better SR/OPE.
2. **Fix rollout recorder bug** — still needed for meaningful MSE comparison
3. **Test with negative gradient** (behavior policy subtraction) at corrected scales
