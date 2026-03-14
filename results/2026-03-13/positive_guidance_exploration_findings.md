# Positive Guidance Exploration: Findings from v0.2.5.2 → v0.2.5.4

**Date:** 2026-03-13
**Notebooks:** v0.2.5.2 (trajectory MSE), v0.2.5.3 (scorer debug), v0.2.5.4 (timestep sweep)

## Summary

Across three experiments, we identified and fixed the reason guidance was failing:
the gradient scale was catastrophically large due to small sigma and accumulation
over 256 denoising steps. With corrected settings (`score_timestep=5`,
`action_scale=0.0003–0.005`), guidance cuts trajectory MSE by 60%.

---

## v0.2.5.2: Something is wrong with guidance

Ran a guidance sweep with `action_scale ∈ [0.05, 0.5]`, `score_timestep=1`:

| Config | SR | State MSE |
|--------|-----|-----------|
| **unguided** | **60%** | 0.0067 |
| pos_only_0.1 | 24% | 0.0050 |
| full_0.2_r0.5 | 68% | 0.0170 |
| full_0.5_r0.5 | 26% | 0.0436 |

- Unguided was already close to oracle (54% SR)
- **Positive guidance suppressed SR** from 60% to 24–38% — the opposite of expected
- Stronger guidance = worse results
- The MSE comparison was also misleading: real target rollouts have 0% SR
  (recording bug), so "lower MSE" meant "closer to broken data"

**Question:** Is the scorer broken, or is the scale wrong?

## v0.2.5.3: The scorer works — the scale is catastrophic

Five diagnostic tests on `RobomimicDiffusionScorer.grad_log_prob()`:

| Test | Result | Meaning |
|------|--------|---------|
| GD convergence (lr=0.01) | **90% converged, 66% improvement** | Scorer direction is correct |
| Direction (lr=0.1) | 20% closer (worse than random) | Overshoots at large step |
| \|grad\|/\|action\| | 12.86x | Gradients 13x larger than actions |

**Root cause — two compounding problems:**

### 1. Gradient accumulation over 256 denoising steps

With `normalize_grad=True`, each gradient is unit-norm. But guidance is applied
at every denoising step, so total perturbation ≈ `action_scale × 256`:

| action_scale | Total perturbation | % of \|action\|≈1.1 |
|---|---|---|
| 0.05 | 12.8 | 1164% |
| 0.10 | 25.6 | 2327% |
| 0.50 | 128.0 | 11636% |

Every config in v0.2.5.2 was massively overpowered.

### 2. Our sigma is 4.5x smaller than SOPE's

The score formula is `-noise_pred / sigma`. Our sigma at t=1 amplifies much more:

| System | Steps | Schedule | sigma[1] | Amplification |
|--------|-------|----------|----------|--------------|
| SOPE DiffusionPolicy | 32 | linear | 0.113 | 8.85x |
| Our RobomimicScorer | 100 | cosine | 0.025 | 39.8x |

The cosine schedule has much less noise at t=1, and 100 steps (vs 32) means t=1
is closer to the clean end where sigma → 0.

### Why v0.2.5.2's accidental "best" config worked

`full_0.2_r0.5` had 68% SR — the highest. This is because the negative behavior
term (`-0.5 × behavior_grad`) partially cancelled the oversized positive term,
accidentally producing a smaller net perturbation. It wasn't guidance working —
it was two wrongs partially cancelling.

## v0.2.5.4: Find the right settings

### Part 1: Score timestep sweep

Tested `score_timestep ∈ [1, 2, 5, 10, 20, 50]`:

| t | sigma | dir%@lr=0.1 | GD convergence | GD improvement |
|---|-------|-------------|----------------|----------------|
| 1 | 0.042 | **10%** | 90% | 60% |
| 2 | 0.058 | 60% | 100% | 63% |
| **5** | **0.105** | **90%** | **100%** | **72%** |
| **10** | **0.182** | **100%** | **100%** | **72%** |
| 20 | 0.333 | 100% | 100% | 67% |
| 50 | 0.722 | 100% | 100% | 50% |

- **t=5–10 is the sweet spot.** Sigma comparable to SOPE (0.105–0.182 vs 0.113),
  100% GD convergence, best 72% improvement.
- t=1: too amplified (direction test fails at large lr)
- t=50: too noisy (gradient loses signal, only 50% improvement)

### Part 2: Corrected guidance at full scale

With `score_timestep=5`, `action_scale ∈ [0.0003, 0.01]`, 10 trajs, T_GEN=30:

| Config | MSE | vs unguided |
|--------|-----|-------------|
| unguided | 0.00484 | baseline |
| **t5_s0.0003** | **0.00194** | **-60%** |
| t5_s0.001 | 0.00201 | -59% |
| t5_s0.005 | 0.00197 | -59% |
| t5_s0.01 | 0.00277 | -43% |

- **Guidance cuts MSE by 60%** — trajectories are substantially more realistic
- t=5 consistently beats t=1 at the same scale
- Results stable across scales 0.0003–0.005 (not overpowering anymore)
- At scale=0.01 MSE starts rising again (total perturbation = 233% of \|action\|)
- 0% SR everywhere due to trimmed settings (10 trajs, T_GEN=30)

## Current Status

**v0.2.5.5 (running):** Full-scale re-run with corrected settings:
- `score_timestep=5`, `action_scale ∈ [0.0003, 0.001, 0.005]`
- 50 trajectories, T_GEN=60
- Both pos-only and full guidance (with behavior subtraction)
- OPE evaluation against oracle

Will determine if the MSE improvement translates to better SR and OPE estimates.
