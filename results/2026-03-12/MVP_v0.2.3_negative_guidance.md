# MVP v0.2.3: Negative Behavior Guidance

**Date:** 2026-03-12
**Notebook:** `experiments/2026-03-12/MVP_v0.2.3_negative_guidance.ipynb`
**Status:** Complete
**Builds on:** MVP v0.2.2 (diffusion score guidance — positive only)

---

## Motivation

v0.2 and v0.2.2 only used **positive guidance** — pushing actions toward the target policy.
SOPE's key contribution is **negative behavior guidance**: subtracting the behavior policy's
score to push *away* from the behavior distribution. Without this, the chunk diffuser's strong
prior toward expert-like trajectories (98% success) dominates, and guidance either has no effect
(v0.2.2: all configs ~98% SR) or just degrades trajectories (v0.2: monotonic decline).

The full SOPE guidance is:
```
guide = nabla_a log pi(a|s) - ratio * nabla_a log beta(a|s)
```

## What's New

1. **Diffusion-based behavior policy** — MLP DDPM trained on expert demo (state, action) pairs.
   Score function: `-noise_pred(a, t=1, s) / sigma[1]`. This is a novel choice vs SOPE
   (which uses BC_Gaussian/TD3), but provides the same gradient.

2. **Full SOPE guidance loop** — matches `default_sample_fn()`:
   - `k_guide=2` iterations per denoising step
   - Positive (target) minus negative (behavior) gradient
   - Per-timestep L2 normalization + clamping to [-1, 1]
   - Adaptive cosine scaling over diffusion timesteps
   - Re-normalize -> re-condition -> unnormalize after each guidance update

3. **Hyperparameter sweep** — 11 configs:
   - Unguided baseline
   - Positive-only at scale 0.1, 0.2
   - Full guidance: sweep ratio (0.25, 0.5, 0.75, 1.0) at scale=0.2
   - Full guidance: sweep scale (0.05, 0.1, 0.5, 1.0) at ratio=0.5

## Architecture

### Behavior Policy (DiffusionBehaviorPolicy)
- **Type:** Single-step MLP DDPM over actions conditioned on states
- **Denoiser:** ActionDenoiser(state_dim=19, action_dim=7, hidden=256)
  - Sinusoidal timestep embedding (dim=64) -> MLP
  - Input: concat(noisy_action, timestep_emb, state) -> 3 hidden layers (256) with Mish -> action_dim
- **Diffusion:** 100 timesteps, cosine beta schedule
- **Training:** 1000 epochs on ~9,466 expert demo (state, action) pairs
- **Score:** evaluate at t=1 (near-clean), return `-noise_pred / sigma[1]`

### Target Policy Scorer (RobomimicDiffusionScorer)
- Same as v0.2.2 — wraps robomimic DiffusionPolicyUNet
- Chunk-level evaluation via `grad_log_prob_chunk()`

### Chunk Diffuser
- Same as v0.2.2 — TemporalUnet, predict_epsilon=False, dim_mults=(1,4,8)
- Pre-trained checkpoint from `diffusion_ckpts/mvp_v022_fulldim/`

## Key Differences from v0.2.2

| | v0.2.2 | v0.2.3 |
|---|---|---|
| Positive guidance | Diffusion score | Diffusion score (same) |
| Negative guidance | **None** | Diffusion behavior policy score |
| k_guide | 1 | 2 |
| Adaptive scaling | No | Yes (cosine) |
| Gradient clamping | No | Yes ([-1, 1]) |

## Expected Outcomes

- **If negative guidance works:** OPE estimates should move from ~16 (unguided, 98% SR)
  toward the oracle value of 0.54 (54% SR). The negative term pushes away from expert-like
  trajectories while the positive term pulls toward the target policy.

- **If it doesn't work:** Either the diffusion behavior policy's score function is too noisy
  (MLP DDPM may not model the expert distribution well enough), or the guidance hyperparameters
  need further tuning. This would suggest trying BC_Gaussian as SOPE originally does.

## Results

### Summary

**Negative guidance had no measurable effect.** All 11 configs produced nearly identical results: ~98% synthetic success rate, OPE ≈ 15.5–16.1, relative error ≈ 2770–2878%. The oracle value is 0.54 (54% SR).

### Sweep Results

| Config | Scale | Ratio | OPE | SR | Rel Error |
|--------|-------|-------|-----|-----|-----------|
| unguided | 0.00 | 0.00 | 15.78 | 98.0% | 2822% |
| pos_only_0.1 | 0.10 | 0.00 | 15.60 | 98.0% | 2789% |
| pos_only_0.2 | 0.20 | 0.00 | 15.50 | 98.0% | **2770%** |
| full_0.2_r0.25 | 0.20 | 0.25 | 15.70 | 98.0% | 2807% |
| full_0.2_r0.5 | 0.20 | 0.50 | 15.82 | 98.0% | 2830% |
| full_0.2_r0.75 | 0.20 | 0.75 | 16.02 | 98.0% | 2867% |
| full_0.2_r1.0 | 0.20 | 1.00 | 15.96 | 98.0% | 2856% |
| full_0.05_r0.5 | 0.05 | 0.50 | 15.98 | 98.0% | 2859% |
| full_0.1_r0.5 | 0.10 | 0.50 | 15.74 | 98.0% | 2815% |
| full_0.5_r0.5 | 0.50 | 0.50 | 15.96 | 98.0% | 2856% |
| full_1.0_r0.5 | 1.00 | 0.50 | 16.08 | 98.0% | 2878% |

**Best config:** pos_only_0.2 (rel_error = 2770%), but effectively no different from unguided.

### Oracle

- V^π = 0.5400 ± 0.4984
- Success rate: 54.0%

### Comparison Across Versions

| Version | Approach | Best Rel Error |
|---------|----------|---------------|
| v0.2 | BC_Gaussian proxy, pos only, 15-dim | **25.93%** |
| v0.2.2 | Diffusion score, pos only, 26-dim | 2533% |
| v0.2.3 | Diffusion score, pos+neg, 26-dim | 2770% |

### Analysis

1. **Guidance is effectively invisible.** Sweeping action_scale from 0.05 to 1.0 and ratio from 0 to 1.0 changed OPE by < 0.6 (15.5 → 16.1). The chunk diffuser's learned prior completely dominates.

2. **The core problem persists from v0.2.2:** The diffusion score function (at t=1) produces gradients that are too weak or too noisy relative to the chunk diffuser's denoising trajectory. The guidance signal gets washed out by the 256 denoising steps.

3. **Negative guidance made things slightly worse**, not better — higher ratio pushed OPE further from oracle. This suggests the behavior policy score may be pushing away from useful trajectories rather than from the expert-biased prior.

4. **v0.2 remains the best approach.** The BC_Gaussian analytical gradient (direct mean/std extraction) gives a much stronger and more coherent guidance signal than the diffusion score function. The 25.93% relative error in v0.2 is dramatically better than any diffusion-score-based approach.

### Key Takeaway

The diffusion score function `−ε(a, t=1, s)/σ₁` does not provide usable policy gradients for SOPE-style guidance in this setting. The signal-to-noise ratio is too low. Future work should:
- Return to BC_Gaussian analytical gradients (v0.2 approach) as the baseline
- Investigate whether the score at different timesteps (t > 1) gives stronger gradients
- Consider whether the chunk-level diffusion framework is incompatible with single-step action scores

### Output Files

- `results/2026-03-12/ope_summary_mvp_v023.png` — OPE bar chart + success rates + trajectory comparison
- `results/2026-03-12/traj_states_v023.png` — Per-dimension trajectory comparison (demo vs unguided vs best)
- `results/2026-03-12/traj_cubez_all_v023.png` — Cube z across all 11 guidance configs
- `results/2026-03-12/behavior_policy_loss_v023.png` — Behavior policy training loss curve
- `results/2026-03-12/mvp_v023_results.json` — Full results JSON with per-trajectory returns

---

## How to Run

```bash
bash scripts/submit_notebook.sh experiments/2026-03-12/MVP_v0.2.3_negative_guidance.ipynb
```
