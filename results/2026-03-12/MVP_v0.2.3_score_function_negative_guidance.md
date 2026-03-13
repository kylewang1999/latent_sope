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

The diffusion score function `−ε(a, t=1, s)/σ₁` does not provide usable policy gradients for SOPE-style guidance in this setting. The signal-to-noise ratio is too low.

---

## Root Cause Analysis (Post-Mortem)

Comparing our setup with SOPE's actual diffusion policy experiments (in `main_diffusion.py` and `opelab/examples/diffusion_policy/config/`) reveals **three critical differences** that likely explain why guidance has zero effect:

### 1. Chunk diffuser trained on expert data (ours) vs medium data (SOPE)

**This is the #1 issue.** SOPE trains its chunk diffuser on D4RL medium-quality offline data — mixed trajectories with moderate success rates. The prior is broad and guidance only needs to make small adjustments between policies of similar quality.

Our chunk diffuser is trained on **expert rollouts** (98% SR). The prior is extremely concentrated on high-quality trajectories. Guidance would need to drag the distribution from 98% SR down to 54% SR — a massive shift that no reasonable gradient signal can accomplish within 256 denoising steps.

**Lesson:** Guidance is designed for fine-tuning, not large distribution shifts. The chunk diffuser's training data determines the "center" of the distribution, and guidance nudges samples within that distribution. Training on expert data and guiding toward 54% SR is like using classifier guidance to turn a cat into a truck.

### 2. Wrong guidance settings — v0.2.3 doesn't match SOPE's diffusion policy config

SOPE uses **different hyperparameters** for diffusion target policies vs SAC policies:

| Setting | SOPE (diffusion target) | Our v0.2.3 |
|---------|------------------------|------------|
| clamp | **false** | true ([-1,1]) — kills gradients |
| k_guide | **1** | 2 |
| use_adaptive | **false** | true (cosine) — attenuates early steps |
| action_scale | 0.05–0.5 (task-specific) | 0.05–1.0 (swept) |
| ratio | 0.25–0.5 | 0.25–1.0 (swept) |

Clamping to [-1,1] and adaptive cosine scaling both **reduce** guidance strength. SOPE explicitly disables these for diffusion policies (while enabling clamp=true for SAC policies), suggesting diffusion scores are already well-behaved and don't need dampening.

### 3. Behavior policy type mismatch

SOPE uses **D4RLPolicy (Gaussian MLP from dataset metadata)** as the behavior policy for negative guidance — even when the target policy is a diffusion model (`main_diffusion.py` line 49: `behavior_policy = D4RLPolicy(env_name).to(device)`). We trained a separate **diffusion behavior policy** (ActionDenoiser MLP DDPM, 1000 epochs).

SOPE's behavior policy has an analytical gradient path: MLP forward → mean/std → Gaussian log-prob → autograd. Our diffusion behavior policy uses the same t=1 score trick, meaning both the positive and negative terms use noisy score approximations. The two noisy signals may partially cancel or produce incoherent gradients.

---

## Proposed Next Steps (v0.2.4)

Ordered by expected impact and effort. Each is an independent hypothesis.

### Experiment A: Train chunk diffuser on target policy data (HIGH impact, MODERATE effort)

**Hypothesis:** If the chunk diffuser's prior is closer to the target policy's trajectory distribution, guidance only needs fine adjustments.

- Collect 50–200 rollouts from the target policy (54% SR diffusion policy)
- Train chunk diffuser on this data instead of expert data
- Unguided stitching should already produce OPE closer to 0.54
- Then apply guidance to fine-tune

This is exactly what SOPE does — their chunk diffuser is trained on D4RL-medium (the behavior data), not expert data.

**Tradeoff:** If target policy data is too narrow (all ~54% SR), the diffuser may overfit. Could mix with expert data (50/50) to keep diversity.

### Experiment B: Fix guidance hyperparameters to match SOPE (HIGH impact, LOW effort)

**Hypothesis:** Clamping and adaptive scaling are killing the gradient signal.

- Set `clamp=false`, `use_adaptive=false`, `k_guide=1` (match SOPE diffusion policy config)
- Re-run scale/ratio sweep with corrected settings

SOPE explicitly disables clamping for diffusion policies. Our clamping to [-1,1] after L2 normalization may be truncating already-weak diffusion score gradients to near-zero.

### Experiment C: Use Gaussian behavior policy for negative guidance (MODERATE impact, LOW effort)

**Hypothesis:** Negative guidance should use a clean analytical gradient (Gaussian MLP), not a noisy diffusion score. Two noisy scores cancel incoherently.

- Reuse BC_Gaussian from v0.2 as behavior policy for negative guidance
- Keep diffusion score for positive (target) term
- This matches SOPE's actual setup: diffusion target + Gaussian behavior

### Experiment D: Much larger guidance scales (MODERATE impact, LOW effort)

**Hypothesis:** With the expert-biased prior, we need 10–100x larger scales than SOPE uses.

- Sweep action_scale = [5, 10, 20, 50, 100] with clamp=false
- SOPE uses 0.05–0.5 because their prior is much weaker

**Risk:** If gradient direction is wrong (noisy score), amplifying it just amplifies noise. Run after Experiment B.

### Experiment E: Score at multiple/higher diffusion timesteps (LOW impact, MODERATE effort)

**Hypothesis:** Score at t=1 may be too sharp/noisy for multimodal diffusion policies. Higher t gives smoother gradients.

- Try t=5, t=10, t=20 (out of 100 behavior policy steps / 32 target policy steps)
- Or average scores across t=[1,2,3,5]

SOPE only uses t=1. Speculative, but worth exploring if other fixes don't fully resolve.

### Experiment F: Guide only at late denoising steps (LOW impact, LOW effort)

**Hypothesis:** Guidance applied early (high noise) gets overwritten by subsequent denoising. Late-step guidance sticks.

- Apply guidance only for the last N denoising steps (e.g., last 50 of 256)

SOPE guides at all steps, but their weaker prior means early guidance isn't fighting a strong attractor.

### Experiment G: SMC / importance resampling (MODERATE impact, HIGH effort)

**Hypothesis:** Instead of gradient guidance, generate many unguided candidates and resample by policy likelihood.

- Generate B=256+ candidate chunks (unguided)
- Score each under target diffusion policy
- Resample proportional to policy score

Sidesteps the gradient magnitude problem entirely. Fallback if gradient guidance can't work with expert prior.

### Recommended Order

1. **B** (fix hyperparameters) — quick, removes known bugs
2. **C** (Gaussian behavior policy) — matches SOPE's actual setup
3. **A** (train on target data) — addresses root cause, most likely to work
4. **D** (larger scales) — cheap to try after B
5. **E** (multi-timestep scores) — speculative
6. **F** (late-step guidance) — easy to try alongside others
7. **G** (SMC resampling) — fallback, different paradigm

**Best first experiment:** B + C together (fix hyperparameters + Gaussian behavior policy). Minimal code changes, aligns with SOPE's actual diffusion policy setup. If the prior is still too strong, then A (retrain chunk diffuser) is the real fix.

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
