# Positive Guidance Implementation (in v0.2.5)

**Date:** 2026-03-13
**Notebook:** `experiments/2026-03-12/MVP_v0.2.5_reward_model_fix.ipynb`
**Source:** `src/latent_sope/robomimic_interface/guidance.py`

## Overview

Positive guidance steers diffusion-generated trajectory chunks toward the target policy π(a|s) by adding ∇_a log π(a|s) to the denoised model mean at each diffusion step. The implementation wraps a robomimic DiffusionPolicyUNet (temporal ConditionalUnet1D) to extract scores, unlike SOPE's reference which uses a single-step MLP (PearceMlp).

## Data Flow Summary

```
Input: states (B, T, 19), actions (B, T, 7) from diffusion model_mean (unnormalized)
  │
  ├─ _build_obs_conditioning(states)
  │    ├─ Take first To=2 states from chunk (frame stacking)
  │    ├─ Decode flat state → obs dict via _states_to_obs_dict_batched
  │    │    └─ Splits 19-dim vector into: object(10), eef_pos(3), eef_quat(4), gripper_qpos(2)
  │    ├─ Run through robomimic's ObservationGroupEncoder (with GroupNorm, not BatchNorm)
  │    └─ Flatten → obs_cond (B, To * obs_features)
  │
  ├─ _build_action_sequence(actions)
  │    ├─ Place T chunk actions at positions [To-1, To-1+T) in Tp=16 sequence
  │    ├─ Repeat-pad before/after with nearest chunk action (avoids OOD zero-padding)
  │    └─ Output: action_seq (B, Tp=16, action_dim=7)
  │
  ├─ UNet forward pass at score_timestep=1 (near-clean)
  │    └─ noise_pred = noise_pred_net(sample=action_seq, timestep=1, global_cond=obs_cond)
  │
  └─ Score extraction at chunk positions
       └─ scores = -noise_pred[:, start:end, :] / sigma[1]
           where sigma[1] = sqrt(1 - alpha_bar[1])
```

## Files Involved

| File | Key Lines | Role |
|------|-----------|------|
| `src/latent_sope/robomimic_interface/guidance.py` | 48–322 | `RobomimicDiffusionScorer`: all scoring methods |
| `src/latent_sope/robomimic_interface/encoders.py` | 70–155 | `LowDimConcatEncoder`: state ↔ obs dict conversion |
| `src/latent_sope/robomimic_interface/checkpoints.py` | 265–325 | `build_algo_from_checkpoint`: constructs the algo the scorer wraps |
| `third_party/robomimic/robomimic/algo/diffusion_policy.py` | 52–121 | `DiffusionPolicyUNet._create_networks()`: builds obs_encoder, noise_pred_net, noise_scheduler |
| `third_party/robomimic/robomimic/models/diffusion_policy_nets.py` | — | `ConditionalUnet1D`: temporal U-Net with FiLM conditioning |
| `diffusers` (external) | — | `DDPMScheduler`: provides alphas_cumprod for sigma computation |

## Detailed Trace

### 1. Scorer Construction (`guidance.py:66–117`)

`RobomimicDiffusionScorer.__init__` extracts from the algo object:
- `algo.nets["policy"]["obs_encoder"]` — the robomimic obs encoder (ObservationGroupEncoder with GroupNorm)
- `algo.nets["policy"]["noise_pred_net"]` — the ConditionalUnet1D
- `algo.noise_scheduler` — DDPMScheduler (1000 timesteps, squaredcos_cap_v2 beta schedule)
- `algo.algo_config.horizon.prediction_horizon` (Tp=16) and `.observation_horizon` (To=2)
- `algo.ac_dim` (7 for Lift)

It then initializes a `LowDimConcatEncoder` with obs_shapes/obs_dims from the algo, precomputes `sigma[1]`, and sets `_action_start = To - 1 = 1`.

The algo itself is created by `build_algo_from_checkpoint` (`checkpoints.py:265–325`), which calls `algo_factory("diffusion_policy", ...)` → `DiffusionPolicyUNet`, then loads weights via `model.deserialize(ckpt_dict["model"])`.

### 2. Score Computation (`guidance.py:255–301`)

`grad_log_prob_chunk(states, actions)` is the primary method:

1. **Obs conditioning** (`_build_obs_conditioning`, lines 181–211): Takes first `To` states from chunk. If chunk is shorter than `To`, pads by repeating the first state. Decodes flat 19-dim states to obs dict via `_states_to_obs_dict_batched` (lines 133–161), which splits by sorted obs key order. Runs through `obs_encoder` time-distributed (via `robomimic.utils.tensor_utils.time_distributed`), flattens to `(B, To * obs_features)`.

2. **Action sequence** (`_build_action_sequence`, lines 213–253): Creates a `(B, 16, 7)` tensor. Chunk actions placed at positions `[1, 1+T_chunk)`. Position 0 repeat-padded with first chunk action. Positions after the chunk repeat-padded with last chunk action.

3. **UNet forward** (lines 284–287): `noise_pred_net(sample=action_seq, timestep=1, global_cond=obs_cond)` → `(B, 16, 7)` noise prediction.

4. **Score extraction** (lines 291–301): `scores = -noise_pred[:, 1:1+T, :] / sigma[1]`. This uses the score-noise relationship: at small t, `∇_a log p(a|s) ≈ -ε_pred / σ[t]`. If chunk is longer than available positions `(Tp - start)`, pads with zeros.

### 3. Single-step delegation (`guidance.py:303–322`)

`grad_log_prob(states, actions)` for `(N, D)` inputs just unsqueezes a time dim, calls `grad_log_prob_chunk`, and squeezes back.

### 4. Application in Guidance Loop (notebook lines 1300–1357)

At each denoising step:
```
model_mean = unnormalize(model_mean)              # to obs space
states_chunk = model_mean[:, :, :19]
actions_chunk = model_mean[:, :, 19:]
target_grad = target_scorer.grad_log_prob_chunk(states_chunk, actions_chunk)  # (B,T,7)
target_grad = target_grad / (target_grad.norm(dim=-1, keepdim=True) + 1e-6)  # L2 normalize per timestep
guide = zeros_like(model_mean)
guide[:, :, 19:] = action_scale * target_grad     # only modify actions
model_mean = model_mean + guide                   # apply in unnormalized space
model_mean = normalize(model_mean)                # back to normalized
model_mean = apply_conditioning(model_mean, cond)  # re-pin conditioned states
model_mean = unnormalize(model_mean)              # for next k_guide iteration
```

After the `k_guide` loop, a final `normalize(model_mean)` before adding diffusion noise.

## Verification Against SOPE Reference

Checked against `third_party/sope/opelab/core/baselines/diffusion/diffusion.py`:
- **`gradlog_diffusion` (lines 91–110)**: SOPE's version for diffusion policies — calls `policy.grad_log_prob(states, actions)` analytically, returns `(N,T,D)` with zeros for states. Our scorer does the same thing via `RobomimicDiffusionScorer.grad_log_prob_chunk`.
- **`default_sample_fn` (lines 152–250)**: The unnormalize → guide → normalize → condition → unnormalize cycle matches exactly. Guide applied to model_mean in unnormalized space, same as notebook.
- **`p_sample_loop` (lines 387–439)**: Post-noise `apply_conditioning` at line 425 matches notebook line 1363.

## Key Design Decisions

1. **score_timestep=1** (not 0): t=0 is fully clean (σ=0, degenerate). t=1 is near-clean with numerical stability.
2. **Repeat-padding** (not zero-padding) for action sequences: The temporal UNet expects coherent multi-step sequences. Zero-padding is OOD.
3. **Frame stacking uses actual chunk states**: First `To` states from the chunk provide real temporal context, unlike duplicating a single state.
4. **L2 normalization per timestep**: Matches SOPE's `normalize_v = not clamp and normalize_grad` when `clamp=False`.
5. **No adaptive schedule** (`use_adaptive=False`): SOPE disables cosine decay for diffusion policy targets.

## Known Limitation

The scorer evaluates the target policy on partially-denoised `model_mean` states. Early in diffusion (high t), these states are noisy and far from real observations — policy scores may be unreliable. SOPE has the same limitation. The disabled `use_adaptive` cosine schedule was SOPE's mitigation for this.
