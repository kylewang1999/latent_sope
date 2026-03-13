# MVP v0.2.5: Reward Model Fix (Episode-Level Binary Reward)

**Date:** 2026-03-12
**Notebook:** `experiments/2026-03-12/MVP_v0.2.5_reward_model_fix.ipynb`
**Builds on:** MVP v0.2.4.2

## What was tried

Fixed the reward function used to score synthetic trajectories. The previous `score_trajectories_gt` summed per-step binary rewards over the full trajectory: if `cube_z > 0.84` at 30 of 60 steps, the return was 30.0. But the oracle uses robosuite's **episode-level** reward: 1.0 if the task succeeds at any point, 0.0 otherwise. Oracle returns are in [0, 1] while OPE returns were in [0, 60] — completely different scales.

The fix: `score_trajectories_gt` now returns 1.0 if `cube_z > 0.84` at **any** timestep, 0.0 otherwise. OPE estimates are now directly comparable to oracle values.

Everything else (diffuser architecture, training data, guidance configs) is identical to v0.2.4.2.

## Key Metrics

- **Oracle V^pi:** 0.5400 (SR 54.0%)
- **Best configs:** `pos_only_0.05` and `full_0.5_r0.25` tied — **OPE=0.48/0.60, rel_error=11.11%**

### Full sweep results

| Config | Scale | Ratio | OPE | SR | Rel Error |
|--------|-------|-------|-----|-----|-----------|
| unguided | 0.00 | 0.00 | 0.76 | 76% | 40.74% |
| pos_only_0.05 | 0.05 | 0.00 | 0.48 | 48% | **11.11%** |
| pos_only_0.1 | 0.10 | 0.00 | 0.44 | 44% | 18.52% |
| pos_only_0.2 | 0.20 | 0.00 | 0.34 | 34% | 37.04% |
| full_0.05_r0.25 | 0.05 | 0.25 | 0.74 | 74% | 37.04% |
| full_0.1_r0.25 | 0.10 | 0.25 | 0.76 | 76% | 40.74% |
| full_0.2_r0.25 | 0.20 | 0.25 | 0.64 | 64% | 18.52% |
| full_0.2_r0.5 | 0.20 | 0.50 | 0.22 | 22% | 59.26% |
| full_0.5_r0.25 | 0.50 | 0.25 | 0.60 | 60% | **11.11%** |
| full_0.5_r0.5 | 0.50 | 0.50 | 0.00 | 0% | 100.00% |

## Comparison to v0.2.4

| Version | Reward type | OPE range | Best rel_error | Best config |
|---------|-------------|-----------|----------------|-------------|
| v0.2.4.2 | Per-step sum | [0, 60] | 18.52% | full_0.2_r0.25 |
| v0.2.5 | Episode binary | [0, 1] | 11.11% | pos_only_0.05 / full_0.5_r0.25 |

Note: v0.2.4's 18.52% was a coincidence — the per-step sum happened to land near the oracle for one config, but the scales were fundamentally mismatched. v0.2.5's 11.11% is a true apples-to-apples comparison.

## Analysis

1. **Reward fix works:** OPE estimates are now in [0, 1], directly comparable to oracle. The relative errors are meaningful.

2. **Unguided diffuser overestimates (76% vs 54% oracle):** The training data is 80% expert demos (100% SR) and 20% target rollouts (0% SR). The diffuser is biased toward expert-like trajectories, inflating synthetic success rate.

3. **Positive guidance reduces SR:** Counterintuitive — positive guidance (toward target policy) actually lowers success rate. This makes sense because the target policy only has 54% SR, so steering toward it pulls away from the expert-biased prior. At small scales (0.05) this helps bring the estimate closer to oracle; at larger scales (0.2) it overshoots.

4. **Negative guidance (with positive) restores SR:** Adding behavior policy subtraction partially cancels the positive guidance effect, pushing SR back up. With high ratio (0.5), it overcorrects and kills SR entirely.

5. **The 0% target SR from rollouts is suspicious:** Oracle says the policy has 54% SR, but the loaded rollouts show 0% SR when evaluated with `cube_z > 0.84`. This may be because the rollout `.h5` files store observations that terminate early (before the cube is fully lifted) — the `total_reward` in the `.h5` metadata might show success but the stored trajectory may not contain the lift event. Worth investigating.

## Positive Guidance Implementation Details

The positive guidance uses `RobomimicDiffusionScorer` (in `src/latent_sope/robomimic_interface/guidance.py`) to compute `∇_a log π(a|s)` from the robomimic DiffusionPolicyUNet target policy.

### How it works

At each denoising step, after computing `model_mean` from `p_mean_variance`, the guidance loop:

1. **Unnormalizes** `model_mean` to obs space, splits into `states_chunk (B,T,19)` and `actions_chunk (B,T,7)`.

2. **Obs conditioning**: Takes the first `To` (observation_horizon=2) states from the chunk, decodes to obs dicts via `LowDimConcatEncoder`, runs through robomimic's obs encoder → `(B, To * obs_features)` conditioning vector.

3. **Action sequence**: Places chunk actions into a `Tp=16` prediction horizon at positions `[To-1, To-1+T)`, repeat-padding before/after with nearest chunk action. This is critical — robomimic's ConditionalUnet1D is a sequence model unlike SOPE's single-step MLP.

4. **UNet forward pass** at `score_timestep=1` (near-clean): `noise_pred = noise_pred_net(action_seq, t=1, obs_cond)`.

5. **Score extraction**: `scores = -noise_pred[:, start:end, :] / sigma[1]`, using the relationship `∇_a log p(a|s) ≈ -ε_pred / σ[t]` at small t.

6. **L2 normalization** per timestep (matching SOPE's `normalize_v` when `clamp=False, normalize_grad=True`).

7. **Scaling**: `guide = action_scale * guide` (no adaptive cosine decay, matching SOPE defaults for diffusion policies).

8. **Application**: Added to `model_mean` in unnormalized space, then re-normalized → re-conditioned → un-normalized for next `k_guide` iteration.

### Verified against SOPE reference

The implementation was checked against SOPE's `default_sample_fn` (lines 152–250) and `p_sample_loop` (lines 387–439) in `third_party/sope/opelab/core/baselines/diffusion/diffusion.py`. All of the following match:
- Unnormalize → guide → normalize → condition → unnormalize cycle
- L2 normalization per timestep
- Guide applied only to action dimensions (zeros for states)
- Post-noise conditioning in outer loop
- Noise zeroing at t=0

### Known limitation

The scorer evaluates the target policy on partially-denoised states from `model_mean`. Early in the diffusion process (high t), these states are noisy and far from real observations, so policy scores may not be meaningful. SOPE has the same issue. The `use_adaptive` cosine schedule (disabled here per SOPE defaults for diffusion policies) was SOPE's attempt to mitigate this.

## Next steps

- Investigate the 0% target rollout SR discrepancy (oracle says 54%, rollouts say 0%)
- The unguided overestimation suggests the training data mix matters — try different target:expert ratios
- Consider evaluating across multiple target policies (multi-policy OPE) now that the reward function is correct
