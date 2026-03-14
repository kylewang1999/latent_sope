# MVP v0.2.5.6: Cross-Policy Guidance — Prove Guidance Effectiveness

**Date:** 2026-03-13
**Notebook:** `experiments/2026-03-13/MVP_v0.2.5.6_cross_policy_guidance.ipynb`
**Builds on:** v0.2.5.5 (guidance settings corrected, but target ≈ behavior made it inconclusive)
**Runtime:** 10.8 min (SLURM, V100)

## Goal

Prove that positive guidance carries target policy information by evaluating 6 different target policies using a single fixed diffuser (trained on 54% SR behavior data). If guidance works, guided OPE should correlate with oracle SR while unguided OPE stays flat.

## Settings

- Diffuser: EMA model from `diffusion_ckpts/mvp_v0252_traj_mse` (trained on test checkpoint data, ~54% SR)
- Guidance: `score_timestep=5`, `action_scale=0.001`, pos-only (no behavior subtraction)
- 50 trajectories per config, T_GEN=60
- Same initial states for all configs (from test checkpoint rollouts)
- Same random seed (42) for unguided and each guided run

## Results

| Policy | Oracle SR | Unguided OPE | Guided OPE | Δ(G-U) |
|--------|-----------|-------------|------------|--------|
| 10demos_epoch10 | 8% | 60% | 58% | -2% |
| 100demos_epoch20 | 42% | 60% | 62% | +2% |
| test_checkpoint | 54% | 60% | 62% | +2% |
| 10demos_epoch30 | 62% | 60% | 58% | -2% |
| 50demos_epoch30 | 82% | 60% | 62% | +2% |
| 200demos_epoch40 | 90% | 60% | 58% | -2% |

**Spearman ρ (guided vs oracle): -0.10** (p=0.85)
**Spearman ρ (unguided vs oracle): NaN** (constant — perfectly flat)

**RESULT: FAIL** — Guidance does not differentiate policies.

## Analysis

1. **Unguided is perfectly flat at 60%** — This is the correct sanity check. The diffuser produces the same output regardless of which target policy we intend to evaluate. The diffuser's learned distribution determines SR, not the target.

2. **Guided varies only ±2% (58–62%)** — The guidance perturbation is essentially noise. There is no monotonic relationship between oracle SR and guided OPE. The 8% oracle policy gets 58% guided OPE; the 90% oracle policy also gets 58%.

3. **Guidance signal is too weak** — At `action_scale=0.001` with 256 denoising steps, the total perturbation per action dimension is `0.001 × 256 = 0.256` (normalized). While v0.2.5.3 showed the scorer gradients point in the right direction, the accumulated perturbation isn't enough to overcome the diffuser's strong prior over the learned trajectory distribution.

4. **The diffuser's prior dominates** — The diffuser was trained on ~250 trajectories (50 rollouts + 200 expert demos) and has learned a strong distribution. Guidance at this scale is a small nudge on top of a powerful generative model. The model's denoising steps quickly correct the small perturbations back toward its learned distribution.

5. **Why v0.2.5.4 showed 60% MSE improvement but this fails** — v0.2.5.4 measured MSE against the *same policy's* rollouts. Guidance toward the behavior policy (= diffuser's training data) reinforces what the model already learned, reducing reconstruction noise. But it doesn't change the *distribution* — it just makes individual trajectories cleaner. Cross-policy guidance requires shifting the distribution, which needs much stronger perturbation.

## Implications

- **Guidance at this scale is cosmetic** — it reduces noise but doesn't shift the distribution
- **Stronger scales risk instability** — v0.2.5.2 showed that scales >0.01 blow up trajectories
- **Fundamental tension**: the scale needed to shift distributions (~10x current) is the same scale that destabilizes denoising
- **Possible fixes**:
  - Only apply guidance in final N denoising steps (not all 256)
  - Use classifier-free guidance approach instead of score-based
  - Train the diffuser on weaker/mixed data so its prior is easier to override
  - Use SOPE's original approach: guide at sample time with a separate reward model, not policy scores

## Comparison to Prior Experiments

| Version | Key Finding |
|---------|-------------|
| v0.2.5.2 | Guidance scale was catastrophically large, broke trajectories |
| v0.2.5.3 | Scorer direction is correct (90% GD convergence at t=5) |
| v0.2.5.4 | t=5 sweet spot, scale 0.0003–0.005 reduces MSE by 60% |
| v0.2.5.5 | Same-policy guidance: marginal improvement (11% → 7.4% OPE error) |
| **v0.2.5.6** | **Cross-policy guidance: no effect (ρ = -0.10). Guidance doesn't differentiate policies.** |
