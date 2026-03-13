# MVP v0.2.5.2: Trajectory MSE — Real vs Synthetic State Comparison

**Date:** 2026-03-13
**Builds on:** MVP v0.2.5 (reward model fix, target+expert diffuser)
**Notebook:** `experiments/2026-03-13/MVP_v0.2.5.2_trajectory_mse.ipynb`

## Goal

Measure how similar the chunk diffusion model's synthetic trajectories are to the
target policy's real trajectories using paired MSE. Each synthetic trajectory is
conditioned on the exact same initial state as the corresponding real rollout.

## Setup

- 50 target rollouts + 200 expert demos → 5611 training chunks
- Chunk diffuser: TemporalUnet (3.7M params), 50 epochs, x0-prediction, 256 diffusion steps
- BC_Gaussian behavior policy: 500 epochs on target data
- Target scorer: RobomimicDiffusionScorer (diffusion policy)
- 10 guidance configs swept (unguided, 3 pos-only, 6 full guidance)
- 50 synthetic trajectories per config, 60 steps each
- Oracle V^pi = 0.5400 (SR = 54%)

## Key Results

### Trajectory MSE by Guidance Config

| Config | State MSE | RMSE | Synth SR |
|--------|-----------|------|----------|
| **pos_only_0.1** (best) | **0.0050** | **0.071** | 24% |
| pos_only_0.05 | 0.0053 | 0.073 | 24% |
| pos_only_0.2 | 0.0053 | 0.073 | 38% |
| unguided | 0.0067 | 0.082 | 60% |
| full_0.05_r0.25 | 0.0069 | 0.083 | 44% |
| full_0.1_r0.25 | 0.0090 | 0.095 | 40% |
| full_0.2_r0.25 | 0.0106 | 0.103 | 28% |
| full_0.5_r0.25 | 0.0155 | 0.125 | 54% |
| full_0.2_r0.5 | 0.0170 | 0.130 | 68% |
| full_0.5_r0.5 | 0.0436 | 0.209 | 26% |

### Initial State Verification

All configs: **MATCH** (max abs diff = 5.96e-08). Synthetic trajectories start
from the exact same states as real rollouts.

### Per-Dimension MSE (Unguided, Worst 5)

| Dimension | MSE | RMSE |
|-----------|-----|------|
| cube_qw | 0.0745 | 0.273 |
| cube_qz | 0.0268 | 0.164 |
| eef_qy | 0.0199 | 0.141 |
| cube_qy | 0.0016 | 0.041 |
| grip2cube_z | 0.0012 | 0.034 |

Quaternion components dominate the error. Position dimensions (cube_xyz, eef_xyz)
have much lower MSE (~0.0001–0.001).

### Training

- Diffuser final loss: 0.106 (50 epochs, 555s)
- BC_Gaussian final NLL: -6.90 (500 epochs)

## Findings

1. **Positive-only guidance gives lowest MSE.** `pos_only_0.1` achieves 25% lower
   state MSE than unguided (0.0050 vs 0.0067). This suggests light positive guidance
   helps the diffuser stay closer to real trajectory dynamics.

2. **Full guidance (with negative/behavior term) increases MSE.** Adding the behavior
   policy negative gradient pushes trajectories further from real data. The stronger
   the guidance scale or ratio, the worse the MSE — `full_0.5_r0.5` has 6.5x the
   MSE of the best config.

3. **MSE and synthetic SR are not straightforwardly correlated.** Unguided has the
   second-highest SR (60%) but moderate MSE. The best-MSE config (`pos_only_0.1`)
   has only 24% SR. Meanwhile `full_0.2_r0.5` has 68% SR but very high MSE (0.017).
   This means guidance can inflate synthetic SR without making trajectories more
   realistic.

4. **Real target SR = 0%** due to the known rollout recorder bug (cube_z never
   reaches 0.84 in recorded data). This means any nonzero synthetic SR comes from
   the diffuser generating states beyond what it saw in training — the expert demos
   (which do reach cube_z > 0.84) are leaking success behavior into synthetic
   trajectories even from target-policy initial states.

5. **Quaternion dimensions dominate the error.** The 4 quaternion components
   (cube_qz, cube_qw, eef_qy, eef_qw) account for the majority of state MSE.
   Position dimensions are well-reconstructed (RMSE < 0.02). This suggests the
   diffuser struggles with rotational dynamics, which could be improved by
   quaternion-aware normalization or separate treatment.

## Comparison to Prior Experiments

- v0.2.5 did not compute trajectory MSE — only OPE relative error
- This experiment shows that even the "best" guidance config for OPE (which
  minimizes relative error to oracle) may not produce the most realistic
  trajectories. MSE and OPE accuracy are different objectives.

## Post-Hoc Analysis: The MSE Comparison Is Fundamentally Broken

After reviewing the outputs, we identified that the trajectory MSE results above
are **not interpretable as intended** due to the rollout recording bug.

### The problem

Real target SR = 0% because the rollout recorder never stores the final success
state (cube_z > 0.84). So the MSE is measured against **truncated/broken**
reference trajectories that never show a successful lift. This inverts the
interpretation:

- **Lower MSE = closer to the broken data**, not closer to the true target behavior
- The "best MSE" config (`pos_only_0.1`, MSE=0.005) has only 24% SR — it's
  matching trajectories that never lift the cube
- **Unguided** has the highest SR (60%, closest to oracle 54%) but higher MSE
  because it *correctly* lifts the cube, diverging from the broken reference

### Guidance is suppressing success, not steering toward it

| Config type | Effect on SR | Effect on MSE | Interpretation |
|-------------|-------------|---------------|----------------|
| pos_only (target grad only) | 60% → 24–38% | Decreases | Pushes toward broken reference |
| full (target − ratio×behavior) | 60% → 28–68% | Increases | Partially recovers lifting via negative term |
| Strong full (0.5_r0.5) | 60% → 26% | 6.5x increase | Destabilizes trajectories entirely |

The target scorer gradients appear to be **suppressing** the lifting behavior
rather than reinforcing it. This needs to be debugged before guidance can be
meaningfully evaluated.

### Why we can't visualize "guidance steering" from this experiment

1. **No separate behavior-only baseline.** The diffuser trains on target+expert
   data, so unguided output is already a mix. There's no "pure behavior policy"
   trajectory to steer *from*.
2. **Broken reference trajectories.** Without correct target rollouts (recording
   bug fixed), trajectory MSE measures similarity to the wrong thing.
3. **Guidance hurts rather than helps.** The target scorer gradients reduce SR,
   so there's no positive steering effect to visualize.

## Next Steps: Debug the Target Scorer Gradients

Before re-running guidance experiments, the `RobomimicDiffusionScorer` gradients
must be validated. Concrete debugging plan:

### 1. Sanity check: does the scorer prefer the policy's own actions?

Take a state from a real rollout. Compute `grad_log_prob` at:
- The action the target policy actually took → gradient should be small (near mode)
- A random action → gradient should be large and point toward the real action
- Test: does `a_rand + lr * grad_rand` move closer to `a_real`?

### 2. Check gradient magnitudes vs action scale

Compare `|grad|` to `|action|`. If the gradient norm is orders of magnitude
larger than action magnitudes, the `action_scale` configs (0.05–0.5) may be
way too large, causing the guidance to overpower the diffusion model.

### 3. Check sigma at score_timestep=1

The scorer divides by `sigma = sqrt(1 - alpha_bar[1])`. If sigma ≈ 0 at t=1,
gradients are amplified massively. Print `target_scorer.sigma` to verify.

### 4. Visualize gradient field on a real trajectory

Plot the gradient vector at each timestep of a real trajectory. If gradients
are smooth and coherent, the scorer is at least internally consistent. If
spiky/noisy, the score extraction is unreliable.

### 5. Check frame-stacking alignment

Verify `observation_horizon`, `prediction_horizon`, `action_start` against
`CHUNK_SIZE=4`. If most UNet positions correspond to padding rather than real
chunk actions, the scores at chunk positions may be garbage.
