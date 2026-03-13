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
