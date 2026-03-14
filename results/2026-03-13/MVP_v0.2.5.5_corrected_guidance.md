# MVP v0.2.5.5: Corrected Guidance — Full Scale Run

**Date:** 2026-03-13
**Notebook:** `experiments/2026-03-13/MVP_v0.2.5.5_corrected_guidance.ipynb`
**Builds on:** v0.2.5.4 (score_timestep=5 is sweet spot, action_scale 0.0003–0.005 works)
**Runtime:** 17 min (local GPU, interactive session)

## Goal

Full-scale guidance sweep with corrected settings from v0.2.5.4:
- `score_timestep=5` (sigma=0.105, comparable to SOPE's 0.113)
- `action_scale ∈ [0.0003, 0.001, 0.005]`
- Both pos-only (target gradient only) and full (with behavior subtraction, ratio=0.25, 0.5)
- 50 trajectories, T_GEN=60, 9 configs total

## Settings

- Diffuser: EMA model from `mvp_v0252_traj_mse`, chunk_size=4, 256 diffusion steps
- Target scorer: `RobomimicDiffusionScorer`, score_timestep=5, sigma=0.105
- Behavior scorer: `BCGaussian` (2-layer MLP, 256 hidden), trained 500 epochs on target rollout data, NLL=-7.76
- Normalization: pooled from 50 target rollouts + 200 expert demos
- Oracle: V^π = 0.540, SR = 54% (from oracle_50.json)

## Results

| Config | Scale | Ratio | SR | State MSE | OPE | Rel Error |
|--------|-------|-------|----|-----------|-----|-----------|
| unguided | 0.0 | 0.0 | 60% | 0.00673 | 0.600 | 11.1% |
| pos_0.0003 | 0.0003 | 0.0 | 58% | 0.00697 | 0.580 | **7.4%** |
| pos_0.001 | 0.001 | 0.0 | 60% | 0.00592 | 0.600 | 11.1% |
| pos_0.005 | 0.005 | 0.0 | 60% | 0.00637 | 0.600 | 11.1% |
| full_0.0003_r0.25 | 0.0003 | 0.25 | 68% | 0.00589 | 0.680 | 25.9% |
| full_0.001_r0.25 | 0.001 | 0.25 | 68% | 0.00667 | 0.680 | 25.9% |
| full_0.005_r0.25 | 0.005 | 0.25 | 62% | **0.00546** | 0.620 | 14.8% |
| full_0.001_r0.5 | 0.001 | 0.5 | 62% | 0.00570 | 0.620 | 14.8% |
| full_0.005_r0.5 | 0.005 | 0.5 | 58% | 0.00548 | 0.580 | **7.4%** |

**Best OPE:** pos_0.0003 and full_0.005_r0.5 (7.4% relative error)
**Best MSE:** full_0.005_r0.25 (MSE=0.00546)

## Analysis

1. **Unguided baseline is already decent** — 11.1% relative error, 60% SR vs 54% oracle. The diffuser slightly overestimates success rate.

2. **Pos-only guidance is marginal** — The three pos-only configs (0.0003, 0.001, 0.005) produce SR 58–60%, barely different from unguided. The target policy gradient alone doesn't meaningfully steer trajectories.

3. **Full guidance (ratio=0.25) hurts OPE** — Behavior subtraction with ratio=0.25 inflates SR to 68% (25.9% error). The behavior policy gradient pushes trajectories *away* from the behavior distribution, but since behavior ≈ target here (same checkpoint), this just adds noise that biases toward success.

4. **Higher ratio (0.5) recovers accuracy** — Stronger behavior subtraction at ratio=0.5 brings SR back to 58–62%, with best configs matching 7.4% error. The stronger subtraction may be canceling out more of the target gradient, effectively returning closer to unguided.

5. **MSE and OPE don't correlate** — Best MSE (full_0.005_r0.25, 0.00546) has 14.8% OPE error; best OPE configs have higher MSE. Trajectory fidelity ≠ reward accuracy.

6. **All configs overestimate** — Every config produces SR ≥ 58% vs 54% oracle. The diffuser has a systematic positive bias in success rate, likely because:
   - Training data includes expert demos (200) which are all successes
   - The known obs recording bug means target rollout failures look different from real failures

## Comparison to Prior Experiments

| Version | Best OPE Rel Error | Notes |
|---------|-------------------|-------|
| v0.2.5.1 | N/A | Guidance didn't work (wrong scale) |
| v0.2.5.2 | ~11% (unguided only) | Baseline trajectory MSE |
| v0.2.5.3 | N/A | Scorer debug |
| v0.2.5.4 | ~11% (unguided) | Found t=5 sweet spot |
| **v0.2.5.5** | **7.4%** | Marginal improvement from guidance |

## Conclusions

- Guidance provides at best a modest improvement (11.1% → 7.4%) over unguided diffusion for this single-policy OPE setting.
- The systematic overestimation bias is the bigger problem — likely needs fixing the obs recording bug and retraining.
- For the single-policy case, the unguided diffuser may be sufficient. Guidance matters more for multi-policy OPE where you need to distinguish between policies.
