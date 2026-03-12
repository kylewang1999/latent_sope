# MVP v0.2.4: Target Data Diffuser + Corrected SOPE Guidance

**Date:** 2026-03-12
**Notebook:** `experiments/2026-03-12/MVP_v0.2.4_target_data_diffuser.ipynb`

## What was tried

Trained the chunk diffuser on **target policy rollouts** (50 episodes, pre-collected March 9) instead of expert demos (98% SR). The hypothesis was that SOPE's diffuser is trained on mixed-quality data (D4RL-medium), so training on expert data creates too strong a prior for guidance to overcome. Also corrected guidance hyperparameters to match SOPE's diffusion policy config: `clamp=false`, `k_guide=1`, `use_adaptive=false`.

Swept 10 guidance configs: unguided, positive-only (scale 0.05–0.2), and full positive+negative (scale 0.05–0.5, ratio 0.25–0.5).

## Key Metrics

- **Oracle V^pi:** 0.5400 (SR 54.0%)
- **Target policy SR from loaded rollouts:** 0.0% (!)
- **All guidance configs:** OPE=0.0000, SR=0.0%, relative error=100%
- **Training:** 50 epochs, 1230 chunks, final loss=0.225, 105s
- **BC behavior policy:** final NLL=-8.04

## Comparison to prior experiments

| Version | Description | Best Rel Error |
|---------|-------------|---------------|
| v0.2 | BC_Gaussian proxy, pos only, 15-dim | 25.93% |
| v0.2.2 | Diffusion score, pos only, 26-dim | 2533% |
| v0.2.3 | Diffusion score, pos+neg, 26-dim | 2770% |
| **v0.2.4** | Target data diffuser, corrected guidance | **100.00%** |

## Analysis

**The experiment failed due to a data issue, not a modeling issue.**

The 50 pre-collected target policy rollouts had **0% success rate** when evaluated with the `cube_z > 0.84` threshold, even though the oracle (same checkpoint, 50 rollouts) shows 54% SR. This means:

1. The diffuser was trained entirely on failure trajectories — it faithfully learned the failure distribution
2. All generated trajectories also fail, giving OPE=0.0 across the board
3. Guidance cannot fix this — there's no success signal in the training data at all

**Root cause investigation needed:**
- Were the 50 rollouts collected from a different (worse) checkpoint?
- Is there a mismatch in how success is computed? (reward field in .h5 vs cube_z threshold)
- The oracle used 50 fresh rollouts at eval time — maybe the pre-collected rollouts from March 9 used different env settings or a stale checkpoint

**Next steps:**
- Verify the pre-collected rollouts: check cube_z distribution, compare to oracle rollouts
- Re-collect rollouts from the correct checkpoint with verified success tracking
- If rollouts are correct but SR=0%, the target policy may be stochastic and 50 rollouts too few — increase to 200+
