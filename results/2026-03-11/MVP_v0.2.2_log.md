# MVP v0.2.2: Diffusion Score Guidance — Results Log

**Date:** 2026-03-11
**Notebook:** `experiments/2026-03-11/MVP_v0.2.2_diffusion_score_guidance.ipynb`
**Plan:** `results/2026-03-11/diffusion_guidance_implementation_plan.md`

---

## Experiment Summary

**Goal:** Replace the BC_Gaussian proxy guidance (v0.2) with the actual target policy's diffusion score function for SOPE guidance. This matches what SOPE does in its reference implementation — the noise prediction network at t=1 gives `nabla_a log pi(a|s)` directly via `-noise_pred / sigma[1]`.

**Key changes from v0.2:**
- Full 19-dim states + 7-dim actions = **26-dim transitions** (v0.2 used 11+4=15)
- Retrained chunk diffusion model on full-dim data
- `RobomimicDiffusionScorer` wraps the target diffusion policy to provide `grad_log_prob()`
- No proxy policy — score comes from the actual target policy

---

## Configuration

| Parameter | Value |
|-----------|-------|
| State dim | 19 (object + eef_pos + eef_quat + gripper_qpos) |
| Action dim | 7 (full robomimic action space) |
| Transition dim | 26 |
| Chunk size | 4 |
| Diffusion steps | 256 |
| UNet dim/mults | 32 / (1, 4, 8) |
| Predict mode | x0 (not epsilon) |
| Training | 10 epochs x 1000 steps, batch=128, lr=3e-4 |
| Data | 200 human demos (8866 chunks) |
| Model params | 3,686,618 |
| Guidance | Diffusion score at t=1 (sigma=0.0418) |
| Gradient normalization | Yes (SOPE default) |
| Synthetic trajectories | 50, horizon=60, gamma=1.0 |

---

## Training

Loss decreased smoothly from 0.318 to 0.027 over 10 epochs. Still decreasing — more epochs could help.

| Epoch | Loss |
|-------|------|
| 1 | 0.3175 |
| 2 | 0.1367 |
| 5 | 0.0503 |
| 10 | 0.0271 |

**Checkpoint saved:** `diffusion_ckpts/mvp_v022_fulldim/`

---

## Oracle

- **Oracle V^pi = 0.54** (50 rollouts, gamma=1.0, undiscounted cumulative reward)
- Success rate: 54%
- std: 0.498

---

## Results: Guidance Scale Sweep

| Scale | OPE Estimate | Success Rate | Rel Error |
|-------|-------------|-------------|-----------|
| unguided | 15.72 | 98% | 2811% |
| 0.01 | 15.30 | 98% | 2733% |
| 0.05 | 13.80 | 96% | 2456% |
| **0.1** | **12.62** | **96%** | **2237%** |
| 0.5 | 16.76 | 100% | 3004% |
| 1.0 | 36.54 | 100% | 6667% |

**Best scale: 0.1** (rel error = 2237%)

---

## Analysis

### 1. Guidance does NOT work in v0.2.2

The results are dramatically worse than v0.2. All OPE estimates are wildly inflated (12.6–36.5 vs oracle 0.54). The relative errors are in the thousands of percent — this is not meaningful OPE.

For comparison, **v0.2 (BC_Gaussian proxy)** achieved:
- Best rel error: **25.93%** (scale=0.1, OPE=0.40 vs oracle=0.54)
- Unguided: 2878% rel error → guided brought it down to 26%

v0.2.2 (diffusion score) shows no comparable improvement from guidance.

### 2. The core problem: synthetic trajectories are too "successful"

The unguided model already produces 98% success rate (vs oracle 54%). This means the chunk diffusion model, trained on 200 human demos (which are all successful), has learned to always generate trajectories where the cube is lifted. It cannot generate failure trajectories.

**Guidance makes this worse, not better.** The diffusion score pushes actions even more toward the target policy's preferred actions, which are successful lifting actions. At scale=1.0, success rate hits 100% and OPE balloons to 36.54.

### 3. Why v0.2 worked but v0.2.2 doesn't

v0.2 used a **BC_Gaussian target policy** (not the diffusion policy). The BC_Gaussian had ~0% success rate on its own, so the "oracle" was effectively the gap between demo behavior and the weak BC_Gaussian. The guidance in v0.2 was steering trajectories *away* from always-succeeding toward the BC_Gaussian's failure mode — which happened to be the right direction.

v0.2.2 uses the **actual diffusion target policy** (54% success rate). The guidance pushes toward this policy's actions, but since the demo data is 100% successful and the diffusion model already generates near-100% success trajectories, there's no room for guidance to correct downward.

### 4. Fundamental issue: demo-only data can't represent the target policy's failures

The chunk diffusion model is trained on 200 human demos — all successful. It has never seen a failure trajectory. So it cannot generate trajectories with 54% success rate. This is a **data coverage problem**, not a guidance problem.

To get meaningful OPE, the diffusion model needs to see both success and failure trajectories. Options:
- Train on **target policy rollouts** (not demos) — these will have the right success/failure mix
- Use a **behavior policy** that generates mixed-quality data
- Add **noise augmentation** to demos to simulate failures

### 5. Scale=0.1 is still "best" but meaningless

The best scale (0.1) slightly reduces the estimate from 15.72 to 12.62 — a modest ~20% reduction. But the estimate is still 23x too high. The guidance is technically pointing in the right direction (lower estimates are closer to 0.54), but the magnitude is nowhere near sufficient to overcome the data coverage gap.

Scale=0.5 and 1.0 are counterproductive — they overshoot and push trajectories to be even more successful.

---

## Comparison: v0.2 vs v0.2.2

| Metric | v0.2 (BC_Gaussian proxy) | v0.2.2 (Diffusion score) |
|--------|--------------------------|--------------------------|
| Target policy | BC_Gaussian (~0% success) | DiffusionPolicy (54% success) |
| State/Action dims | 11+4=15 | 19+7=26 |
| Guidance source | Proxy (BC_Gaussian log_prob) | Actual policy (diffusion score) |
| Best rel error | **25.93%** | 2237% |
| Unguided rel error | 2878% | 2811% |
| Guidance helps? | Yes (2878% → 26%) | No (2811% → 2237%) |

v0.2's success was likely an artifact of evaluating a ~0% success BC_Gaussian target against all-success demos — the guidance could steer toward failure. v0.2.2 evaluates the real 54% target policy, exposing that the core pipeline can't produce calibrated estimates without data coverage of failures.

---

## SLURM Execution

**SLURM Job:** 7326959 (node d14-06, V100-PCIE-32GB)
**Started:** Wed Mar 11 13:40:14 PDT 2026
**Finished:** Wed Mar 11 13:48:31 PDT 2026 (~8 min)
**Status:** Completed successfully — notebook executed and saved in-place.

## Artifacts

- **Results JSON:** `results/2026-03-11/mvp_v022_results.json`
- **OPE summary plot:** `results/2026-03-11/ope_summary_mvp_v022.png`
- **Training loss plot:** `results/2026-03-11/training_loss_v022.png`
- **Trajectory plots:** `results/2026-03-11/traj_states_v022.png`, `traj_actions_v022.png`, `traj_cubez_all_scales_v022.png`
- **Marginal stats:** `results/2026-03-11/marginal_stats_v022.png`
- **Return histograms:** `results/2026-03-11/return_histograms_v022.png`
- **Rollout data:** `results/2026-03-11/rollouts_mvp_v022/`
- **Diffusion checkpoint:** `diffusion_ckpts/mvp_v022_fulldim/`

---

## Next Steps

1. **Train chunk diffusion on target policy rollouts** instead of demos — this gives the model data coverage over both success and failure trajectories
2. **Revisit v0.3 pipeline** which collects rollouts from the target policy (5-50 rollouts) and trains on those
3. Consider hybrid data: demos + target policy rollouts to improve coverage
4. The diffusion score extraction (`RobomimicDiffusionScorer`) works correctly — the issue is purely data coverage
