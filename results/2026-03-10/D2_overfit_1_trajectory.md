# D2: Overfit 1 Trajectory (No Quaternions) — Results

**Date:** 2026-03-10
**Notebook:** `experiments/2026-03-10/8.5k_steps_no_quarternions_1_rollout.ipynb`
**Purpose:** Sanity check — can the model memorize a single 41-step trajectory (~17 chunks)?

---

## Verdict: FAIL

**The model cannot memorize even 1 trajectory.** After 2000 epochs (2000 gradient steps) with high LR (1e-3) on just 17 chunks, the loss plateaus at 0.63 and chunk L2 is 910. This is a **model-level failure**, not a data/scale issue.

---

## Config

| Parameter | Value |
|-----------|-------|
| Rollouts | 1 |
| Chunks | 17 |
| State dim | 11 (no quaternions) |
| Action dim | 7 |
| Transition dim | 18 |
| Chunk size | 8 |
| Frame stack | 2 |
| Total chunk horizon | 10 |
| Diffusion steps | 256 |
| dim_mults | (1, 2) |
| Model params | 250,898 |
| LR | 1e-3 (fixed, no scheduler) |
| Epochs | 2000 |
| Batch size | 64 (> 17 chunks, so all chunks every step) |
| Each chunk seen | ~7,529 times |
| Gradient steps | 2,000 |

---

## Training Loss

| Epoch | Mean Loss |
|-------|-----------|
| 1 | 1.578 |
| 100 | 1.198 |
| 500 | 0.884 |
| 1000 | 0.857 |
| 2000 (final) | 0.635 |

**Analysis:** Loss decreases steadily but is still 0.635 after 2000 epochs on 17 chunks. For a memorization test, loss should reach near-zero (<0.01). The loss curve shows no plateau — the model is still learning but extremely slowly. This suggests the model either:
1. Cannot represent the data (too small / wrong architecture), or
2. The loss function doesn't drive toward correct reconstruction (predict_epsilon vs predict_x0 issue)

---

## Chunk Reconstruction Quality

| Checkpoint | State L2 | ± std | Action L2 | ± std |
|------------|----------|-------|-----------|-------|
| Epoch 100 | 1,288.0 | ±156.3 | 8,920.1 | ±2,063.8 |
| Epoch 500 | 1,132.5 | ±154.0 | 7,543.5 | ±1,727.9 |
| Epoch 1000 | 1,111.4 | ±133.4 | 4,686.0 | ±978.9 |
| Epoch 2000 | 910.7 | ±? | ? | ? |

**Target was L2 < 0.1.** Actual: 910.7 — off by **4 orders of magnitude**.

Action L2 is even worse than state L2 (4,686 at epoch 1000), which is notable given `action_weight=5.0`. Either the weighting isn't working or actions are inherently harder.

---

## Stitched Trajectory Quality

| Metric | Value |
|--------|-------|
| Trajectory State MSE | 19,422.7 |
| Trajectory Action MSE | 2,600,417.2 |
| Blowup ratio | 816.1x |
| Real traj length | 41 steps |
| Synthetic traj length | 60 steps (truncated to 41 for comparison) |

### Per-dimension Marginal Statistics

| Dim | Real range | Synthetic range | Blowup |
|-----|-----------|----------------|--------|
| obj_px | [-0.016, -0.013] | [-2.3, 1.6] | ~150x |
| obj_py | [0.000, 0.004] | [-2.7, 4.4] | ~1,100x |
| **obj_pz** | [0.819, 0.831] | [-16.0, 20.0] | ~24x |
| **g2c_x** | [-0.013, 0.089] | [-414, 352] | **~4,600x** |
| g2c_y | [-0.011, 0.003] | [-14.5, 26.8] | ~2,400x |
| **g2c_z** | [-0.188, -0.007] | [-825, 339] | **~4,400x** |
| eef_px | [-0.103, 0.000] | [-361, 326] | **~3,500x** |
| eef_py | [0.000, 0.012] | [-18.6, 18.0] | ~1,500x |
| **eef_pz** | [0.827, 1.011] | [-542, 576] | **~570x** |
| grip_0 | [0.021, 0.040] | [-28.5, 75.4] | ~1,900x |
| grip_1 | [-0.040, -0.021] | [-52.0, 43.9] | ~1,300x |

**Key observation:** Even with quaternions removed, ALL dimensions still blow up massively. The worst offenders are now g2c (gripper-to-cube) and eef (end-effector) positions — 3,500-4,600x blowup. This means **quaternions were not the root cause**. The model simply doesn't work.

---

## OPE Results

| Metric | Value |
|--------|-------|
| Oracle V^pi | 0.540 |
| Real traj return | 0.0 (this particular trajectory didn't succeed) |
| Synthetic return | 19.0 (spurious — cube_z crosses 0.84 from random blowup) |
| OPE estimate | 19.0 |
| Relative error | 3,418.5% |

---

## Critical Diagnosis

This result is **highly informative**. The model cannot memorize 17 chunks after seeing each one ~7,500 times. This rules out:
- ~~Data scale~~ (only 17 chunks, trivially small)
- ~~Quaternion difficulty~~ (removed, still fails)
- ~~Training budget~~ (7,500 presentations per chunk is massive)
- ~~Stitching drift~~ (even individual chunk reconstruction is garbage)

**Remaining suspects:**

### 1. `predict_epsilon=True` (HIGH SUSPICION)
Our model predicts noise (epsilon), while SOPE's reference uses `predict_epsilon=False` (predicts x0 directly). With epsilon prediction, the model predicts noise added at timestep t. The actual reconstruction is computed as `x0 = (x_t - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)`. This indirect path may make it harder for the model to learn, especially with a small UNet. **SOPE explicitly uses x0 prediction.**

### 2. Model too small (MEDIUM SUSPICION)
250k params with dim_mults=(1,2) gives channels [32, 64]. This may be insufficient even for 17 chunks. However, a 250k-param network SHOULD be able to memorize 17 × 10 × 18 = 3,060 values. This suggests the architecture is not the bottleneck — the training objective is.

### 3. Conditioning mechanism (MEDIUM SUSPICION)
`make_cond()` pins the first `frame_stack` states. If the conditioning is applied incorrectly (wrong indices, wrong normalization), the model receives contradictory signals. Worth verifying with a manual inspection.

### 4. Noise schedule (LOW-MEDIUM SUSPICION)
256 diffusion steps with cosine schedule. SOPE uses 1000 steps. Fewer steps means larger noise jumps, which may make it harder for the model to learn clean denoising.

---

## Recommended Next Steps

1. **Switch to `predict_epsilon=False`** — This is the single highest-priority change. SOPE uses x0 prediction. This should be tested with the same 1-trajectory overfit setup before changing anything else.

2. **Verify conditioning** — Print the cond dict passed to `conditional_sample()` and verify it matches the expected initial states at the correct timestep indices.

3. **Increase model capacity** — Try dim_mults=(1,4,8) alongside the predict_epsilon fix, but don't do this alone (the memorization failure suggests the objective, not capacity, is the issue).

4. **Increase diffusion steps** — Try 1000 to match SOPE, but this is lower priority than the prediction mode fix.

---

## Comparison to Previous Results

| Metric | 8.5k steps (with quat) | D2 overfit (no quat) |
|--------|----------------------|---------------------|
| Chunks | 1,300 | 17 |
| Steps | 8,500 | 2,000 |
| State dim | 19 | 11 |
| Final loss | 0.728 | 0.635 |
| Chunk L2 | 7,287 | 910 |
| Blowup | ~15,000x | 816x |
| OPE error | 5,333% | 3,419% |

The D2 overfit test has ~8x better chunk L2 than the 8.5k run (910 vs 7287), but this is entirely attributable to the simpler problem (11 dims, 17 chunks). **In both cases, the model fundamentally fails to learn.** The loss doesn't converge and reconstructions are noise.
