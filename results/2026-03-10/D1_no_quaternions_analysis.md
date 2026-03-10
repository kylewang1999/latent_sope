# D1 Results: 8.5k Steps — No Quaternions

**Date:** 2026-03-10
**Notebook:** `experiments/2026-03-10/8.5k_steps_no_quarternions.ipynb`
**Hypothesis:** Quaternions (unit-norm, antipodal symmetry) are geometrically hostile to L2 diffusion. Dropping them reduces state dim 19 → 11, transition dim 26 → 18.

---

## Config

| Parameter | 8.5k Baseline (with quats) | D1 (no quats) |
|-----------|---------------------------|----------------|
| State dim | 19 | 11 |
| Action dim | 7 | 7 (unchanged) |
| Transition dim | 26 | 18 |
| dim_mults | (1, 2) | (1, 2) |
| Parameters | ~252k | ~251k |
| Training steps | 8,500 | 8,500 |
| Diffusion steps | 256 | 256 |
| Everything else | Same | Same |

**Dropped state dimensions:** object quaternion (indices 3–6), EEF quaternion (indices 13–16).

**Kept state dimensions (new layout):**

| New Index | Content |
|-----------|---------|
| 0–2 | cube_pos (x, y, z) |
| 3–5 | gripper_to_cube (x, y, z) |
| 6–8 | eef_pos (x, y, z) |
| 9–10 | gripper_qpos (0, 1) |

**Action dimensions (unchanged, OSC_POSE controller):**

| Index | Content |
|-------|---------|
| 0–2 | EEF position delta (dx, dy, dz) |
| 3–5 | EEF orientation delta (axis-angle rotation) |
| 6 | Gripper action (-1 = open, +1 = close) |

---

## Results Summary

| Metric | 8.5k Baseline (with quats) | D1 (no quats) | Change |
|--------|---------------------------|----------------|--------|
| Final training loss | 0.728 | 0.567 | 22% lower |
| Chunk L2 (states) | 7,287 ± 1,672 | 503 ± 76 | **14.5x better** |
| Chunk L2 (actions) | 7,569 ± 1,621 | 3,740 ± 757 | 2.0x better |
| Trajectory state MSE | 1,175,015 | 35,039 | **33.5x better** |
| Trajectory action MSE | 4,163,263 | 4,608,946 | Slightly worse |
| OPE estimate | 29.340 | 26.160 | Similar (both garbage) |
| OPE relative error | 5,333% | 4,744% | Marginal |
| % states clamped | 99.7% | 99.6% | No change |
| Synthetic state range | [-17,280, +14,051] | [-1,672, +1,377] | ~10x tighter |
| Success rate (unclamped) | 100% (spurious) | 100% (spurious) | Same |
| Success rate (clamped) | 0% | 0% | Same |
| Error growth (7 chunks) | 3.2x | 13.4x | Worse |
| Per-chunk growth factor | 1.21x | 1.54x | Worse |

---

## Training Curve

| Epoch | Mean Loss |
|-------|-----------|
| 1 | 1.383 |
| 50 | 0.889 |
| 100 | 0.773 |
| 200 | 0.664 |
| 500 | 0.567 |

Convergence ratio (last 10% / first 10%): **0.550** — loss still dropping at end of training. Not converged, same as baseline.

---

## Chunk Quality Progression

| Epoch | State L2 | Action L2 |
|-------|----------|-----------|
| 50 | 1,156 ± 138 | 9,913 ± 1,981 |
| 100 | 999 ± 153 | 8,487 ± 1,766 |
| 200 | 741 ± 107 | 6,243 ± 1,728 |
| 500 | 503 ± 76 | 3,740 ± 757 |

State L2 improved consistently from 1,156 → 503 across checkpoints. Action L2 also improved (9,913 → 3,740) but remains very high.

---

## Per-Dimension Blowup (Synthetic vs Real)

| Dim | Real range | Syn range | Blowup factor |
|-----|-----------|-----------|---------------|
| obj_px | [-0.03, 0.04] | [-584, 394] | ~14,000x |
| obj_py | [-0.03, 0.03] | [-483, 460] | ~15,000x |
| obj_pz | [0.00, 0.84] | [-32, 26] | ~70x |
| g2c_x | [-0.04, 0.14] | [-1,121, 1,087] | ~12,000x |
| g2c_y | [-0.05, 0.05] | [-353, 283] | ~6,000x |
| g2c_z | [-0.21, 0.01] | [-1,672, 1,346] | ~14,000x |
| eef_px | [-0.12, 0.07] | [-1,057, 862] | ~10,000x |
| eef_py | [-0.04, 0.04] | [-489, 370] | ~11,000x |
| eef_pz | [0.00, 1.04] | [-1,620, 1,377] | ~2,900x |
| grip_0 | [0.00, 0.04] | [-271, 227] | ~12,000x |
| grip_1 | [-0.04, 0.00] | [-245, 248] | ~12,000x |

**All dimensions still blow up catastrophically during stitching**, with 1,000–15,000x blowup across every dimension. Removing quaternions didn't fix the fundamental problem.

---

## Interpretation

### What improved

1. **Chunk quality (states) improved 14.5x.** State L2 dropped from 7,287 → 503. This is a meaningful improvement — quaternions were indeed hurting state reconstruction. The reduced 11-dim, 18-transition-dim problem is easier for the model.

2. **Training loss is lower** (0.567 vs 0.728), suggesting the model fits the reduced space better with the same capacity and budget.

3. **Synthetic state range is ~10x tighter** (±1,672 vs ±17,280). Still far out of distribution, but the scale of the blowup is reduced.

### What didn't improve

1. **OPE is still garbage.** 4,744% relative error (vs 5,333%) — essentially unchanged. The pipeline still doesn't produce usable value estimates.

2. **99.6% of states are still out-of-range** (vs 99.7%). Clamping still collapses everything to 0.

3. **Action L2 barely improved** (3,740 vs 7,569 — only 2x) despite state L2 improving 14.5x. This is likely because **action dims 3–5 (orientation deltas) depend on EEF quaternion state that was dropped.** The model can't predict orientation actions without orientation state, so those 3 action dims are effectively noise from the model's perspective.

4. **Stitching error compounding got worse** (13.4x total growth vs 3.2x, per-chunk 1.54x vs 1.21x). This is surprising — the better chunks should stitch better. The likely explanation: at 8.5k steps, individual chunks are still bad enough that stitching compounds less (you can't make noise worse), but with somewhat-better chunks, the compounding errors become the dominant failure mode.

### The action-state inconsistency

The action space was kept at 7 dims (OSC_POSE: 3 position deltas + 3 orientation deltas + 1 gripper), but the orientation deltas (indices 3–5) depend on the current EEF orientation (the dropped quaternion). This means:

- The diffusion model must learn 3 action dimensions that appear stochastic given the reduced state
- This wastes model capacity and pollutes the loss signal
- The poor action L2 improvement (2x vs 14.5x for states) supports this: the model improved on position-coupled actions but not orientation-coupled ones

**Recommendation:** Future no-quaternion experiments should also drop action dims 3–5, reducing to a 4-dim action space (dx, dy, dz, gripper). This makes the state-action space fully consistent.

---

## Diagnosis

**Dropping quaternions was necessary but not sufficient.** The 14.5x chunk L2 improvement confirms quaternions were a real problem for the diffusion model. But 503 is still far above the target (L2 < 1.0), and the model still hasn't converged (loss ratio 0.55). The remaining bottleneck is the same as the baseline: **undertrained model with insufficient capacity.**

This result maps to **Phase A4** in the architecture log decision tree (chunk L2 > 100, still garbage), but with a strong signal that quaternion removal helps. The correct path forward is:

1. **Combine D1 (no quaternions) with the 500k-step run** — the 500k run addresses model capacity (dim_mults (1,4,8)) and training budget. Adding quaternion removal on top should compound the gains.
2. **Also drop orientation action dims (3–5)** to make state-action space consistent.
3. If chunk L2 drops below 100 with the combined changes, stitching error compounding (now 1.54x per chunk) becomes the next bottleneck to address (B1: add 1-state overlap).

---

## Bug Note

Cell 6 hit a `NameError: name 'latents_full' is not defined` because the filtered rollouts already existed on disk (the `dst_path.exists()` early-return path skips the h5 reading loop, so `latents_full` is never assigned). The cell succeeded functionally — all filtered rollouts were loaded correctly — and all subsequent cells ran fine. The error is cosmetic and only affects the print statement.
