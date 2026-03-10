# Noisy & Blowing-up Chunks and Trajectories

**Date:** 2026-03-09
**Notebooks:** `latent_sope_5_rollouts_executed.ipynb`, `latent_sope_50_rollouts_executed.ipynb`

---

## High-Level Picture

### What happened
The diffusion model generates synthetic chunks and trajectories that are physically nonsensical. Real trajectories stay smoothly within [0, 1.0], while synthetic ones explode to ranges of [-11,000, +15,000] — 2-3 orders of magnitude larger. The OPE estimate is off by ~5500–6700% relative error.

### Why it matters
This is a total pipeline failure: the diffusion model has not learned the data distribution. Every downstream step (stitching, reward scoring, OPE) inherits garbage. The 100% synthetic "success rate" (vs 0% real) is a hallmark — the model generates values so wild that cube z randomly crosses the 0.84 threshold, producing meaningless reward scores.

### The two root causes (high-level)

1. **The model is massively undertrained.** 125 chunks (5 rollouts) or 1250 chunks (50 rollouts) with 5–50 epochs yields only ~10–850 gradient steps. Diffusion models typically need 10k–100k+ steps to converge. The denoiser is essentially outputting structured noise.

2. **Errors compound exponentially during stitching.** Even moderate per-chunk errors become catastrophic over ~7 autoregressive stitching steps (horizon 60 / chunk size 8). Each chunk conditions on the previous chunk's noisy output, creating a positive feedback loop.

### The paradox: 50 rollouts is worse than 5
Despite 10x more data, the 50-rollout model produces worse results (state MSE 914k vs 72k). This is because the 50-rollout model uses 256 diffusion steps (vs 64), giving a poorly-trained denoiser 4x more opportunities to drift during sampling. More data doesn't help when the model hasn't converged.

---

## Low-Level Details

### Numerical evidence

| Metric | 5 rollouts | 50 rollouts |
|--------|-----------|-------------|
| Chunk L2 (states) | 2991 ± 601 | 8907 ± 1939 |
| Chunk L2 (actions) | 4169 ± 667 | 10597 ± 1958 |
| Trajectory state MSE | 71,988 | 914,590 |
| Trajectory action MSE | 252,969 | 3,883,755 |
| Real state range | [-1, 1] | [-1, 1] |
| Synthetic state range | [-3154, 2999] | [-11094, 15095] |
| Real success rate | 0% | 0% |
| Synthetic success rate | 100% | 100% |
| Oracle V^pi | 0.400 | 0.540 |
| OPE estimate | 27.3 | 30.7 |
| Relative error | 6728% | 5585% |

### Worst offenders: `obj_qy` and `obj_qz`

These quaternion components have 100–1000x the RMSE of other dimensions:

| Dim | Real range | Syn range (5 rollouts) | Syn range (50 rollouts) |
|-----|-----------|----------------------|------------------------|
| obj_qy | [0, 1.0] | [-1536, 1288] | [-5173, 7610] |
| obj_qz | [-1.0, 1.0] | [-3154, 2999] | [-11094, 15095] |

Other dimensions also blow up but less dramatically. For comparison, `obj_pz` (cube height) has real range [0, 0.84] but synthetic range [-24, 25] in the 50-rollout case.

### Per-dimension marginal statistics (50-rollout, selected dims)

| Dim | Real mean | Syn mean | Real std | Syn std |
|-----|-----------|----------|----------|---------|
| obj_pz | 0.73 | 1.08 | 0.26 | 7.77 |
| obj_qy | 0.59 | 316.9 | 0.34 | 1628.7 |
| obj_qz | 0.14 | 493.1 | 0.64 | 3705.7 |
| eef_pz | 0.81 | -15.0 | 0.29 | 310.0 |
| grip_0 | 0.03 | 10.0 | 0.01 | 52.0 |

### Detailed hypotheses

**H1: Insufficient training (primary cause)**
- 5 rollouts: ~125 chunks, 5 epochs → ~10 gradient steps (2 batches × 5 epochs)
- 50 rollouts: ~1250 chunks, 50 epochs → ~850 gradient steps (17 batches × 50 epochs)
- SOPE reference on D4RL likely uses 100k+ steps. We are 100–10,000x undertrained.
- The training loss needs to be checked — if it hasn't converged, this explains everything.

**H2: Quaternion manifold structure ignored**
- Quaternions satisfy q_w² + q_x² + q_y² + q_z² = 1 (unit hypersphere constraint)
- The diffusion model treats each component as an independent unconstrained real
- Z-score normalization doesn't respect this manifold — normalized quaternions don't lie on any meaningful surface
- A small error in normalized space can produce an unnormalized quaternion far from the unit sphere

**H3: Normalization amplification**
- Normalizer: `(x - mean) / std`, unnormalizer: `x * std + mean`
- Dimensions with small training variance (e.g. `grip_0` std ≈ 0.014) get amplified: a normalized-space error of 1.0 → physical error of ~70x the natural range
- `obj_qz` std ≈ 0.64 gives ~1.5x amplification — not huge alone, but multiplicative with H1

**H4: Autoregressive stitching compounds errors exponentially**
- `generate_full_trajectory()` runs ~7 stitching iterations (horizon 60 / chunk_size 8)
- Each iteration conditions on the last `frame_stack=2` states from the previous chunk
- Noisy conditioning → out-of-distribution input → even noisier output → feedback loop
- Evidence: trajectory MSE (71,988) is ~24x worse than chunk MSE (2,991) for the 5-rollout case
- Conditioning is extracted from the *normalized* sample (`sample.trajectories[:, -frame_stack:, :state_dim]`), so errors propagate in normalized space

**H5: More diffusion steps amplify a bad denoiser**
- 5-rollout model: 64 diffusion steps during sampling
- 50-rollout model: 256 diffusion steps during sampling
- Each denoising step applies the learned ε-prediction: `x_{t-1} = f(x_t, t, ε_θ)`
- If ε_θ is poorly learned, each step adds noise rather than removing it
- 256 steps × bad denoiser = 4x more drift than 64 steps × bad denoiser

### What to try next

**Quick wins (address H1):**
1. Check training loss curves — did loss converge?
2. Scale to 200+ epochs on 50 rollouts (target: 10k+ gradient steps)
3. Compare to SOPE reference training schedule

**Reduce sampling error (address H5):**
4. Try 64 or 128 diffusion steps for 50-rollout model
5. Verify that reducing steps improves or maintains chunk quality

**Representation fixes (address H2, H3):**
6. Consider 6D rotation representation instead of raw quaternions
7. Post-hoc quaternion normalization on generated outputs
8. Clamp outputs to physically valid ranges as a diagnostic baseline

**Diagnostics:**
9. Verify normalization round-trip: does normalize → unnormalize = identity on training data?
10. Plot per-step MSE during stitching to confirm error compounding pattern
11. Try single-chunk generation (no stitching) to isolate chunk quality from stitching errors
