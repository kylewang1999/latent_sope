# Try Next: Fixing Noisy Blowup in Chunk Diffusion

**Date:** 2026-03-09
**Context:** The diffusion model generates physically nonsensical trajectories (ranges of [-11k, +15k] vs real [-1, 1]). Root causes: massive undertraining (~850 gradient steps vs 10k+ needed) and error compounding during autoregressive stitching. See `noisy_blowup_analysis.md` for full diagnosis.

---

## Phase 1: Quick Wins — Address Undertraining (H1)

These should be done first since undertraining is the primary root cause. Everything downstream is garbage until the model actually learns the data distribution.

### 1.1 Check training loss curves
- [ ] Re-run training on the 50-rollout data (`rollout_latents_50/`) and **log the training loss at every epoch**
- [ ] Plot loss vs. epoch and loss vs. gradient step
- [ ] Determine whether loss has converged, is still decreasing, or has plateaued
- [ ] If loss hasn't converged, that confirms H1 and justifies scaling up training

### 1.2 Scale up training to 10k+ gradient steps
- [ ] Calculate the exact number of gradient steps needed: with ~1250 chunks and batch_size=64, that's ~20 batches/epoch. To hit 10k steps, need ~500 epochs
- [ ] Run training on 50 rollouts for **500 epochs** (target: ~10,000 gradient steps)
- [ ] If 500 epochs is too slow, try 200 epochs first (~4,000 steps) as an intermediate checkpoint
- [ ] Log loss every 10 epochs to track convergence
- [ ] Save checkpoints at epochs 50, 100, 200, 500 to compare quality at different training stages

### 1.3 Compare to SOPE reference training schedule
- [ ] Check `third_party/sope/` for how many gradient steps / epochs D4RL experiments use
- [ ] Look in `third_party/sope/opelab/examples/` and any config files for training hyperparameters
- [ ] Document the reference training budget (steps, epochs, dataset size) for comparison

---

## Phase 2: Reduce Sampling Error (H5)

The 50-rollout model uses 256 diffusion steps during sampling, which gives a poorly-trained denoiser 4x more chances to drift compared to 64 steps. Try reducing this.

### 2.1 Test fewer diffusion steps
- [ ] After retraining (Phase 1), generate chunks using **64, 128, and 256** diffusion steps
- [ ] Compare chunk L2 error across the three settings
- [ ] Use the best-performing step count for all subsequent experiments
- [ ] Note: the number of diffusion steps during *training* and *sampling* may differ — check `SopeDiffusionConfig` and `GaussianDiffusion` to understand which parameter controls sampling steps

### 2.2 Single-chunk quality check
- [ ] Before running the full stitching pipeline, generate **single chunks** (no autoregressive stitching) and measure L2 error
- [ ] This isolates chunk quality from stitching-induced error compounding
- [ ] Target: chunk L2 error should be < 1.0 (same order of magnitude as the data) before attempting stitching
- [ ] Use `l2_chunk_error()` from `eval/metrics.py`

---

## Phase 3: Diagnostics — Confirm Error Compounding (H4)

These diagnostics help understand *where* in the pipeline errors enter and how they grow.

### 3.1 Verify normalization round-trip
- [ ] Take a batch of real training data
- [ ] Run it through `normalize → unnormalize` and check if the result matches the input exactly
- [ ] Compute max absolute error across all dimensions
- [ ] If round-trip error is non-trivial, investigate whether float32 precision or the normalization implementation is introducing drift

### 3.2 Plot per-step MSE during stitching
- [ ] Modify `generate_full_trajectory()` (or add a diagnostic wrapper) to record the per-chunk MSE at each stitching iteration
- [ ] Plot MSE vs. stitching step (expect exponential growth if error compounding is the issue)
- [ ] Compare the growth rate before vs. after Phase 1 retraining
- [ ] This will tell us whether stitching is viable at all with current architecture, or if we need architectural changes

### 3.3 Clamped-output baseline
- [ ] As a quick diagnostic, clamp all generated states to the training data's [min, max] range per dimension after each stitching step
- [ ] Re-run the full pipeline (stitching → reward → OPE) with clamped outputs
- [ ] This won't fix the model but will show whether the OPE pipeline *could* work if the diffusion model stayed in-distribution
- [ ] If clamped OPE estimate is reasonable, it confirms the pipeline is correct and only the model quality needs improvement

---

## Phase 4: Representation Fixes (H2, H3)

Only pursue these if Phase 1 + 2 don't fully resolve the issue. Quaternion blowup may self-resolve once the model is properly trained.

### 4.1 Post-hoc quaternion normalization
- [ ] After generating trajectories, normalize the quaternion components (indices 3–6 for `object` key, indices 13–16 for `robot0_eef_quat`) to unit length
- [ ] Check if this alone brings quaternion values back to valid ranges
- [ ] Re-run reward scoring and OPE with post-hoc normalized quaternions
- [ ] This is a band-aid, not a fix — but useful for diagnosing whether quaternion blowup is the dominant error source

### 4.2 Consider alternative rotation representations
- [ ] If quaternion blowup persists after proper training, consider switching to 6D rotation representation (Zhou et al. 2019)
- [ ] This would require changes to the encoder (`LowDimConcatEncoder`) and decoder
- [ ] Low priority — only do this if Phases 1–3 fail to produce reasonable results

---

## Execution Order

Run these in order of priority. Stop and evaluate after each phase before moving to the next.

1. **Phase 1.1** (check loss curves) — 10 min. If loss hasn't converged, proceed to 1.2
2. **Phase 1.2** (scale training to 500 epochs) — ~30-60 min depending on hardware
3. **Phase 1.3** (check SOPE reference) — 10 min reading code
4. **Phase 2.2** (single-chunk quality check) — 5 min. Do this before stitching
5. **Phase 2.1** (test diffusion step counts) — 15 min
6. **Phase 3.1** (normalization round-trip) — 5 min
7. **Phase 3.2** (per-step MSE during stitching) — 20 min (requires code changes)
8. **Phase 3.3** (clamped-output baseline) — 10 min
9. **Phase 4.1** (post-hoc quaternion normalization) — 10 min
10. **Phase 4.2** (6D rotation) — only if everything else fails

**Expected outcome after Phase 1+2:** Chunk L2 error drops to O(1), synthetic trajectories stay in [-5, 5] range, OPE relative error drops below 100%.
