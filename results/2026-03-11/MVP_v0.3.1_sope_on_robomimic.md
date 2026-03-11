# MVP v0.3.1: Multi-Target-Policy OPE with Diffusion Policy Targets

**Date:** 2026-03-10
**Notebook:** `experiments/2026-03-10/MVP_v0.3.1_sope_on_robomimic.ipynb` (to be created)
**Status:** Planned
**Builds on:** MVP v0.3 (failed — BC_Gaussian too weak) + v0.2 (guidance proxy approach)

---

## Objective

Test whether SOPE can **rank** multiple target policies of varying quality, using **Spearman ρ** and **Regret@1**.

## Key Lesson from v0.3

BC_Gaussians are too weak to be target policies (0% success on Lift). But they work well as **guidance proxies** (v0.2 showed 25.9% relative error). So we separate the two roles:

- **Target policy** = robomimic diffusion policy (strong enough to actually solve Lift)
- **Guidance proxy** = BC_Gaussian trained on each target policy's rollouts (only needs to approximate `grad_log_prob`)

## Design: Data Partition

Same partition as v0.3 — target policies trained on data the SOPE diffusion model has never seen:

- **Demos 0-149 (150):** Train the SOPE chunk diffusion model (behavior data / world model)
- **Demos 150-199 (50):** Train robomimic diffusion policies on subsets of these held-out demos

## Pipeline

### Phase 1: Train Target Policies (SLURM, overnight)

Train 5 robomimic diffusion policies on different amounts of held-out demos:

| Policy | Training demos | Expected quality |
|--------|---------------|-----------------|
| diffpol_50 | demos 150-199 (50) | Best |
| diffpol_40 | demos 150-189 (40) | Good |
| diffpol_30 | demos 150-179 (30) | Medium |
| diffpol_20 | demos 150-169 (20) | Weak |
| diffpol_10 | demos 150-159 (10) | Weakest |

**Implementation:**
- Use robomimic's `filter_key` mechanism to select specific demo ranges from `low_dim_v15.hdf5`
- Need to create custom HDF5 filter masks for each subset (robomimic's `filter_dataset_size.py` picks random demos — we need contiguous ranges from the held-out partition)
- Base config from existing checkpoint: `diffusion_policy_trained_models/test/20260309132349/config.json`
- Save intermediate checkpoints (`every_n_epochs: 50`) to track convergence
- Training command: `python -m robomimic.scripts.train --config <config.json> --dataset <hdf5>`

**Estimated time:** ~2000 epochs each. Need to benchmark 1 run first to estimate wall time.

### Phase 2: Oracle Rollouts

Roll out each trained diffusion policy in the Lift environment (20 rollouts each, horizon=60).

### Phase 3: Guidance Proxies

For each target policy:
1. Collect ~50 rollouts to get (state, action) pairs
2. Train a BC_Gaussian on those pairs (same as v0.2: 11→64→64→4, 200 epochs)
3. BC_Gaussian provides `grad_log_prob` for guided SOPE sampling

### Phase 4: Guided SOPE + Evaluation

- Reuse the SOPE chunk diffusion model from v0.3 (trained on 150 behavior demos)
- Generate 50 guided synthetic trajectories per target policy (scale=0.1)
- Score with ground-truth reward (cube_z > 0.84)
- Compute Spearman ρ, Regret@1, per-policy relative error

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Data partition | 150 behavior / 50 target | Same as v0.3 |
| Target policies | 5 robomimic diffusion policies | Trained on {50,40,30,20,10} held-out demos |
| SOPE diffusion model | Reuse from v0.3 | `diffusion_ckpts/mvp_v03_diffusion/` |
| State dim | 11 (no quaternions) | Same as v0.1-v0.3 |
| Action dim | 4 (pos + gripper) | For SOPE chunk diffusion + BC guidance proxy |
| Target policy actions | Full 7-dim | Robomimic policies use native action space |
| Guidance proxy | BC_Gaussian per target policy | Trained on target policy rollouts |
| Guidance scale | 0.1 | Best from v0.2 |
| Oracle rollouts | 20 per policy | |
| Synthetic trajectories | 50 per policy | |
| Reward | Ground-truth (cube_z > 0.84) | |

## Success Criteria

1. **Oracle values show spread**: Policies trained on more demos should have higher V^π
2. **Spearman ρ > 0**: OPE rankings positively correlated with oracle rankings
3. **Regret@1 is small**: OPE identifies the best (or near-best) policy
4. **Pipeline completes end-to-end**

## Estimated Runtime

| Step | Estimated Time | Notes |
|------|---------------|-------|
| Phase 1: Train 5 diffusion policies | TBD (SLURM overnight) | Need to benchmark |
| Phase 2: Oracle rollouts (5 × 20) | ~25 min | Same as v0.3 |
| Phase 3: Collect rollouts + train BC proxies | ~15 min per policy | 50 rollouts + BC training |
| Phase 4: Guided SOPE + eval | ~20 min | Same as v0.3 |
| **Total (excl. Phase 1)** | **~1-1.5 hr** | |

## Open Questions

1. How long does one robomimic diffusion policy training take on V100? Need to benchmark before submitting 5 jobs.
2. Will 10-demo diffusion policies achieve nonzero success on Lift? If not, we may need a minimum of 20 demos.
3. Should we save intermediate training checkpoints and use those as additional target policies (varying quality from one training run)?

## Results

*To be filled after running.*

## Observations

*To be filled after running.*
