# MVP v0.3: Multi-Target-Policy OPE Evaluation

**Date:** 2026-03-10
**Notebook:** `experiments/2026-03-10/MVP_v0.3_sope_on_robomimic.ipynb`
**Status:** Complete
**Runtime:** ~22 minutes (A100 80GB, interactive node)
**Builds on:** MVP v0.1 (pipeline) + v0.2 (guidance)

---

## Objective

Test whether SOPE can **rank** multiple target policies of varying quality using Spearman ρ (rank correlation) and Regret@1 (policy selection).

## Key Design: Data Partition for Fairness

Previous versions trained the diffusion model on all 200 demos, and v0.2's target policy (diffusion policy) was also trained on the same data. In v0.3, we partition the data so that target policies are trained on data the diffusion model has **never seen**:

- **Demos 0-149 (150):** Train the diffusion model (behavior data / OPE "world model")
- **Demos 150-199 (50):** Train BC_Gaussian target policies on subsets of these held-out demos

This ensures guidance genuinely steers toward different distributions rather than pushing toward something the diffusion model already knows.

## Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Data partition** | 150 behavior / 50 target | Fair separation |
| **Target policies** | 5 BC_Gaussians on {50, 40, 30, 20, 10} held-out demos | Varying quality |
| State dim | 11 (no quaternions) | Same as v0.1/v0.2 |
| Action dim | 4 (no orientation) | Same as v0.1/v0.2 |
| Diffusion training | 10 epochs × 1000 steps | Same as v0.1 |
| Diffusion architecture | TemporalUnet, dim=32, mults=(1,4,8) | Same as v0.1 |
| predict_epsilon | False (x0-prediction) | Same as v0.1 |
| BC_Gaussian | 11→64→64→4 MLP + learnable log_std | Same as v0.2 |
| BC training | 500 epochs, lr=1e-3 | Sufficient convergence |
| Guidance scale | 0.1 | Best from v0.2 sweep |
| Oracle rollouts | 20 per policy (100 total) | Reduced for speed |
| Synthetic trajectories | 50 per policy | Same as v0.1/v0.2 |
| Generation horizon | 60 steps | Lift episode length |
| Reward | Ground-truth (cube_z > 0.84) | Zero approximation error |

### Key differences from v0.1/v0.2

- **Data partition**: Diffusion trained on 150 demos (not 200), target policies on held-out 50
- **Fresh diffusion model**: Cannot reuse v0.1 checkpoint (trained on all 200)
- **Multiple target policies**: 5 BC_Gaussians vs 1 target policy in v0.1/v0.2
- **Each BC_Gaussian is its own exact guidance proxy**: No proxy approximation error

## Pipeline Steps

1. **Load & partition** 200 demos → 150 behavior + 50 target
2. **Train chunk diffusion** on 150 behavior demos (10 epochs × 1000 steps)
3. **Train 5 BC_Gaussians** on {50, 40, 30, 20, 10} of the held-out target demos
4. **Oracle rollouts** for each BC_Gaussian (20 rollouts × 5 policies in Lift env)
5. **Guided SOPE** for each target policy (50 synthetic trajectories, scale=0.1)
6. **Multi-policy evaluation**: Spearman ρ, Regret@1, per-policy relative error
7. **Visualization**: scatter plot, ranking comparison, per-policy trajectories

## Expected Runtime

| Step | Estimated Time | Notes |
|------|---------------|-------|
| Step 1: Load & partition | ~1s | HDF5 reads |
| Step 2: Train diffusion | ~5 min | 10k steps on ~6,600 chunks |
| Step 3: Train 5 BC_Gaussians | ~3-5 min | Small MLPs, 500 epochs each |
| Step 4: Oracle rollouts | ~25 min | 5 × 20 rollouts × ~15s |
| Step 5-6: Guided SOPE (×5) | ~15-25 min | 5 × 50 trajs with 256 diffusion steps + autograd |
| Step 7-8: Eval + viz | ~1s | |
| **Total** | **~50-60 min** | |

## Success Criteria

1. **Oracle values show spread**: Different data fractions → different policy quality
2. **Spearman ρ > 0**: OPE rankings have positive correlation with oracle rankings
3. **Regret@1 is small**: OPE correctly identifies the best (or near-best) policy
4. **Pipeline completes end-to-end**: All 8 steps run without errors

## Saved Artifacts

- `diffusion_ckpts/mvp_v03_diffusion/diffusion_model_ema.pt` — Diffusion model (behavior split)
- `diffusion_ckpts/mvp_v03_diffusion/norm_stats.npz` — Normalization stats
- `diffusion_ckpts/mvp_v03_bc_policies/bc_*demos.pt` — 5 BC_Gaussian checkpoints

## Results

### Oracle Values (Ground Truth V^π)

All 5 BC_Gaussian target policies scored **V^π = 0.000** with **0% success rate** across all 20 oracle rollouts each. None of the policies successfully lifted the cube.

| Policy | Demos | Oracle V^π | Oracle Std | Success Rate |
|--------|-------|-----------|-----------|-------------|
| bc_50demos | 50 | 0.000 | 0.000 | 0.0% |
| bc_40demos | 40 | 0.000 | 0.000 | 0.0% |
| bc_30demos | 30 | 0.000 | 0.000 | 0.0% |
| bc_20demos | 20 | 0.000 | 0.000 | 0.0% |
| bc_10demos | 10 | 0.000 | 0.000 | 0.0% |

### OPE Estimates (Guided SOPE)

The diffusion model + guidance produced nonzero OPE estimates despite zero oracle values:

| Policy | Oracle V^π | OPE Estimate | OPE Std | Synthetic Success |
|--------|-----------|-------------|---------|-------------------|
| bc_50demos | 0.00 | 10.88 | 5.042 | 92.0% |
| bc_40demos | 0.00 | 10.04 | 5.993 | 82.0% |
| bc_30demos | 0.00 | 11.26 | 5.399 | 92.0% |
| bc_20demos | 0.00 | 9.02 | 4.856 | 96.0% |
| bc_10demos | 0.00 | 10.54 | 5.471 | 96.0% |

### Ranking Metrics

| Metric | Value |
|--------|-------|
| Spearman ρ | NaN (all oracle values tied at 0) |
| Regret@1 | 0.0000 (trivially — all policies equal) |
| Regret@2 | 0.0000 |
| Regret@3 | 0.0000 |
| Mean MSE | 107.6831 |
| Mean Relative Error | ~10^11 % (division by ~0) |

### OPE Ranking vs Oracle Ranking

| Rank | Oracle Best→Worst | OPE Best→Worst |
|------|-------------------|----------------|
| 1 | bc_10demos | bc_30demos |
| 2 | bc_20demos | bc_50demos |
| 3 | bc_30demos | bc_10demos |
| 4 | bc_40demos | bc_40demos |
| 5 | bc_50demos | bc_20demos |

(Oracle ranking is arbitrary since all values are tied at 0.)

## Observations

### Why it failed: BC_Gaussian policies are too weak

The fundamental issue is that **none of the 5 BC_Gaussian target policies can solve Lift**. The 2-layer MLP (11→64→64→4) with zero-padded orientation actions is not expressive enough to perform the task, regardless of how many demos it's trained on.

**Root cause analysis:**
1. **Reduced action space**: The 4-dim actions (pos + gripper, orientation zeroed out) may not provide sufficient control for the Lift task. The original diffusion policy uses full 7-dim actions.
2. **Small architecture**: A 64-hidden-dim MLP is very limited compared to the diffusion policy's UNet architecture.
3. **No temporal context**: BC_Gaussian sees only the current state, while the successful diffusion policy conditions on frame-stacked history.
4. **Held-out demos only**: Training on just 10-50 of the held-out demos (150-199) may not provide enough coverage, especially with a simple architecture.

### Diffusion model hallucination

Despite oracle V^π = 0 for all policies, the guided diffusion model generates trajectories where cube_z exceeds the lift threshold (~82-96% synthetic success). This means guidance is pushing trajectories into "lift" regions of state space, but these don't correspond to real policy behavior — the diffusion model is hallucinating feasible trajectories.

### What this tells us

- The pipeline itself works end-to-end (all 8 steps completed successfully)
- The data partition design is sound
- The BC_Gaussian guidance mechanism produces noticeable variation in OPE estimates across policies (9.02 to 11.26)
- But the experiment is **uninformative for ranking** because the target policies all have zero ground-truth value

### Next steps: MVP v0.3b — Diffusion policy checkpoints as target policies

**Decision:** BC_Gaussians are too weak to be target policies. Instead, use **robomimic diffusion policy checkpoints at different training stages** as target policies (early = weak, late = strong). BC_Gaussians are demoted to **guidance proxies only** (same role as in v0.2).

**Revised design:**
1. **Train multiple robomimic diffusion policies** (or save intermediate checkpoints from one training run) to get policies of varying quality
2. **Oracle V^π**: Roll out each diffusion policy checkpoint in the Lift env
3. **Guidance proxy**: For each target policy, collect rollouts and train a BC_Gaussian on them (like v0.2)
4. **Guided SOPE**: Use each BC_Gaussian proxy for guidance during trajectory generation
5. **Evaluate**: Spearman ρ, Regret@1 across the policy set

**Key insight from v0.2:** BC_Gaussian works well as a guidance proxy (it only needs to approximate `grad_log_prob`, not actually solve the task). The v0.3 mistake was making it the target policy too.

**Blockers:**
- We currently only have **one** robomimic diffusion policy checkpoint (`last.pth`). The `models/` directory is empty (no intermediate checkpoints saved during training).
- Need to either: (a) retrain with intermediate checkpoint saving, (b) train multiple policies on different data amounts, or (c) train robomimic BC-RNN policies at different quality levels.
- Training robomimic policies is expensive — likely need SLURM jobs.
