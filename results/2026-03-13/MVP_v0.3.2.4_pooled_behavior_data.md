# MVP v0.3.2.4: Pooled Multi-Policy Behavior Data (Option A)

**Date:** 2026-03-13
**Status:** RUNNING — SLURM job 7376470 (3hr walltime)
**Notebook:** `experiments/2026-03-13/MVP_v0.3.2.4_pooled_behavior_data.ipynb`

## Motivation

v0.3.3 (Option B) used 80 rollouts from a single 52% SR policy as behavior data. This is the cleanest match to SOPE's D4RL medium assumption, but gives only ~3,600 training chunks — potentially too few for the diffuser to learn a good chunk distribution.

Option A pools **all 720 rollouts from all 8 target policies** as behavior data (~43% average SR). This gives ~40k training chunks (10x more data) with broader state-space coverage, at the cost of heterogeneous quality (0%–90% SR mixed together).

## Setup

### Behavior data (D_beta)
- **720 rollouts** pooled from 8 policies:

| Policy | Rollouts | Oracle SR |
|---|---|---|
| 50demos_epoch10 | 80 | 0% |
| 10demos_epoch10 | 100 | 8% |
| 200demos_epoch10 | 80 | 18% |
| 100demos_epoch20 | 100 | 42% |
| 10demos_epoch20 | 80 | 52% |
| 10demos_epoch30 | 100 | 62% |
| 100demos_epoch40 | 100 | 76% |
| 200demos_epoch40 | 80 | 90% |

- **No expert demos** in training data
- Normalization computed from pooled data only

### Target policies (evaluation)
- `200demos_epoch10`: 18% SR (same as v0.3.3)
- `200demos_epoch40`: 90% SR (same as v0.3.3)

### Architecture & training
- TemporalUnet: base_dim=32, dim_mults=(1,4,8), no attention
- GaussianDiffusion: 256 steps, x0-prediction, action_weight=5.0
- chunk_size=4, stride=1 (~40k chunks expected)
- 50 epochs, batch=64, lr=3e-4, grad_clip=1.0
- BC behavior policy: 2-layer Gaussian MLP (hidden=256), 500 steps on pooled data

### Guidance sweep
- action_scales: [0.0, 0.05, 0.1, 0.2, 0.5]
- ratios: [0.0, 0.25, 0.5]
- 13 configs x 2 policies = 26 evaluations
- 50 synthetic trajectories per eval, horizon=60
- k_guide=1, normalize_grad=True, no adaptive/clamp

### Checkpoints saved
- `diffusion_ckpts/mvp_v0324_pooled_behavior/diffusion_model.pt`
- `diffusion_ckpts/mvp_v0324_pooled_behavior/diffusion_model_ema.pt`
- `diffusion_ckpts/mvp_v0324_pooled_behavior/norm_mean.npy` / `norm_std.npy`
- `diffusion_ckpts/mvp_v0324_pooled_behavior/bc_behavior.pt`

## Key questions

1. **Does 10x more data improve the diffuser?** v0.3.3 had ~3.6k chunks. v0.3.2.4 has ~40k. More data should give better reconstruction and smoother trajectories.
2. **Does heterogeneous quality hurt guidance?** SOPE trains on homogeneous medium data. Our mix spans 0%–90% SR. The BC behavior policy will be a "mixture" policy — its grad_log_prob may be less meaningful than a single-policy BC.
3. **Does the unguided baseline change?** With heterogeneous data, unguided trajectories should reflect the ~43% average SR mix, not a 52% single-policy anchor.

## Comparison to prior experiments

| Experiment | Behavior data | Chunks | Target policies | Key result |
|---|---|---|---|---|
| v0.3.2.x | 200 expert + 80 target | ~12k | 2 | Guidance destroyed — expert bias too strong |
| v0.3.3 (Option B) | 80 rollouts, 1 policy (52% SR) | ~3.6k | 2 | PENDING (SLURM) |
| **v0.3.2.4 (Option A)** | **720 rollouts, 8 policies (0–90% SR)** | **~40k** | **2** | **PENDING** |

## Results

**FAILED** — SLURM job 7376470, crashed near end of notebook with `TypeError`.

### All guidance configs: 0% SR, OPE=0.00

Every single guidance configuration produced identical results — 0% synthetic SR and OPE=0.00 for both target policies:

| Scale | Ratio | 200demos_epoch10 OPE (SR) | 200demos_epoch40 OPE (SR) | Spearman rho | MAE |
|-------|-------|---------------------------|---------------------------|-------------|-----|
| 0.00 | 0.00 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.05 | 0.00 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.05 | 0.25 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.05 | 0.50 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.10 | 0.00 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.10 | 0.25 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.10 | 0.50 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.20 | 0.00 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.20 | 0.25 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.20 | 0.50 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.50 | 0.00 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.50 | 0.25 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |
| 0.50 | 0.50 | 0.00 (0%) | 0.00 (0%) | NaN | 0.540 |

Oracle: 200demos_epoch10 = 0.18 (18%), 200demos_epoch40 = 0.90 (90%)

### Crash

The notebook crashed at the results summary cell with:
```
TypeError: 'NoneType' object is not subscriptable
```
`best_cfg` was `None` because all Spearman rho values were NaN (all OPE outputs identical), so `best_cfg[0]` failed. This is a minor bug in the reporting code — the real problem is the 0% SR.

### Analysis

The 0% synthetic SR across all configs (including unguided) means the diffuser itself is not generating trajectories that reach `cube_z > 0.84`. This is almost certainly the **rollout recorder bug** (Bug #1 in CLAUDE.md): the recorder drops the final success observation, so training data never contains states where `cube_z > 0.84`. Even the 90% SR policy's rollouts show 0% success when scored by `cube_z > 0.84` on the recorded states.

With 720 rollouts from policies spanning 0–90% SR, the training data has plenty of near-success trajectories (cube_z up to ~0.835) but zero actual success states. The diffuser faithfully reproduces this distribution — generating trajectories that plateau just below the threshold.

Guidance cannot fix this: positive guidance steers actions toward the target policy, but the underlying state distribution in the training data never reaches the success region. No amount of action guidance will make cube_z jump from 0.835 to 0.84 when the diffuser has never seen that state.

### Conclusion

This experiment confirms that the rollout recorder bug completely blocks OPE evaluation. Until the recorder is fixed to capture the post-step success observation, all experiments will produce 0% synthetic SR regardless of guidance settings, data quantity, or policy quality. The fix in `rollout.py` (storing `next_obs` on success termination) is a prerequisite for all further work.
