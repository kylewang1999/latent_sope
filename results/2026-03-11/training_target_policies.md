# Training a Gradient of Diffusion Target Policies

**Date:** 2026-03-11
**Goal:** Train 5 diffusion policies with varying success rates for multi-policy OPE evaluation.

## Motivation

To properly evaluate stitch-OPE, we need multiple target policies spanning a range of performance levels. Using intermediate checkpoints from a single training run (approach 1) creates an artificially easy test — policies share the same optimization trajectory, so distribution shift between behavior and target is minimal.

Instead, we train separate policies on **different-sized subsets of the demonstration data** (approach 2). A policy trained on 10 demos will have genuinely different behavior than one trained on 200 demos, creating meaningful distribution shift that stitch-OPE must handle.

## Design

| Target Policy | Filter Key | # Demos | # Sequences | Expected Success |
|---------------|------------|---------|-------------|-----------------|
| `lift_diffusion_10demos`  | `10_demos`  | 10  | ~511  | Low (0–30%)    |
| `lift_diffusion_25demos`  | `25_demos`  | 25  | ~1218 | Low-Mid (20–50%) |
| `lift_diffusion_50demos`  | `50_demos`  | 50  | ~2447 | Mid (40–70%)   |
| `lift_diffusion_100demos` | `100_demos` | 100 | ~4898 | Mid-High (60–90%) |
| `lift_diffusion_200demos` | `None`      | 200 | ~9666 | High (80–100%) |

**Behavior policy:** The 200-demo policy (strongest). Its rollouts form the offline dataset for stitch-OPE. All 200 human demos are used, giving maximum state-space coverage.

**No data leakage concern:** The OPE pipeline consumes rollouts from the behavior policy (Step 1), not the original human demos. Demo overlap between behavior and target policy training is irrelevant.

**Why this is a fair OPE test:** Target policies trained on fewer demos learn genuinely different strategies (e.g., one grasp approach vs. multiple). The behavior data (from the 200-demo policy) must be reweighted/guided to match each target policy's distribution — this is the core challenge of OPE.

## Training Setup

All policies share the same architecture and hyperparameters — only the training data size differs.

- **Algorithm:** `diffusion_policy` (DDPM UNet)
- **Architecture:** UNet down_dims=[256, 512, 1024], kernel=5, n_groups=8
- **Training:** 40 epochs, 100 grad steps/epoch, batch_size=100, AdamW lr=1e-4 cosine
- **Eval:** Disabled (we run our own oracle eval post-hoc via `oracle_value()`)
- **Checkpoints:** Saved every 10 epochs + last.pth
- **Seed:** 1 (fixed across all runs)
- **Dataset:** `datasets/lift/ph/low_dim_v15.hdf5` with filter keys created by `filter_dataset_size.py`

### Filter keys created

```
10_demos:  10 demos (511 sequences)
25_demos:  25 demos (1218 sequences)
50_demos:  50 demos (2447 sequences)
100_demos: 100 demos (4898 sequences)
(200 demos: all data, no filter = 9666 sequences)
```

## Files

- **Configs:** `scripts/train_target_policies/config_{10,25,50,100,200}demos.json`
- **SLURM script:** `scripts/train_target_policies/train_all.sbatch`
- **Submit helper:** `scripts/train_target_policies/submit.sh` (splits into 2 SLURM jobs)
- **Output dir:** `third_party/robomimic/diffusion_policy_trained_models/lift_diffusion_{N}demos/<timestamp>/`

### Running

**Via SLURM:**
```bash
cd ~/latent_sope && bash scripts/train_target_policies/submit.sh
```

**Locally:**
```bash
cd ~/latent_sope && bash scripts/train_target_policies/train_all.sbatch       # all 5
cd ~/latent_sope && bash scripts/train_target_policies/train_all.sbatch 10    # just one
```

**Estimated time:** ~6 min per policy, ~30 min total for all 5.

## Training Results

**SLURM Job:** 7326750 (node d14-06, V100-PCIE-32GB)
**Started:** Wed Mar 11 13:40:14 PDT 2026
**Finished:** Wed Mar 11 14:24:42 PDT 2026 (~45 min total)

| Policy | Final Loss (Epoch 40) | Duration | Checkpoint | Notes |
|--------|----------------------|----------|------------|-------|
| 10 demos  | — | — | `20260311115828` | Trained separately earlier (pre-SLURM run) |
| 25 demos  | **FAILED** | ~1 min | None | `EOFError: EOF when reading a line` — likely missing filter key or data issue |
| 50 demos  | 0.0298 | ~14 min | `20260311134204` | Loss converged smoothly |
| 100 demos | 0.0335 | ~15 min | `20260311135551` | Loss converged smoothly |
| 200 demos | 0.0349 | ~14 min | `20260311141036` | Loss converged smoothly |

### Loss Trajectories (Epoch 1 → 40)

| Epoch | 50 demos | 100 demos | 200 demos |
|-------|----------|-----------|-----------|
| 1     | 0.956    | 0.957     | 0.956     |
| 5     | 0.086    | 0.085     | 0.088     |
| 10    | 0.053    | 0.053     | 0.055     |
| 20    | 0.040    | 0.043     | 0.043     |
| 30    | 0.034    | 0.037     | 0.038     |
| 40    | 0.030    | 0.033     | 0.035     |

### Notes

- 25 demos crashed immediately with `EOFError` — likely the filter key was not properly written to the HDF5 file. Needs investigation.
- All other policies have similar initial loss (~0.956) but final loss slightly increases with more data (0.030 → 0.035), which is expected — more diverse demos are harder to fit.
- Memory usage stable at ~1880 MB throughout.
- 10 demos was trained in a separate earlier run (timestamp `20260311115828`), not part of this SLURM job.

### Oracle evaluation still needed

Success rates and oracle V^π values are **not yet computed** — need to run `oracle_value()` on each checkpoint (50 rollouts each).

## Next Steps

1. **Fix 25 demos filter key** — check if `25_demos` key exists in the HDF5 file, re-create if needed, retrain
2. Run oracle evaluation (50 rollouts each) to get ground-truth V^π for all policies
3. Collect behavior data: 200 rollouts from the 200-demo policy
4. Run stitch-OPE on each target policy
5. Evaluate: Spearman rank correlation, per-policy relative error, regret@k
