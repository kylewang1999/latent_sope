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

## Results

_To be filled in after training completes._

| Policy | Best Epoch | Success Rate | Oracle V^π | Notes |
|--------|-----------|--------------|------------|-------|
| 10 demos  | | | | |
| 25 demos  | | | | |
| 50 demos  | | | | |
| 100 demos | | | | |
| 200 demos | | | | |

## Next Steps

1. After training: run oracle evaluation (50 rollouts each) to get ground-truth V^π
2. Collect behavior data: 200 rollouts from the 200-demo policy
3. Run stitch-OPE on each target policy
4. Evaluate: Spearman rank correlation, per-policy relative error, regret@k
