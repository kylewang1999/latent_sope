# Latent Rollout → Chunk Diffuser Pipeline

This document describes the end-to-end pipeline for collecting visual latent rollouts from a trained behavior policy and training a SOPE chunk diffuser on them.

---

## Overview

```
Behavior Policy Checkpoint
        │
        ▼
1. Collect Rollouts          collect_rollout_latents.py
   (run policy in env,
    extract image latents)
        │
        ▼
2. Train Chunk Diffuser      train_sope_diffuser.py
   (TemporalUnet +
    GaussianDiffusion on
    latent-action chunks)
        │
        ▼
3. Evaluate                  eval_diffuser_rmse.py
```

---

## Stage 1 — Collect Latent Rollouts

**Script:** `scripts/collect_rollout_latents.py`

Loads a trained robomimic Diffusion Policy checkpoint, runs it in the simulation environment, and records the visual latents produced by the policy's image encoder (ResNet18 + SpatialSoftmax → 64-dim) at every timestep. Each rollout is saved as an `.h5` file.

### Command

```bash
python scripts/collect_rollout_latents.py \
    --tasks lift can \
    --epochs 50 100 200 300 400 500 600 \
    --n_rollouts 20 \
    --horizon 500 \
    --device cuda \
    --output_dir data/rollouts
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--tasks` | `lift can` | Which tasks to collect rollouts for |
| `--epochs` | `50 100 200 300 400 500 600` | Which policy checkpoint epochs to use |
| `--n_rollouts` | `20` | Number of rollouts per task/epoch |
| `--horizon` | `500` | Max timesteps per rollout |
| `--device` | `cuda` | Device to run the policy on |
| `--output_dir` | `data/rollouts` | Where to save `.h5` rollout files |

### Output

```
data/rollouts/
  lift/
    epoch_0050/
      rollout_0000.h5
      rollout_0001.h5
      ...
  can/
    epoch_0600/
      rollout_0000.h5
      ...
```

Each `.h5` file contains:
- `latents` — `(T, 64)` float32 visual latents
- `actions` — `(T, 7)` float32 actions
- `rewards`, `dones` — trajectory signals
- Attributes: `success`, `total_reward`, `horizon`, `frame_stack`

### Key source files

- `src/latent_sope/robomimic_interface/checkpoints.py`
  - `load_checkpoint()` — loads a robomimic `.pth` checkpoint from disk
  - `build_rollout_policy_from_checkpoint()` — reconstructs the callable policy
  - `build_env_from_checkpoint()` — reconstructs the simulation environment

- `src/latent_sope/robomimic_interface/rollout.py`
  - `PolicyFeatureHook` — attaches a forward hook to the policy's `ObservationEncoder` to capture the visual latent at each step without modifying the policy
  - `RolloutLatentRecorder` — accumulates `(latent, action, reward, done)` per timestep
  - `rollout()` — runs the env loop: reset → policy → step → record → repeat
  - `save_rollout_latents()` — serializes a trajectory to `.h5`

---

## Stage 2 — Train Chunk Diffuser

**Script:** `scripts/train_sope_diffuser.py`

Loads the collected rollout `.h5` files, slices them into overlapping `(frame_stack + chunk_size)` windows, and trains a SOPE-style chunk diffuser (TemporalUnet backbone + DDPM) to model the joint distribution of `(latent, action)` sequences conditioned on the past `frame_stack` latents.

### Command

```bash
python scripts/train_sope_diffuser.py \
    --task lift \
    --epochs 200 \
    --batch_size 256 \
    --chunk_size 8 \
    --frame_stack 0 \
    --diffusion_steps 256 \
    --device cuda
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | *(required)* | `lift` or `can` |
| `--data_dir` | `data/rollouts/{task}` | Directory of rollout `.h5` files (searched recursively) |
| `--checkpoint_dir` | `data/sope_checkpoints/{task}` | Where to save diffuser checkpoints |
| `--epochs` | `200` | Training epochs |
| `--batch_size` | `256` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--chunk_size` | `8` | Number of future steps per chunk (`W`) |
| `--frame_stack` | `0` | Number of past frames to condition on |
| `--stride` | `2` | Step between chunk start indices |
| `--diffusion_steps` | `256` | DDPM denoising steps |
| `--predict_epsilon` | `True` | Predict noise (`True`) or x0 (`False`) |
| `--lr_schedule` | `None` | `None` or `cosine` |
| `--save_every` | `10` | Save checkpoint every N epochs |
| `--resume` | `False` | Resume from latest checkpoint |
| `--device` | `cuda` | Training device |

> **Note:** `chunk_size + frame_stack` must be divisible by 8 (required by `dim_mults=(1,2,4,8)`).

### Output

```
data/sope_checkpoints/lift/
  configs.json
  sope_diffuser_epoch_0010.pt
  sope_diffuser_epoch_0020.pt
  ...
  sope_diffuser_latest.pt
```

Each checkpoint contains the model weights, epoch/step counters, configs, and normalization stats.

### Key source files

- `src/latent_sope/robomimic_interface/dataset.py`
  - `RolloutChunkDatasetConfig` — configures chunk size, frame stack, stride, normalization
  - `RolloutChunkDataset` — slices rollout trajectories into `(states_from, actions_from, states_to, actions_to)` chunks
  - `make_rollout_chunk_dataloader()` — aggregates multiple `.h5` files into a single dataloader with global normalization stats

- `src/latent_sope/diffusion/sope_diffuser.py`
  - `SopeDiffusionConfig` — configures the diffusion model (dims, loss type, action weight, etc.)
  - `SopeDiffuser` — wraps `TemporalUnet` (denoiser) and `GaussianDiffusion` (DDPM logic)
  - `make_cond()` — builds the inpainting conditioning dict: pins past `frame_stack` latents into the trajectory prefix at each denoising step
  - `loss()` — concatenates context and target chunks along the time axis and computes the DDPM loss

- `src/latent_sope/diffusion/train.py`
  - `TrainingConfig` — training hyperparameters (lr, grad clip, scheduler, etc.)
  - `train()` — the main training loop: dataloader → loss → backward → checkpoint

### Conditioning style

The diffuser uses **inpainting-style** conditioning. At every reverse diffusion step, the known past latent states (`states_from`) are hard-written back into the noisy trajectory tensor `x`. This works because the conditioning signal lives in the same latent space as the trajectory being diffused.

---

## Chunk layout

```
time:   [ t-S, ..., t-1 | t, t+1, ..., t+W ]
         ◄─ frame_stack ─►◄─── chunk_size ───►
         states_from          states_to / actions_to
         (conditioned on,     (diffused / predicted)
          pinned by inpainting)
```

The full sequence fed to the TemporalUnet has length `frame_stack + chunk_size`, with shape `(B, frame_stack + chunk_size, latent_dim + action_dim)`.

---

## Environment setup

Checkpoints are expected at:
```
third_party/robomimic/diffusion_policy_trained_models/
  lift_mh/lift_mh_diffusion/<run>/models/model_epoch_N.pth
  can_mh/can_mh_diffusion/<run>/models/model_epoch_N.pth
```

MuJoCo rendering requires:
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
```
(These are set automatically by the collection script.)
