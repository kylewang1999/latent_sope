# Latent SOPE — Component Reference

## Data Flow

```
Robomimic Policy  →  Rollout + PolicyFeatureHook
                          ↓
                  RolloutLatentTrajectory (.h5/.npz)
                          ↓
                  RolloutChunkDataset (chunked sequences)
                          ↓
                  DataLoader (batched chunks)
                          ↓
                  SopeDiffuser.loss()  →  Training loop
                          ↓
                  Checkpoint (weights + configs + norm stats)
```

---

## `robomimic_interface/`

Bridges robomimic policies/environments with the latent diffusion pipeline.

### `checkpoints.py` — Checkpoint loading

`RobomimicCheckpoint` wraps a robomimic experiment directory (`run_dir`) and
its `.pth` weights file. Helper functions reconstruct live objects from a
checkpoint:

| Function | Returns |
|----------|---------|
| `load_checkpoint(run_dir, ckpt_path)` | `RobomimicCheckpoint` with parsed config + loaded state dict |
| `build_algo_from_checkpoint(ckpt)` | robomimic `PolicyAlgo` (BC, DiffusionPolicyUNet, etc.) |
| `build_rollout_policy_from_checkpoint(ckpt)` | callable `RolloutPolicy` |
| `build_env_from_checkpoint(ckpt)` | robomimic environment |
| `load_demo(h5, demo_key, obs_keys)` | obs/action arrays from an HDF5 demo |
| `prepare_obs(obs, device, obs_stats)` | batched + normalized obs tensors ready for policy forward |

### `encoders.py` — Feature extraction

Two encoder strategies for converting observations into latent vectors:

- **`LowDimConcatEncoder`** — Concatenates selected low-dim obs keys into a
  flat vector. Supports `decode_to_obs_dict(z)` for reconstruction. No learned
  parameters.

- **`HighDimObsEncoder`** — Registers a forward hook on a named submodule of
  the policy network (e.g. `"nets.policy.obs_encoder"`) and captures its
  activations. Use `encode_obs_batch(obs_batch, device)` to get `(T, Dz)`
  latents.

`resolve_module(root, dotted_path)` navigates nested `nn.Module` attributes.

### `dataset.py` — Chunk dataset

`RolloutChunkDataset` slices a `RolloutLatentTrajectory` into overlapping
chunks for diffusion training. Each sample is a dict:

| Key | Shape | Description |
|-----|-------|-------------|
| `states_from` | `(frame_stack, Dz)` | Past states ending at t−1 (conditioning) |
| `actions_from` | `(frame_stack, Da)` | Past actions (conditioning) |
| `states_to` | `(chunk_size+1, Dz)` | Future state chunk starting at t |
| `actions_to` | `(chunk_size, Da)` | Future action chunk starting at t |
| `metadata` | dict | Optional: demo id, time indices, etc. |

`RolloutChunkDatasetConfig` controls `chunk_size`, `stride`, `frame_stack`,
`source` (`"latents"` or `"obs"`), dimensions, and normalization.

`make_rollout_chunk_dataloader(paths, config, batch_size, ...)` loads multiple
rollout files, concatenates them into one dataset, computes super-trajectory
normalization stats, and returns a `DataLoader`.

### `rollout.py` — Trajectory recording

- **`PolicyFeatureHook`** — Forward hook that captures encoder outputs during
  `policy.get_action()`. Resolves the hook target from a set of candidate
  module paths. Call `pull_feat()` to retrieve the latest latent.

- **`RolloutLatentRecorder`** — Step-by-step recorder that accumulates
  latents, actions, rewards, and dones into a `RolloutLatentTrajectory`.

- **`rollout(policy, env, horizon, ...)`** — Runs a policy in an env for
  `horizon` steps, optionally writing video frames and recording latents via a
  `RolloutLatentRecorder`. Returns `RolloutStats`.

- **`save_rollout_latents` / `load_rollout_latents`** — Serialize trajectories
  to `.npz` or `.h5` (with compression).

---

## `diffusion/`

Chunk-level diffusion model built on top of `third_party/sope`.

### `sope_diffuser.py` — Model wrapper

`SopeDiffuser` combines a `TemporalUnet` backbone with a `GaussianDiffusion`
sampler from the SOPE library.

- **Input representation**: each training sample is a `(B, W, transition_dim)`
  tensor where `transition_dim = state_dim + action_dim` and
  `W = chunk_horizon + frame_stack`.
- **Conditioning**: `make_cond(batch)` builds a dict mapping timestep indices
  to frame-stack states, which the diffusion model treats as inpainting
  constraints.
- **`loss(batch, cond)`** — Forward diffusion + denoising loss (L2 by default).
- **`sample(num_samples, cond)`** — Reverse-process sampling to generate
  trajectory chunks.

`SopeDiffusionConfig` holds architecture params (`dim_mults`, `attention`),
diffusion params (`diffusion_steps`, `predict_epsilon`, `loss_type`,
`action_weight`), and guidance flags (WIP).

`cross_validate_configs(cfg_dataset, cfg_diffusion)` checks that dataset and
diffusion configs agree on `latents_dim`, `action_dim`, and that `chunk_horizon
+ frame_stack` is divisible by the UNet's downsampling factor.

### `train.py` — Training loop

`train(cfg_dataset, cfg_diffusion, cfg_training)` runs the full training
pipeline:

1. Seed, resolve device, validate configs.
2. Build `DataLoader` from rollout file paths.
3. Instantiate `SopeDiffuser` + Adam optimizer.
4. Epoch loop: forward → loss → backward → gradient clip → step.
5. Periodic logging and checkpoint saving.

`TrainingConfig` controls `epochs`, `batch_size`, `lr`, `grad_clip`,
`max_steps`, `log_every`, `save_every`, and data paths.

Checkpoints contain the diffusion state dict, epoch/step counters, both
configs, and normalization statistics.

---

## `eval/`

### `metrics.py` — Reconstruction metrics

`l2_chunk_error(x_hat, x_gt)` compares predicted and ground-truth chunks of
shape `(N, W, D)`. Returns an `L2ChunkError` dataclass with `mean_l2`,
`std_l2`, and `rmse_per_dim` (shape `(D,)`).

---

## `utils/`

### `common.py` — Logging, config, and I/O helpers

- **Logging**: `get_console_logger()` creates a rich-formatted logger;
  `make_log_dir()` creates timestamped output directories.
- **Config**: `load_configs()` / `save_configs()` use Hydra + OmegaConf.
- **Video**: `display_video()` renders image arrays as inline MP4 in Jupyter.
- **Checkpointing**: `save_nnx_module()` / `restore_nnx_module()` persist
  Flax NNX models via Orbax.
- **WandB**: `wandb_log_artifact()` logs files/directories as artifacts.
- **Misc**: `timeit` decorator, `catch_keyboard_interrupt` decorator,
  `save_dict_to_json`.

### `misc.py` — Seeding and device resolution

- `set_global_seed(seed)` — Seeds `random`, `numpy`, `torch`, and CUDA.
- `resolve_device(prefer_cuda)` — Returns `"cuda"` or `"cpu"`.
- `DeviceConfig` — Frozen dataclass holding device string and dtype with a
  `torch_dtype()` accessor.
