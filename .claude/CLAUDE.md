# Latent SOPE

Latent-space offline policy evaluation using SOPE-guided trajectory chunk diffusion, built on top of robomimic policies.

## Project Structure

```
src/latent_sope/
  diffusion/
    sope_diffuser.py    # SopeDiffuser: wraps TemporalUnet + GaussianDiffusion from third_party/sope
    train.py            # Training loop for chunk diffusion (latents-only mode)
  robomimic_interface/
    checkpoints.py      # Load robomimic checkpoints, build PolicyAlgo / RolloutPolicy / env
    encoders.py         # LowDimConcatEncoder, HighDimObsEncoder (hook-based feature extraction)
    dataset.py          # RolloutChunkDataset: chunks rollout trajectories into (states_from, actions_from, states_to, actions_to)
    rollout.py          # PolicyFeatureHook, RolloutLatentRecorder, rollout(), save/load latent trajectories (.npz/.h5)
    collect.py          # [Step 1] Batch offline data collection: collect_rollouts(), discover_obs_keys()
  eval/
    oracle.py           # [Step 0] Ground-truth V^pi: oracle_value(), save/load_oracle_result()
    metrics.py          # L2 chunk reconstruction error
  utils/
    common.py           # Logging (rich), config (hydra/omegaconf), video display, nnx save/load, wandb helpers
    misc.py             # Seeding, device resolution
scripts/
  latent_sope.ipynb     # Main pipeline notebook (Steps 0-7)
  hello_robomimic.ipynb # Environment test notebook
  hello_stitch_ope.ipynb # SOPE demo on D4RL
third_party/
  sope/                 # SOPE repo (TemporalUnet, GaussianDiffusion, diffuser baselines)
  robomimic/            # robomimic (editable install via submodule)
  clean_diffuser/       # CleanDiffuser (editable install, --no-deps)
```

## Key Concepts

- **Chunk diffusion**: Diffusion model over (state, action) trajectory chunks of length `chunk_horizon + frame_stack`. The model denoises concatenated `[states, actions]` sequences conditioned on `frame_stack` past states.
- **Latent trajectories**: Rollouts produce `RolloutLatentTrajectory` objects saved as `.h5`/`.npz`. Latents come from policy encoder hooks (`PolicyFeatureHook`) or low-dim obs concatenation.
- **SOPE guidance**: Uses `GaussianDiffusion` from `third_party/sope` with optional policy/behavior_policy for guided sampling (not yet fully wired up - see TODOs in `sope_diffuser.py`).

## Environment Setup

- Python 3.10 required (mujoco_py + d4rl constraint)
- CUDA 12-based wheels (PyTorch cu126, JAX cuda12)
- Run `bash bootstrap_env.sh` after `conda activate latent_sope`
- Requires `LD_LIBRARY_PATH` to include `~/.mujoco/mujoco210/bin` **and** `/usr/lib/nvidia`
- Run `bash bootstrap_egl.sh` for headless rendering (EGL)
- Submodules: `git submodule update --init --recursive`

## Known Setup Bugs (found 2026-03-09)

The README setup instructions have several gaps that will cause failures. See below.

### Bug 1: `bootstrap_env.sh` installs Cython 3.x, which breaks `mujoco_py`

`mujoco_py` compiles a Cython extension (`cymj.pyx`) on first import. Cython 3.x introduced
`noexcept` enforcement that is incompatible with `mujoco_py`'s callback signatures. The
bootstrap script does not pin Cython, so pip resolves Cython 3.x (pulled in transitively by
numba or other deps), and `import mujoco_py` fails with:
```
Cannot assign type 'void (const char *) except * nogil' to 'void (*)(const char *) noexcept nogil'
```

**Fix:** Add `pip install "cython<3"` *after* all other installs in `bootstrap_env.sh` (must be
last because `--force-reinstall` or transitive deps can pull Cython 3.x back).

### Bug 2: `bootstrap_env.sh` is missing GLEW headers (`GL/glew.h`)

Even after Cython compiles `cymj.pyx`, the C compilation step fails because `mujoco_py`'s
`eglshim.c` includes `<GL/glew.h>`, which is not available on CARC nodes (no `glew-devel`
system package). The bootstrap script does not install it.

**Fix:** Add `conda install -c conda-forge glew mesalib mesa-libgl-cos7-x86_64 -y` to the
bootstrap script.

### Bug 3: `bootstrap_env.sh` is missing `patchelf`

After Cython + C compilation succeeds, `mujoco_py`'s builder calls `patchelf --remove-rpath`
on the built `.so`, but `patchelf` is not installed.

**Fix:** Add `pip install patchelf` to the bootstrap script.

### Bug 4: README `LD_LIBRARY_PATH` is missing `/usr/lib/nvidia`

The README (step 4) says to add:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
```
But `mujoco_py` also requires the NVIDIA library path. Without it, `mujoco_py` errors with:
```
Missing path to your environment variable ... Please add: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

**Fix:** The instruction should be:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

### Bug 5: Jupyter kernel needs explicit env vars for `mujoco_py` + EGL

When running the notebook via Jupyter, the kernel does not inherit the conda activate hooks
(EGL vars) or the shell's `LD_LIBRARY_PATH`. This means `mujoco_py` fails to import and
offscreen rendering breaks inside the notebook even if the terminal works fine.

**Fix:** After `ipykernel install`, patch the kernel.json to include `env` with
`LD_LIBRARY_PATH`, `C_INCLUDE_PATH`, `CPATH`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
and `PYTHONNOUSERSITE=1` (to avoid polluting with `~/.local` site-packages from a different
Python version).

### Bug 6 (environment-specific): `PYTHONNOUSERSITE` needed when multiple Python versions exist

If the user has packages in `~/.local/lib/python3.11/site-packages` (from a base env or prior
install), Python 3.10 in the conda env will still pick up 3.11 user-site packages due to how
`site.py` works. This causes subtle import conflicts. `conda run` is particularly affected -
it reports Python 3.11 even when the env has 3.10, because the user-site `site.py` takes over.

**Fix:** Set `PYTHONNOUSERSITE=1` in scripts and kernel specs, or avoid `conda run` in favor
of directly invoking the env's Python (`/path/to/envs/latent_sope/bin/python`).

## OPE Pipeline Progress

The pipeline is documented in `scripts/latent_sope.ipynb`. Steps 0-3 are complete, Steps 4-7 need work.

### Step 0: Ground Truth — DONE
- `eval/oracle.py`: `oracle_value(ckpt, num_rollouts, horizon)` → `OracleResult`
- `save_oracle_result()` / `load_oracle_result()` persist to JSON
- Only supports gamma=1.0; for discounted use `oracle_value_from_trajectories()`
- Tested end-to-end on Lift diffusion policy checkpoint

### Step 1: Collect Offline Data — DONE
- `robomimic_interface/collect.py`: `collect_rollouts(ckpt, output_dir, num_rollouts)` → `CollectionResult`
- `discover_obs_keys(ckpt)` auto-discovers low-dim obs keys
- Saves `.h5` files via `save_rollout_latents()`, consumed directly by Step 2
- Uses `LowDimConcatEncoder` (feat_type="low_dim_concat") — latents = concatenated obs keys
- Tested: 3 rollouts on Lift, latents shape (T, 2, 19), actions (T, 7)

### Step 2: Chunk the Offline Data — DONE
- `dataset.py`: `make_rollout_chunk_dataloader(paths, config)` → DataLoader + NormalizationStats
- Takes .h5 paths from Step 1, chunks into `(states_from, actions_from, states_to, actions_to)`
- Normalization computed at super-trajectory level across all rollout files
- Config: `RolloutChunkDatasetConfig(chunk_size=8, stride=2, frame_stack=2, source="latents")`
- Tested: batch shapes `states_from (B,2,19)`, `states_to (B,9,19)`, `actions_to (B,8,7)`
- **Gotcha**: latents are (T, frame_stack, D) — use `shape[-1]` not `shape[1]` for feature dim

### Step 3: Train Chunk Diffusion — DONE
- `diffusion/train.py`: `train()` loop with gradient clipping, checkpointing
- `diffusion/sope_diffuser.py`: `SopeDiffuser` wraps TemporalUnet + GaussianDiffusion
- `cross_validate_configs()` checks dataset↔diffusion dim alignment
- Notebook inlines training loop for visibility; `train.train()` also works standalone
- Checkpoint saved to `policy_train_dir/diffusion_ckpts/sope_diffuser_latest.pt`
- Tested: 5 epochs on 10 rollouts, loss decreasing, checkpoint saves correctly
- Only `source='latents'` works (`source='obs'` raises ValueError in train.py:108)
- No validation loss / eval during training (not yet implemented)

### Current Scale & Scaling Plan
- **Current notebook settings**: K=10 oracle rollouts, N_ROLLOUTS=5 offline, EPOCHS=5 training
- Each rollout takes ~15s (Lift, horizon=60). Each ~60-step trajectory yields ~25 chunks (chunk_size=8, stride=2).
- 5 rollouts = ~125 chunks = only 2 batches of 64. The diffusion model barely sees any data.
- **Next milestone (medium)**: 50 rollouts + 50 epochs. Step 1 ≈ 12 min, Step 3 ≈ 5 min. Total ≈ 20 min.
- **Full scale**: 200 rollouts + 100 epochs. Step 1 ≈ 50 min, Step 3 ≈ 30 min. Total ≈ 1.5 hr.
- Oracle (Step 0) should also scale: K=50–100 for tighter estimate (~12–25 min).

### Step 4: Policy Guidance — NEEDS BUILDING
- SOPE's `GaussianDiffusion` has guidance via `gradlog_diffusion()` which calls `policy.grad_log_prob(state, action)`
- Robomimic policies don't expose `grad_log_prob`. Need wrapper per policy type:
  - BC_Gaussian: extract mean/std → analytic Gaussian log-prob
  - BC_GMM: log-sum-exp over mixture components
  - DiffusionPolicyUNet: use diffusion score
- `SopeDiffuser.sample()` passes `guided=self.cfg.guided` but ignores `guidance_hyperparams`
- With LowDimConcatEncoder, latents=obs so policy can consume states directly
- For MVP: skip guidance, sample unguided first

### Step 5: Stitching Loop — NEEDS BUILDING
- `SopeDiffuser.sample()` generates single chunks but stitching loop is TODO (sope_diffuser.py:28)
- Need to port `Diffuser.generate_full_trajectory()` from third_party/sope
- Prototype exists in latent_sope.ipynb Step 5 cell (`stitch_trajectory_latent()`)
- Sub-tasks: autoregressive conditioning, termination predicate, initial state sampling, end_indices tracking
- Robomimic tasks use fixed horizon (no early termination), simplifying this

### Step 6: Reward Estimation — NEEDS BUILDING
- Option A (MVP): decode latents→obs via `LowDimConcatEncoder.decode_to_obs_dict()`, use env reward function
- Option B: train MLP reward model R(z,a)→r on offline (latent, action, reward) tuples
- Need discounted return computation: sum(gamma^t * r_t)
- Nothing exists in src yet for this

### Step 7: OPE Evaluation — NEEDS BUILDING
- Compare OPE estimate to oracle value from Step 0
- Metrics needed: MSE, Spearman rank correlation (across multiple policies), Regret@k
- `eval/metrics.py` only has `l2_chunk_error()` currently
- Need multi-policy evaluation harness

### Critical Path (MVP)
1. Use LowDimConcatEncoder (latents = obs, keeps reward/guidance in obs space)
2. Start unguided (skip Step 4)
3. Build stitching loop (Step 5)
4. Score with true reward via decoder (Step 6 Option A)
5. Add guidance later once unguided pipeline works end-to-end

## Development Notes

- The codebase uses both JAX/Flax (in `utils/common.py` for nnx modules) and PyTorch (for diffusion and robomimic)
- Training currently only supports `source='latents'` mode (not `source='obs'`)
- `RolloutChunkDataset` produces batches with keys: `states_from`, `actions_from`, `states_to`, `actions_to`, `metadata`
- Normalization is computed at the super-trajectory level across all rollout files, not per-trajectory
- Configs are frozen dataclasses: `SopeDiffusionConfig`, `RolloutChunkDatasetConfig`, `TrainingConfig`
- `cross_validate_configs()` checks dimension alignment between dataset and diffusion configs
- Test checkpoint: `third_party/robomimic/diffusion_policy_trained_models/test/20260309132349/`
- Conda env Python: `/home1/reishuen/miniconda3/envs/latent_sope/bin/python`
- Always use `PYTHONNOUSERSITE=1` when running outside conda activate (avoids Python 3.11 site-packages conflicts)
- `@timeit` decorator on `rollout()` is noisy for batch operations; `oracle.py` and `collect.py` suppress it by temporarily raising CONSOLE_LOGGER level
