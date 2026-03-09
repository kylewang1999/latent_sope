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
  eval/
    metrics.py          # L2 chunk reconstruction error
  utils/
    common.py           # Logging (rich), config (hydra/omegaconf), video display, nnx save/load, wandb helpers
    misc.py             # Seeding, device resolution
scripts/
  hello_robomimic.ipynb # Environment test notebook
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

## Development Notes

- The codebase uses both JAX/Flax (in `utils/common.py` for nnx modules) and PyTorch (for diffusion and robomimic)
- Training currently only supports `source='latents'` mode (not `source='obs'`)
- `RolloutChunkDataset` produces batches with keys: `states_from`, `actions_from`, `states_to`, `actions_to`, `metadata`
- Normalization is computed at the super-trajectory level across all rollout files, not per-trajectory
- Configs are frozen dataclasses: `SopeDiffusionConfig`, `RolloutChunkDatasetConfig`, `TrainingConfig`
- `cross_validate_configs()` checks dimension alignment between dataset and diffusion configs
