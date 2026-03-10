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
    metrics.py          # L2 chunk reconstruction error, OPE eval (ope_eval → OPEResult)
    reward_model.py     # [Step 6] Ground-truth reward (LiftRewardFn) + learned MLP fallback
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

See `memory/setup_bugs.md` for full details. Summary of fixes needed for `bootstrap_env.sh`:
1. Pin `cython<3` (after all other installs) — mujoco_py Cython 3.x incompatibility
2. `conda install -c conda-forge glew mesalib mesa-libgl-cos7-x86_64 -y` — missing GLEW headers
3. `pip install patchelf` — missing patchelf
4. `LD_LIBRARY_PATH` must include both `~/.mujoco/mujoco210/bin` and `/usr/lib/nvidia`
5. Jupyter kernel.json needs env vars: `LD_LIBRARY_PATH`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `PYTHONNOUSERSITE=1`
6. Always use `PYTHONNOUSERSITE=1` when running outside conda activate

## OPE Pipeline Progress

The pipeline is documented in `scripts/latent_sope.ipynb`. Steps 0-3, 5-7 are complete. Step 4 (guidance) skipped for MVP.

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

### Step 4: Policy Guidance — SKIPPED FOR MVP
- SOPE's `GaussianDiffusion` has guidance via `gradlog_diffusion()` which calls `policy.grad_log_prob(state, action)`
- Robomimic policies don't expose `grad_log_prob`. Need wrapper per policy type:
  - BC_Gaussian: extract mean/std → analytic Gaussian log-prob
  - BC_GMM: log-sum-exp over mixture components
  - DiffusionPolicyUNet: use diffusion score
- `SopeDiffuser.sample()` passes `guided=self.cfg.guided` but ignores `guidance_hyperparams`
- With LowDimConcatEncoder, latents=obs so policy can consume states directly
- Currently skipped: sampling unguided first to validate pipeline end-to-end

### Step 5: Stitching Loop — DONE
- `SopeDiffuser.generate_full_trajectory()` in `sope_diffuser.py` — fully implemented
- Autoregressive chunk generation: condition chunk k+1 on last `frame_stack` states of chunk k
- Conditioning always in **normalized space**; outputs stored **unnormalized**
- Next chunk conditioning extracted from still-normalized diffusion sample (avoids round-trip error)
- `apply_conditioning()` re-pins conditioned states after every denoising step
- Fixed-horizon cutoff (no termination predicate needed for robomimic tasks)
- `end_indices` tracking per-trajectory
- Tested: generates (B, T, state_dim) states and (B, T, action_dim) actions

### Step 6: Reward Estimation — DONE (Ground-Truth)
- **Decision: Use ground-truth analytical reward** (not learned MLP)
- `LiftRewardFn` in `eval/reward_model.py`: checks `cube_z > table_height + 0.04` → reward 1.0 or 0.0
- `score_trajectories_gt()`: decodes latents → obs dict via `LowDimConcatEncoder.decode_to_obs_dict()`, applies reward fn, computes discounted returns
- Pure numpy, no training needed, zero approximation error
- Learned MLP (`RewardMLP`, `train_reward_model`, `score_trajectories`) kept in same file as fallback
- Tested end-to-end with stitched trajectories

#### Lift Reward Function Details
- Robosuite Lift reward: sparse, `2.25 * reward_scale / 2.25 = 1.0` when cube lifted
- Success condition: `cube_z > table_height + 0.04` where `table_height = 0.8` → threshold `0.84`
- `object` obs key (10-dim): `[cube_pos(3), cube_quat(4), gripper_to_cube_pos(3)]`
- Cube z-position is at **index 2** of the `object` key (index 2 of the 19-dim latent)
- Resting cube height ≈ 0.8208m (on table surface)

#### Latent Vector Layout (LowDimConcatEncoder, sorted obs_keys)
| Index | Key | Dim | Content |
|-------|-----|-----|---------|
| 0–9   | `object` | 10 | cube_pos(3) + cube_quat(4) + gripper_to_cube(3) |
| 10–12 | `robot0_eef_pos` | 3 | end-effector XYZ |
| 13–16 | `robot0_eef_quat` | 4 | end-effector quaternion |
| 17–18 | `robot0_gripper_qpos` | 2 | gripper joint positions |

#### How SOPE Does Reward Scoring (Reference)
- SOPE uses **two options**: `reward_fn(env, state, action)` (analytical, env-specific) or learned `RewardEnsembleEstimator` (MLP `[64,64,1]`, MSE loss, 1000 iterations)
- D4RL experiments always use learned model; Gym experiments use analytical when available
- Analytical reward_fn in SOPE reconstructs MuJoCo state via `env.set_state(qpos, qvel)` then calls `env.step()`
- Our approach is cleaner: directly check obs values without simulator, since Lift reward only depends on cube z-position
- Learned model is kept as fallback for tasks with complex/unknown reward functions

### Step 7: OPE Evaluation — DONE
- `eval/metrics.py`: `ope_eval(oracle_value, synthetic_returns)` → `OPEResult`
- `OPEResult` dataclass: `oracle_value`, `ope_estimate`, `mse`, `relative_error`, `ope_std`
- Tested end-to-end
- Multi-policy evaluation harness (Spearman rank correlation, Regret@k) not yet implemented

### Next Steps
1. **Signs-of-life run**: 50 rollouts (already collected in `rollout_latents_50/`) + 50 epochs
   - Expect: generated states stay in-distribution (cube z near 0.82), OPE estimate in [0, 1]
   - If noisy: scale to 200 rollouts + 100 epochs (~1.5 hr)
2. Add policy guidance (Step 4) once unguided pipeline produces reasonable estimates
3. Multi-policy evaluation harness for rank correlation

## Development Notes

- The codebase uses both JAX/Flax (in `utils/common.py` for nnx modules) and PyTorch (for diffusion and robomimic)
- Training currently only supports `source='latents'` mode (not `source='obs'`)
- `RolloutChunkDataset` produces batches with keys: `states_from`, `actions_from`, `states_to`, `actions_to`, `metadata`
- Normalization is computed at the super-trajectory level across all rollout files, not per-trajectory
- Configs are frozen dataclasses: `SopeDiffusionConfig`, `RolloutChunkDatasetConfig`, `TrainingConfig`
- `cross_validate_configs()` checks dimension alignment between dataset and diffusion configs
- Test checkpoint: `third_party/robomimic/diffusion_policy_trained_models/test/20260309132349/`
- Existing rollout data: `rollout_latents/` (5 rollouts), `rollout_latents_50/` (50 rollouts)
- Conda env Python: `/home1/reishuen/miniconda3/envs/latent_sope/bin/python`
- Always use `PYTHONNOUSERSITE=1` when running outside conda activate (avoids Python 3.11 site-packages conflicts)
- `@timeit` decorator on `rollout()` is noisy for batch operations; `oracle.py` and `collect.py` suppress it by temporarily raising CONSOLE_LOGGER level

## SOPE Reference Implementation (third_party/sope)

Key files for understanding the reference implementation:
- **Stitching loop**: `third_party/sope/opelab/core/baselines/diffuser.py:229–351` — `generate_full_trajectory()`
- **Diffusion sampling**: `third_party/sope/opelab/core/baselines/diffusion/diffusion.py` — `GaussianDiffusion.conditional_sample()`, `p_sample_loop()`
- **Conditioning**: `third_party/sope/opelab/core/baselines/diffusion/helpers.py:159–172` — `apply_conditioning()`
- **Guidance**: `third_party/sope/opelab/core/baselines/diffusion/diffusion.py:31–110` — `gradlog()`, `gradlog_diffusion()`
- **Reward model**: `third_party/sope/opelab/core/reward.py` — `RewardEnsembleEstimator`
- **Eval harness**: `third_party/sope/opelab/examples/helpers.py:184–326` — `evaluate_policies()`
