# Training Pipeline, Environment Bootstrap, Policy Preparation, And Data Workflows

This note consolidates the repository's training-pipeline, training-entrypoint,
environment-bootstrap, policy-preparation, and rollout-dataset workflows.

Relevant code:

- [bootstrap_env.sh](../bootstrap_env.sh)
- [bootstrap_egl.sh](../bootstrap_egl.sh)
- [src/utils.py](../src/utils.py)
- [src/train.py](../src/train.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/eval.py](../src/eval.py)
- [scripts/create_rollout_dataset.py](../scripts/create_rollout_dataset.py)
- [scripts/prepare_policy_hm-image.py](../scripts/prepare_policy_hm-image.py)
- [scripts/train_sope.py](../scripts/train_sope.py)
- [scripts/train_sope_gym.py](../scripts/train_sope_gym.py)
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
- [src/robomimic_interface/checkpoints.py](../src/robomimic_interface/checkpoints.py)
- [src/sope_interface/dataset.py](../src/sope_interface/dataset.py)
- [third_party/robomimic/setup.py](../third_party/robomimic/setup.py)

## 1. Summary

The active training stack uses:

- [bootstrap_env.sh](../bootstrap_env.sh) and [bootstrap_egl.sh](../bootstrap_egl.sh)
  as the canonical environment-bootstrap path
- [src/train.py](../src/train.py) as the canonical orchestration module
- [src/diffusion.py](../src/diffusion.py) as the canonical diffusion wrapper
- [scripts/prepare_policy_hm-image.py](../scripts/prepare_policy_hm-image.py)
  as the policy-staging helper for the main robomimic checkpoint workflow
- [scripts/create_rollout_dataset.py](../scripts/create_rollout_dataset.py)
  as the rollout-dataset generation entrypoint
- [scripts/train_sope.py](../scripts/train_sope.py) as the latent-trajectory
  training CLI, including postprocessed robomimic demos
- [scripts/train_sope_gym.py](../scripts/train_sope_gym.py) as the SOPE Gym
  state-dataset CLI
- [src/eval.py](../src/eval.py) as the reusable evaluation path

`train_sope(...)` and `train_rewardpred(...)` are now the canonical entrypoints
for both internally constructed loaders and externally supplied loaders. The old
loader-only wrappers no longer own separate implementations.

## 2. Data And Chunking Path

The latent-trajectory path builds chunk datasets from saved latent-trajectory
files under
[src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py).
The intended chunk contract is:

- `states_from`: frame-stacked prefix ending at $t_0 - 1$
- `actions_from`: aligned action prefix
- `states_to`: latent states from $t_0$ through $t_0 + W$
- `actions_to`: actions from $t_0$ through $t_0 + W - 1$

When training from many rollout trajectory files, the train and eval split is
performed at the trajectory-file level so chunks from the same trajectory do
not leak across splits.

### 2.1 Conditioning interpretation

The local SOPE-aligned reading of these fields is:

- condition on `states_from`
- predict the future transition chunk carried by `states_to` and `actions_to`
- do not treat `actions_from` as part of the observed prefix that must be
  clamped during denoising

This distinction matters because the local FiLM-conditioned path uses the
prefix as external context, not as generated content that must be overwritten
inside the sampled trajectory tensor at every reverse step.

### 2.2 SOPE Gym adapter

The SOPE Gym path under
[src/sope_interface/dataset.py](../src/sope_interface/dataset.py) converts
flat arrays into the same chunk contract by:

1. loading observations, actions, rewards, and terminals
2. reconstructing episode boundaries
3. splitting train and eval at the episode level
4. emitting the same `states_from` / `actions_from` / `states_to` /
   `actions_to` fields

## 3. Policy Preparation And Rollout Dataset Generation

Prepared policy artifacts and generated rollout datasets belong to the same
operational path because the rollout-backed training and evaluation stack
expects policy checkpoints under `data/policy/<policy-name>` and rollout files
under `data/rollout/<policy-name>`.

### 3.1 Policy preparation

[`scripts/prepare_policy_hm-image.py`](../scripts/prepare_policy_hm-image.py)
prepares the multi-human image diffusion policy archive under
`data/policy/rmimic-lift-mh-image-v15-diffusion_260123`.

Its behavior is:

- install `gdown` only if importing it fails
- download the Google Drive archive into `data/`
- extract into a temporary directory under `data/`
- skip macOS metadata such as `__MACOSX` and `._*`
- locate the real payload by searching for a directory containing both
  `config.json` and `models/`
- rename and stage that payload under `data/policy/`
- remove the downloaded zip after a successful prepare

### 3.2 Rollout dataset generation

[`scripts/create_rollout_dataset.py`](../scripts/create_rollout_dataset.py)
generates one rollout-latent `.h5` file per trajectory from a prepared
robomimic policy under `data/policy/<policy-name>`.

Default output location:

`data/rollout/<policy-name>`

Default behavior:

- resolve the policy checkpoint root from `data/policy/<policy-name>`
- roll out the requested number of trajectories
- save one `.h5` file per trajectory
- write rendered videos under `data/rollout/<policy-name>/videos/`
- store helpful HDF5 attrs such as `object_init_loc`, `rollout_index`,
  `policy_train_dir`, and `checkpoint_path`

Saved filenames include the rollout index prefix, for example
`0007_<object_init_loc>_len60.h5`.

### 3.3 Parallel rollout execution and rendering

Rollout generation uses process-level parallelism, not thread-level
parallelism.

Design choice:

- robosuite / MuJoCo environment reset and XML reconstruction were not reliable
  under Python threads
- worker processes avoid the thread-safety issues seen with concurrent
  `reset_to(...)` calls
- CUDA-backed workers use the `spawn` start method so each process initializes
  CUDA cleanly

Each worker process owns its own checkpoint object, rollout policy,
environment, and feature hook, while the parent process owns the single
`tqdm` progress bar and updates it as futures complete.

The `--num-render` flag defaults to `10`. When `num_render > 0`, the rendering
stride is

$$\begin{align}
\texttt{render\_stride} = \max\left(1, \left\lfloor \frac{\texttt{num\_traj}}{\texttt{num\_render}} \right\rfloor \right).
\end{align}$$

Setting `num_render = 0` disables video rendering.

### 3.4 Interaction with the dataloader

The generated `.h5` files are intended to work directly with
[src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py).

Practical constraints:

- rollout files should remain directly under `data/rollout/<policy-name>/`
- `videos/` does not interfere because the loader only gathers top-level
  rollout files
- `source="latents"` works because the files contain `latents` and `actions`
- `source="obs"` works when the rollout files include `obs`, which the current
  rollout script stores

## 4. Entry Point Consolidation

`train_sope(...)` accepts optional prebuilt:

- `loader`
- `eval_loader`
- `stats`
- `train_data_refs`
- `eval_data_refs`

When those are omitted, it preserves the rollout-backed behavior:

1. split rollout files with `_split_rollout_paths(...)`
2. build train and eval loaders
3. assign shared normalization stats to the eval dataset when needed
4. build `SopeDiffuser`
5. delegate to the shared training loop

`train_rewardpred(...)` mirrors the same pattern for reward-model training while
keeping the reward dataset unnormalized.

[`scripts/train_sope_gym.py`](../scripts/train_sope_gym.py) now calls
`train_sope(...)` directly with prebuilt SOPE Gym loaders instead of routing
through a separate loader-only wrapper.

## 5. Scheduling And Loop Behavior

The training loop uses a single epoch-level progress bar and converts
count-style scheduling knobs into epoch intervals:

- `num_evals` uses floored division with a clamp to at least `1` when enabled
- `num_saves` uses a conservative ceil-based interval

Training-time evaluation is orchestrated through
[src/eval.py](../src/eval.py). The loop logs:

- training loss metrics
- epoch summaries
- held-out chunk-evaluation metrics such as loss and RMSE under `eval_metrics/*`
- held-out chunk diagnostics under `eval_diagnostics/*`
- no guided training-time evaluation metrics; guided sampling remains a
  checkpoint-backed offline evaluation handled by
  [`scripts/test_ope_guided.py`](../scripts/test_ope_guided.py)

When `wandb` is enabled, the train/eval split and evaluation/save cadence are
included in the logged config metadata.

## 6. Evaluation Path

[src/eval.py](../src/eval.py) can:

1. load a saved SOPE diffusion checkpoint
2. optionally attach a robomimic policy checkpoint for guidance
3. generate trajectories autoregressively
4. compare generated trajectories against saved rollout trajectories

### 6.1 Interpreting chunk loss and rollout error

Low training loss does not by itself guarantee low chunk RMSE or rollout MSE.
When those metrics disagree, the first places to check are:

- a mismatch between the intended parameterization and the active sampling path
- a conditioning interpretation that is inconsistent with the dataset layout
- normalization or prefix-handling mistakes

These checks should be done before concluding that the denoiser backbone itself
is the main source of error.

### 6.2 Debug and ablation modes

Two debugging paths remain useful when investigating this stack:

- EEF-only loss masking: the model still denoises full state-action chunks, but
  the loss is restricted to the end-effector position slice to test whether the
  backbone can learn a simpler projection of the task
- EEF-only unconditioned mode: the dataset, conditioning, and evaluation path
  are simplified so the model trains on a reduced state subset without the usual
  prefix-conditioning contract

These are debugging tools, not the main contract that the rollout-backed SOPE
wrapper is trying to preserve.

## 7. Environment Bootstrap

The repository bootstrap path is intentionally part of the same operational
surface as training and rollout generation, because rebuilding the environment
incorrectly can break robomimic, MuJoCo rendering, or the guidance stack before
any experiment code runs.

### 7.1 Bootstrap Strategy

[`bootstrap_env.sh`](../bootstrap_env.sh) now:

- installs PyTorch explicitly
- installs modern JAX explicitly
- installs robomimic runtime dependencies explicitly
- installs a modern Hugging Face stack explicitly
- installs robomimic editable with `--no-deps --no-build-isolation`
- installs clean_diffuser editable with `--no-deps --no-build-isolation`

This avoids robomimic's stale dependency pins from
[`third_party/robomimic/setup.py`](../third_party/robomimic/setup.py), which
still include:

- `diffusers==0.11.1`
- `transformers==4.41.2`
- `huggingface_hub==0.23.4`

### 7.2 Why This Is Needed

The incompatibility comes from mixing an old `diffusers` release with a modern
JAX release. In that configuration, later robomimic imports can fail because
old `diffusers` code expects symbols such as `jax.random.KeyArray` that are no
longer present in newer JAX.

The chosen fix is preventive rather than reparative:

- do not let robomimic install its pinned Hugging Face packages in the first
  place
- keep the desired modern stack in place throughout bootstrap

### 7.3 EGL Hook Behavior

[`bootstrap_egl.sh`](../bootstrap_egl.sh) installs conda activation hooks that
default MuJoCo rendering to EGL:

- `MUJOCO_GL=egl`
- `PYOPENGL_PLATFORM=egl`
- `MUJOCO_EGL_DEVICE_ID=0` by default

It also checks the active environment for the old-`diffusers` / new-JAX hazard
and upgrades the Hugging Face stack when the problematic combination is
detected.

## 8. Validation

The lightweight validation for the current training stack is:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python -m py_compile src/utils.py src/train.py src/diffusion.py src/eval.py scripts/train_sope.py scripts/train_sope_gym.py src/robomimic_interface/dataset.py src/robomimic_interface/rollout.py src/robomimic_interface/checkpoints.py src/sope_interface/dataset.py
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python -m py_compile scripts/create_rollout_dataset.py scripts/prepare_policy_hm-image.py
bash -n bootstrap_env.sh
bash -n bootstrap_egl.sh
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 scripts/train_sope.py --help
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 scripts/train_sope_gym.py --help
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 scripts/create_rollout_dataset.py --help
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 scripts/prepare_policy_hm-image.py --help
```

Rerun a small smoke training job only when a change affects loader semantics,
training control flow, or checkpoint contents.
