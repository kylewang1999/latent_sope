# Policy And Data Workflows

Relevant code:
- [scripts/create_rollout_dataset.py](../scripts/create_rollout_dataset.py)
- [scripts/prepare_policy_hm-image.py](../scripts/prepare_policy_hm-image.py)
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)

## 1. Summary

This note consolidates the repository workflows for staging robomimic policy artifacts and converting them into rollout trajectory datasets.

The current directory convention is:
- prepared policies live under `data/policy/<policy-name>`
- generated rollout trajectories live under `data/rollout/<policy-name>`

## 2. Policy Preparation

[`scripts/prepare_policy_hm-image.py`](../scripts/prepare_policy_hm-image.py) prepares the multi-human image diffusion policy archive under:

`data/policy/rmimic-lift-mh-image-v15_diffusion_260123`

Behavior:
- installs `gdown` only if importing it fails
- downloads the Google Drive zip archive into `data/`
- extracts into a temporary directory under `data/`
- skips macOS metadata such as `__MACOSX` and `._*`
- finds the real payload by looking for a directory containing both `config.json` and `models/`
- renames and stages that payload under `data/policy/`
- removes the downloaded zip after a successful prepare

## 3. Rollout Dataset Generation

[`scripts/create_rollout_dataset.py`](../scripts/create_rollout_dataset.py) generates one rollout-latent `.h5` file per trajectory from a prepared robomimic policy under `data/policy/<policy-name>`.

Default output location:

`data/rollout/<policy-name>`

Default behavior:
- resolves the policy checkpoint root from `data/policy/<policy-name>`
- rolls out the requested number of trajectories
- saves one `.h5` file per trajectory
- writes rendered videos under `data/rollout/<policy-name>/videos/`
- stores helpful HDF5 attrs such as `object_init_loc`, `rollout_index`, `policy_train_dir`, and `checkpoint_path`

The saved filenames now include the rollout index prefix, for example:

`0007_<object_init_loc>_len60.h5`

## 4. Parallel Rollouts

Rollout generation uses process-level parallelism, not thread-level parallelism.

Design choice:
- robosuite / MuJoCo environment reset and XML reconstruction were not reliable under Python threads
- worker processes avoid the thread-safety issues seen with concurrent `reset_to(...)` calls
- CUDA-backed workers use the `spawn` start method so each process initializes CUDA cleanly

Each worker process owns its own:
- checkpoint object
- rollout policy
- environment
- feature hook

The parent process owns the single `tqdm` progress bar and updates it as futures complete.

## 5. Rendering Cadence

The `--num-render` flag defaults to `10`.

The rendering stride is computed as:

$$\begin{align}
\texttt{render\_stride} = \max\left(1, \left\lfloor \frac{\texttt{num\_traj}}{\texttt{num\_render}} \right\rfloor \right)
\end{align}$$

when `num_render > 0`. Setting `num_render = 0` disables video rendering.

## 6. Interaction With The Dataloader

The generated `.h5` files are intended to work directly with [`src/robomimic_interface/dataset.py`](../src/robomimic_interface/dataset.py).

Practical constraints:
- rollout files should remain directly under `data/rollout/<policy-name>/`
- `videos/` does not interfere because the loader only gathers top-level rollout files
- `source="latents"` works because the files contain `latents` and `actions`
- `source="obs"` works when the rollout files include `obs`, which the current rollout script stores

## 7. Validation

Run:

```bash
python3 -m py_compile scripts/create_rollout_dataset.py scripts/prepare_policy_hm-image.py src/robomimic_interface/dataset.py src/robomimic_interface/rollout.py
python3 scripts/create_rollout_dataset.py --help
python3 scripts/prepare_policy_hm-image.py --help
```

These commands check syntax and CLI wiring without running a full download or rollout job.
