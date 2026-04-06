# Training Pipeline And Entrypoints

This note consolidates the repository's training-pipeline and training-entrypoint
documentation.

Relevant code:

- [src/utils.py](../src/utils.py)
- [src/train.py](../src/train.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/eval.py](../src/eval.py)
- [scripts/train_sope.py](../scripts/train_sope.py)
- [scripts/train_sope_gym.py](../scripts/train_sope_gym.py)
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
- [src/robomimic_interface/checkpoints.py](../src/robomimic_interface/checkpoints.py)
- [src/sope_interface/dataset.py](../src/sope_interface/dataset.py)

## 1. Summary

The active training stack uses:

- [src/train.py](../src/train.py) as the canonical orchestration module
- [src/diffusion.py](../src/diffusion.py) as the canonical diffusion wrapper
- [scripts/train_sope.py](../scripts/train_sope.py) as the rollout-backed CLI
- [scripts/train_sope_gym.py](../scripts/train_sope_gym.py) as the SOPE Gym
  state-dataset CLI
- [src/eval.py](../src/eval.py) as the reusable evaluation path

`train_sope(...)` and `train_rewardpred(...)` are now the canonical entrypoints
for both internally constructed loaders and externally supplied loaders. The old
loader-only wrappers no longer own separate implementations.

## 2. Data And Chunking Path

The rollout-backed path builds chunk datasets from saved rollout files under
[src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py).
The intended chunk contract is:

- `states_from`: frame-stacked prefix ending at $t_0 - 1$
- `actions_from`: aligned action prefix
- `states_to`: latent states from $t_0$ through $t_0 + W$
- `actions_to`: actions from $t_0$ through $t_0 + W - 1$

When training from many rollout trajectory files, the train and eval split is
performed at the trajectory-file level so chunks from the same trajectory do
not leak across splits.

### 2.1 SOPE Gym adapter

The SOPE Gym path under
[src/sope_interface/dataset.py](../src/sope_interface/dataset.py) converts
flat arrays into the same chunk contract by:

1. loading observations, actions, rewards, and terminals
2. reconstructing episode boundaries
3. splitting train and eval at the episode level
4. emitting the same `states_from` / `actions_from` / `states_to` /
   `actions_to` fields

## 3. Entry Point Consolidation

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

## 4. Scheduling And Loop Behavior

The training loop uses a single epoch-level progress bar and converts
count-style scheduling knobs into epoch intervals:

- `num_evals` uses floored division with a clamp to at least `1` when enabled
- `num_saves` uses a conservative ceil-based interval

Training-time evaluation is orchestrated through
[src/eval.py](../src/eval.py). The loop logs:

- training loss metrics
- epoch summaries
- held-out unguided metrics such as loss and RMSE
- held-out diagnostics
- placeholder guided namespaces for future guided evaluation wiring

When `wandb` is enabled, the train/eval split and evaluation/save cadence are
included in the logged config metadata.

## 5. Evaluation Path

[src/eval.py](../src/eval.py) can:

1. load a saved SOPE diffusion checkpoint
2. optionally attach a robomimic policy checkpoint for guidance
3. generate trajectories autoregressively
4. compare generated trajectories against saved rollout trajectories

## 6. Validation

The lightweight validation for the current training stack is:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python -m py_compile src/utils.py src/train.py src/diffusion.py src/eval.py scripts/train_sope.py scripts/train_sope_gym.py src/robomimic_interface/dataset.py src/robomimic_interface/rollout.py src/robomimic_interface/checkpoints.py src/sope_interface/dataset.py
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 scripts/train_sope.py --help
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 scripts/train_sope_gym.py --help
```

Rerun a small smoke training job only when a change affects loader semantics,
training control flow, or checkpoint contents.
