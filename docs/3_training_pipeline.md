# Training And Chunk Evaluation Pipeline

Relevant code:
- [src/utils.py](../src/utils.py)
- [src/train.py](../src/train.py)
- [scripts/train_sope.py](../scripts/train_sope.py)
- [scripts/train_sope_gym.py](../scripts/train_sope_gym.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/eval.py](../src/eval.py)
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
- [src/robomimic_interface/checkpoints.py](../src/robomimic_interface/checkpoints.py)
- [src/sope_interface/dataset.py](../src/sope_interface/dataset.py)

## 1. Summary

This note consolidates the current SOPE training and chunk-evaluation pipeline.

The active codepath now uses:
- `src/utils.py` as the shared utility and path surface
- `src/train.py` as the canonical SOPE training module
- `src/diffusion.py` as the canonical SOPE diffusion module
- `scripts/train_sope.py` as the CLI entrypoint for diffusion training plus
  reward-predictor training
- `scripts/train_sope_gym.py` as the SOPE Gym state-dataset entrypoint
- `src/eval.py` as the reusable diffusion-checkpoint evaluation path

## 2. Training Data Path

By default, [`scripts/train_sope.py`](../scripts/train_sope.py) trains from rollout trajectory files under:

`data/rollout/rmimic-lift-ph-lowdim_diffusion_260130`

The script accepts either a single rollout file or a directory of rollout files. When given a directory, it infers latent and action shapes from the first rollout file it finds.

The training CLI also exposes `--num-workers`, which is forwarded into the
PyTorch `DataLoader` construction for both the train and held-out eval loaders.

The CLI now also exposes `--dim-mults` and forwards it directly into
`SopeDiffusionConfig.dim_mults`, so the canonical diffusion backbone width/depth
can be changed from the training entrypoint without editing code.

## 3. Dataset And Chunking

The training path uses [`src/robomimic_interface/dataset.py`](../src/robomimic_interface/dataset.py) to build chunk datasets from saved rollout latents.

The intended chunk layout is:
- `states_from`: the frame-stacked prefix ending at $t_0 - 1$
- `actions_from`: the aligned frame-stacked action prefix
- `states_to`: latent states from $t_0$ through $t_0 + W$
- `actions_to`: actions from $t_0$ through $t_0 + W - 1$

`train_sope.py` currently uses `source="latents"` by default.

### 3.1 SOPE Gym adapter

The repository also supports SOPE Gym `pdataset`-style assets through
[`src/sope_interface/dataset.py`](../src/sope_interface/dataset.py) and
[`scripts/train_sope_gym.py`](../scripts/train_sope_gym.py).

That adapter converts flat SOPE Gym arrays into the same chunk contract used by
the rollout-backed loader:

1. load `observations`, `actions`, `rewards`, and `terminals`
2. reconstruct episode boundaries from `terminals`
3. split train and eval at the episode level
4. build `states_from`, `actions_from`, `states_to`, and `actions_to` chunks

The SOPE Gym path supports two normalization modes:

- `asset`: reuse the shipped `normalization.json`
- `computed`: recompute shared train-split normalization from chunk data

For low-dimensional gym states such as `pdataset`, evaluation logs transition,
state, and action RMSE only. The robomimic-specific EEF metric is skipped
because that state layout does not expose the `robot0_eef_pos` slice.

## 4. Train Eval Split

When the input resolves to many rollout trajectory files, training now uses a trajectory-level train / eval split controlled by `train_fraction`, which defaults to `0.8`.

Design choices:
- The split happens at the rollout-file level, not the chunk level, to avoid leakage between train and eval chunks from the same trajectory.
- The file list is shuffled using the training seed, so the split is deterministic for a fixed seed.
- `train_fraction` is stored in the saved checkpoint metadata so later evaluation can reconstruct the same held-out split.
- Eval uses the held-out trajectory files only.
- When dataset normalization is enabled, the eval dataset reuses the train normalization statistics.

## 5. Scheduling

Training cadence is driven by count-like CLI knobs that are converted into epoch intervals inside [`src/train.py`](../src/train.py).

- `num_saves` controls checkpoint cadence through:

$$\begin{align}
\texttt{save\_every} = \max\left(1, \left\lceil \frac{\texttt{epochs}}{\texttt{num\_saves}} \right\rceil \right)
\end{align}$$

This uses ceil-based conversion so the requested checkpoint count is not undershot.

- `num_evals` controls held-out evaluation cadence through:

$$\begin{align}
\texttt{eval\_every} = \max\left(1, \left\lfloor \frac{\texttt{epochs}}{\texttt{num\_evals}} \right\rfloor \right)
\end{align}$$

This uses floored division so evaluation runs no more often than the requested count implies.

## 6. Shared Training Loop

`src/train.py` now routes both diffusion and reward training through
`train_general(...)`, which owns the shared epoch-level mechanics:

- seed setup
- device transfer
- optimizer stepping and gradient clipping
- scheduler stepping
- W&B run lifecycle
- checkpoint cadence
- eval cadence

Task-specific behavior is injected through `TrainLoopCallbacks`.

Current entrypoints:

- `train_sope_with_loaders(...)` validates configs, builds `SopeDiffuser`,
  prepares diffusion callbacks, and delegates to `train_general(...)`
- `train_rewardpred_with_loaders(...)` validates raw low-dim reward inputs,
  builds `RewardPredictor`, prepares reward callbacks, and delegates to
  `train_general(...)`
- `train_reward(...)` and `train_reward_with_loaders(...)` remain as
  compatibility aliases to `train_rewardpred(...)` and
  `train_rewardpred_with_loaders(...)`

The training hyperparameters are also split explicitly for
[`scripts/train_sope.py`](../scripts/train_sope.py):

- diffusion keeps the legacy unprefixed knobs such as `--epochs`,
  `--batch-size`, and `--lr`
- reward training uses `--reward-epochs`, `--reward-batch-size`, and
  `--reward-lr`

The default reward schedule is intentionally different from diffusion:

- reward `epochs=100`
- reward `batch_size=1024`
- reward `lr=1e-3`

Reward optimizer LR comes from `RewardPredictorConfig.lr`, while reward
`epochs` and `batch_size` come from the reward-side `TrainingConfig`.

## 7. Script Behavior

[`scripts/train_sope.py`](../scripts/train_sope.py) now runs two training jobs
sequentially in one invocation:

1. diffusion chunk training
2. reward-predictor training

Both runs share the same checkpoint directory and rollout split, but they use
separate W&B runs by default:

- `<run_name>_diffusion`
- `<run_name>_rewardpred`

The checkpoint stems remain distinct:

- `sope_diffuser_*.pt`
- `sope_reward_predictor_*.pt`

## 8. Chunk Evaluation

The evaluator now supports chunk-level reconstruction metrics instead of only
full-trajectory stitching.

- The default path evaluates one sampled chunk at a time, conditioned on the
  saved `frame_stack` states from the dataset.
- Metrics are computed against the held-out ground-truth chunk on the
  non-conditioned future window only.
- Reported errors include transition, state, and action RMSE.
- The evaluator also reports normalized-space chunk RMSE, a persistence
  baseline on the same target, and simple dataset scale summaries for the
  relevant state-action tensors.

Metric-space convention when `normalize=True`:

- training-time chunk RMSE diagnostics from the diffusion loss path are
  normalized-space metrics
- eval-set summary RMSE metrics use the unnormalized chunk metrics under
  `gen_unnormalized`
- normalized eval-loss-path chunk RMSE is not logged under the main eval
  namespace to avoid mixing spaces

### 8.1 Shared train-time and standalone eval path

`src/train.py` now calls `evaluate_sope(...)` from
[`src/eval.py`](../src/eval.py) instead of keeping a private evaluation helper.

Behavior is intended to stay aligned between train-time and standalone
checkpoint evaluation:

- average held-out diffusion loss is computed over the eval loader
- unguided chunk RMSE and persistence-baseline metrics come from the shared
  chunk-evaluation path
- the returned report carries one primary metric key, which defaults to
  `gen_unnormalized`

### 8.2 Split compatibility

Evaluation reproduces the same trajectory-level split rule used by training:

- rollout files are shuffled with the training seed
- the default split is the held-out `eval` split
- when dataset normalization is enabled, eval chunks reuse normalization
  statistics computed from the train split

### 8.3 Recent simplifications

- `RMSEMetrics` is now used directly as the mutable chunk accumulator in
  [`src/eval.py`](../src/eval.py)
- per-step RMSE arrays were removed, leaving aggregate chunk metrics only

## 9. Logging

The training loop uses a single epoch-level `tqdm` progress bar and logs:
- batch metrics such as `train/loss`
- epoch summaries such as `epoch/loss`
- held-out evaluation metrics under `eval_metrics:unguided/...`, currently including `loss`, `transition_rmse`, `state_rmse`, and `action_rmse`
- held-out diagnostics under `eval_diagnostics:unguided/...`, currently including the mean unnormalized ground-truth chunk value across batch, timestep, and transition dimensions
- placeholder guided namespaces `eval_metrics:guided/...` and `eval_diagnostics:guided/...` reserved for future guided evaluation wiring

When `wandb` is enabled, the train / eval split, save cadence, and eval cadence are included in the logged config metadata.

Training-time held-out evaluation is now orchestrated directly inside
`evaluate_sope(...)` in [`src/eval.py`](../src/eval.py): it computes eval loss,
updates the epoch progress bar postfix, and logs summary scalars without routing the metric payload back through
[`src/train.py`](../src/train.py).

## 10. Evaluation Path

[`src/eval.py`](../src/eval.py) can load a saved SOPE diffusion checkpoint, optionally attach a robomimic policy checkpoint for guidance, autoregressively stitch full trajectories, and compare predicted latent trajectories against saved rollout trajectories.

## 11. Validation

Run:

```bash
python3 -m py_compile src/utils.py src/train.py scripts/train_sope.py scripts/train_sope_gym.py src/eval.py src/robomimic_interface/dataset.py src/robomimic_interface/rollout.py src/robomimic_interface/checkpoints.py src/sope_interface/dataset.py
python3 scripts/train_sope.py --help
python3 scripts/train_sope_gym.py --help
python3 scripts/train_sope.py --wandb-mode disabled --max-steps 1 --num-saves 200 --num-evals 0 --reward-epochs 1 --reward-batch-size 32 --reward-lr 1e-3
```

The first command checks syntax for the active training stack. The CLI help
checks validate both the rollout-backed and SOPE Gym training entrypoints
without launching a run. The final command is a short rollout-backed smoke test
that verifies both diffusion and reward checkpoints are written with their own
training hyperparameters.
