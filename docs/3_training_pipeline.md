# Training Pipeline

Relevant code:
- [src/utils.py](../src/utils.py)
- [src/train.py](../src/train.py)
- [scripts/train_sope.py](../scripts/train_sope.py)
- [src/sope_diffuser.py](../src/sope_diffuser.py)
- [src/eval.py](../src/eval.py)
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
- [src/robomimic_interface/checkpoints.py](../src/robomimic_interface/checkpoints.py)

## Summary

This note consolidates the current SOPE training and evaluation pipeline.

The active codepath now uses:
- `src/utils.py` as the shared utility and path surface
- `src/train.py` as the canonical SOPE training module
- `scripts/train_sope.py` as the CLI entrypoint for chunk-diffusion training
- `src/eval.py` as the reusable diffusion-checkpoint evaluation path

## Training Data Path

By default, [`scripts/train_sope.py`](../scripts/train_sope.py) trains from rollout trajectory files under:

`data/rollout/rmimic-lift-ph-lowdim_diffusion_260130`

The script accepts either a single rollout file or a directory of rollout files. When given a directory, it infers latent and action shapes from the first rollout file it finds.

The training CLI also exposes `--num-workers`, which is forwarded into the
PyTorch `DataLoader` construction for both the train and held-out eval loaders.

The CLI now also exposes `--dim-mults` and forwards it directly into
`SopeDiffusionConfig.dim_mults`, so Temporal U-Net width/depth can be changed
from the training entrypoint without editing code.

## Dataset And Chunking

The training path uses [`src/robomimic_interface/dataset.py`](../src/robomimic_interface/dataset.py) to build chunk datasets from saved rollout latents.

The intended chunk layout is:
- `states_from`: the frame-stacked prefix ending at $t_0 - 1$
- `actions_from`: the aligned frame-stacked action prefix
- `states_to`: latent states from $t_0$ through $t_0 + W$
- `actions_to`: actions from $t_0$ through $t_0 + W - 1$

`train_sope.py` currently uses `source="latents"` by default.

## Train Eval Split

When the input resolves to many rollout trajectory files, training now uses a trajectory-level train / eval split controlled by `train_fraction`, which defaults to `0.8`.

Design choices:
- The split happens at the rollout-file level, not the chunk level, to avoid leakage between train and eval chunks from the same trajectory.
- The file list is shuffled using the training seed, so the split is deterministic for a fixed seed.
- `train_fraction` is stored in the saved checkpoint metadata so later evaluation can reconstruct the same held-out split.
- Eval uses the held-out trajectory files only.
- When dataset normalization is enabled, the eval dataset reuses the train normalization statistics.

## Scheduling

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

## Logging

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

## Evaluation Path

[`src/eval.py`](../src/eval.py) can load a saved SOPE diffusion checkpoint, optionally attach a robomimic policy checkpoint for guidance, autoregressively stitch full trajectories, and compare predicted latent trajectories against saved rollout trajectories.

## Validation

Run:

```bash
python3 -m py_compile src/utils.py src/train.py scripts/train_sope.py src/eval.py src/robomimic_interface/dataset.py src/robomimic_interface/rollout.py src/robomimic_interface/checkpoints.py
python3 scripts/train_sope.py --help
```

The first command checks syntax for the active training stack. The second validates the training CLI wiring without launching a run.
