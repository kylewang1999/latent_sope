# SOPE Gym `pdataset` Adapter

## Scope

This note documents the adapter and training entrypoint added for the SOPE Gym
assets downloaded by
[third_party/sope/scripts/download_assets.sh](../third_party/sope/scripts/download_assets.sh),
with the initial target dataset at
[data/sope_gym_data/pdataset](../data/sope_gym_data/pdataset).

Relevant code:

- [src/sope_interface/dataset.py](../src/sope_interface/dataset.py)
- [scripts/inspect_sope_gym_pdataset.ipynb](../scripts/inspect_sope_gym_pdataset.ipynb)
- [scripts/train_sope_gym.py](../scripts/train_sope_gym.py)
- [src/train.py](../src/train.py)
- [src/eval.py](../src/eval.py)

## Before

The SOPE diffusion training path only accepted rollout files saved from the
robomimic integration. That path expected chunk datasets exposing:

- `states_from`
- `actions_from`
- `states_to`
- `actions_to`

The downloaded SOPE Gym assets are a single flat stream of `.npy` arrays
(`observations`, `actions`, `rewards`, `terminals`) plus a dataset-level
`normalization.json`, so they could not be consumed directly by
[src/train.py](../src/train.py) or the chunk evaluators in
[src/eval.py](../src/eval.py).

## After

`src/sope_interface/dataset.py` now converts SOPE Gym assets into the same
chunk contract used by
[src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py):

1. Load the flat `.npy` arrays and optional `normalization.json`.
2. Reconstruct episodes from `terminals`.
3. Split train and eval data at the episode level, not at the chunk level, to
   avoid leakage across splits.
4. Build chunk tensors with the same semantics as the rollout adapter:
   `states_from/actions_from` hold the historical prefix and
   `states_to/actions_to` hold the future chunk.

The SOPE Gym adapter supports two normalization modes:

- `asset`: use the shipped `normalization.json`.
- `computed`: recompute one shared `(state, action)` normalization over the
  training chunks, matching the current rollout-backed training path more
  closely.

`scripts/train_sope_gym.py` builds train and eval loaders from those episode
splits and then reuses the existing diffusion training loop in
[src/train.py](../src/train.py). Eval metrics continue to come from
[src/eval.py](../src/eval.py), so the logged chunk RMSE path is shared with the
rollout-backed workflow. For low-dimensional gym states like `pdataset`,
`eval.py` logs transition, state, and action RMSE only; the robomimic-specific
EEF metric is skipped because the state does not contain that slice.

## Assumptions

- Episode boundaries are determined entirely by `terminals.npy`.
- The final observation of a chunk is taken from the saved observation stream,
  matching the existing rollout chunk contract that uses `W + 1` states for
  `W` actions.
- `pdataset` is treated as a state-space dataset with `frame_stack=1` by
  default.

## Validation

Small validation to rerun after changes:

1. Open [scripts/inspect_sope_gym_pdataset.ipynb](../scripts/inspect_sope_gym_pdataset.ipynb)
   and inspect the displayed batch contract.
2. Run a short dry train:

```bash
python3 scripts/train_sope_gym.py \
  --data data/sope_gym_data/pdataset \
  --epochs 1 \
  --max-steps 2 \
  --batch-size 64 \
  --num-workers 0 \
  --wandb-mode disabled
```

If the adapter changes behavior, re-run both the notebook inspection and a
short train with `--normalization-source asset` and `--normalization-source computed`.
