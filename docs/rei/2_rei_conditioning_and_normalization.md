# Rei Conditioning And Normalization

This note records two separate observations about the `rei/` worktree:

1. how the reusable `rei/src/latent_sope/...` code handles robomimic rollout
   conditioning and normalization
2. whether Rei's `v0.2.5.14` notebook family actually uses that reusable path

The distinction matters because the branch contains both reusable library code
and notebook-local experimental code, and they do not implement the same
conditioning scheme.

## Reusable `rei/src` Code Path

Relevant files:

- [`rei/src/latent_sope/robomimic_interface/dataset.py`](../../../rei/src/latent_sope/robomimic_interface/dataset.py)
- [`rei/src/latent_sope/diffusion/sope_diffuser.py`](../../../rei/src/latent_sope/diffusion/sope_diffuser.py)
- [`rei/src/latent_sope/diffusion/train.py`](../../../rei/src/latent_sope/diffusion/train.py)
- [`rei/src/latent_sope/eval/metrics.py`](../../../rei/src/latent_sope/eval/metrics.py)
- [`rei/src/latent_sope/robomimic_interface/checkpoints.py`](../../../rei/src/latent_sope/robomimic_interface/checkpoints.py)

### Conditioning Layout In `dataset.py`

`RolloutChunkDataset` slices each rollout into four tensors:

- `states_from`: previous latent states ending at `t0 - 1`
- `actions_from`: previous actions ending at `t0 - 1`
- `states_to`: latent states from `t0` through `t0 + W`
- `actions_to`: actions from `t0` through `t0 + W - 1`

The historical prefix is collected with `_collect_frame_stack(...)`, which pads
early timesteps by repeating the earliest available frame.

### What `SopeDiffuser.loss(...)` Actually Conditions On

In `rei/src/latent_sope/diffusion/sope_diffuser.py`, the training input is
constructed by concatenating:

- the historical prefix `[states_from, actions_from]`
- the future chunk `[states_to[:-1], actions_to]`

along the time axis into a single diffusion trajectory.

However, the explicit SOPE conditioning dict returned by `make_cond(...)` only
contains the prefix states:

- `cond[t] = batch["states_from"][:, t, :]`

So the reusable `rei/src` path uses:

- prefix states as the hard inpainting condition
- prefix actions as part of the supervised trajectory tensor, not as explicit
  conditioning values

This is closer to "state-conditioned chunk diffusion with prefixed action
history in the target tensor" than to "condition on the full `(state, action)`
history."

### Normalization In The Reusable Path

`make_rollout_chunk_dataloader(...)` computes one shared normalization stat over
all loaded chunks using concatenated `[state, action]` transitions from
`states_to[:, :-1, :]` and `actions_to`.

If `config.normalize=True`, `dataset.py` applies those stats to:

- `states_from`
- `states_to`
- `actions_to`

but does **not** normalize `actions_from`.

That means the reusable training tensor in `SopeDiffuser.loss(...)` mixes:

- normalized prefix states
- raw prefix actions
- normalized future states
- normalized future actions

So the branch's reusable dataset path contains a normalization inconsistency in
the historical action prefix.

### Whether `norm` / `unorm` Are Used In The Reusable Path

Yes, on the wrapper side they are used.

`SopeDiffuser` builds `normalizer` and `unnormalizer` from saved dataset stats
and passes them into SOPE's `GaussianDiffusion`.

In addition:

- `generate_full_trajectory(...)` normalizes the initial state before sampling
- sampled chunks are explicitly unnormalized before being returned
- the next chunk's conditioning state is carried forward in normalized space

So the reusable `rei/src` sampling path does rely on normalization functions.

### Eval Metric Space In The Reusable Path

`rei/src/latent_sope/eval/metrics.py` does not normalize or unnormalize
internally. It computes L2 and MSE in whatever space the caller passes in.

Because `generate_full_trajectory(...)` returns unnormalized states and actions,
trajectory-level eval would be in raw units if that helper were used directly.

### Robomimic Policy-Side Normalization

Separately from SOPE chunk normalization, `build_rollout_policy_from_checkpoint`
preserves robomimic observation and action normalization stats when wrapping a
checkpoint into a `RolloutPolicy` on the fallback path.

So there are two normalization layers in play:

- robomimic policy normalization for acting / scoring
- SOPE chunk normalization for diffusion training and sampling

## `v0.2.5.14` Notebook Family

Relevant notebooks:

- [`rei/experiments/2026-03-25/MVP_v0.2.5.14_fixed_guidance.ipynb`](../../../rei/experiments/2026-03-25/MVP_v0.2.5.14_fixed_guidance.ipynb)
- [`rei/experiments/2026-03-25/MVP_v0.2.5.14b_12_policies.ipynb`](../../../rei/experiments/2026-03-25/MVP_v0.2.5.14b_12_policies.ipynb)
- [`rei/experiments/2026-03-25/MVP_v0.2.5.14c_action_sweep.ipynb`](../../../rei/experiments/2026-03-25/MVP_v0.2.5.14c_action_sweep.ipynb)
- [`rei/experiments/2026-03-26/MVP_v0.2.5.14d_action_sweep_large.ipynb`](../../../rei/experiments/2026-03-26/MVP_v0.2.5.14d_action_sweep_large.ipynb)
- [`rei/experiments/2026-03-26/MVP_v0.2.5.14e_12pol_action_sweep.ipynb`](../../../rei/experiments/2026-03-26/MVP_v0.2.5.14e_12pol_action_sweep.ipynb)
- [`rei/experiments/2026-03-26/MVP_v0.2.5.14f_timestep_sweep.ipynb`](../../../rei/experiments/2026-03-26/MVP_v0.2.5.14f_timestep_sweep.ipynb)
- [`rei/experiments/2026-03-26/MVP_v0.2.5.14g_action_sweep_high.ipynb`](../../../rei/experiments/2026-03-26/MVP_v0.2.5.14g_action_sweep_high.ipynb)
- [`rei/experiments/2026-03-26/MVP_v0.2.5.14h_high_action_sweep.ipynb`](../../../rei/experiments/2026-03-26/MVP_v0.2.5.14h_high_action_sweep.ipynb)
- [`rei/experiments/2026-03-26/MVP_v0.2.5.14i_grid_search.ipynb`](../../../rei/experiments/2026-03-26/MVP_v0.2.5.14i_grid_search.ipynb)

### What They Use

The `v0.2.5.14` notebooks do **not** use:

- `RolloutChunkDataset`
- `make_rollout_chunk_dataloader(...)`
- `SopeDiffuser`
- `SopeDiffuser.generate_full_trajectory(...)`

Instead, they:

- instantiate `TemporalUnet` and `GaussianDiffusion` directly
- build notebook-local `normalize_fn` and `unnormalize_fn`
- implement their own `generate_trajectories(...)` sampling loop
- use `RobomimicDiffusionScorer` from
  `rei/src/latent_sope/robomimic_interface/guidance.py`

### Notebook Conditioning Scheme

The notebook-local trajectory generator uses a much simpler conditioning rule
than the reusable `rei/src` path.

It initializes:

- `conditions = {0: cond_init}`

where `cond_init` is the normalized initial state.

After each generated chunk, it updates:

- `conditions = {0: last_state_norm}`

using only the final generated state of the previous chunk.

So the `v0.2.5.14` family conditions on only one state at one timestep. It does
not use:

- frame-stacked `states_from`
- `actions_from`
- the reusable `make_cond(...)` logic in `rei/src`

### Notebook Normalization Scheme

The notebooks compute their own transition statistics inline from pooled
`all_states` and `all_actions`, then define:

- `normalize_fn(x) = (x - mean) / std`
- `unnormalize_fn(x) = x * std + mean`

Those functions are passed directly into `GaussianDiffusion`.

They are also used explicitly inside the notebook-local guidance loop:

1. denoising mean `mm` is unnormalized into real state/action space
2. `RobomimicDiffusionScorer.grad_log_prob_chunk(...)` is evaluated in that raw
   space
3. the guided mean is renormalized before the diffusion update proceeds
4. final sampled chunks are unnormalized before storage and evaluation

So the `v0.2.5.14` notebooks definitely use normalization, but they use their
own notebook-local normalization path rather than the reusable `rei/src`
dataset-wrapper path.

## Practical Takeaways

- The reusable `rei/src` code and the `v0.2.5.14` notebooks should not be read
  as implementing the same conditioning contract.
- The reusable `rei/src` path uses prefix-state conditioning and carries prefix
  actions in the diffusion target tensor.
- The reusable `rei/src` path has a normalization inconsistency because
  `actions_from` remains raw when normalization is enabled.
- The `v0.2.5.14` notebooks bypass that reusable path and use a notebook-local,
  single-state conditioning loop with notebook-local normalization functions.
- Any conclusion drawn from the `v0.2.5.14` notebook family should therefore be
  interpreted as evidence about that notebook-specific sampling procedure, not
  about the reusable `rei/src` pipeline as a whole.
