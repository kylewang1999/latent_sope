# Autoregressive Trajectory Generation In SopeDiffuser

Relevant code:
- [src/diffusion.py](../src/diffusion.py)
- [src/eval.py](../src/eval.py)
- [third_party/sope/opelab/core/baselines/diffuser.py](../third_party/sope/opelab/core/baselines/diffuser.py)
- [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py)
- [third_party/sope/opelab/examples/helpers.py](../third_party/sope/opelab/examples/helpers.py)

## 1. Summary

This note covers both:

- the local autoregressive full-trajectory generation path now implemented in
  [`SopeDiffuser`](../src/diffusion.py)
- the original upstream SOPE rollout-generation contract in
  [`third_party/sope/opelab/core/baselines/diffuser.py`](../third_party/sope/opelab/core/baselines/diffuser.py)

The two are related, but they are not the same sampling contract.

The local change moved autoregressive full-trajectory generation into
[`SopeDiffuser`](../src/diffusion.py) so sampling behavior lives with the
diffusion model wrapper instead of being duplicated in the evaluation module.

Before this change, [`src/eval.py`](../src/eval.py) owned a standalone
`generate_full_trajectory(...)` helper that manually:
- normalized the initial state
- built the flattened FiLM conditioning prefix
- sampled one chunk at a time
- fed the last generated normalized states back into the next chunk

After this change, `SopeDiffuser.generate_full_trajectory(...)` is the
canonical implementation of that autoregressive loop, and
[`src/eval.py`](../src/eval.py) calls the model method directly.

## 2. Local Wrapper Behavior

The generated trajectory remains chunk-autoregressive:

$$\begin{align}
\text{prefix}_{k+1} = \hat{s}_{k, -F:}
\end{align}$$

where $F$ is `frame_stack` and $\hat{s}_{k, -F:}$ are the last generated
normalized states from chunk $k$.

The method still:
- conditions on repeated copies of the normalized initial state for the first chunk
- samples `chunk_horizon` future steps per iteration
- truncates the last chunk when `max_length` is not a multiple of `chunk_horizon`
- returns unnormalized NumPy arrays for states and actions
- uses `SopeDiffusionConfig.trajectory_horizon` as the default rollout length when
  `max_length` is omitted

`conditioning_mode="none"` is still rejected for full-trajectory generation,
matching the previous eval-only helper.

The default `trajectory_horizon` is `60`. That matches the repository's
rollout-dataset generation path for robomimic Lift in
[`scripts/create_rollout_dataset.py`](../scripts/create_rollout_dataset.py),
which uses `--horizon 60` by default for the
`rmimic-lift-ph-lowdim_diffusion_260130` rollouts that the SOPE training and
evaluation scripts consume.

## 3. Upstream SOPE Contract

The original upstream SOPE rollout-generation path uses a different
conditioning and stitching rule.

### 3.1 Conditioning

Upstream rollout generation conditions on a dict like:

```python
conditions = {0: normalized_initial_state}
```

The initial condition is built by:

1. concatenating the initial state with zero actions
2. normalizing the full transition vector
3. slicing back to the state coordinates

After each sampled chunk, the next condition is updated from the last generated
state only, not from a multi-step prefix window.

### 3.2 Chunk Stitching

Upstream SOPE samples a chunk of length `mini_trajectory_size`, but stores only
`mini_trajectory_size - 1` steps in the final rollout.

That is an overlap-by-one autoregressive stitching rule: the last state of one
sampled chunk becomes the seed state for the next chunk instead of being
appended as an additional emitted step.

This differs from the local
[`SopeDiffuser.generate_full_trajectory(...)`](../src/diffusion.py), which
stores all generated future steps from each chunk and reconditions on the last
`frame_stack` normalized states.

### 3.3 Normalization And Guidance Space

Upstream SOPE uses a joint normalizer over the full transition vector $(s, a)$.

During guided sampling in
[`third_party/sope/opelab/core/baselines/diffusion/diffusion.py`](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py):

1. the reverse-process mean is moved into unnormalized trajectory space
2. guidance gradients are applied there
3. the result is renormalized before the DDPM step continues

So the diffusion process operates on normalized coordinates, but the guidance
update is applied in unnormalized trajectory space.

The operative update can be summarized as:

$$\begin{align}
\mu \leftarrow \mu + \lambda_t \left( \nabla_a \log \pi(a \mid s) - r \nabla_a \log \beta(a \mid s) \right)
\end{align}$$

where $\mu$ is the current reverse-process mean in unnormalized trajectory
space, $\lambda_t$ is either a fixed or schedule-scaled multiplier, and $r$ is
the negative-guidance ratio.

### 3.4 Tanh Action Mode

Upstream SOPE supports `tanh_action=True` in the top-level
[`Diffuser`](../third_party/sope/opelab/core/baselines/diffuser.py).

That mode:

1. enables the guidance path that uses `policy.gaussian_log_prob(...)`
2. applies `torch.tanh(...)` to the action coordinates after unnormalization

The diffusion model still predicts a full joint transition chunk; only the
action slice is squashed.

### 3.5 Termination Handling

Upstream autoregressive generation is trajectory-aware:

- it tracks which rollouts are still alive
- it stops generating a rollout once `is_terminated_fn(state)` becomes true
- it records end indices so later reward accumulation uses the realized horizon

The local `SopeDiffuser.generate_full_trajectory(...)` path does not currently
implement this early-stop behavior.

## 4. Comparison

Relative to upstream SOPE, the active local wrapper in
[`src/diffusion.py`](../src/diffusion.py):

- uses FiLM conditioning instead of inpainting-style conditioning
- conditions on a flattened `frame_stack` state prefix, not a single state
- stores all generated chunk steps instead of overlap-by-one stitching
- does not currently implement upstream `tanh_action` support
- does not currently implement per-trajectory termination-aware rollout

So the local implementation should be treated as a related but different
autoregressive contract rather than a line-by-line port of upstream SOPE.

## 5. Validation

Run:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python -m py_compile src/diffusion.py src/eval.py scripts/create_rollout_dataset.py
```

This is a syntax-level validation for the moved autoregressive path. A full
behavioral re-check should rerun any trajectory-evaluation workflow that
previously called `src.eval.generate_full_trajectory(...)`, especially if the
result is being compared against the upstream SOPE rollout contract.
