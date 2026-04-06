# Autoregressive Trajectory Generation And OPE

This note consolidates the repository's autoregressive rollout-generation and
rollout-time OPE estimate notes.

Relevant code:

- [src/diffusion.py](../src/diffusion.py)
- [src/eval.py](../src/eval.py)
- [third_party/sope/opelab/core/baselines/diffuser.py](../third_party/sope/opelab/core/baselines/diffuser.py)
- [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py)
- [third_party/sope/opelab/examples/helpers.py](../third_party/sope/opelab/examples/helpers.py)

## 1. Summary

The local `SopeDiffuser` now owns both:

1. autoregressive full-trajectory generation
2. rollout-time OPE return estimation

The rollout path is related to upstream SOPE's diffusion baseline, but it is
not the same conditioning and stitching contract.

## 2. Local Autoregressive Generation

`SopeDiffuser.generate_full_trajectory(...)` is the canonical local rollout
implementation. It:

- normalizes the initial condition internally
- samples one chunk at a time
- feeds the trailing generated state prefix into the next chunk
- truncates the last chunk when needed
- returns unnormalized state and action arrays

The chunk-autoregressive recurrence is:

$$\begin{align}
\text{prefix}_{k+1} = \hat{s}_{k, -F:},
\end{align}$$

where $F$ is `frame_stack`.

## 3. Upstream SOPE Comparison

Upstream SOPE rollout generation instead:

- conditions with an in-painting dictionary such as
  `conditions = {0: normalized_initial_state}`
- uses its own chunk stitching rule
- supports trajectory-aware early termination
- includes optional `tanh_action` handling

Relative to that upstream contract, the active local wrapper:

- uses FiLM-style context conditioning instead of in-painting
- conditions on a flattened state prefix rather than a single state
- stores all generated chunk steps instead of overlap-by-one stitching
- does not currently implement upstream `tanh_action`
- does not currently stop generation early on termination

## 4. Rollout-Time Reward Path

`SopeDiffuser.ope_estimate(...)` applies the learned reward predictor only after
sampling. The path is:

1. generate rollout states and actions
2. score each generated transition with the reward model
3. discount and sum predicted immediate rewards
4. average the returns across the rollout batch

The per-step reward estimate is

$$\begin{align}
\hat{r}_t = \hat{r}_\phi(s_t, a_t),
\end{align}$$

and the per-trajectory OPE return is

$$\begin{align}
\hat{J}_i
=
\sum_{t=0}^{T-1}
\gamma_{\text{ope}}^t \hat{r}_\phi(s_{i,t}, a_{i,t}).
\end{align}$$

The scalar estimate returned by `ope_estimate(...)` is the batch mean:

$$\begin{align}
\hat{J}
\approx
\frac{1}{B}\sum_{i=1}^{B}\hat{J}_i.
\end{align}$$

`SopeDiffusionConfig.ope_gamma` only affects this discounted-return
computation. It does not change diffusion training or guidance.

## 5. Normalization Contract

The public rollout interface stays in unnormalized state/action space, but
sampling and autoregressive reconditioning still happen in normalized transition
space internally.

That means:

- the initial condition is normalized with the same transition-level statistics
  used during training
- sampled chunks are denormalized only for the public return value
- the next autoregressive prefix is built from the normalized generated state
  suffix, not from the denormalized output arrays

## 6. Validation

The lightweight validation for this note's codepath is:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python -m py_compile src/diffusion.py src/eval.py
```

If the rollout contract changes, rerun the smallest available trajectory
evaluation that compares generated trajectories against saved rollout assets.
