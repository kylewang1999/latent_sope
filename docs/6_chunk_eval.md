# Chunk Evaluation

Relevant code:
- [src/eval.py](../src/eval.py)
- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py)
- [scripts/eval_sope.py](../scripts/eval_sope.py)

This note records the current SOPE evaluation path for diffusion checkpoints.

### 2026-03-29 Refactor: Mutable `RMSEMetrics` Chunk Accumulators

The chunk evaluator in [src/eval.py](../src/eval.py) now uses mutable
`RMSEMetrics` instances directly as the running accumulator for chunk metrics.

Behavior is intended to be unchanged:

- the batch loop still aggregates the same squared-error and scale totals
- final reported RMSE and mean diagnostics are computed with the same
  denominators as before

This removes the intermediate dict accumulator and the long
`_compute_chunk_metrics(...)` unpacking path, so chunk metric wiring now stays
inside one container type from accumulation through reporting.

### 2026-03-29 Simplification: Remove Per-Step RMSE Outputs

The evaluator no longer reports per-step RMSE arrays for transition, state, or
action metrics.

The active aggregate metrics use the average over batch elements, chunk
horizon, and feature dimension, which matches the chunk-level RMSE values used
by training-time evaluation and standalone checkpoint evaluation.

### 2026-03-29 Refactor: Shared Training Eval Helper

The train loop now calls `evaluate_sope(...)` from
[src/eval.py](../src/eval.py) instead of carrying a private
`_evaluate_sope(...)` implementation in
[src/train.py](../src/train.py).

Behavior is intended to be unchanged:

- average held-out diffusion loss is still computed over the eval loader
- unguided chunk RMSE and persistence-baseline metrics still come from
  the shared chunk-evaluation path

This reduces duplication between training-time eval and standalone evaluation
entrypoints, so future metric additions only need to be wired in one place.

`evaluate_sope(...)` now also accepts `primary_metric_key` to choose which
unguided chunk metric entry receives the scalar held-out diffusion loss in the
returned report. The default remains `gen_unnormalized`, which matches the
current training-time summary path.

## What Changed

The evaluator now supports chunk-level reconstruction metrics instead of only
full-trajectory stitching.

- The default path evaluates one sampled chunk at a time, conditioned on the
  saved `frame_stack` states from the dataset.
- Metrics are computed against the held-out ground-truth chunk on the
  non-conditioned future window only.
- Reported errors include transition, state, and action RMSE.
- The evaluator now also reports normalized-space chunk RMSE, a persistence
  baseline on the same target, and simple dataset scale summaries for the
  relevant state-action tensors.

## Split Compatibility

Evaluation now reproduces the same trajectory-level split rule used by
training:

- rollout files are shuffled with the training seed
- the default split is the held-out `eval` split
- when dataset normalization is enabled, eval chunks reuse normalization
  statistics computed from the train split

This keeps chunk evaluation aligned with the train / eval separation in
[src/train.py](../src/train.py).

## Normalization Contract

The diffuser expects every transition in the training sequence to use the same
normalization scheme when dataset normalization is enabled.

The dataset-level statistics are computed over concatenated future transitions
`[state, action]` in
[src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py).
Those same state and action statistics must then be applied consistently to:

- `states_from`
- `actions_from`
- `states_to`
- `actions_to`

This matters because the training loss in
[src/sope_diffuser.py](../src/sope_diffuser.py) concatenates the historical
prefix and the future chunk into one diffusion target sequence. If one slice is
left in raw units while the others are normalized, the model sees a single
trajectory with inconsistent scales across timesteps.

### 2026-03-28 Fix: Normalize `actions_from`

Before this fix, the dataset normalized:

- `states_from`
- `states_to`
- `actions_to`

but left `actions_from` in raw action units.

That created a mixed-scale training target:

- prefix states: normalized
- prefix actions: raw
- future states: normalized
- future actions: normalized

The code now normalizes `actions_from` with the same action mean and standard
deviation used for `actions_to`, so the full diffusion sequence lives in one
consistent normalized space.

Expected impact:

- training loss is easier to interpret because all transition slices use the
  same scale
- chunk sampling should no longer be penalized by a raw-vs-normalized mismatch
  in the historical action prefix
- unusually large chunk MSE is less likely to come from the input pipeline
  itself and more likely to reflect model quality or raw data scale

## Current Diagnostic Additions

The remaining debugging question is whether the large held-out chunk MSE mostly
comes from raw feature scale or from genuinely poor one-chunk prediction. The
evaluator now records three additional views to answer that question.

### Normalized-space chunk MSE

The evaluator now measures sampled-vs-ground-truth error directly in the same
normalized coordinate system that the diffusion model trains in:

$$\begin{align}
\text{MSE}_{\text{norm}}
= \frac{1}{N H D}
\sum_{i=1}^{N}
\sum_{\tau=1}^{H}
\sum_{d=1}^{D}
\left(
\hat x^{\text{norm}}_{i,\tau,d} - x^{\star,\text{norm}}_{i,\tau,d}
\right)^2.
\end{align}$$

If raw-space MSE is huge while normalized-space MSE is moderate, the dominant
effect is likely raw feature scale rather than a catastrophic modeling failure.

### Persistence baseline

The evaluator now also computes a trivial future predictor that repeats the
last conditioned state and action across the prediction horizon:

$$\begin{align}
\hat s_{t+\tau}^{\text{base}} &= s_t, \\
\hat a_{t+\tau}^{\text{base}} &= a_t,
\end{align}$$

for $\tau \in \{1, \dots, H\}$.

This baseline is evaluated in both normalized and raw space. If the learned
diffuser does not beat this baseline, the large chunk MSE is probably not just
an artifact of evaluation scale.

### Dataset scale summaries

`scripts/eval_sope.py` now emits global mean, standard deviation, minimum, and
maximum for:

- `states_from`
- `states_to`
- `actions_from`
- `actions_to`

in both raw space and normalized space when normalization is enabled. This is
meant to expose large latent or action ranges quickly before deeper debugging.

## 2026-03-28 Update: State-Conditioned Prefix Training

To align the wrapper with the original SOPE conditioning scheme, training now
treats the historical prefix as conditioning information only.

In [src/sope_diffuser.py](../src/sope_diffuser.py):

- prefix timesteps are built as `[states_from, 0]`
- future timesteps remain `[states_to[:, :-1], actions_to]`
- the diffusion loss weight is set to zero on all prefix timesteps
- sampling still conditions only on the prefix state slices

This avoids asking the model to reconstruct historical prefix actions even
though the sampler never clamps them.

One consequence is that `train/a0_loss` is no longer a useful diagnostic in
this configuration, because the first action slot belongs to the conditioned
prefix rather than to the predicted future chunk.

## SOPE Normalization Subtleties

There are three different normalization questions in the current SOPE path, and
they should not be conflated:

1. dataset normalization for training targets
2. diffuser sampling space inside `GaussianDiffusion`
3. policy-guidance space used during guided sampling

### Dataset normalization

The dataset computes feature-wise mean and standard deviation over concatenated
future transitions:

$$\begin{align}
x^{\text{future}} = [s_{t:t+H-1}, a_{t:t+H-1}].
\end{align}$$

Those statistics are then reused to normalize both the historical prefix and
the future chunk before the batch is fed into the model. The intent is that the
entire diffusion training sequence lives in one consistent normalized space.

### Diffuser sampling space

The diffusion model itself samples in normalized space.

`GaussianDiffusion.p_sample_loop(...)` passes `normalizer` and `unnormalizer`
into the sample function, but the returned `Sample.trajectories` are still in
normalized coordinates. The reason is:

- the sampler temporarily calls `unnormalizer(...)` so guidance code can inspect
  or modify trajectories in data space
- before returning a reverse-diffusion step, it converts the sample back with
  `normalizer(...)`

So the output of `conditional_sample(...)` remains normalized, and the explicit
`diffuser.unnormalizer(sample.trajectories)` call in
[src/eval.py](../src/eval.py) is required for raw-space chunk comparison.

In other words, the internal unnormalize-normalize cycle inside the sampler is
not a final output transform. It is only an internal workspace conversion.

### Policy-guidance space

Guidance is more subtle.

The current SOPE sampler computes guidance on an unnormalized copy of the
trajectory proposal. That means the policy-facing score interface is currently
treated as though it consumes raw state-action units:

$$\begin{align}
\texttt{policy.grad\_log\_prob}(s, a)
\end{align}$$

is called on data-space tensors, not on the normalized diffusion state.

This assumption may be valid for policies whose score or log-probability is
defined directly in raw environment units. It is not automatically valid for
every policy wrapper.

### Why this matters for robomimic diffusion policies

Robomimic diffusion policies are trained with their own action normalization
conventions. In that setting, the current SOPE behavior creates a potential
mismatch:

- SOPE unnormalizes the chunk proposal before guidance
- the robomimic denoiser may expect normalized action inputs instead

So there are two distinct contracts:

- SOPE sampler contract today: policy score API receives raw-space tensors
- robomimic policy contract: action score may naturally live in robomimic's own
  normalized action space

If those spaces differ, the policy adapter must reconcile them explicitly,
rather than assuming SOPE's raw-space guidance inputs are already correct.

### Practical reading guide

When debugging normalization-related metrics, use this checklist:

- If training batches look inconsistent across prefix and future timesteps, look
  at dataset normalization.
- If `conditional_sample(...)` outputs seem to need conversion before metric
  computation, remember that sampler outputs are normalized by design.
- If guided sampling behaves strangely while unguided sampling looks reasonable,
  inspect the policy adapter's expected input space before changing the diffuser
  normalization path.

## Guided Evaluation

Guided chunk evaluation can attach a robomimic diffusion policy through
[src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py).

- The adapter exposes `grad_log_prob(state, action)` so SOPE's guided sampler
  can use robomimic as a score source.
- The current approximation assumes the incoming `state` tensor is already an
  encoded observation feature, and it evaluates the robomimic denoiser at a
  fixed diffusion timestep.
- This is enough for chunk-level guidance experiments today, but it is still a
  simplification of robomimic's full sequence-conditioned action diffusion.

## Unguided vs Guided Chunk Generation

In `evaluate_diffusion_chunk_mse`, both modes generate one future chunk from
the same conditioning history in `batch["states_from"]`; the difference is
only in how the reverse diffusion sampler chooses each denoising step. See
[src/eval.py](../src/eval.py).

Let the conditioning history be
$c = \{s_{t-S+1}, \dots, s_t\}$ with frame stack $S$, and let the future chunk be
$x_{t+1:t+H}$ with chunk horizon $H$, where each transition contains both state
and action features.

Unguided generation samples only from the learned chunk model:

$$\begin{align}
\hat x_{t+1:t+H}^{\text{unguided}} \sim p_\theta(x_{t+1:t+H} \mid c).
\end{align}$$

Operationally, this is `diffuser.diffusion.conditional_sample(..., guided=False)`,
followed by:

- unnormalization back to the original state-action scale
- dropping the prepended conditioned frames
- comparing only the future window against the ground-truth chunk

$$\begin{align}
x_{t+1:t+H}^{\star}
= \text{unnormalize}\left(
\left[
\texttt{states\_to}[:, :-1, :],\;
\texttt{actions\_to}
\right]
\right).
\end{align}$$

Guided generation uses the same diffusion model, but perturbs the reverse
sampling updates with a policy-derived action score. In the current codepath,
this score comes from an attached policy interface such as
`policy.grad_log_prob(state, action)`. At a high level, the sampler becomes:

$$\begin{align}
\hat x_{t+1:t+H}^{\text{guided}}
\sim p_\theta(x_{t+1:t+H} \mid c, \pi),
\end{align}$$

where $\pi$ does not replace the diffusion model. Instead, it adds a guidance
term that biases the denoising trajectory toward actions favored by the policy.
Schematically, the reverse-step score is modified from the unguided model score
$\nabla_x \log p_\theta(x \mid c)$ to something of the form

$$\begin{align}
\nabla_x \log p_\theta(x \mid c)
&\;+\;
\lambda \nabla_a \log \pi(a \mid s),
\end{align}$$

with guidance strength controlled by sampler hyperparameters such as
`action_scale`, `k_guide`, `normalize_grad`, and related options passed through
`guidance_kw`.

For chunk-MSE evaluation, both modes are scored in exactly the same way after
sampling. The metric difference is therefore not in the loss definition, but in
the source of the sampled chunk:

$$\begin{align}
\text{MSE}
= \frac{1}{N H D}
\sum_{i=1}^{N}
\sum_{\tau=1}^{H}
\sum_{d=1}^{D}
\left(
\hat x_{i,\tau,d} - x^\star_{i,\tau,d}
\right)^2,
\end{align}$$

where:

- $N$ is the number of evaluated chunks
- $H$ is the chunk horizon
- $D$ is either `state_dim + action_dim`, `state_dim`, or `action_dim`
  depending on whether the report is for transition, state, or action MSE

So the practical distinction is:

- `unguided`: "What future chunk does the diffusion model itself sample?"
- `guided`: "What future chunk does the same diffusion model sample after its
  reverse steps are nudged toward policy-preferred actions?"

## Validation

The intended smoke test is:

```bash
python3 -m py_compile src/eval.py src/robomimic_interface/policy.py scripts/eval_sope.py
python3 scripts/eval_sope.py --max-chunks 16 --json
```

Future evaluation work should plug value-estimator metrics into the report
structure in `src/eval.py` rather than creating a separate evaluator.
