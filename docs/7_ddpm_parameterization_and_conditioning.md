# DDPM Parameterization And Conditioning

This note records three implementation details that matter for the current
robomimic chunk-diffusion debugging work:

1. `predict_epsilon=True` vs `predict_epsilon=False` in SOPE's DDPM
2. SOPE's conditioning style versus robomimic diffusion policy conditioning
3. Why FiLM-style context conditioning is a plausible next step for the current
   19D robomimic setup

Relevant code:

- [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py)
- [third_party/sope/opelab/core/baselines/diffuser.py](../third_party/sope/opelab/core/baselines/diffuser.py)
- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py)
- [third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/train.py](../src/train.py)
- [src/eval.py](../src/eval.py)
- [docs/reports/200331.md](./reports/200331.md)

## 1. DDPM Parameterization

The forward process is

$$\begin{align}
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I)
\end{align}$$

Equivalently, the forward corruption distribution is

$$\begin{align}
q(x_t \mid x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}\,x_0,
\left(1-\bar{\alpha}_t\right) I
\right)
\end{align}$$

The reverse model uses a denoiser $f_\theta(x_t, t)$, but there are two common
choices for what $f_\theta$ is asked to predict.

The posterior used by the reverse process is also available in closed form:

$$\begin{align}
q(x_{t-1}\mid x_t, x_0)
=
\mathcal{N}
\left(
x_{t-1};
\tilde{\mu}_t(x_t, x_0),
\tilde{\beta}_t I
\right)
\end{align}$$

with

$$\begin{align}
\tilde{\mu}_t(x_t, x_0)
&=
\frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,x_0
+
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t
\end{align}$$

and

$$\begin{align}
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t
\end{align}$$

In practice the sampler replaces $x_0$ with the denoiser's estimate
$\hat{x}_0$, giving

$$\begin{align}
q(x_{t-1}\mid x_t, \hat{x}_0)
=
\mathcal{N}
\left(
x_{t-1};
\tilde{\mu}_t(x_t, \hat{x}_0),
\tilde{\beta}_t I
\right)
\end{align}$$

This is the quantity implemented by `q_posterior(...)` in
[third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py).

### 1.1 Predict Noise

With `predict_epsilon=True`, the network predicts $\epsilon$.

The model output is

$$\begin{align}
\hat{\epsilon}_\theta = f_\theta(x_t, t)
\end{align}$$

and the clean sample estimate is reconstructed by

$$\begin{align}
\hat{x}_0
=
\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\epsilon}_\theta}
{\sqrt{\bar{\alpha}_t}}
\end{align}$$

The training objective is therefore

$$\begin{align}
\mathcal{L}_{\epsilon}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\left\|
\epsilon - f_\theta(x_t, t)
\right\|_2^2
\right]
\end{align}$$

In words: the model is supervised to explain what noise was added, not to
directly return the clean chunk.

In SOPE this appears in
[third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py):

- `GaussianDiffusion.predict_start_from_noise(...)` interprets model output as
  noise and reconstructs $\hat{x}_0$
- `GaussianDiffusion.p_losses(...)` computes loss against sampled `noise`

Concretely:

- [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py#L356)
- [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py#L509)

### 1.2 Predict Clean Sample

With `predict_epsilon=False`, the network predicts $x_0$ directly.

The model output is now

$$\begin{align}
\hat{x}_{0,\theta} = f_\theta(x_t, t)
\end{align}$$

and the training objective becomes

$$\begin{align}
\mathcal{L}_{x_0}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\left\|
x_0 - f_\theta(x_t, t)
\right\|_2^2
\right]
\end{align}$$

The reverse-process estimate of the clean sample is therefore direct:

$$\begin{align}
\hat{x}_0 = f_\theta(x_t, t)
\end{align}$$

In the same SOPE code path:

- `predict_start_from_noise(...)` returns model output directly as $\hat{x}_0`
- `p_losses(...)` computes loss against `x_start`

This changes the supervised target without changing the overall DDPM family.

### 1.3 Why The Two Modes Can Behave Differently

The two parameterizations are mathematically related, but they are not
optimization-equivalent for a finite model trained with finite data.

The relation between them is exact:

$$\begin{align}
\epsilon
=
\frac{x_t - \sqrt{\bar{\alpha}_t}\,x_0}
{\sqrt{1-\bar{\alpha}_t}}
\end{align}$$

and therefore, if one target were predicted perfectly, the other could be
recovered exactly. The issue is not expressivity in the infinite-data limit; the
issue is the optimization path induced by the target choice.

Practical differences:

- `epsilon`-prediction supervises a noise target and reconstructs $x_0$
  indirectly through the diffusion schedule
- `x_0`-prediction supervises the clean trajectory chunk directly
- under `epsilon`-prediction, small prediction errors in $\hat{\epsilon}_\theta$
  are propagated through the schedule-dependent map from $\hat{\epsilon}_\theta$
  to $\hat{x}_0$
- evaluation in this repository is driven by chunk-space RMSE and sampled future
  accuracy, not only by DDPM training loss
- conditioning masks and non-uniform loss weights further break the intuition
  that the two formulations should train identically

The key practical distinction is:

$$\begin{align}
\mathcal{L}_{\epsilon} \text{ optimizes noise-space accuracy,}
\qquad
\mathcal{L}_{x_0} \text{ optimizes sample-space accuracy.}
\end{align}$$

When the downstream metric is chunk RMSE in state-action space, the
`predict_epsilon=False` formulation can therefore improve the metric of
interest even if both formulations are valid DDPM parameterizations.

For the current robomimic debugging problem, this means that a run can have a
reasonable DDPM loss under `predict_epsilon=True` while still producing worse
sample-space RMSE than `predict_epsilon=False`.

### 1.4 Why `predict_epsilon` Can Have Worse Sample MSE

It is tempting to say the two formulations should behave the same because they
are mathematically equivalent. That statement is only true at the level of an
exact optimum:

- infinite data
- sufficient model capacity
- exact optimization
- exact recovery of one target from the other

Those conditions do not hold in the current robomimic chunk-diffusion setup.

The practical issue is target mismatch.

With `predict_epsilon=True`, the model is trained on

$$\begin{align}
\mathcal{L}_{\epsilon}
=
\mathbb{E}
\left[
\left\|
\epsilon - \hat{\epsilon}_\theta
\right\|_2^2
\right]
\end{align}$$

but the quantity later used by the sampler is

$$\begin{align}
\hat{x}_0
=
\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\epsilon}_\theta}
{\sqrt{\bar{\alpha}_t}}
\end{align}$$

So the sampler does not directly consume the supervised target. It consumes a
transformed version of the model prediction.

This means:

$$\begin{align}
\hat{\epsilon}_\theta \approx \epsilon
\not\Rightarrow
\hat{x}_0 \approx x_0
\text{ with equal error across all } t
\end{align}$$

because the map from $\hat{\epsilon}_\theta$ to $\hat{x}_0$ depends on the
noise schedule. In particular, when $\bar{\alpha}_t$ is small, the conversion
from predicted noise to clean sample can magnify the effect of prediction error.

By contrast, with `predict_epsilon=False`, the loss directly supervises the
quantity that the reverse process uses:

$$\begin{align}
\mathcal{L}_{x_0}
=
\mathbb{E}
\left[
\left\|
x_0 - \hat{x}_{0,\theta}
\right\|_2^2
\right]
\end{align}$$

and the sampler then plugs $\hat{x}_{0,\theta}$ directly into the posterior.

So the two formulations are:

- equivalent as probabilistic parameterizations of the same DDPM family
- not equivalent as finite-sample optimization problems

This gap can become more visible when:

- the downstream metric is chunk-space MSE rather than noise-space MSE
- the model is trained with conditioning
- the loss is masked or reweighted across coordinates or timesteps
- the data regime is relatively small or structured

That is why, in practice, `predict_epsilon=True` can produce worse chunk RMSE
than `predict_epsilon=False` even though the underlying DDPM mathematics is
consistent in both cases.

### 1.5 Generation Pipeline Difference

The reverse-generation loop is structurally the same in both parameterizations:

1. start from Gaussian noise $x_T$
2. run the denoiser at timestep $t$
3. convert the denoiser output into an estimate $\hat{x}_0$
4. compute the posterior
   $q(x_{t-1} \mid x_t, \hat{x}_0)$
5. sample or take the mean to obtain $x_{t-1}$
6. re-apply conditioning if the sampler uses conditioned coordinates

The only branch is step 3: how the denoiser output becomes $\hat{x}_0$.

#### Generation With `predict_epsilon=True`

The denoiser outputs

$$\begin{align}
\hat{\epsilon}_\theta = f_\theta(x_t, t)
\end{align}$$

and the sampler reconstructs

$$\begin{align}
\hat{x}_0
=
\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\epsilon}_\theta}
{\sqrt{\bar{\alpha}_t}}
\end{align}$$

Then the posterior step uses this reconstructed clean sample:

$$\begin{align}
q(x_{t-1} \mid x_t, \hat{x}_0)
\end{align}$$

with posterior mean

$$\begin{align}
\tilde{\mu}_t(x_t, \hat{x}_0)
&=
\frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\hat{x}_0
+
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t
\end{align}$$

So the generation path is:

- predict noise
- convert predicted noise to a clean-sample estimate
- run the posterior update from that estimate

#### Generation With `predict_epsilon=False`

The denoiser outputs

$$\begin{align}
\hat{x}_{0,\theta} = f_\theta(x_t, t)
\end{align}$$

directly, so no conversion is needed:

$$\begin{align}
\hat{x}_0 = \hat{x}_{0,\theta}
\end{align}$$

The posterior step is then applied to that direct estimate:

$$\begin{align}
q(x_{t-1} \mid x_t, \hat{x}_{0,\theta})
\end{align}$$

with the same closed-form posterior mean

$$\begin{align}
\tilde{\mu}_t(x_t, \hat{x}_{0,\theta})
&=
\frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\hat{x}_{0,\theta}
+
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t
\end{align}$$

So the generation path is:

- predict clean sample directly
- run the posterior update from that estimate

#### Practical Implication

The outer reverse-DDPM loop is the same in both cases. What changes is the
numerical route by which the sampler obtains $\hat{x}_0$, and that matters
because the posterior update depends entirely on the quality of the clean-sample
estimate.

In SOPE this branch is implemented in
[third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py):

- `predict_start_from_noise(...)`:
  [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py#L356)
- `p_mean_variance(...)`:
  [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py#L378)

## 2. SOPE Conditioning Versus Robomimic Conditioning

### 2.1 SOPE Uses In-Painting-Style Conditioning

SOPE's diffusion model conditions by explicitly pinning known state values at
selected timesteps. The helper `apply_conditioning(...)` is used during both
training and sampling inside
[third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py).

This style is closest to trajectory in-painting:

- the conditioning information is placed directly into the sequence tensor
- conditioned positions are re-pinned after corruption / denoising steps
- the denoiser still operates on a trajectory-shaped tensor rather than on a
  separate context embedding

In the older local wrapper, this was exposed through `make_cond(...)` and
prefix-state construction in the removed `src/sope_diffuser.py` path.

### 2.2 Robomimic Diffusion Policy Uses Explicit Context Conditioning

Robomimic's diffusion policy backbone does not use SOPE-style in-painting on
the denoised sequence.

Instead:

1. observations are encoded by `ObservationGroupEncoder`
2. encoded observation history is flattened into a single conditioning vector
3. that vector is injected into `ConditionalUnet1D`

The relevant path is:

- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py#L62)
- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py#L193)
- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py#L213)

### 2.3 The Robomimic Backbone Is FiLM-Style

Inside
[third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py),
`ConditionalResidualBlock1D` implements FiLM-style modulation:

- the conditioning vector is mapped to per-channel scale and bias
- the residual block applies `scale * out + bias`

Relevant lines:

- [third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py#L74)
- [third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py#L97)
- [third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py#L103)

This is explicit context conditioning, not in-painting.

## 3. Canonical Backbone And Consolidation

The repository now standardizes on the FiLM-conditioned robomimic backbone in
[src/diffusion.py](../src/diffusion.py).

### 3.1 Canonical API

The canonical public surface is now:

- `NormalizationStats`
- `make_normalizers(...)`
- `SopeDiffusionConfig`
- `SopeDiffuser`
- `cross_validate_configs(...)`

Training and evaluation now use that single API directly:

- [src/train.py](../src/train.py) instantiates the canonical diffuser
- [src/eval.py](../src/eval.py) loads checkpoints through the same config and
  wrapper path
- the training scripts import `SopeDiffusionConfig` from
  [src/diffusion.py](../src/diffusion.py)

### 3.2 Why This Path Won

The canonical path:

- keeps SOPE's DDPM objective, scheduler coefficients, and guidance math
- swaps the denoiser to robomimic's `ConditionalUnet1D`
- flattens `states_from` into one FiLM conditioning vector of shape
  `(B, frame_stack * state_dim)`
- predicts only the future transition chunk
  `states_to[:, :-1, :] || actions_to`

So the active design changes the conditioning path and denoiser backbone, not
the outer DDPM loop.

### 3.3 Compatibility Decision

This consolidation intentionally breaks compatibility with:

- imports from `src.sope_diffuser`
- older checkpoints that depended on `diffuser_kind`
- older checkpoints that relied on the removed module path

Any experiments that still need the removed path must be retrained or migrated
manually.

## 4. Implication For The Current Robomimic Bug

The current main-branch report isolates the failure to the conditioned 19D
robomimic setup:

- DDPM loss decreases
- sampled future RMSE stays poor
- the simpler unconditioned EEF-only control experiment works

See [docs/reports/200331.md](./reports/200331.md).

That pattern suggests the issue is not "diffusion cannot learn anything at
all", but rather "the current conditioning formulation is not giving the model
usable future-prediction context on the full 19D state."

This makes FiLM-style context conditioning a reasonable next design to test.

Why:

- the task is structurally "predict future chunk given observed history"
- the current prefix-state approach forces the history to live inside the same
  sequence tensor as the prediction target
- robomimic's own diffusion policy already solves a closely related conditioning
  problem with explicit encoded context

This does not prove that FiLM is universally better than SOPE's in-painting.
It only means FiLM is a better match for the current robomimic debugging target.

## 5. Before / After Summary

Before this note:

- `predict_epsilon` and conditioning differences were being discussed across
  chat and reports, but not in one implementation-oriented reference

After this note:

- the code paths for DDPM target parameterization are linked directly
- the distinction between SOPE conditioning and robomimic conditioning is
  explicit
- the rationale for trying FiLM conditioning on the current 19D robomimic setup
  is documented in one place

## 6. Validation To Re-Run

If the conditioning path is changed, the minimum follow-up checks should be:

1. one-chunk held-out RMSE on the same 19D robomimic setup from
   [docs/reports/200331.md](./reports/200331.md)
2. comparison against the current prefix-state in-painting baseline
3. the existing EEF-only control to verify the refactor did not break the base
   diffusion path
4. `python3 -m py_compile src/diffusion.py src/train.py src/eval.py scripts/train_sope.py scripts/train_sope_gym.py scripts/train_sope_film.py`
5. a one-step training smoke test and checkpoint save/load/eval round-trip on
   the canonical path
