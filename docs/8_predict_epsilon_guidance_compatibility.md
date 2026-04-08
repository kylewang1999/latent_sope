# Predict Epsilon Compatibility For Chunk Sampling And Guidance

Relevant code:

- [src/diffusion.py](../src/diffusion.py)
- [src/sampling.py](../src/sampling.py)
- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py)
- [third_party/sope/opelab/core/policy.py](../third_party/sope/opelab/core/policy.py)
- [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py)
- [third_party/sope/opelab/core/baselines/diffusion/helpers.py](../third_party/sope/opelab/core/baselines/diffusion/helpers.py)
- [third_party/sope/opelab/core/baselines/diffusion/temporal.py](../third_party/sope/opelab/core/baselines/diffusion/temporal.py)
- [third_party/sope/opelab/core/baselines/diffuser.py](../third_party/sope/opelab/core/baselines/diffuser.py)
- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py)

## 1. Summary

This note explains how the chunk diffusion model's `predict_epsilon` setting and
the guidance policy's denoiser parameterization affect sampling.

Short answer:

- chunk-level `predict_epsilon` changes how the chunk denoiser output is
  interpreted, but not the fixed DDPM posterior-variance schedule
- policy-level `predict_epsilon` changes how a policy denoiser output must be
  converted into a score or `grad_log_prob`
- the chunk model and the guidance policy do not need to use the same
  parameterization
- in the current repository, robomimic guidance is only implemented correctly
  for epsilon-predicting policies

## 2. Chunk Diffusion Effect

The SOPE chunk diffuser uses a fixed beta schedule and the usual DDPM posterior:

$$\begin{align}
q(x_t \mid x_0)
&=
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}\,x_0,
(1 - \bar{\alpha}_t) I
\right), \\
q(x_{t-1} \mid x_t, x_0)
&=
\mathcal{N}
\left(
x_{t-1};
\mu_t(x_t, x_0),
\sigma_t^2 I
\right).
\end{align}$$

In [`third_party/sope/opelab/core/baselines/diffusion/diffusion.py`](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py),
`predict_epsilon` only changes how the model output is converted into $\hat{x}_0$:

$$\begin{align}
\hat{x}_0(x_t, t)
=
\frac{1}{\sqrt{\bar{\alpha}_t}}
\left(
x_t - \sqrt{1 - \bar{\alpha}_t}\,\hat{\epsilon}_\theta(x_t, t)
\right)
\end{align}$$

when `predict_epsilon=True`, while `predict_epsilon=False` means the network
output is interpreted directly as $\hat{x}_0$.

That affects:

- the training target
- the predicted reverse-process mean
- the sample trajectory quality and calibration

That does not affect:

- the stored beta schedule
- the posterior variance $\sigma_t^2$
- the Gaussian noise magnitude injected at each reverse step

So for unguided chunk sampling, changing chunk-level `predict_epsilon` changes
the reverse-process mean path, not the variance schedule.

## 3. Guided Chunk Sampling Effect

The local guided sampler in [`src/diffusion.py`](../src/diffusion.py) first
computes the chunk model's reverse moments, then adds a guidance drift term to
the mean before sampling:

$$\begin{align}
\mu_t^{\text{guided}}
=
\mu_t^{\text{chunk}}
+ g_t,
\end{align}$$

where $g_t$ is the externally computed guidance term.

The sample update remains

$$\begin{align}
x_{t-1}
=
\mu_t^{\text{guided}}
+ \sigma_t z,
\qquad
z \sim \mathcal{N}(0, I).
\end{align}$$

Therefore:

- chunk-level `predict_epsilon` still changes the chunk model's mean path
- guidance changes the mean again through an added drift term
- the reverse-process variance schedule still comes from the chunk diffuser, not
  from the policy parameterization

In this repository, the chunk diffuser owns the sampling noise schedule even
when a policy provides guidance.

## 4. Policy Parameterization Effect

The policy side matters differently. The guidance policy is not sampled with the
chunk diffuser's DDPM update. Instead, its denoiser output is converted into a
score-like gradient.

For an epsilon-predicting policy, the standard diffusion-score relation is

$$\begin{align}
\nabla_{x_t} \log p_t(x_t \mid s)
\approx
- \frac{\hat{\epsilon}_\theta(x_t, t, s)}{\sqrt{1 - \bar{\alpha}_t}}.
\end{align}$$

That is the approximation currently used by
[`src/robomimic_interface/policy.py`](../src/robomimic_interface/policy.py).

If a policy instead predicts $\hat{x}_0$ directly, the score is not obtained by
reusing the epsilon formula with $\hat{x}_0$ in place of $\hat{\epsilon}$.
The correct conversion must go through the Gaussian posterior implied by the
chosen parameterization.

So policy-level `(no)-predict-epsilon` affects:

- the meaning of the policy denoiser output
- the formula needed to compute `grad_log_prob`
- the magnitude and direction of the guidance term

It does not directly change the chunk sampler's variance schedule.

## 5. Estimating Grad-Log-Prob For Chunk Diffusion With `predict_epsilon=False`

If the chunk diffusion model itself is used as a scorer, or if we want to
reason about a guidance term induced by the chunk denoiser, then
`predict_epsilon=False` changes the score conversion in a specific way.

When the denoiser predicts $\hat{x}_0(x_t, t, c)$ directly for conditioning
context $c$, the forward noising relation is still

$$\begin{align}
x_t
=
\sqrt{\bar{\alpha}_t}\,x_0
+ \sqrt{1 - \bar{\alpha}_t}\,\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I).
\end{align}$$

The DDPM score for the noisy marginal satisfies the Tweedie-style identity

$$\begin{align}
\nabla_{x_t} \log p_t(x_t \mid c)
&=
- \frac{x_t - \sqrt{\bar{\alpha}_t}\,\mathbb{E}[x_0 \mid x_t, c]}{1 - \bar{\alpha}_t}.
\end{align}$$

Replacing the posterior mean $\mathbb{E}[x_0 \mid x_t, c]$ with the model
prediction $\hat{x}_0(x_t, t, c)$ gives the practical estimator

$$\begin{align}
\widehat{\nabla_{x_t} \log p_t(x_t \mid c)}
&\approx
\frac{\sqrt{\bar{\alpha}_t}\,\hat{x}_0(x_t, t, c) - x_t}{1 - \bar{\alpha}_t}.
\end{align}$$

This is the direct-$x_0$ analogue of the usual epsilon formula
$-\hat{\epsilon}_\theta / \sqrt{1 - \bar{\alpha}_t}$.

These two views are consistent. If

$$\begin{align}
\hat{x}_0
=
\frac{1}{\sqrt{\bar{\alpha}_t}}
\left(
x_t - \sqrt{1 - \bar{\alpha}_t}\,\hat{\epsilon}_\theta
\right),
\end{align}$$

then substituting into the direct-$x_0$ score expression recovers

$$\begin{align}
\frac{\sqrt{\bar{\alpha}_t}\,\hat{x}_0 - x_t}{1 - \bar{\alpha}_t}
=
- \frac{\hat{\epsilon}_\theta}{\sqrt{1 - \bar{\alpha}_t}}.
\end{align}$$

For this repository's chunk trajectories, $x_t$ concatenates state and action
channels. If the goal is a guidance vector over actions only, first estimate the
full chunk score and then take only the action slice:

$$\begin{align}
\hat{g}_t^{\text{chunk}}
&=
\left[
\widehat{\nabla_{x_t} \log p_t(x_t \mid c)}
\right]_{\text{action dims}}.
\end{align}$$

Operationally, that means:

1. run the chunk denoiser at timestep $t$ to obtain $\hat{x}_0(x_t, t, c)$
2. form $\left(\sqrt{\bar{\alpha}_t}\,\hat{x}_0 - x_t\right) / (1 - \bar{\alpha}_t)$
3. zero out or discard the state channels if guidance should act only on actions
4. apply any existing guidance scaling or normalization after this conversion

One numerical caveat matters near $t = 0$: the denominator
$1 - \bar{\alpha}_t$ can become very small, so implementations should clamp it
away from zero before dividing.

## 6. Do The Chunk Model And Policy Have To Match?

No. They are separate models with separate roles.

The chunk diffuser only needs internal consistency between:

1. its forward noising process
2. its training target
3. its reverse-time `predict_start_from_noise(...)` logic

The policy only needs internal consistency between:

1. its own diffusion scheduler and parameterization
2. the denoiser output interpretation
3. the score conversion used by `grad_log_prob`

There is no mathematical requirement that both models use the same choice of
epsilon prediction versus direct $x_0$ prediction.

What must match is the semantics inside each model's own diffusion system.

## 7. Current Repository Status

For the current codebase:

- the chunk diffuser supports either chunk-level `predict_epsilon=True` or
  `False` because the SOPE diffusion wrapper handles both training and sampling
  consistently
- guided chunk sampling still uses the chunk diffuser's own posterior-variance
  schedule regardless of policy type
- the robomimic guidance adapter currently assumes the policy is
  epsilon-predicting when it converts denoiser output into a score
- the adapter also uses a fixed score timestep rather than the active chunk
  sampler timestep, so guidance is already an approximation even in the
  epsilon-prediction case

The practical consequence is:

- chunk-side `predict_epsilon=False` is acceptable if the chunk model is trained
  and sampled consistently
- policy-side `predict_epsilon=False` is not correctly supported by the current
  robomimic guidance adapter

## 8. Why `gradlog_diffusion(...)` Zeros The State Dimensions

In the legacy SOPE guidance path, `gradlog_diffusion(...)` constructs a
trajectory-shaped tensor, fills the action slice with `policy.grad_log_prob`,
and explicitly writes zeros into the state slice.

That behavior is intentional rather than an omission.

The key reason is that the policy guidance signal is interpreted as an
action-conditional score:

$$\begin{align}
\nabla_a \log \pi(a \mid s),
\end{align}$$

not as a joint trajectory score
$\nabla_{(s, a)} \log p(s, a)$.

For the non-diffusion policies in SOPE, this is explicit:

- `gradlog(...)` detaches `state_t`
- it enables gradients only on `action_t`
- it returns a tensor whose nonzero entries live only in the action channels

For the diffusion-policy adapter, the contract is the same. The method
`DiffusionPolicy.grad_log_prob(...)` in
[`third_party/sope/opelab/core/policy.py`](../third_party/sope/opelab/core/policy.py)
computes a denoising score on the action sample conditioned on the state
embedding. It does not expose a score for changing the state itself.

Therefore, if the sampler were to update the state coordinates directly using
this signal, it would be optimizing the state to make the policy look more
likely, which is not the intended guidance objective.

Operationally, the guidance update in the SOPE sampler is:

$$\begin{align}
\mu_t^{\text{guided}}
=
\mu_t^{\text{model}}
+ \lambda_t g_t,
\end{align}$$

where $g_t$ is nonzero only in the action channels.

This means the current substep applies policy guidance only to the action part
of the trajectory, while leaving the state part untouched by the explicit
gradient update.

## 9. Can Later States Still Change?

Yes, but only indirectly.

At a single reverse step, adding the guide does not directly modify the state
entries because the guide tensor has zeros in those dimensions. In that narrow
sense, the state slice is protected from direct policy-gradient edits.

However, the updated trajectory is then fed back into the next denoising step.
The trajectory denoiser is a joint temporal U-Net over the full transition
vector and horizon, so later predictions for state channels can depend on the
actions that were modified in earlier guidance steps.

The practical consequence is:

- current-step states are not directly nudged by `gradlog_diffusion(...)`
- subsequent denoising iterations can still move state channels because the
  denoiser couples state and action coordinates across time
- conditioned state entries are a special case because
  `apply_conditioning(...)` writes them back after each reverse step in the
  original SOPE sampler

This is the intended compromise for SOPE-style action guidance:

- use the policy score only where its semantics are valid, namely the action
  coordinates
- let the trajectory model absorb those action edits and propagate them into a
  dynamically consistent rollout over later denoising steps

The current implementation clue is the local sampler itself:

- [src/sampling.py](../src/sampling.py) computes policy guidance through
  `gradlog_diffusion(...)`
- both target and behavior policies are required to expose
  `grad_log_prob(state, action)`
- the guide tensor still has nonzero entries only in the action channels

## 10. Recommended Interpretation And Local Sampler Status

For experiments in this repository, read the compatibility rule as:

- chunk-model parameterization controls chunk denoising semantics
- policy parameterization controls guidance-score semantics
- they do not need to match each other
- the current guidance implementation should be treated as valid only for
  epsilon-predicting robomimic policies

The local refactor history that used to live in separate sampler notes is now
part of this same interpretation:

- the active one-step sampler and reverse loop live in
  [src/sampling.py](../src/sampling.py)
- the local guidance path is diffusion-only and no longer falls back to
  `gradlog(...)` for non-diffusion policies
- removed local sampler-only args such as `gmode`, `reverse`, `transform`,
  `state_scale`, `neg_grad_scale`, and `use_action_grad_only` are not part of
  the current repo contract

If policy-side `no-predict-epsilon` support is needed, update
[`src/robomimic_interface/policy.py`](../src/robomimic_interface/policy.py) so
`grad_log_prob(...)` derives the correct score from the policy's actual
parameterization instead of assuming
$-\hat{\epsilon}_\theta / \sqrt{1 - \bar{\alpha}_t}$.

If chunk-side scoring from a direct-$x_0$ denoiser is needed, use the
direct-$x_0$ score estimate
$\left(\sqrt{\bar{\alpha}_t}\,\hat{x}_0 - x_t\right) / (1 - \bar{\alpha}_t)$
before slicing to the action channels.

## 11. Validation

If guidance or parameterization semantics change, rerun at least:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python -m py_compile src/diffusion.py src/sampling.py src/robomimic_interface/policy.py
```

If policy-side parameterization support is extended, rerun one small unguided
chunk sample and one guided chunk sample and verify:

- the chunk sampler still uses the same posterior-variance schedule
- only the guidance drift changes
- guidance magnitudes remain numerically stable across timesteps
