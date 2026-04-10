# Robomimic Diffusion Guidance, FiLM Backbone, And Visual Encoder Training

Relevant code:

- [third_party/sope/opelab/core/baselines/diffusion/helpers.py](../third_party/sope/opelab/core/baselines/diffusion/helpers.py)
- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py)
- [third_party/robomimic/robomimic/config/diffusion_policy_config.py](../third_party/robomimic/robomimic/config/diffusion_policy_config.py)
- [third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)
- [third_party/robomimic/robomimic/models/obs_core.py](../third_party/robomimic/robomimic/models/obs_core.py)
- [third_party/robomimic/robomimic/models/base_nets.py](../third_party/robomimic/robomimic/models/base_nets.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/sampling.py](../src/sampling.py)
- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py)

## 1. Summary

Robomimic's diffusion policy is trained as an epsilon predictor over action
sequences, not as a tractable density model with a native
$\log \pi(a \mid s)$ interface. In the current repository:

- the usable guidance signal is the denoising score implied by that
  epsilon-prediction parameterization
- the local sampler expects diffusion-style policies exposing
  `grad_log_prob(state, action)`
- `FilmConditionedBackbone` is a thin adapter around robomimic's
  `ConditionalUnet1D`
- the default robomimic visual encoder is trained jointly with the diffusion
  U-Net unless the chosen backbone explicitly freezes itself

## 2. What Robomimic Actually Trains

Robomimic's diffusion policy:

1. encodes the observation history
2. samples Gaussian noise and a diffusion timestep
3. corrupts clean action chunks with the scheduler
4. predicts the noise with `noise_pred_net`
5. minimizes MSE against the sampled noise

The model is therefore a sequence model over action chunks. The score used for
local guidance is the usual DDPM score approximation:

$$\begin{align}
\nabla_{a_t} \log p_t(a_t \mid s)
\approx
- \frac{\hat{\epsilon}_\theta(a_t, t, s)}{\sqrt{1 - \bar{\alpha}_t}}.
\end{align}$$

## 3. Local Guidance Contract

The local SOPE-style sampler does not ask robomimic for an exact likelihood.
Instead it requires a policy adapter with the contract

$$\begin{align}
\texttt{policy.grad\_log\_prob(states, actions)}
\rightarrow
\frac{\partial \log p(\text{actions} \mid \text{states})}
{\partial \text{actions}}.
\end{align}$$

In the current repo that means:

- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py)
  reconstructs the robomimic policy internals and exposes `grad_log_prob(...)`
- [src/sampling.py](../src/sampling.py) assumes both target and behavior
  policies follow that same diffusion-policy contract
- [src/diffusion.py](../src/diffusion.py) exposes the local FiLM guidance knobs
  `action_score_postprocess`, `action_neg_score_weight`, and `clamp_linf`
- guidance edits only action channels; the local sampler does not treat the
  policy as a scorer over state coordinates

The current local chunk contract is:

- `states`: `[B, H, Ds]`
- `actions`: `[B, H, Da]`
- `grad_log_prob(states, actions)`: `[B, H, Da]`

Current caveat:

- the adapter still uses a fixed score timestep rather than the active chunk
  sampler timestep, so guidance remains an approximation even though it is now
  wired end to end for diffusion-policy adapters

### 3.1 Parameterization compatibility

The chunk diffuser and the guidance policy are separate diffusion contracts. In
particular:

- chunk-side and policy-side parameterizations do not need to match
- the chunk diffuser always uses its own beta schedule, posterior mean path, and
  posterior-variance schedule
- changing chunk-side `predict_epsilon` changes the denoiser training target and
  the interpretation of the chunk reverse mean, but it does not change the
  Gaussian noise scale injected by the chunk sampler
- changing policy-side parameterization changes only the score-conversion
  formula used inside `grad_log_prob(state, action)`

The current robomimic adapter implements only the epsilon-prediction score
conversion. So chunk-side `predict_epsilon=False` is supported as long as the
chunk model remains internally consistent, but policy-side `predict_epsilon=False`
would require replacing the epsilon-based score conversion with the corresponding
`predict-x0` form.

## 4. Why The Backbone Is FiLM-Style

Robomimic's `ConditionalUnet1D` does not instantiate a class literally called
`FiLMLayer`, but its conditioning path is still FiLM-style.

The timestep embedding and optional observation-conditioning vector are
concatenated into one global conditioning vector:

$$\begin{align}
\text{global\_feature}
\in
\mathbb{R}^{B \times \text{cond\_dim}},
\qquad
\text{cond\_dim}
=
\text{diffusion\_step\_embed\_dim}
+
\text{global\_cond\_dim}.
\end{align}$$

Each conditioned residual block maps that vector into per-channel scale and bias
parameters that modulate intermediate convolution features. This is why the
local diffusion wrapper treats robomimic as the canonical FiLM-conditioned
backbone.

### 4.1 Contrast with upstream SOPE conditioning

Upstream SOPE uses in-painting-style conditioning: `apply_conditioning(...)`
overwrites selected trajectory entries inside the sampled tensor, and the
conditioning is re-applied during reverse sampling.

The local wrapper instead passes the conditioning prefix as an external FiLM
context to the denoiser. So the local implementation should be described as
SOPE-inspired rather than identical to upstream SOPE conditioning.

## 5. Visual Encoder Training Contract

Robomimic's default image path uses `VisualCore`, which by default uses a
trainable `ResNet18Conv` backbone with `SpatialSoftmax`.

During standard diffusion-policy BC training:

- the observation encoder lives inside the `policy` module together with
  `noise_pred_net`
- the forward pass runs through `obs_encoder` before the denoiser
- the optimizer is built over the enclosing `policy` module

So for the default configuration, the visual encoder is trained jointly with
the diffusion U-Net.

This is not universal. Pretrained backbones such as `R3MConv` and `MVPConv`
expose a `freeze` flag and can remain frozen even though the outer robomimic
policy is still optimized.

## 6. Validation

The smallest meaningful checks for this part of the stack are:

1. reconstruct a robomimic `PolicyAlgo` from checkpoint artifacts and confirm
   access to EMA weights, `noise_pred_net`, `obs_encoder`, and scheduler state
2. verify that `grad_log_prob(...)` returns the expected action-gradient shape
3. run one guided and one unguided chunk sample and confirm only action
   coordinates receive explicit score edits
4. rerun `python -m py_compile src/diffusion.py src/sampling.py src/robomimic_interface/policy.py`
