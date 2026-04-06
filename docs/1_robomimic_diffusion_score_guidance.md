# Robomimic Diffusion Guidance And FiLM Backbone

This note consolidates the repository's robomimic-specific diffusion guidance
and `ConditionalUnet1D` conditioning notes.

Relevant code:

- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py)
- [third_party/robomimic/robomimic/config/diffusion_policy_config.py](../third_party/robomimic/robomimic/config/diffusion_policy_config.py)
- [third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)
- [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py)
- [third_party/sope/opelab/core/policy.py](../third_party/sope/opelab/core/policy.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/robomimic_interface/checkpoints.py](../src/robomimic_interface/checkpoints.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)

## 1. Summary

Robomimic's diffusion policy is trained as an epsilon predictor over action
sequences, not as a tractable density model with a native
$\log \pi(a \mid s)$ interface. For SOPE-style guidance, the usable signal is
the denoising score implied by the DDPM parameterization:

$$\begin{align}
\nabla_{a_t} \log p_t(a_t \mid s)
\approx
- \frac{\hat{\epsilon}_\theta(a_t, t, s)}{\sqrt{1 - \bar{\alpha}_t}}.
\end{align}$$

In this repository, the same robomimic backbone is also the canonical FiLM-like
conditioning implementation. `FilmConditionedBackbone` in
[src/diffusion.py](../src/diffusion.py) is only a thin adapter around
robomimic's `ConditionalUnet1D`; the actual conditioning happens inside the
robomimic residual blocks through per-channel scale and bias modulation.

## 2. What Robomimic Actually Trains

Robomimic's diffusion policy:

1. encodes the observation history
2. samples Gaussian noise and a diffusion timestep
3. corrupts clean action chunks with the scheduler
4. predicts the noise with `noise_pred_net`
5. minimizes MSE against the sampled noise

The model is therefore a sequence model over action chunks. The public network
contract is:

- `sample`: `(B, T, input_dim)`
- `timestep`: scalar or `(B,)`
- `global_cond`: `(B, global_cond_dim)` or `None`

Internally the sample is moved to channel-first layout:

$$\begin{align}
\text{sample} &\in \mathbb{R}^{B \times T \times \text{input\_dim}} \\
x &= \text{moveaxis}(\text{sample}, -1, -2)
\in \mathbb{R}^{B \times \text{input\_dim} \times T}.
\end{align}$$

## 3. SOPE Guidance Contract

SOPE's guided diffusion path expects a policy object that can return an
action-space score compatible with the sampled chunk tensor:

$$\begin{align}
\texttt{policy.grad\_log\_prob(states, actions)}
\rightarrow
\frac{\partial \log p(\text{actions} \mid \text{states})}
{\partial \text{actions}}.
\end{align}$$

That means the robomimic integration problem is not only "extract a score from
the denoiser." The adapter must also:

- reconstruct the raw `PolicyAlgo` from a checkpoint
- access EMA weights, `noise_pred_net`, `obs_encoder`, and scheduler state
- align robomimic sequence shapes with SOPE's guidance call site
- align action normalization conventions
- pass the active SOPE diffusion timestep into the guidance path

## 4. Why The Backbone Is FiLM-Style

Robomimic's `ConditionalUnet1D` does not instantiate
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
parameters that modulate the intermediate convolution features. This is the
practical reason the repo-level diffusion wrapper treats robomimic as the
canonical FiLM-conditioned backbone.

## 5. Integration Caveats

### 5.1 Timestep mapping

The guidance adapter must use the same diffusion timestep convention as the
active SOPE sampler. Any mismatch between scheduler indices or timestep scaling
changes the score magnitude.

### 5.2 Shape mismatch

Robomimic is naturally sequence-level, while SOPE's current call site is often
flattened to per-step `(N \cdot T, D)` action tensors. The two plausible
integration choices are:

1. change the guidance path to operate on full action chunks
2. use a single-step approximation that discards some sequence structure

The first option is more faithful to the model that robomimic actually trains.

### 5.3 Action normalization

The score is defined in whatever action space the denoiser sees. If SOPE
samples in normalized action space while the external interface expects raw
actions, the guidance term must be transformed consistently.

### 5.4 Observation conditioning

Guidance depends on the same observation encoding path as robomimic rollout.
Using `RolloutPolicy` alone is not enough when direct access to encoder and
scheduler internals is required.

## 6. Repository Status

The repo already notes that guided sampling is not yet wired end to end through
the local SOPE wrapper:

- `guided=True` is still incomplete in the current sampling path
- guidance hyperparameters are not fully threaded through
- the active sampler still assumes any attached guidance policy already matches
  the expected `grad_log_prob(...)` contract

## 7. Validation

The smallest meaningful checks for this part of the stack are:

1. reconstruct a robomimic `PolicyAlgo` from a checkpoint and confirm access to
   EMA weights, `noise_pred_net`, `obs_encoder`, and scheduler state
2. verify the returned action-gradient shape matches the SOPE action tensor
3. compare score magnitudes before and after normalization
4. run one guided and one unguided chunk sample and confirm only the action
   slice changes
5. rerun `python -m py_compile src/diffusion.py src/robomimic_interface/checkpoints.py src/robomimic_interface/rollout.py`
