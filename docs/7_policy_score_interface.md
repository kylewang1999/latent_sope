# Policy Score Interface And Postprocess Contract For Guided Sampling

Relevant code:

- [src/sampling.py](../src/sampling.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py)

## 1. Summary

[`guided_sample_step`](../src/sampling.py) now treats the policy-score API and
action-score postprocessing as one explicit local contract instead of encoding
those choices implicitly through older upstream-derived guidance switches.

The public FiLM sampling API under [`FilmGaussianDiffusion.conditional_sample`](../src/diffusion.py#L372)
now uses:

- `action_score_scale`
- `action_score_postprocess`
- `action_neg_score_weight`
- `clamp_linf`

## 2. Action-Score Contract

The local interface guard has been renamed to
[`_requires_action_score_interface`](../src/sampling.py). The contract is
deliberately narrow: the sampler only requires that the policy object expose
`grad_log_prob(state, action)` on chunk-shaped tensors.

This keeps the FiLM guidance code agnostic to the policy head type. A diffusion
policy still satisfies the contract, but a future non-diffusion head such as an
RNN-GMM policy can also plug in as long as it exposes the same action-score
entry point.

For the current local sampler, the expected shapes are:

- `state`: `[B, H, Ds]`
- `action`: `[B, H, Da]`
- return value: `[B, H, Da]`

## 3. Behavior Before And After

### 3.1 Before

The local sampler exposed upstream-derived switches whose interaction encoded
several different behaviors:

- whether to L2-normalize action scores
- whether to clip action scores
- whether the behavior-score weight affected the behavior score at all

This made the local guidance contract harder to read than the actual
implementation needed to be.

### 3.2 After

The FiLM-guided chunk sampler now uses an explicit local contract for
action-score postprocessing:

- `action_score_scale: float`
- `action_score_postprocess: Literal["none", "l2", "clamp"]`
- `action_neg_score_weight: float`
- `clamp_linf: float`

## 4. Postprocess Semantics

Inside [`prepare_guidance`](../src/sampling.py), the sampler retrieves raw
action-only scores on chunk tensors and then applies one of three local
heuristics independently at each chunk timestep:

- `"none"`: keep the raw score unchanged
- `"l2"`: L2-normalize each per-timestep action-score vector
- `"clamp"`: clip both target and behavior scores to `[-clamp_linf, clamp_linf]`

The final local guide is then

$$\begin{align}
\text{guide}
=
\text{postprocess}(g_{\pi})
-
\text{action\_neg\_score\_weight}\,\text{postprocess}(g_{\beta}),
\end{align}$$

when negative guidance is enabled, and just the postprocessed target score
otherwise.

This is intentionally not a behavior-preserving alias layer for upstream SOPE.
In particular, `action_neg_score_weight` is always active and is applied after
postprocessing the behavior score.

## 5. Implementation Notes

- [`GuidanceOptions`](../src/sampling.py) now validates
  `action_score_postprocess` eagerly during construction and rejects
  unsupported values.
- [`prepare_guidance`](../src/sampling.py) now performs the full local
  guidance path in one place: flatten per-timestep `(state, action)` pairs,
  query raw action scores, postprocess them according to the selected mode,
  combine target and behavior scores, scale the result, and zero-pad the
  action-only guide back to the chunk shape.
- [`FilmGaussianDiffusion.conditional_sample`](../src/diffusion.py#L372)
  exposes the renamed public sampler kwargs.

## 6. Scheduler Device Handling

`DiffusionPolicy.grad_log_prob(...)` converts robomimic's predicted epsilon
into a score by reading `noise_scheduler.alphas_cumprod` at the configured
guidance timestep.

That scheduler tensor is not part of the robomimic module tree, so moving the
policy network to CUDA does not guarantee that `alphas_cumprod` moves with it.
The adapter now materializes `alphas_cumprod` on the current action / denoiser
device before gathering the per-query $\bar{\alpha}_t$ values.

Before this fix, guided OPE runs could fail on CUDA with a device-mismatch
error when indexing a CPU scheduler tensor using a CUDA timestep tensor. After
the fix, the score-conversion path uses the same device as the denoiser output.

## 7. Validation

The minimal validation for this change is:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 -m py_compile src/sampling.py src/diffusion.py src/robomimic_interface/policy.py
```
