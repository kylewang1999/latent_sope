# Policy Score Interface for Guided Sampling

## 1. Change Summary

[`guided_sample_step`](../src/sampling.py) now treats action-score postprocessing as
an explicit local contract instead of encoding it implicitly through the older
upstream-derived guidance switches.

The public FiLM sampling API under [`FilmGaussianDiffusion.conditional_sample`](../src/diffusion.py#L372)
now uses:

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

## 3. Postprocess Semantics

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

## 4. Validation

The minimal validation for this change is:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 -m py_compile src/sampling.py src/diffusion.py src/robomimic_interface/policy.py
```
