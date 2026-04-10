# Action-Score Postprocess Contract For FiLM Guidance

Relevant code:

- [src/sampling.py](../src/sampling.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py)

## 1. Summary

The FiLM-guided chunk sampler now uses an explicit local contract for
action-score postprocessing:

- `action_score_scale: float`
- `action_score_postprocess: Literal["none", "l2", "clamp"]`
- `action_neg_score_weight: float`
- `clamp_linf: float`

This replaces the earlier local reuse of upstream SOPE's
branch encoding for score normalization, score clipping, and behavior-score
weighting. The goal is not upstream-parity naming; the goal is a more
transparent local API for the robomimic-guided FiLM sampler.

## 2. Behavior Before And After

### 2.1 Before

The local sampler exposed upstream-derived switches whose interaction encoded
several different behaviors:

- whether to L2-normalize action scores
- whether to clip action scores
- whether the behavior-score weight affected the behavior score at all

This made the local contract harder to read than the actual implementation
needed to be.

### 2.2 After

The new local behavior is:

- `"none"` keeps raw target and behavior scores unchanged
- `"l2"` L2-normalizes each per-timestep score vector
- `"clamp"` clips score tensors elementwise to `[-clamp_linf, clamp_linf]`
- `action_neg_score_weight` always scales the already-postprocessed behavior score

So, with negative guidance enabled, the implemented rule is

$$\begin{align}
\text{guide}
=
\text{postprocess}(g_{\pi})
-
\text{action\_neg\_score\_weight}\,\text{postprocess}(g_{\beta}).
\end{align}$$

When negative guidance is disabled, the guide is just the postprocessed target
score.

## 3. Implementation Notes

- [`GuidanceOptions`](../src/sampling.py) now validates
  `action_score_postprocess` eagerly during construction and rejects unsupported values.
- [`prepare_guidance`](../src/sampling.py) now performs the full local guidance
  path in one place: flatten per-timestep `(state, action)` pairs, query raw
  action scores, postprocess them according to the selected mode, combine target
  and behavior scores, scale the result, and zero-pad the action-only guide back
  to the chunk shape.
- [`FilmGaussianDiffusion.conditional_sample`](../src/diffusion.py#L372) exposes
  the renamed public sampler kwargs.

## 4. Validation

The smallest validation to rerun after this change is:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 -m py_compile src/sampling.py src/diffusion.py src/robomimic_interface/policy.py
```
