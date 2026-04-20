# Port-Over FiLM-Conditioned Diffusion

## 1. Summary

This note documents the self-contained FiLM-conditioned sequence diffusion
package added under [`port_over/`](../port_over/).

The port is intended for reuse in other projects that need a denoiser and DDPM
wrapper over generic `(B, T, D)` tensors, optionally conditioned on a FiLM
vector of shape `(B, Z)`.

Relevant code:

- [`port_over/film_unet.py`](../port_over/film_unet.py)
- [`port_over/diffusion.py`](../port_over/diffusion.py)
- [`port_over/sampling.py`](../port_over/sampling.py)
- [`port_over/__init__.py`](../port_over/__init__.py)

## 2. What Was Ported

### 2.1. Denoiser Blocks

The UNet-side port keeps the FiLM-conditioned building blocks from:

- [`src/diffusion.py`](../src/diffusion.py)
- [`third_party/robomimic/robomimic/models/diffusion_policy_nets.py`](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)

Specifically, the port includes:

- sinusoidal timestep embeddings
- 1D downsample and upsample layers
- convolution, group norm, and Mish blocks
- FiLM-conditioned residual blocks
- `ConditionalUnet1D`
- `FilmConditionedBackbone`

The FiLM behavior is preserved: the timestep embedding is concatenated with the
global condition and injected through per-channel scale and bias terms in each
residual block.

### 2.2. Diffusion Wrapper

The DDPM wrapper keeps the generic reverse and forward diffusion math from:

- [`src/diffusion.py`](../src/diffusion.py)
- [`third_party/sope/opelab/core/baselines/diffusion/diffusion.py`](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py)

The new wrapper exposes:

- `predict_start_from_noise`
- `q_posterior`
- `q_sample`
- `p_mean_variance`
- `p_losses`
- `loss`
- `p_sample_loop`
- `conditional_sample`

It supports both parameterizations:

- `predict_epsilon=True`: train the model to predict diffusion noise
- `predict_epsilon=False`: train the model to predict `x0` directly

### 2.3. Sampling Helper

The default sampler is based on the `guided_sample_step` structure in
[`src/sampling.py`](../src/sampling.py), but the port intentionally implements
only unguided DDPM sampling.

## 3. What Was Intentionally Removed

The port excludes all project-specific logic that would make it harder to reuse
outside this repository:

- normalization and unnormalization hooks
- policy guidance and behavior-policy subtraction
- per-dimension and per-timestep weighted losses
- observation or action split assumptions in the public API
- trajectory chunk conditioning by overwriting coordinates
- autoregressive rollout stitching and prefix-state contracts
- repo-specific progress bars and diagnostics

One remaining default is that `clip_denoised=True` clamps reconstructed `x0` to
`[-1, 1]`. This is disabled by default.

## 4. Public API

### 4.1. Tensor Shapes

The package assumes:

- denoised data tensor: `(B, T, D)`
- FiLM condition tensor: `(B, Z)`

There is no required state/action split. `D` is treated as a single generic
feature dimension.

### 4.2. Main Classes

- `ConditionalUnet1D(input_dim, global_cond_dim, diffusion_step_embed_dim=256, down_dims=(256, 512, 1024), kernel_size=5, n_groups=8)`
- `FilmConditionedBackbone(transition_dim, global_cond_dim, down_dims, diffusion_step_embed_dim=256, kernel_size=5, n_groups=8)`
- `FilmGaussianDiffusion(model, data_dim, n_timesteps=1000, loss_type="l2", clip_denoised=False, predict_epsilon=True, beta_schedule="cosine")`

## 5. Minimal Usage

```python
import torch

from port_over import FilmConditionedBackbone, FilmGaussianDiffusion


B, T, D, Z = 8, 32, 6, 10
x = torch.randn(B, T, D)
cond = torch.randn(B, Z)

model = FilmConditionedBackbone(
    transition_dim=D,
    global_cond_dim=Z,
    down_dims=(128, 256, 512),
)

diffusion = FilmGaussianDiffusion(
    model=model,
    data_dim=D,
    n_timesteps=100,
    loss_type="l2",
    predict_epsilon=True,
)

loss, info = diffusion.loss(x, cond)
sample = diffusion.conditional_sample(shape=(B, T, D), cond=cond)
print(loss.item(), sample.trajectories.shape)
```

If no FiLM conditioning is needed, build the backbone with `global_cond_dim=0`
and omit `cond`.

## 6. Validation Commands

Run validation in the `latent_sope` conda environment:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 -m py_compile port_over/*.py
```

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 - <<'PY'
import torch
from port_over import FilmConditionedBackbone, FilmGaussianDiffusion

B, T, D, Z = 2, 8, 4, 6
x = torch.randn(B, T, D)
cond = torch.randn(B, Z)

model = FilmConditionedBackbone(D, Z, down_dims=(32, 64))
diffusion = FilmGaussianDiffusion(model=model, data_dim=D, n_timesteps=8)
loss, _ = diffusion.loss(x, cond)
sample, info = diffusion.conditional_sample((B, T, D), cond=cond, return_chain=True, return_info=True)

assert torch.isfinite(loss)
assert sample.trajectories.shape == (B, T, D)
assert sample.chains is not None
assert info["guidance"] is None
PY
```

Re-run those checks if the ported UNet architecture, diffusion math, or public
tensor contracts change.
