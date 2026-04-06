# `ConditionalUnet1D` Structure And FiLM-Style Conditioning

Relevant code:
- [third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)
- [src/diffusion.py](../src/diffusion.py)

## Summary

This note documents the structure of robomimic's
[`ConditionalUnet1D`](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)
and explains why its conditioning path is FiLM-style even though it does not
instantiate
[`FiLMLayer`](../third_party/robomimic/robomimic/models/base_nets.py).

In this repository, [`FilmConditionedBackbone`](../src/diffusion.py) is only a
thin adapter around `ConditionalUnet1D`. The actual conditioning happens
inside each `ConditionalResidualBlock1D`, where a conditioning vector is mapped
to per-channel scale and bias parameters and then applied to the intermediate
1D convolution features.

## Tensor Layout

The public `ConditionalUnet1D.forward(...)` contract is:

- `sample`: `(B, T, input_dim)`
- `timestep`: scalar or `(B,)`
- `global_cond`: `(B, global_cond_dim)` or `None`

Inside the network, the sample is moved to channel-first layout:

$$\begin{align}
\text{sample} &\in \mathbb{R}^{B \times T \times \text{input\_dim}} \\
x &= \text{moveaxis}(\text{sample}, -1, -2) \in \mathbb{R}^{B \times \text{input\_dim} \times T}
\end{align}$$

The timestep embedding path produces:

$$\begin{align}
\text{time\_emb} \in \mathbb{R}^{B \times \text{diffusion\_step\_embed\_dim}}
\end{align}$$

and if `global_cond` is present, the model concatenates them to produce:

$$\begin{align}
\text{global\_feature} \in \mathbb{R}^{B \times \text{cond\_dim}},
\qquad
\text{cond\_dim} = \text{diffusion\_step\_embed\_dim} + \text{global\_cond\_dim}
\end{align}$$

This same `global_feature` is passed into every conditioned residual block in
the down path, middle blocks, and up path.

## Network Structure

Let

$$\begin{align}
\text{all\_dims} = [\text{input\_dim}] + \text{down\_dims}.
\end{align}$$

Then:

- the down path contains one stage for each adjacent pair in `all_dims`
- the middle path contains two `ConditionalResidualBlock1D` blocks at the
  deepest channel width
- the up path contains one stage for each adjacent pair in
  `reversed(all_dims[1:])`
- `final_conv` maps the last decoder feature map back to `input_dim`

For the default robomimic setting `down_dims=[256, 512, 1024]`, the module
graph is:

```mermaid
flowchart TD
    A[sample: (B,T,input_dim)] --> B[moveaxis to (B,input_dim,T)]
    B --> D1[Down Block 1<br/>ConditionalResidualBlock1D input_dim→256<br/>ConditionalResidualBlock1D 256→256<br/>Downsample1d]
    D1 --> D2[Down Block 2<br/>ConditionalResidualBlock1D 256→512<br/>ConditionalResidualBlock1D 512→512<br/>Downsample1d]
    D2 --> D3[Down Block 3<br/>ConditionalResidualBlock1D 512→1024<br/>ConditionalResidualBlock1D 1024→1024<br/>Identity]

    T[timestep] --> TE[diffusion_step_encoder<br/>SinusoidalPosEmb → Linear → Mish → Linear]
    G[global_cond: (B,global_cond_dim)] --> CAT[concat if present]
    TE --> CAT
    CAT --> GF[global_feature: (B,cond_dim)]

    GF -.-> D1
    GF -.-> D2
    GF -.-> D3

    D3 --> M1[Mid Block 1<br/>ConditionalResidualBlock1D 1024→1024]
    M1 --> M2[Mid Block 2<br/>ConditionalResidualBlock1D 1024→1024]
    GF -.-> M1
    GF -.-> M2

    D2 -. skip .-> U1
    M2 --> U1[Up Block 1<br/>concat skip: 1024 + 512<br/>ConditionalResidualBlock1D 1536→512<br/>ConditionalResidualBlock1D 512→512<br/>Upsample1d]
    GF -.-> U1

    D1 -. skip .-> U2
    U1 --> U2[Up Block 2<br/>concat skip: 512 + 256<br/>ConditionalResidualBlock1D 768→256<br/>ConditionalResidualBlock1D 256→256<br/>Identity]
    GF -.-> U2

    U2 --> F[final_conv<br/>Conv1dBlock 256→256<br/>Conv1d 256→input_dim]
    F --> O[moveaxis back to (B,T,input_dim)]
```

Two implementation details matter here:

1. each down stage pushes its post-residual output onto the skip stack before
   downsampling
2. the up path is shorter than the down path because it iterates over
   `reversed(in_out[1:])`, so the deepest down stage does not get a symmetric
   decoder stage

## Why The Conditioning Is FiLM-Style

The conditioning mechanism lives in
[`ConditionalResidualBlock1D`](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py).

Given:

- feature input `x` with shape `(B, C_{\text{in}}, T)`
- conditioning vector `cond` with shape `(B, \text{cond\_dim})`

the block first computes a feature map:

$$\begin{align}
F = \text{Conv1dBlock}(x) \in \mathbb{R}^{B \times C_{\text{out}} \times T}
\end{align}$$

It then maps `cond` through `cond_encoder`:

$$\begin{align}
\text{cond\_encoder}(\text{cond}) \in \mathbb{R}^{B \times 2C_{\text{out}} \times 1}
\end{align}$$

which is reshaped and split into:

$$\begin{align}
\gamma, \beta \in \mathbb{R}^{B \times C_{\text{out}} \times 1}
\end{align}$$

and applied by broadcast over the temporal dimension:

$$\begin{align}
\tilde{F}_{b,c,t} = \gamma_{b,c} F_{b,c,t} + \beta_{b,c}
\end{align}$$

This is the defining FiLM pattern: a conditioning-dependent per-channel affine
transformation of intermediate features.

The current robomimic parameterization uses:

$$\begin{align}
\tilde{F} = \gamma \odot F + \beta
\end{align}$$

rather than the `FiLMLayer` variant

$$\begin{align}
\tilde{F} = (1 + \gamma) \odot F + \beta,
\end{align}$$

but both are FiLM-family conditioning schemes. The difference is only the
specific affine parameterization.

## Relationship To `FilmConditionedBackbone`

[`FilmConditionedBackbone`](../src/diffusion.py) does not itself implement
FiLM logic. Its job is only to adapt the SOPE diffusion interface to the
robomimic denoiser API by passing:

- `x`
- `time`
- flattened prefix conditioning as `global_cond`

into `ConditionalUnet1D`.

So the correct boundary is:

- `FilmConditionedBackbone`: adapter
- `ConditionalUnet1D`: conditioned U-Net
- `ConditionalResidualBlock1D`: where the FiLM-style modulation actually occurs

## Validation

If the robomimic denoiser structure changes, re-read:

- [`ConditionalUnet1D.__init__`](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)
- [`ConditionalUnet1D.forward`](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)
- [`ConditionalResidualBlock1D.forward`](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)

to verify that the stage counts, skip connections, and FiLM-style affine
modulation described here still match the implementation.
