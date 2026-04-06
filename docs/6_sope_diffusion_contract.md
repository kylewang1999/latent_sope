# SOPE Diffusion Contract And EEF Debug Modes

Relevant code:
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)
- [src/sope_diffuser.py](../src/sope_diffuser.py)
- [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py)
- [third_party/sope/opelab/core/baselines/diffusion/helpers.py](../third_party/sope/opelab/core/baselines/diffusion/helpers.py)
- [third_party/sope/opelab/examples/d4rl/diffusion_trainer.py](../third_party/sope/opelab/examples/d4rl/diffusion_trainer.py)

This note consolidates four closely related questions:

1. how the local chunk fields map onto original SOPE
2. how SOPE's `q_sample`, `p_losses`, and reverse sampling path map onto the
   standard DDPM equations
3. how the EEF-only loss-mask ablation works in the full-state contract
4. how the newer EEF-only unconditioned debug mode changes the dataset,
   conditioning, and evaluation behavior

The goal is to make the conditioning contract explicit.

## 1. Standard DDPM Reminder

DDPM has:

1. a fixed forward process $q$
2. a learned reverse process $p_\theta$

### 1.1 Forward noising

$$\begin{align}
q(x_t \mid x_0)
=
\mathcal{N}\left(
x_t;
\sqrt{\bar\alpha_t}\, x_0,
(1-\bar\alpha_t)I
\right)
\end{align}$$

and the closed-form sample is:

$$\begin{align}
x_t
=
\sqrt{\bar\alpha_t}\, x_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I).
\end{align}$$

### 1.2 Reverse denoising

The reverse model predicts either noise or $x_0$. In the epsilon-prediction
parameterization,

$$\begin{align}
\epsilon_\theta(x_t, t)
\end{align}$$

is used to reconstruct

$$\begin{align}
\hat x_0(x_t, t)
=
\frac{1}{\sqrt{\bar\alpha_t}}
\left(
x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)
\right).
\end{align}$$

Training usually minimizes:

$$\begin{align}
\mathcal{L}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\left\|
\epsilon - \epsilon_\theta(x_t, t)
\right\|_2^2
\right].
\end{align}$$

## 2. How SOPE Implements DDPM

In vendored SOPE:

- `q_sample(...)` implements the closed-form forward noising equation
- `p_losses(...)` is the training objective
- `p_sample_loop(...)` is the reverse denoising loop used at inference

### 2.1 `q_sample`

In [diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py),

```python
sample = (
    extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
    extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
)
```

so `q_sample(x_start, t, noise)` is exactly:

$$\begin{align}
x_t
=
\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon.
\end{align}$$

### 2.2 `p_losses`

`p_losses(...)` does:

1. sample `noise`
2. build `x_noisy = q_sample(x_start, t, noise)`
3. apply conditioning to `x_noisy`
4. run the denoiser `self.model(x_noisy, t)`
5. apply conditioning again to `x_recon`
6. compare against `noise` or `x_start`

In code:

```python
x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
x_noisy = apply_conditioning(x_noisy, cond, self.observation_dim, reverse=self.reverse)

x_recon = self.model(x_noisy, t)
x_recon = apply_conditioning(x_recon, cond, self.observation_dim, reverse=self.reverse)
```

This means `apply_conditioning` does not change the model weights or the model
architecture. It modifies the tensors passed into and returned from the model.

More precisely:

- the forward noising process first corrupts the whole trajectory
- conditioning then overwrites the conditioned slice of the noisy sample
- the denoiser $\epsilon_\theta$ sees that conditioned noisy tensor as input
- after the model prediction, the same conditioned slice is overwritten again

So conditioning is enforced around the denoiser, not inside the denoiser
module itself.

### 2.3 Reverse sampling

At inference, [p_sample_loop(...)](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py)
starts from Gaussian noise:

```python
x = torch.randn(shape)
```

Then:

1. condition `x` once before the loop
2. for each reverse timestep, run one denoising step
3. re-apply conditioning after every step

So the conditioned slice is enforced throughout the whole backward denoising
process.

## 3. What `apply_conditioning` Actually Does

The helper [apply_conditioning(...)](../third_party/sope/opelab/core/baselines/diffusion/helpers.py)
does not use a mask over all transition dimensions. It hard-codes one block:

```python
if reverse:
    x[:, t, action_dim:] = val.clone()
else:
    x[:, t, :state_dim] = val.clone()
```

With the default `reverse=False`, each `cond[t]` overwrites only:

$$\begin{align}
x[:, t, :\text{state\_dim}]
\end{align}$$

That means:

- conditioned timesteps are still part of the trajectory
- the conditioned state channels are clamped
- the action channels at those same timesteps remain latent

## 4. Original SOPE Conditioning Contract

In original SOPE D4RL training, each example is a full state-action trajectory:

$$\begin{align}
x = [(s_0, a_0), \dots, (s_{T-1}, a_{T-1})].
\end{align}$$

The trainer constructs:

```python
conds = {0: batch[:, 0, :state_dim]}
loss, infos = diffusion_model.loss(batch, conds)
```

So original SOPE trains on state-action trajectories while conditioning only on
known state slices.

That is coherent because the action channels, including the action at the
conditioned timestep, are part of the trajectory the model is supposed to
generate.

## 5. Local Chunk Fields

For local chunking at start index `t0`, the dataset builds:

- `states_from = [s_{t0-S}, \dots, s_{t0-1}]`
- `actions_from = [a_{t0-S}, \dots, a_{t0-1}]`
- `states_to = [s_{t0}, s_{t0+1}, \dots, s_{t0+W}]`
- `actions_to = [a_{t0}, a_{t0+1}, \dots, a_{t0+W-1}]`

where:

- `S = frame_stack`
- `W = chunk_size`

This decomposes the rollout into:

- a historical prefix: `states_from`, `actions_from`
- a future chunk to predict: `states_to[:-1]`, `actions_to`

## 6. SOPE-Aligned Mapping for Local Chunks

To match original SOPE, the local wrapper should map the chunk fields as:

- conditioning input:
  `states_from`
- generated future chunk:
  `states_to[:-1]`, `actions_to`

The subtle part is the prefix timesteps.

Those prefix timesteps are still present in the diffusion trajectory tensor, but
only their state channels are observed through `cond`. Their action channels are
not explicit conditions.

So the SOPE-aligned training target is:

$$\begin{align}
x_{\text{train}}
=
\big[
(s_{t0-S}, 0), \dots, (s_{t0-1}, 0),
(s_{t0}, a_{t0}), \dots, (s_{t0+W-1}, a_{t0+W-1})
\big].
\end{align}$$

and the conditioning dict is:

$$\begin{align}
\text{cond}
=
\{0: s_{t0-S}, \dots, S-1: s_{t0-1}\}.
\end{align}$$

This is now implemented in [src/diffusion.py](../src/diffusion.py):

## 7. EEF-Only Loss-Mask Ablation

The rollout latents used by [`scripts/train_sope.py`](../scripts/train_sope.py)
come from the low-dim concatenation path in
[src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
and [src/robomimic_interface/encoders.py](../src/robomimic_interface/encoders.py).

For robomimic low-dim Lift observations, the default sorted key order yields:

- `object`: 10 dims
- `robot0_eef_pos`: 3 dims
- `robot0_eef_quat`: 4 dims
- `robot0_gripper_qpos`: 2 dims

Therefore the per-timestep `robot0_eef_pos` slice is `$[10:13)$`.

`SopeDiffusionConfig.diffuser_eef_pos_only` is a loss-mask ablation on top of
the full-state transition contract:

- rollout, dataset, and model tensor shapes remain unchanged
- only the future `robot0_eef_pos` slice contributes to diffusion loss
- other state channels and all action channels remain present in model input
  and output but are not supervised

This is useful for checking whether poor chunk RMSE is concentrated in
non-EEF channels while preserving the original SOPE-style conditioning path.

## 8. EEF-Only Unconditioned Debug Mode

The narrower debug mode uses a dataset-schema change instead of a loss mask.

With:

- `RolloutChunkDatasetConfig.state_projection="eef_pos"`
- `RolloutChunkDatasetConfig.disable_conditioning=True`
- `SopeDiffusionConfig.conditioning_mode="none"`

the robomimic rollout dataset now:

- projects low-dimensional states to `robot0_eef_pos`
- keeps the same batch keys for compatibility
- zeroes action tensors so SOPE still sees the expected transition width
- computes normalization stats over active EEF state dimensions plus zeroed
  action placeholders

The diffuser now:

- returns `None` from `make_cond(...)`
- trains on only the future chunk horizon instead of prefix-plus-future
- ignores action channels in the diffusion loss for this mode

This debug path is intentionally not the same thing as
`diffuser_eef_pos_only=True`:

- `diffuser_eef_pos_only=True` keeps the full-state dataset and only masks the
  loss
- `conditioning_mode="none"` with `state_projection="eef_pos"` actually
  reduces the active state space to 3D EEF position and removes prefix-state
  conditioning

Full autoregressive trajectory generation remains unsupported for this
unconditioned debug mode.

## 9. Evaluation-Side EEF Diagnostics

The chunk evaluator in [src/eval.py](../src/eval.py) now computes
`robot0_eef_pos` diagnostics from the same slice used by the training-time
loss-mask ablation or the 3D projected debug path.

When an EEF slice is available, eval reports:

- `rmse_eef_pos`
- `mean_eef_pos`
- `mean_eef_pos_gt`

Those diagnostics are included in the `eval_metrics:*` and
`eval_diagnostics:*` summaries sent to `wandb`.

If the state layout does not expose an EEF slice, those metrics remain unset
instead of failing evaluation. In the unconditioned 3D debug path, action and
transition metrics are also left unset because they are not meaningful.

## 10. Validation

Small validation to rerun after changes:

1. `python3 -m py_compile src/sope_diffuser.py src/eval.py src/train.py`
2. one-batch diffusion loss smoke test for the active training mode
3. chunk evaluation smoke test confirming the expected metric set is populated

The current canonical implementation differs from that historical contract:

- the active `SopeDiffuser` in [src/diffusion.py](../src/diffusion.py) predicts
  only the future chunk
- `make_cond(...)` flattens `states_from` into FiLM conditioning
- there is no separate prefix-in-trajectory training path anymore

## 11. Why `actions_from` Is Not SOPE Conditioning

`actions_from` is historical context from the local dataset, but it is not part
of SOPE's explicit conditioning API.

Under the current wrapper, `actions_from` is useful for:

- debugging
- persistence baselines
- future experiments with explicit past-action conditioning

but it is not clamped by `apply_conditioning(...)`.

So the correct interpretation is:

- `states_from`: conditioned observed prefix states
- `actions_from`: auxiliary historical metadata, not explicit SOPE condition
- `states_to[:-1]`: future state targets
- `actions_to`: future action targets

## 12. Why the Old Local Mapping Was Inconsistent

Previously, the wrapper built:

$$\begin{align}
\big[
(s_{t0-S}, a_{t0-S}), \dots, (s_{t0-1}, a_{t0-1}),
(s_{t0}, a_{t0}), \dots
\big]
\end{align}$$

while still conditioning only on `states_from`.

This did not break tensor dimensions, but it created a semantic mismatch:

- prefix actions appeared in `x_start` as if they were part of the observed
  prefix trajectory
- `apply_conditioning(...)` never clamps those action channels
- sampling therefore treated them as latent, not observed

Original SOPE does train on state-action trajectories with state-only
conditioning, but there the unconditioned action channels are part of the
trajectory to be generated. In the old local wrapper, some of those
unconditioned action channels belonged to the historical prefix instead.

## 13. Conditioning and the Denoiser $\epsilon_\theta$

It is easy to misread `apply_conditioning(...)` as modifying the model itself.
It does not.

The denoiser is still the model call:

$$\begin{align}
\epsilon_\theta(x_t, t).
\end{align}$$

Conditioning is implemented by overwriting parts of the trajectory tensor around
that call:

### 13.1 In training

$$\begin{align}
x_t &= q(x_t \mid x_0), \\
\tilde x_t &= \text{apply\_conditioning}(x_t, \text{cond}), \\
\hat\epsilon &= \epsilon_\theta(\tilde x_t, t), \\
\hat\epsilon_{\text{cond}} &= \text{apply\_conditioning}(\hat\epsilon, \text{cond}).
\end{align}$$

### 13.2 In reverse sampling

At every reverse step:

$$\begin{align}
x_t
\rightarrow
\text{reverse-step}
\rightarrow
x_{t-1}
\rightarrow
\text{apply\_conditioning}(x_{t-1}, \text{cond}).
\end{align}$$

So conditioning affects:

- the noisy input presented to the denoiser during training
- the model output used in the training loss
- the initial noisy sample at inference
- every reverse denoising step during inference

But it always acts by overwriting selected tensor slices, not by changing the
definition of $\epsilon_\theta$ itself.

## 14. Practical Summary

When reading the current local SOPE wrapper:

- the diffusion model still operates on full state-action trajectories
- `apply_conditioning(...)` clamps only the conditioned state slice by default
- conditioned timesteps remain part of the generated trajectory tensor
- the action channels at conditioned timesteps are still latent unless explicit
  action conditioning is added
- the local chunk wrapper now matches that contract by treating `states_from` as
  the condition and not treating `actions_from` as supervised observed prefix
  content
