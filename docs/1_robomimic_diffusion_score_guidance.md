# Robomimic Diffusion Score Guidance For SOPE

This note documents how to recover a score-like guidance term from a robomimic diffusion policy and apply it to SOPE's guided trajectory diffusion.

Relevant code:

- Robomimic diffusion policy: [`third_party/robomimic/robomimic/algo/diffusion_policy.py`](../third_party/robomimic/robomimic/algo/diffusion_policy.py)
- Robomimic diffusion config: [`third_party/robomimic/robomimic/config/diffusion_policy_config.py`](../third_party/robomimic/robomimic/config/diffusion_policy_config.py)
- SOPE guided diffusion path: [`third_party/sope/opelab/core/baselines/diffusion/diffusion.py`](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py)
- SOPE diffusion policy wrapper: [`third_party/sope/opelab/core/policy.py`](../third_party/sope/opelab/core/policy.py)
- Robomimic checkpoint reconstruction in this repo: [`src/latent_sope/robomimic_interface/checkpoints.py`](../src/latent_sope/robomimic_interface/checkpoints.py)
- SOPE chunk diffuser wrapper in this repo: [`src/latent_sope/diffusion/sope_diffuser.py`](../src/latent_sope/diffusion/sope_diffuser.py)
- Robomimic latent / feature hook helper: [`src/latent_sope/robomimic_interface/rollout.py`](../src/latent_sope/robomimic_interface/rollout.py)

## Summary

Robomimic's diffusion policy does not expose an exact tractable $\\log \pi(a \mid s)$ or $\\nabla_a \log \pi(a \mid s)$ API. The model is trained as an epsilon predictor: given noisy actions $a_t$, timestep $t$, and observation conditioning, it predicts the forward-process noise $\\hat\\epsilon_\\theta(a_t, t, s)$.

The usable guidance signal is therefore the denoising score implied by the DDPM parameterization:

$$\begin{align}
\nabla_{a_t} \log p_t(a_t \mid s)
\approx
- \frac{\hat{\epsilon}_\theta(a_t, t, s)}{\sqrt{1 - \bar{\alpha}_t}}
\end{align}$$

This is the object that should back SOPE's `policy.grad_log_prob(...)` interface for diffusion policies.

## What Robomimic Actually Trains

Robomimic's diffusion policy:

- builds an observation encoder and a `noise_pred_net`
- samples Gaussian noise and timesteps
- corrupts clean actions with the scheduler
- regresses the predicted noise against the sampled noise with MSE

The implementation lives in [`third_party/robomimic/robomimic/algo/diffusion_policy.py`](../third_party/robomimic/robomimic/algo/diffusion_policy.py). The default config enables DDPM with epsilon prediction in [`third_party/robomimic/robomimic/config/diffusion_policy_config.py`](../third_party/robomimic/robomimic/config/diffusion_policy_config.py).

Concretely, the policy is conditioned on:

- an observation window of length `observation_horizon = 2`
- an action prediction window of length `prediction_horizon = 16`

by default.

That means the robomimic model is fundamentally a sequence model over action chunks, not a per-step conditional density model.

## What SOPE Expects

SOPE's guided diffusion path calls `gradlog_diffusion(...)` in [`third_party/sope/opelab/core/baselines/diffusion/diffusion.py`](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py). That helper currently:

- splits a trajectory chunk into `states` and `actions`
- flattens `(N, T, D)` into `(N \cdot T, D)`
- calls `policy.grad_log_prob(states, actions)`
- writes the returned action-gradient back into the chunk tensor

So the current SOPE contract for diffusion guidance is:

$$\begin{align}
\texttt{policy.grad\_log\_prob(states, actions)}
\rightarrow
\frac{\partial \log p(\text{actions} \mid \text{states})}{\partial \text{actions}}
\end{align}$$

SOPE's own `DiffusionPolicy` wrapper in [`third_party/sope/opelab/core/policy.py`](../third_party/sope/opelab/core/policy.py) already implements this idea by turning the diffusion model output into a score-like term.

## Recommended Adapter Contract

Use the raw robomimic `PolicyAlgo`, not just `RolloutPolicy`, because the guidance path needs direct access to:

- `algo.nets["policy"]["noise_pred_net"]`
- `algo.nets["policy"]["obs_encoder"]`
- `algo.noise_scheduler`
- `algo.ema`

In this repository, reconstruct the raw algo with [`build_algo_from_checkpoint(...)`](../src/latent_sope/robomimic_interface/checkpoints.py), not only the rollout wrapper.

At guidance time:

1. Use EMA weights if available, matching robomimic's inference path.
2. Build the robomimic observation conditioning exactly as robomimic does:
   - encode the observation window with `obs_encoder`
   - flatten it into `obs_cond`
3. Feed the current noisy action sample and mapped timestep into `noise_pred_net`.
4. Convert predicted noise into a score estimate with the DDPM scaling term.

The intended adapter is:

$$\begin{align}
\texttt{grad\_log\_prob(states, actions, t)}
&=
- \frac{\hat{\epsilon}_\theta(actions, t, obs\_cond(states))}{\sqrt{1 - \bar{\alpha}_t}}
\end{align}$$

where $\\bar{\alpha}_t$ comes from robomimic's diffusion scheduler.

## Timestep Mapping

SOPE chunk diffusion and robomimic policy diffusion do not necessarily use the same number of diffusion steps.

If SOPE runs with timestep index $t_{\text{sope}} \in \{0, \dots, T_{\text{sope}} - 1\}$ and robomimic uses $T_{\text{rm}}$ training timesteps, map them with a simple monotone rescaling:

$$\begin{align}
t_{\text{rm}}
&=
\text{round}\left(
\frac{t_{\text{sope}}}{\max(T_{\text{sope}} - 1, 1)}
\cdot
(T_{\text{rm}} - 1)
\right)
\end{align}$$

This is not mathematically exact across different beta schedules, but it is the minimal consistent mapping when the two samplers do not share a schedule object.

## Shape Mismatch: Exact Vs Approximate Integration

This is the main caveat.

Robomimic's diffusion policy is trained on action sequences. By default it consumes:

- two observation frames
- a sixteen-step action sequence

SOPE's current `gradlog_diffusion(...)` path instead flattens each chunk into independent per-step `(state_t, action_t)` pairs before calling `policy.grad_log_prob(...)`.

Those interfaces are not exactly aligned.

There are two integration options:

### Option A: Sequence-level guidance

This is the principled version.

Modify SOPE so `gradlog_diffusion(...)` passes:

- an observation window
- an action window
- the current diffusion timestep

into the robomimic score adapter.

Then compute guidance over the whole action sequence and write the resulting action-window gradient back into the SOPE chunk tensor.

### Option B: Single-step approximation

This is simpler but approximate.

Force the robomimic diffusion policy into a single-step setting, for example:

- `observation_horizon = 1`
- `prediction_horizon = 1`

Then the robomimic policy behaves much more like the per-step contract SOPE currently assumes.

## Action Normalization Caveat

Robomimic expects diffusion-policy actions to be normalized to $[-1, 1]$. The training code checks this explicitly in [`third_party/robomimic/robomimic/algo/diffusion_policy.py`](../third_party/robomimic/robomimic/algo/diffusion_policy.py).

SOPE's guided sampler currently unnormalizes `model_mean` before calling the policy guidance path in [`third_party/sope/opelab/core/baselines/diffusion/diffusion.py`](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py).

That creates a mismatch unless one of the following is true:

- guidance is computed in normalized action space, or
- the gradient is transformed back through the action-normalization Jacobian

The safer implementation is to keep robomimic guidance in the same normalized action space that the robomimic diffusion model was trained on.

## Observation Conditioning Caveat

Robomimic's observation conditioning is not just a flat state vector. The policy:

- encodes observations with its own `obs_encoder`
- flattens the encoded observation window
- conditions the denoiser on that encoded tensor

If SOPE uses the same encoder latents as its `z_t`, those latents can serve as the conditioning input directly. This repository already has hook-based utilities for extracting robomimic features in [`src/latent_sope/robomimic_interface/rollout.py`](../src/latent_sope/robomimic_interface/rollout.py).

If SOPE instead stores raw low-dimensional observations in `z_t`, then the adapter must recreate robomimic's observation-conditioning path before evaluating the denoiser.

## Minimal Implementation Sketch

The adapter logic should look like:

```python
class RobomimicDiffusionScore:
    def __init__(self, algo, sope_num_steps):
        self.algo = algo
        self.nets = algo.ema.averaged_model if algo.ema is not None else algo.nets
        self.scheduler = algo.noise_scheduler
        self.obs_horizon = algo.algo_config.horizon.observation_horizon
        self.pred_horizon = algo.algo_config.horizon.prediction_horizon
        self.sope_num_steps = sope_num_steps

    def map_timestep(self, t_sope):
        rm_steps = self.scheduler.config.num_train_timesteps
        frac = t_sope.float() / max(self.sope_num_steps - 1, 1)
        return torch.clamp((frac * (rm_steps - 1)).round().long(), 0, rm_steps - 1)

    def grad_log_prob(self, states, actions, t_sope):
        t_rm = self.map_timestep(t_sope)
        obs_cond = states[:, :self.obs_horizon].reshape(states.size(0), -1)
        eps_hat = self.nets["policy"]["noise_pred_net"](
            sample=actions[:, :self.pred_horizon],
            timestep=t_rm,
            global_cond=obs_cond,
        )
        alpha_bar = self.scheduler.alphas_cumprod[t_rm].to(actions.device)
        sigma_t = torch.sqrt(1.0 - alpha_bar).view(-1, 1, 1)
        return -eps_hat / (sigma_t + 1e-6)
```

This sketch is sequence-level. It is not a correct drop-in for SOPE's current flattened `(N \cdot T, D)` interface without also changing the SOPE guidance call site.

## Current Repository Status

This repository already notes that guided sampling is not wired end to end in [`src/latent_sope/diffusion/sope_diffuser.py`](../src/latent_sope/diffusion/sope_diffuser.py). In particular:

- `guided=True` is still marked as TODO
- guidance hyperparameters are not yet threaded through fully
- the current SOPE sampling path assumes the policy guidance object already matches the expected `grad_log_prob(...)` contract

So the missing work is not only "extract a score from robomimic." It is also:

- adapt the robomimic denoiser into a SOPE guidance object
- align sequence shapes
- align action normalization
- pass the current SOPE diffusion timestep into the policy score path

## Validation To Run After Implementing

The smallest meaningful checks are:

1. Reconstruct a robomimic diffusion-policy `PolicyAlgo` from a checkpoint and verify the adapter can access EMA weights, `noise_pred_net`, and scheduler state.
2. On one batch, confirm the returned gradient shape matches the SOPE action tensor shape exactly.
3. Compare gradient magnitudes before and after normalization to make sure the score is not exploding at small $1 - \bar{\alpha}_t$.
4. Run a one-chunk guided SOPE sample with `guided=False` and `guided=True` and verify that only the action slice changes.
5. If using normalized action guidance, verify that robomimic's action range assumption remains satisfied throughout the guidance call.

Residual risk remains highest around:

- timestep mapping across different samplers
- sequence-shape alignment
- guidance in unnormalized action space
