"""Guided sampling helpers for the FiLM-conditioned chunk diffuser.

Important attributes of this re-implementation:

1. FiLM conditioning, not in-painting:
   `cond` is passed into `model.p_mean_variance(...)` on every reverse step.
   This sampler does not re-apply `apply_conditioning(...)`-style hard
   overwrites to the sampled chunk. That is correct for the current FiLM
   contract, but it is not equivalent to upstream SOPE's in-painting regime.
   If parts of the trajectory tensor must remain exact after each reverse step,
   add an explicit projection step outside this helper.

Important checks before trusting guided samples from this module:

1. Policy-score space must match chunk space:
   guidance is computed after `model_mean` is unnormalized and before the result
   is renormalized for the DDPM update. The policy adapter must therefore expect
   the same state and action representation produced by `unnormalizer(...)`.
   For the current robomimic adapter, the main thing to verify is the action
   space: robomimic diffusion policies are trained on normalized actions, while
   rollout files may store executed environment actions. If those differ,
   guidance direction may still look plausible while its scale is wrong.

2. Robomimic score timestep is still approximate:
   `DiffusionPolicy.grad_log_prob(...)` currently uses a fixed
   `DiffusionPolicyScoreConfig.score_timestep` instead of the active chunk
   sampler timestep `t`. This keeps the interface simple, but the resulting
   score is only an approximation to the desired diffusion-time-conditioned
   policy score.

3. Chunk-level score contract is still heuristic for robomimic policies:
   `prepare_guidance(...)` now passes the full sampled chunk to
   `policy.grad_log_prob(state, action)` as `[B, H, *]` tensors. The current
   robomimic adapter still reduces that chunk to per-step surrogate queries
   internally before reshaping back to `[B, H, Da]`, so guidance remains an
   approximation rather than the exact score of a joint action-chunk density.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Protocol

import numpy as np
import torch

import third_party.sope.opelab.core.baselines.diffusion.utils as diffusion_utils


@dataclass(frozen=True)
class Sample:
    trajectories: torch.Tensor  # [B, T, Ds+Da]: final sampled transition chunk in normalized model space.
    chains: Optional[torch.Tensor]  # [B, K + 1, T, D] or None: full reverse chain, including the initial Gaussian noise.


def get_schedule_multiplier(
    t: int | torch.Tensor,
    n_timesteps: int,
    schedule_type: str = "cosine",
) -> float:
    if isinstance(t, torch.Tensor):
        t = t.item()
    t_frac = t / n_timesteps

    if schedule_type == "cosine":
        return 0.5 * (1 + math.cos(math.pi * t_frac))
    if schedule_type == "linear":
        return 1 - t_frac
    if schedule_type == "sigmoid":
        k = 10
        mid = 0.5
        return 1 / (1 + math.exp(k * (t_frac - mid)))
    raise ValueError(f"Unknown schedule type: {schedule_type}")


def make_timesteps(batch_size: int, i: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), i, device=device, dtype=torch.long)


SampleInfo = dict[str, Optional[torch.Tensor]]


class ScorePolicy(Protocol):
    def grad_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor: ...


class TrajChunkDiffusionModel(Protocol):
    betas: torch.Tensor
    n_timesteps: int
    observation_dim: int
    action_dim: int
    policy: Optional[ScorePolicy]
    behavior_policy: Optional[ScorePolicy]
    normalizer: Optional[Callable[[Any], Any]]
    unnormalizer: Optional[Callable[[Any], Any]]

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class GuidanceOptions:
    num_guidance_iters: int = 2  # Repeats the guide update this many times per reverse step.
    use_adaptive: bool = True  # Upstream switch between constant and cosine-scheduled guidance scaling.
    use_neg_grad: bool = True  # Subtract the behavior-policy score from the target-policy score.
    action_score_scale: float = 0.2  # Multiplier applied after combining scores into the final guide.
    action_score_postprocess: Literal["none", "l2", "clamp"] = "l2"
    action_neg_score_weight: float = 1.0  # Behavior-score weight applied after score postprocessing.
    clamp_linf: float = 1.0  # Bound used when `action_score_postprocess == "clamp"`.

    def __post_init__(self) -> None:
        assert self.action_score_postprocess in {"none", "l2", "clamp"}, \
            "Unsupported action_score_postprocess=" \
            f"{self.action_score_postprocess!r}. Expected one of: 'none', 'l2', 'clamp'."
        assert self.action_score_scale > 0, "action_score_scale must be positive."
        assert self.action_neg_score_weight >= 0, "action_neg_score_weight must be non-negative."
        assert self.clamp_linf >= 0, "clamp_linf must be non-negative."


def _init_sample_info(
    model_mean: torch.Tensor,
    state_dim: int,
    return_info: bool,
) -> SampleInfo:
    info: SampleInfo = {
        "model_prediction": None,
        "guidance": None,
    }
    if return_info:
        info["model_prediction"] = model_mean[..., state_dim:].detach().clone()
    return info

def _update_sample_info(
    info: SampleInfo,
    guide: torch.Tensor, # (B, T, Da+Ds) where the [..., :state_dim] are padded with zeros
    state_dim: int,
    return_info: bool,
) -> None:
    if return_info:
        info["guidance"] = guide[..., state_dim:].detach().clone()

def _requires_action_score_interface(
    policy: Any,
    *,
    role: str,
) -> ScorePolicy:
    assert policy is not None, f"{role} requires a policy."
    grad_log_prob = getattr(policy, "grad_log_prob", None)
    assert callable(grad_log_prob), f"{role} must expose grad_log_prob(state, action)"
    return policy


def _validate_sample_fn_args(
    *,
    normalizer: Optional[Any],
    unnormalizer: Optional[Any],
    guided: bool,
    policy: Optional[Any],
    use_neg_grad: bool,
    behavior_policy: Optional[Any],
) -> None:
    assert normalizer is not None and unnormalizer is not None, "Traj chunk diffusion sampling requires normalizer and unnormalizer."
    if not guided: 
        return
    else:
        assert policy is not None, "guided=True requires a target policy."
        _requires_action_score_interface(policy, role="target policy")
        if use_neg_grad:
            assert behavior_policy is not None, "Negative guidance requires a behavior policy."
            _requires_action_score_interface(behavior_policy, role="behavior policy")


def prepare_guidance(
    *,
    model: TrajChunkDiffusionModel,
    model_mean: torch.Tensor, # conditiona chunk denoising mean (B, T, Ds + Da)
    t: torch.Tensor,
    state_dim: int,
    action_dim: int,
    options: GuidanceOptions,
    policy: Optional[Any],
    behavior_policy: Optional[Any],
) -> torch.Tensor:
    """Build an action-only guide, then pad zeros on state channels to match `model_mean`."""
    states = model_mean[..., :state_dim]
    actions = model_mean[..., state_dim:]

    def compute_action_score(policy: Optional[Any], *, role: str) -> torch.Tensor:
        policy = _requires_action_score_interface(policy, role=role)
        score = policy.grad_log_prob(states, actions)
        assert score.shape == actions.shape, (
            f"{role} must return shape {tuple(actions.shape)} from grad_log_prob, "
            f"got {tuple(score.shape)}."
        )
        return score

    def postprocess_action_score(score: torch.Tensor) -> torch.Tensor:
        if options.action_score_postprocess == "none":
            return score
        if options.action_score_postprocess == "l2":
            norm = torch.norm(score, dim=-1, keepdim=True) + 1e-6
            return score / norm
        else:
            return torch.clamp(score, min=-options.clamp_linf, max=options.clamp_linf)

    def scale_action_guide(action_guide: torch.Tensor) -> torch.Tensor:
        if not options.use_adaptive:
            return options.action_score_scale * action_guide
        scale_multiplier = get_schedule_multiplier(
            model.n_timesteps - t[0].item(),
            model.n_timesteps,
            schedule_type="cosine",
        )
        return scale_multiplier * options.action_score_scale * action_guide

    target_score = compute_action_score(policy, role="target policy")
    target_score = postprocess_action_score(target_score)
    if options.use_neg_grad:
        behavior_score = compute_action_score(behavior_policy, role="behavior policy")
        behavior_score = postprocess_action_score(behavior_score)
        action_guide = target_score - (options.action_neg_score_weight * behavior_score)
    else:
        action_guide = target_score

    guide = torch.zeros_like(model_mean) # (B, T, Ds + Da)
    guide[..., state_dim:] = scale_action_guide(action_guide) # Overwrite action scores at (B,T,Ds:)
    return guide


def _finalize_conditional_sample(
    model_mean: torch.Tensor, # (B,T,Ds+Da)
    model_std: torch.Tensor, # (B,T,Ds+Da)
    x: torch.Tensor, # (B,T,Ds+Da)
    t: torch.Tensor, # (B,)
    normalizer: Callable[[Any], Any],
    info: SampleInfo,
) -> tuple[torch.Tensor, SampleInfo]:
    model_mean = normalizer(model_mean) # (B,T,Ds+Da)
    with torch.no_grad():
        noise = torch.randn_like(x)
        noise[t == 0] = 0  # clears noise at diffusion time t=0
    return model_mean + model_std * noise, info


def guided_sample_step(
    model: TrajChunkDiffusionModel,
    x: torch.Tensor,
    t: torch.Tensor,
    state_dim: int,
    action_dim: int,
    guided: bool,
    guidance_hyperparams: Optional[dict[str, Any]],
    *,
    policy: Optional[Any] = None,
    behavior_policy: Optional[Any] = None,
    unnormalizer: Optional[Any] = None,
    normalizer: Optional[Any] = None,
    cond: Optional[torch.Tensor] = None,
    return_info: bool = False,
) -> tuple[torch.Tensor, SampleInfo]:
    """Apply SOPE-style guidance to the reverse-step Gaussian returned by `model`.

    `model.p_mean_variance(x, t, cond)` is expected to return the unguided DDPM
    reverse kernel for the current diffusion step:
    - `model_mean` is the pre-guidance mean of `x_{t-1} | x_t, cond` in the
    model's normalized trajectory space, while
    - `posterior_variance` and
    - `posterior_log_variance` set the Gaussian noise scale used after the mean
    shift. This helper only edits the mean; it reuses the model-provided
    variance unchanged.
    """

    with torch.no_grad():
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, t=t, cond=cond)
        model_std = torch.exp(0.5 * model_log_variance)

    options = GuidanceOptions(**dict(guidance_hyperparams or {}))
    _validate_sample_fn_args(
        normalizer=normalizer,
        unnormalizer=unnormalizer,
        guided=guided,
        policy=policy,
        use_neg_grad=options.use_neg_grad,
        behavior_policy=behavior_policy,
    )

    model_mean = unnormalizer(model_mean)
    info = _init_sample_info(model_mean, state_dim, return_info)
    
    if not guided:
        return _finalize_conditional_sample(model_mean, model_std, x, t, normalizer, info)

    for _ in range(options.num_guidance_iters):
        guide = prepare_guidance(
            model=model,
            model_mean=model_mean,
            t=t,
            state_dim=state_dim,
            action_dim=action_dim,
            options=options,
            policy=policy,
            behavior_policy=behavior_policy,
        )

        model_mean = model_mean + guide
        _update_sample_info(info, guide, state_dim, return_info)

    return _finalize_conditional_sample(model_mean, model_std, x, t, normalizer, info)


def run_p_sample_loop(
    model: TrajChunkDiffusionModel,
    shape: tuple[int, int, int],
    cond: Optional[torch.Tensor],
    *,
    verbose: bool = False,
    return_chain: bool = False,
    sample_fn: Any = guided_sample_step,
    guided: bool = False,
    guidance_hyperparams: Optional[dict[str, Any]] = None,
    return_info: bool = False,
    **sample_kwargs: Any,
) -> tuple[Sample, dict[str, np.ndarray]]:
    device = model.betas.device
    with torch.no_grad():
        batch_size = shape[0]
        x = torch.randn(shape).to(device=device)
        chain = [x] if return_chain else None

    guidance_grads_over_time = []
    model_preds_over_time = []
    progress = diffusion_utils.Progress(model.n_timesteps) if verbose else diffusion_utils.Silent()

    for i in reversed(range(0, model.n_timesteps)):
        t = make_timesteps(batch_size, i, device)
        step_sample_kwargs = dict(sample_kwargs)
        if model.policy is not None:
            step_sample_kwargs["policy"] = model.policy
            step_sample_kwargs["behavior_policy"] = model.behavior_policy
        if model.normalizer is None or model.unnormalizer is None:
            raise ValueError("FiLM diffusion sampling requires normalization helpers.")
        step_sample_kwargs["normalizer"] = model.normalizer
        step_sample_kwargs["unnormalizer"] = model.unnormalizer

        x, info = sample_fn(
            model,
            x,
            t,
            state_dim=model.observation_dim,
            action_dim=model.action_dim,
            guided=guided,
            guidance_hyperparams=guidance_hyperparams,
            return_info=return_info,
            cond=cond,
            **step_sample_kwargs,
        )
        if return_info:
            if info["model_prediction"] is not None:
                model_preds_over_time.append(info["model_prediction"].detach().cpu().numpy().copy())
            if info["guidance"] is not None:
                guidance_grads_over_time.append(info["guidance"].detach().cpu().numpy().copy())
        progress.update({"t": i})
        if return_chain:
            chain.append(x)

    progress.stamp()

    info_payload: dict[str, np.ndarray] = {}
    if return_info:
        info_payload = {
            "model_predictions": (
                np.stack(model_preds_over_time) if model_preds_over_time else np.empty((0,))
            ),
            "guidance": (
                np.stack(guidance_grads_over_time) if guidance_grads_over_time else np.empty((0,))
            ),
        }

    
    chain = torch.stack(chain, dim=1) if return_chain else None # (B, K + 1, T, Ds + Da)
    x = x.detach()
    return Sample(x, chain), info_payload
