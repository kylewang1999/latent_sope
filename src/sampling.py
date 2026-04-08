from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import numpy as np
import torch

from third_party.sope.opelab.core.baselines.diffusion.diffusion import (
    Sample,
    get_schedule_multiplier,
    gradlog_diffusion,
    make_timesteps,
)
import third_party.sope.opelab.core.baselines.diffusion.utils as diffusion_utils


class DiffusionScorePolicy(Protocol):
    def grad_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor: ...


class FilmSamplingModel(Protocol):
    betas: torch.Tensor
    n_timesteps: int
    observation_dim: int
    action_dim: int
    policy: Optional[DiffusionScorePolicy]
    behavior_policy: Optional[DiffusionScorePolicy]
    normalizer: Optional[Callable[[Any], Any]]
    unnormalizer: Optional[Callable[[Any], Any]]

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class FilmGuidanceOptions:
    k: int  # Upstream `k_guide`: repeats the guide update `k` times per reverse step.
    normalize_grad: bool  # Upstream switch for the "ratio/clamp" combine path; also feeds the derived `normalize_v` flag when `clamp` is off.
    use_adaptive: bool  # Upstream switch between constant `action_scale` and cosine-scheduled guidance scaling.
    use_neg_grad: bool  # Upstream switch for subtracting the behavior-policy gradient from the target-policy gradient.
    action_scale: float  # Upstream multiplier applied after combining gradients into the final guide.
    clamp: bool  # Upstream switch for elementwise clipping to `[-l_inf, l_inf]`; this also disables `normalize_v`.
    l_inf: float  # Upstream clip bound used when `clamp` is enabled.
    ratio: float  # Upstream weight on the negative gradient before subtraction in the `use_neg_grad` path.
    normalize_v: bool  # Derived upstream flag passed to `gradlog[_diffusion]` to L2-normalize per-timestep action gradients.


def resolve_film_guidance_options(
    guidance_hyperparams: Optional[dict[str, Any]],
) -> FilmGuidanceOptions:
    if not guidance_hyperparams:
        guidance_hyperparams = {}

    normalize_grad = guidance_hyperparams.get("normalize_grad", True)
    clamp = guidance_hyperparams.get("clamp", False)
    return FilmGuidanceOptions(
        k=guidance_hyperparams.get("k_guide", 2),
        normalize_grad=normalize_grad,
        use_adaptive=guidance_hyperparams.get("use_adaptive", True),
        use_neg_grad=guidance_hyperparams.get("use_neg_grad", True),
        action_scale=guidance_hyperparams.get("action_scale", 0.2),
        clamp=clamp,
        l_inf=guidance_hyperparams.get("l_inf", 1.0),
        ratio=guidance_hyperparams.get("ratio", 1.0),
        normalize_v=not clamp and normalize_grad,
    )


def _init_film_sample_info(
    model_mean: torch.Tensor,
    state_dim: int,
    return_info: bool,
) -> list[Optional[torch.Tensor]]:
    info: list[Optional[torch.Tensor]] = [None, None]
    if return_info:
        info[0] = model_mean[..., state_dim:].detach().clone()
    return info


def _require_diffusion_policy_interface(
    policy: Any,
    *,
    role: str,
) -> DiffusionScorePolicy:
    grad_log_prob = getattr(policy, "grad_log_prob", None)
    if not callable(grad_log_prob):
        raise TypeError(
            f"{role} must expose grad_log_prob(state, action) for diffusion-only guidance."
        )
    return policy


def _validate_film_sample_inputs(
    *,
    normalizer: Optional[Any],
    unnormalizer: Optional[Any],
    guided: bool,
    policy: Optional[Any],
    use_neg_grad: bool,
    behavior_policy: Optional[Any],
) -> None:
    if normalizer is None or unnormalizer is None:
        raise ValueError("FiLM diffusion sampling requires normalizer and unnormalizer.")
    if not guided:
        return
    if policy is None:
        raise ValueError("guided=True requires a target policy.")
    _require_diffusion_policy_interface(policy, role="target policy")
    if use_neg_grad:
        if behavior_policy is None:
            raise ValueError("Negative guidance requires a behavior policy.")
        _require_diffusion_policy_interface(behavior_policy, role="behavior policy")


def _compute_film_policy_gradient(
    *,
    policy: Any,
    model_mean: torch.Tensor,
    state_dim: int,
    action_dim: int,
    normalize: bool,
    role: str,
) -> torch.Tensor:
    policy = _require_diffusion_policy_interface(policy, role=role)
    return gradlog_diffusion(
        policy,
        model_mean,
        state_dim,
        action_dim,
        normalize=normalize,
    )


def _compute_film_negative_gradient(
    *,
    behavior_policy: Optional[Any],
    model_mean: torch.Tensor,
    state_dim: int,
    action_dim: int,
    use_neg_grad: bool,
    normalize: bool,
) -> torch.Tensor | int:
    if not use_neg_grad:
        return 0
    if behavior_policy is None:
        raise ValueError("Negative guidance requires a behavior policy.")
    return _compute_film_policy_gradient(
        policy=behavior_policy,
        model_mean=model_mean,
        state_dim=state_dim,
        action_dim=action_dim,
        normalize=normalize,
        role="behavior policy",
    )


def _combine_film_guide(
    gradient: torch.Tensor,
    neg_grad: torch.Tensor | int,
    options: FilmGuidanceOptions,
) -> torch.Tensor:
    if not options.use_neg_grad:
        if options.normalize_grad and options.clamp:
            gradient = torch.clamp(gradient, min=-options.l_inf, max=options.l_inf)
        return gradient

    if not options.normalize_grad:
        return gradient - neg_grad

    if not options.clamp:
        return gradient - (options.ratio * neg_grad)

    gradient = torch.clamp(gradient, min=-options.l_inf, max=options.l_inf)
    neg_grad = torch.clamp(
        options.ratio * neg_grad,
        min=-options.l_inf,
        max=options.l_inf,
    )
    return gradient - neg_grad


def _scale_film_guide(
    guide: torch.Tensor,
    *,
    model: FilmSamplingModel,
    t: torch.Tensor,
    options: FilmGuidanceOptions,
) -> torch.Tensor:
    if not options.use_adaptive:
        return options.action_scale * guide

    scale_multiplier = get_schedule_multiplier(
        model.n_timesteps - t[0].item(),
        model.n_timesteps,
        schedule_type="cosine",
    )
    return scale_multiplier * options.action_scale * guide


def _update_film_sample_info(
    info: list[Optional[torch.Tensor]],
    guide: torch.Tensor,
    state_dim: int,
    return_info: bool,
) -> None:
    if return_info:
        info[1] = guide[..., state_dim:].detach().clone()


def _finalize_film_sample(
    model_mean: torch.Tensor,
    model_std: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    normalizer: Callable[[Any], Any],
    info: list[Optional[torch.Tensor]],
) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
    model_mean = normalizer(model_mean)
    with torch.no_grad():
        noise = torch.randn_like(x)
        noise[t == 0] = 0
    return model_mean + model_std * noise, info


def guided_sampling(
    model: FilmSamplingModel,
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
) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
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

    options = resolve_film_guidance_options(guidance_hyperparams)
    info = _init_film_sample_info(model_mean, state_dim, return_info)
    _validate_film_sample_inputs(
        normalizer=normalizer,
        unnormalizer=unnormalizer,
        guided=guided,
        policy=policy,
        use_neg_grad=options.use_neg_grad,
        behavior_policy=behavior_policy,
    )

    model_mean = unnormalizer(model_mean)
    if not guided:
        return _finalize_film_sample(model_mean, model_std, x, t, normalizer, info)

    for _ in range(options.k):
        gradient = _compute_film_policy_gradient(
            policy=policy,
            model_mean=model_mean,
            state_dim=state_dim,
            action_dim=action_dim,
            normalize=options.normalize_v,
            role="target policy",
        )
        neg_grad = _compute_film_negative_gradient(
            behavior_policy=behavior_policy,
            model_mean=model_mean,
            state_dim=state_dim,
            action_dim=action_dim,
            use_neg_grad=options.use_neg_grad,
            normalize=options.normalize_v,
        )
        guide = _combine_film_guide(gradient, neg_grad, options)
        guide = _scale_film_guide(guide, model=model, t=t, options=options)

        model_mean = model_mean + guide
        model_mean = unnormalizer(normalizer(model_mean))
        _update_film_sample_info(info, guide, state_dim, return_info)

    return _finalize_film_sample(model_mean, model_std, x, t, normalizer, info)


def run_film_p_sample_loop(
    model: FilmSamplingModel,
    shape: tuple[int, int, int],
    cond: Optional[torch.Tensor],
    *,
    verbose: bool = True,
    return_chain: bool = False,
    sample_fn: Any = guided_sampling,
    guided: bool = False,
    guidance_hyperparams: Optional[dict[str, Any]] = None,
    return_info: bool = False,
    **sample_kwargs: Any,
) -> Any:
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
            if info[0] is not None:
                model_preds_over_time.append(info[0].detach().cpu().numpy().copy())
            if info[1] is not None:
                guidance_grads_over_time.append(info[1].detach().cpu().numpy().copy())
        progress.update({"t": i})
        if return_chain:
            chain.append(x)

    progress.stamp()

    if return_info:
        info_payload = {
            "model_predictions": (
                np.stack(model_preds_over_time) if model_preds_over_time else np.empty((0,))
            ),
            "guidance": (
                np.stack(guidance_grads_over_time) if guidance_grads_over_time else np.empty((0,))
            ),
        }

    if return_chain:
        chain = torch.stack(chain, dim=1)
    x = x.detach()
    if return_info:
        return Sample(x, None, chain), info_payload
    return Sample(x, None, chain)
