from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure `opelab` is importable when using third_party/sope in-place.
_SOPE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "sope"
if str(_SOPE_ROOT) not in sys.path:
    sys.path.append(str(_SOPE_ROOT))
_ROBOMIMIC_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "robomimic"
if str(_ROBOMIMIC_ROOT) not in sys.path:
    sys.path.append(str(_ROBOMIMIC_ROOT))

from third_party.robomimic.robomimic.models.diffusion_policy_nets import (  # type: ignore
    ConditionalUnet1D,
)
from third_party.sope.opelab.core.baselines.diffusion.diffusion import (  # type: ignore
    GaussianDiffusion,
    Sample,
    gradlog,
    gradlog_diffusion,
    get_schedule_multiplier,
    make_timesteps,
)
import third_party.sope.opelab.core.baselines.diffusion.utils as diffusion_utils  # type: ignore
from src.sope_diffuser import NormalizationStats, make_normalizers


def _film_sample_fn(
    model: "FilmGaussianDiffusion",
    x: torch.Tensor,
    t: torch.Tensor,
    state_dim: int,
    action_dim: int,
    guided: bool,
    guidance_hyperparams: Optional[dict[str, Any]],
    *,
    gmode: bool = False,
    policy: Optional[Any] = None,
    behavior_policy: Optional[Any] = None,
    transform: Optional[Any] = None,
    unnormalizer: Optional[Any] = None,
    normalizer: Optional[Any] = None,
    cond: Optional[torch.Tensor] = None,
    return_info: bool = False,
    reverse: bool = False,
) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
    """Sampling helper that keeps SOPE guidance math but skips in-paint reconditioning."""

    del transform

    with torch.no_grad():
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, t=t, cond=cond)
        model_std = torch.exp(0.5 * model_log_variance)

    if not guidance_hyperparams:
        guidance_hyperparams = {}

    k = guidance_hyperparams.get("k_guide", 2)
    normalize_grad = guidance_hyperparams.get("normalize_grad", True)
    use_adaptive = guidance_hyperparams.get("use_adaptive", True)
    use_neg_grad = guidance_hyperparams.get("use_neg_grad", True)
    action_scale = guidance_hyperparams.get("action_scale", 0.2)
    clamp = guidance_hyperparams.get("clamp", False)
    l_inf = guidance_hyperparams.get("l_inf", 1.0)
    ratio = guidance_hyperparams.get("ratio", 1.0)
    normalize_v = not clamp and normalize_grad

    info: list[Optional[torch.Tensor]] = [None, None]
    if return_info:
        info[0] = model_mean[..., state_dim:].detach().clone()

    if normalizer is None or unnormalizer is None:
        raise ValueError("FiLM diffusion sampling requires normalizer and unnormalizer.")

    model_mean = unnormalizer(model_mean)
    if guided:
        if policy is None:
            raise ValueError("guided=True requires a target policy.")
        if use_neg_grad and behavior_policy is None:
            raise ValueError("Negative guidance requires a behavior policy.")

        for _ in range(k):
            if policy.__class__.__name__ == "DiffusionPolicy":
                gradient = gradlog_diffusion(
                    policy, model_mean, state_dim, action_dim, normalize=normalize_v
                )
            else:
                gradient = gradlog(
                    policy,
                    model_mean,
                    state_dim,
                    action_dim,
                    normalize=normalize_v,
                    gmode=gmode,
                    verbose=False,
                    reverse=reverse,
                )

            neg_grad = 0
            if use_neg_grad:
                neg_grad = gradlog(
                    behavior_policy,
                    model_mean,
                    state_dim,
                    action_dim,
                    normalize=normalize_v,
                    gmode=gmode,
                    verbose=False,
                    reverse=reverse,
                )

            if use_neg_grad:
                if normalize_grad:
                    if clamp:
                        gradient = torch.clamp(gradient, min=-l_inf, max=l_inf)
                        neg_grad = torch.clamp(ratio * neg_grad, min=-l_inf, max=l_inf)
                    else:
                        neg_grad = ratio * neg_grad
                guide = gradient - neg_grad
            else:
                if normalize_grad and clamp:
                    gradient = torch.clamp(gradient, min=-l_inf, max=l_inf)
                guide = gradient

            if use_adaptive:
                scale_multiplier = get_schedule_multiplier(
                    model.n_timesteps - t[0].item(),
                    model.n_timesteps,
                    schedule_type="cosine",
                )
                guide = scale_multiplier * action_scale * guide
            else:
                guide = action_scale * guide

            model_mean = model_mean + guide
            model_mean = unnormalizer(normalizer(model_mean))
            if return_info:
                info[1] = guide[..., state_dim:].detach().clone()

    model_mean = normalizer(model_mean)
    with torch.no_grad():
        noise = torch.randn_like(x)
        noise[t == 0] = 0
    return model_mean + model_std * noise, info


class FilmConditionedBackbone(nn.Module):
    """Adapter from the SOPE diffusion contract to robomimic's ConditionalUnet1D."""

    def __init__(
        self,
        *,
        horizon: int,
        transition_dim: int,
        global_cond_dim: int,
        down_dims: Tuple[int, ...],
    ) -> None:
        """Construct the robomimic UNet adapter for the SOPE diffusion contract."""
        super().__init__()
        del horizon  # robomimic's UNet is horizon-agnostic at construction time
        self.global_cond_dim = int(global_cond_dim)
        self.backbone = ConditionalUnet1D(
            input_dim=transition_dim,
            global_cond_dim=self.global_cond_dim,
            down_dims=list(down_dims),
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the robomimic denoiser with an optional flattened FiLM context."""
        if self.global_cond_dim == 0:
            global_cond = None
        elif global_cond is None:
            raise ValueError(
                f"Expected FiLM conditioning tensor with dim {self.global_cond_dim}, got None."
            )
        return self.backbone(x, time, global_cond=global_cond)


class FilmGaussianDiffusion(GaussianDiffusion):
    """SOPE GaussianDiffusion variant that threads FiLM context into the denoiser.

    Inheritance boundary:
    - inherited unchanged via explicit `super()` wrappers:
      `__init__`, `get_loss_weights`, `predict_start_from_noise`,
      `q_posterior`, `q_sample`, `loss`, and `forward`
    - overridden for FiLM conditioning:
      `p_mean_variance`, `p_sample_loop`, `conditional_sample`, and `p_losses`
    """

    def __init__(
        self,
        model: nn.Module,
        horizon: int,
        observation_dim: int,
        action_dim: int,
        n_timesteps: int = 1000,
        loss_type: str = "l2",
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        action_weight: float = 1.0,
        loss_discount: float = 1.0,
        loss_weights: Optional[dict[int, float]] = None,
        policy: Optional[Any] = None,
        behavior_policy: Optional[Any] = None,
        normalizer: Optional[Any] = None,
        unnormalizer: Optional[Any] = None,
        transform: Optional[Any] = None,
        guided: bool = True,
        gmode: bool = False,
        reverse: bool = False,
    ) -> None:
        """Inherit SOPE's DDPM buffers and core attrs with an explicit constructor."""
        super().__init__(
            model=model,
            horizon=horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=n_timesteps,
            loss_type=loss_type,
            clip_denoised=clip_denoised,
            predict_epsilon=predict_epsilon,
            action_weight=action_weight,
            loss_discount=loss_discount,
            loss_weights=loss_weights,
            policy=policy,
            behavior_policy=behavior_policy,
            normalizer=normalizer,
            unnormalizer=unnormalizer,
            transform=transform,
            guided=guided,
            gmode=gmode,
            reverse=reverse,
        )

    def get_loss_weights(
        self,
        action_weight: float,
        discount: float,
        weights_dict: Optional[dict[int, float]],
    ) -> torch.Tensor:
        """Reuse SOPE's timestep and per-dimension loss weighting unchanged."""
        return super().get_loss_weights(action_weight, discount, weights_dict)

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Reuse SOPE's epsilon-vs-x0 parameterization logic unchanged."""
        return super().predict_start_from_noise(x_t, t, noise)

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse SOPE's closed-form posterior coefficients unchanged."""
        return super().q_posterior(x_start, x_t, t)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reuse SOPE's forward noising process unchanged."""
        return super().q_sample(x_start, t, noise)

    def _build_batch_rmse_info(
        self,
        x_start: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build normalized-space chunk RMSE diagnostics for one batch."""
        err = x_pred - x_start
        state_err = err[..., : self.observation_dim]
        action_err = err[..., self.observation_dim :]
        info = {
            "chunk_rmse_transition": torch.sqrt(torch.mean(err.square())),
            "chunk_rmse_state": torch.sqrt(torch.mean(state_err.square())),
            "chunk_rmse_action": torch.sqrt(torch.mean(action_err.square())),
        }
        if self.observation_dim == 3:
            eef_err = state_err[..., :3]
            info["chunk_rmse_eef_pos"] = torch.sqrt(torch.mean(eef_err.square()))
        elif self.observation_dim >= 13:
            eef_err = state_err[..., 10:13]
            info["chunk_rmse_eef_pos"] = torch.sqrt(torch.mean(eef_err.square()))
        return info

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reverse-process moments using the FiLM-conditioned denoiser."""
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, cond))
        if self.clip_denoised:
            x_recon.clamp_(-5.0, 5.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon,
            x_t=x,
            t=t,
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample_loop(
        self,
        shape: tuple[int, int, int],
        cond: Optional[torch.Tensor],
        *,
        verbose: bool = True,
        return_chain: bool = False,
        sample_fn: Any = _film_sample_fn,
        guided: bool = False,
        guidance_hyperparams: Optional[dict[str, Any]] = None,
        return_info: bool = False,
        **sample_kwargs: Any,
    ) -> Any:
        """Run the reverse DDPM loop with FiLM context instead of in-paint reconditioning."""
        device = self.betas.device
        with torch.no_grad():
            batch_size = shape[0]
            x = torch.randn(shape).to(device=device)
            chain = [x] if return_chain else None

        guidance_grads_over_time = []
        model_preds_over_time = []
        progress = diffusion_utils.Progress(self.n_timesteps) if verbose else diffusion_utils.Silent()

        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            if self.policy is not None:
                sample_kwargs["policy"] = self.policy
                sample_kwargs["behavior_policy"] = self.behavior_policy
            if self.transform is not None:
                sample_kwargs["transform"] = self.transform
            if self.normalizer is not None:
                sample_kwargs["normalizer"] = self.normalizer
                sample_kwargs["unnormalizer"] = self.unnormalizer
            else:
                raise ValueError("FiLM diffusion sampling requires normalization helpers.")

            x, info = sample_fn(
                self,
                x,
                t,
                state_dim=self.observation_dim,
                action_dim=self.action_dim,
                guided=guided,
                guidance_hyperparams=guidance_hyperparams,
                return_info=return_info,
                gmode=self.gmode,
                reverse=self.reverse,
                cond=cond,
                **sample_kwargs,
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

    def conditional_sample(
        self,
        shape: tuple[int, int, int],
        cond: Optional[torch.Tensor],
        *,
        verbose: bool = True,
        return_chain: bool = False,
        action_scale: float = 0.2,
        state_scale: float = 0.01,
        guided: bool = False,
        use_adaptive: bool = True,
        use_neg_grad: bool = True,
        neg_grad_scale: float = 0.1,
        normalize_grad: bool = True,
        k_guide: int = 2,
        use_action_grad_only: bool = True,
        return_info: bool = False,
        clamp: bool = False,
        l_inf: float = 1.0,
        ratio: float = 1.0,
        **sample_kwargs: Any,
    ) -> Any:
        """Preserve SOPE's public sampling interface for the FiLM-conditioned sampler."""
        del state_scale, neg_grad_scale, use_action_grad_only
        guidance_hyperparams = {
            "action_scale": action_scale,
            "use_adaptive": use_adaptive,
            "use_neg_grad": use_neg_grad,
            "normalize_grad": normalize_grad,
            "k_guide": k_guide,
            "l_inf": l_inf,
            "ratio": ratio,
            "clamp": clamp,
        }
        batch_size = shape[0]
        horizon = shape[1]
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(
            shape,
            cond,
            verbose=verbose,
            return_chain=return_chain,
            guided=guided,
            guidance_hyperparams=guidance_hyperparams,
            return_info=return_info,
            **sample_kwargs,
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor],
        *,
        compute_batch_rmse: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train on noisy future chunks while supplying prefix states as FiLM context."""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.model(x_noisy, t, cond)
        assert noise.shape == model_output.shape
        if self.predict_epsilon:
            loss, info = self.loss_fn(model_output, noise)   # FIXME: Double check: is this consistent with sope's default loss fn?
        else:
            loss, info = self.loss_fn(model_output, x_start)

        if not compute_batch_rmse:
            return loss, info

        with torch.no_grad():
            x_pred = (
                self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
                if self.predict_epsilon
                else model_output
            )
            rmse_info = self._build_batch_rmse_info(x_start, x_pred)
        return loss, {**info, **rmse_info}

    def loss(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor],
        *,
        compute_batch_rmse: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Reuse SOPE's timestep sampling while dispatching into FiLM-aware `p_losses`."""
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, t, cond, compute_batch_rmse=compute_batch_rmse)

    def forward(self, cond: Optional[torch.Tensor], *args: Any, **kwargs: Any) -> Any:
        """Preserve the base module contract by routing calls to conditional sampling."""
        return super().forward(cond, *args, **kwargs)


@dataclass(frozen=True)
class FilmDiffusionConfig:
    """Configuration for FiLM-conditioned trajectory chunk diffusion."""

    chunk_horizon: int = 8
    frame_stack: int = 2
    state_dim: int = 19
    action_dim: int = 7
    diffusion_steps: int = 512

    # robomimic ConditionalUnet1D backbone
    dim_mults: Tuple[int, ...] = (1, 2)
    attention: bool = True

    # diffusion loss
    loss_type: str = "l2"
    action_weight: float = 5.0
    loss_discount: float = 1.0
    predict_epsilon: bool = True

    # optimization
    lr: float = 3e-4
    weight_decay: float = 0.0

    # guidance (optional)
    guided: bool = False
    guidance_hyperparams: Optional[Dict[str, Any]] = None
    diffuser_eef_pos_only: bool = False
    conditioning_mode: Literal["prefix_states", "none"] = "prefix_states"

    conditioning_style: Literal["film"] = "film"
    backbone_base_dim: int = 256

    @property
    def total_chunk_horizon(self) -> int:
        """Return the denoised horizon, which is always the future chunk length for FiLM."""
        return self.chunk_horizon


    class FilmDiffuser:
    """FiLM-conditioned chunk diffusion wrapper using robomimic's ConditionalUnet1D."""

    def __init__(
        self,
        cfg: FilmDiffusionConfig,
        normalization_stats: Optional[NormalizationStats] = None,
        device: str = "cuda",
        policy: Optional[Any] = None,
        behavior_policy: Optional[Any] = None,
    ) -> None:
        """Build the FiLM-conditioned backbone, DDPM wrapper, and normalization helpers."""
        self.cfg = cfg
        self.device = torch.device(device)
        self.state_dim = int(cfg.state_dim)
        self.action_dim = int(cfg.action_dim)
        self.transition_dim = self.state_dim + self.action_dim

        self.normalization_stats = normalization_stats
        self.normalizer, self.unnormalizer = make_normalizers(normalization_stats)

        global_cond_dim = 0
        if cfg.conditioning_mode != "none":
            global_cond_dim = int(cfg.frame_stack * cfg.state_dim)
        down_dims = tuple(int(cfg.backbone_base_dim * mult) for mult in cfg.dim_mults)

        self.model = FilmConditionedBackbone(
            horizon=cfg.total_chunk_horizon,
            transition_dim=self.transition_dim,
            global_cond_dim=global_cond_dim,
            down_dims=down_dims,
        ).to(self.device)

        self.diffusion = FilmGaussianDiffusion(
            model=self.model,
            horizon=cfg.total_chunk_horizon,
            observation_dim=self.state_dim,
            action_dim=self.action_dim,
            n_timesteps=cfg.diffusion_steps,
            normalizer=self.normalizer,
            unnormalizer=self.unnormalizer,
            action_weight=cfg.action_weight,
            loss_discount=cfg.loss_discount,
            loss_type=cfg.loss_type,
            predict_epsilon=cfg.predict_epsilon,
        ).to(self.device)
        self.diffusion.policy = policy
        self.diffusion.behavior_policy = behavior_policy
        self._configure_diffusion_loss_targets()

    def _resolve_eef_pos_slice(self) -> slice:
        """Return the low-dim robomimic slice corresponding to `robot0_eef_pos`."""
        if self.state_dim == 3:
            return slice(0, 3)
        start = 10
        stop = 13
        if self.state_dim < stop:
            raise ValueError(
                f"robot0_eef_pos slice [10:13] requires state_dim >= 13, got {self.state_dim}."
            )
        return slice(start, stop)

    def _configure_diffusion_loss_targets(self) -> None:
        """Restrict the loss to EEF-position coordinates when the debug flag is enabled."""
        if not self.cfg.diffuser_eef_pos_only:
            return
        weights = self.diffusion.loss_fn.weights
        base_weights = weights.clone()
        weights.zero_()
        eef_slice = self._resolve_eef_pos_slice()
        weights[:, eef_slice] = base_weights[:, eef_slice]

    def _build_training_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the future transition chunk denoised by the FiLM model."""
        future_actions = batch["actions_to"]
        if self.cfg.conditioning_mode == "none":
            future_actions = torch.zeros_like(future_actions)
        return torch.cat([batch["states_to"][:, :-1, :], future_actions], dim=-1)

    def make_optimizer(self) -> torch.optim.Optimizer:
        """Construct the default optimizer for the FiLM diffusion model."""
        return torch.optim.Adam(
            self.diffusion.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def make_cond(self, batch: dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Flatten prefix states into the FiLM conditioning vector expected by the denoiser."""
        if self.cfg.conditioning_mode == "none":
            return None
        return batch["states_from"].reshape(batch["states_from"].shape[0], -1)

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor] = None,
        *,
        compute_batch_rmse: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the FiLM-conditioned diffusion loss for one batch of chunks."""
        del cond
        x = self._build_training_target(batch)
        film_cond = self.make_cond(batch)
        loss, info = self.diffusion.loss(
            x,
            film_cond,
            compute_batch_rmse=compute_batch_rmse,
        )
        if isinstance(info, dict) and "a0_loss" in info:
            a0_loss = info["a0_loss"]
            if torch.is_tensor(a0_loss) and not torch.isfinite(a0_loss):
                info = dict(info)
                info["a0_loss"] = torch.tensor(float("nan"), device=a0_loss.device)
        return loss, info

    def sample(
        self,
        num_samples: int,
        cond: Optional[torch.Tensor] = None,
        return_chain: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Sample future transition chunks from the FiLM-conditioned diffusion model."""
        shape = (num_samples, self.cfg.total_chunk_horizon, self.transition_dim)
        return self.diffusion.conditional_sample(
            shape,
            cond,
            guided=self.cfg.guided,
            return_chain=return_chain,
            **kwargs,
        )


def cross_validate_configs(cfg_dataset: Any, cfg_diffusion: FilmDiffusionConfig) -> None:
    """Validate that dataset and FiLM diffusion configs agree on dimensions and conditioning."""
    if not isinstance(cfg_diffusion, FilmDiffusionConfig):
        raise TypeError(
            f"cfg_diffusion must be a FilmDiffusionConfig, got {type(cfg_diffusion)}"
        )

    if not hasattr(cfg_dataset, "frame_stack"):
        raise TypeError(
            "cfg_dataset must expose frame_stack, state_dim or latents_dim, and action_dim. "
            f"Got {type(cfg_dataset)}."
        )

    dataset_state_dim = getattr(cfg_dataset, "latents_dim", None)
    if dataset_state_dim is None:
        dataset_state_dim = getattr(cfg_dataset, "state_dim", None)
    if dataset_state_dim is None:
        raise TypeError(
            "cfg_dataset must define either latents_dim or state_dim for diffusion shape validation."
        )

    dataset_action_dim = getattr(cfg_dataset, "action_dim", None)
    if dataset_action_dim is None:
        raise TypeError("cfg_dataset must define action_dim for diffusion shape validation.")

    dataset_disable_conditioning = bool(getattr(cfg_dataset, "disable_conditioning", False))
    if dataset_disable_conditioning and cfg_diffusion.conditioning_mode != "none":
        raise ValueError(
            "Dataset disables conditioning, but diffusion config is not set to conditioning_mode='none'."
        )
    if (not dataset_disable_conditioning) and cfg_diffusion.conditioning_mode == "none":
        raise ValueError(
            "conditioning_mode='none' requires cfg_dataset.disable_conditioning=True so the dataset contract matches sampling/training."
        )

    if int(dataset_state_dim) != cfg_diffusion.state_dim:
        raise ValueError(
            "Config mismatch: dataset state_dim must equal FilmDiffusionConfig.state_dim "
            f"({dataset_state_dim} != {cfg_diffusion.state_dim})."
        )
    if int(dataset_action_dim) != cfg_diffusion.action_dim:
        raise ValueError(
            "Config mismatch: dataset action_dim must equal FilmDiffusionConfig.action_dim "
            f"({dataset_action_dim} != {cfg_diffusion.action_dim})."
        )

    total_horizon = int(cfg_diffusion.total_chunk_horizon)
    required_div = 2 ** (len(cfg_diffusion.dim_mults) - 1)
    if total_horizon % required_div != 0:
        raise ValueError(
            f"Invalid dim_mults for total_horizon={total_horizon}: "
            f"len(dim_mults)={len(cfg_diffusion.dim_mults)} requires divisibility by {required_div}. "
            "Adjust dim_mults or change chunk_horizon."
        )
