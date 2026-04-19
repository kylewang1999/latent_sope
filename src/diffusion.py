from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Literal, Optional, Tuple

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

from third_party.robomimic.robomimic.models.diffusion_policy_nets import (
    ConditionalUnet1D,
)
from third_party.sope.opelab.core.baselines.diffusion.diffusion import (
    GaussianDiffusion,
)
from src.sampling import guided_sample_step, run_p_sample_loop


@dataclass(frozen=True)
class RewardPredictorConfig:
    """Configuration for a SOPE-style transition reward regressor."""

    state_dim: int = 19
    action_dim: int = 7
    hidden_dims: Tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss_type: Literal["mse"] = "mse"


class RewardPredictor(nn.Module):
    """Standalone SOPE-style reward predictor trained on immediate transformed reward."""

    def __init__(
        self,
        cfg: RewardPredictorConfig,
        *,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.state_dim = int(cfg.state_dim)
        self.action_dim = int(cfg.action_dim)
        layers: list[nn.Module] = []
        prev_dim = self.state_dim + self.action_dim
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers).to(self.device)

    def _prepare_inputs(
        self,
        states: torch.Tensor | np.ndarray,
        actions: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if states_t.shape[:-1] != actions_t.shape[:-1]:
            raise ValueError(
                "states and actions must share the same leading dimensions; "
                f"got {states_t.shape} vs {actions_t.shape}."
            )
        if states_t.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, got last dim {states_t.shape[-1]}."
            )
        if actions_t.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got last dim {actions_t.shape[-1]}."
            )
        leading_shape = tuple(states_t.shape[:-1])
        flat_inputs = torch.cat(
            [
                states_t.reshape(-1, self.state_dim),
                actions_t.reshape(-1, self.action_dim),
            ],
            dim=-1,
        )
        return flat_inputs, leading_shape

    def forward(
        self,
        states: torch.Tensor | np.ndarray,
        actions: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        flat_inputs, leading_shape = self._prepare_inputs(states, actions)
        preds = self.model(flat_inputs).squeeze(-1)
        return preds.reshape(leading_shape)

    def predict(
        self,
        states: torch.Tensor | np.ndarray,
        actions: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            preds = self.forward(states, actions)
        if was_training:
            self.train()
        return preds

    def loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # states_to includes the next-state sequence with length T + 1, so the
        # per-transition reward targets align with the first T states and T actions.
        # states: (B, T, state_dim)
        # actions: (B, T, action_dim)
        # targets: (B, T)
        # preds: (B, T)
        states = batch["states_to"][:, :-1, :]
        actions = batch["actions_to"]
        targets = batch["rewards_to"]
        preds = self.forward(states, actions)
        if preds.shape != targets.shape:
            raise ValueError(
                f"Reward predictor outputs must align with targets; got {preds.shape} vs {targets.shape}."
            )

        if self.cfg.loss_type != "mse":
            raise ValueError(f"Unsupported reward loss_type={self.cfg.loss_type!r}.")
        # Reduce uniformly over the full batch-time grid: mean over B * T elements.
        loss = torch.mean((preds - targets) ** 2)

        baseline_targets = targets.reshape(-1)
        baseline_zero_mse = torch.mean(baseline_targets ** 2)
        info = {
            "reward_mse": loss.detach(),
            "reward_pred_mean": preds.detach().mean(),
            "reward_target_mean": targets.detach().mean(),
            "reward_baseline_zero_mse": baseline_zero_mse.detach(),
        }
        return loss, info


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


def make_normalizers(
    stats: Optional[NormalizationStats],
) -> Tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    if stats is None:
        return (lambda x: x), (lambda x: x)

    mean = stats.mean
    std = stats.std

    def _norm(x: Any) -> Any:
        if torch.is_tensor(x):
            mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
            return (x - mean_t) / std_t
        return (x - mean) / std

    def _unnorm(x: Any) -> Any:
        if torch.is_tensor(x):
            mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
            return x * std_t + mean_t
        return x * std + mean

    return _norm, _unnorm


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
        guided: bool = True,
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
            guided=guided,
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
        sample_fn: Any = guided_sample_step,
        guided: bool = False,
        guidance_hyperparams: Optional[dict[str, Any]] = None,
        return_info: bool = False,
        **sample_kwargs: Any,
    ) -> Any:
        """Run the reverse DDPM loop with FiLM context instead of in-paint reconditioning."""
        sample, info = run_p_sample_loop(
            self,
            shape,
            cond,
            verbose=verbose,
            return_chain=return_chain,
            sample_fn=sample_fn,
            guided=guided,
            guidance_hyperparams=guidance_hyperparams,
            return_info=return_info,
            **sample_kwargs,
        )
        if return_info:
            return sample, info
        return sample

    def conditional_sample(
        self,
        shape: tuple[int, int, int],
        cond: Optional[torch.Tensor],
        *,
        verbose: bool = True,
        return_chain: bool = False,
        action_score_scale: float = 0.2,
        guided: bool = False,
        use_adaptive: bool = True,
        use_neg_grad: bool = True,
        action_score_postprocess: Literal["none", "l2", "clamp"] = "l2",
        num_guidance_iters: int = 2,
        return_info: bool = False,
        clamp_linf: float = 1.0,
        action_neg_score_weight: float = 1.0,
        **sample_kwargs: Any,
    ) -> Any:
        """Expose the local FiLM-guidance API for chunk sampling."""
        guidance_hyperparams = {
            "action_score_scale": action_score_scale,
            "use_adaptive": use_adaptive,
            "use_neg_grad": use_neg_grad,
            "action_score_postprocess": action_score_postprocess,
            "num_guidance_iters": num_guidance_iters,
            "clamp_linf": clamp_linf,
            "action_neg_score_weight": action_neg_score_weight,
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
        # FIXME: Might no longer be compatible. Is this actually used in smaple() and generate_full_trajectory()?
        return super().forward(cond, *args, **kwargs)

@dataclass(frozen=True)
class SopeDiffusionConfig:
    """Configuration for the canonical SOPE chunk diffusion model."""

    chunk_horizon: int = 14
    trajectory_horizon: int = 60
    ope_gamma: float = 0.99
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

    conditioning_style: Literal["film"] = "film" # deprecated; now film conditoining is canonical
    backbone_base_dim: int = 256

    @property
    def total_chunk_horizon(self) -> int:
        """Return the denoised horizon, which is always the future chunk length for FiLM."""
        return self.chunk_horizon


class SopeDiffuser:
    """Canonical SOPE chunk diffusion wrapper using robomimic's ConditionalUnet1D."""

    def __init__(
        self,
        cfg: SopeDiffusionConfig,
        normalization_stats: Optional[NormalizationStats] = None,
        device: str = "cuda",
        policy: Optional[Any] = None,
        behavior_policy: Optional[Any] = None,
    ) -> None:
        """Build the canonical backbone, DDPM wrapper, and normalization helpers."""
        self.cfg = cfg
        self.device = torch.device(device)
        self.state_dim = int(cfg.state_dim)
        self.action_dim = int(cfg.action_dim)
        self.transition_dim = self.state_dim + self.action_dim
        self._validate_policy_obs_horizon(policy, role="target policy")
        self._validate_policy_obs_horizon(behavior_policy, role="behavior policy")

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

    def _validate_policy_obs_horizon(
        self,
        policy: Optional[Any],
        *,
        role: str,
    ) -> None:
        if policy is None:
            return
        assert hasattr(policy, "observation_horizon"), (
            f"{role} must expose observation_horizon for robomimic horizon validation."
        )
        assert int(self.cfg.frame_stack) == int(policy.observation_horizon), (
            "SOPE frame_stack must match the robomimic observation_horizon when "
            f"using {role} guidance: got frame_stack={self.cfg.frame_stack}, "
            f"{role}.observation_horizon={policy.observation_horizon}."
        )

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
        """Construct the default optimizer for the canonical diffusion model."""
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
        """Compute the canonical diffusion loss for one batch of chunks."""
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
        """Sample future transition chunks from the canonical diffusion model."""
        shape = (num_samples, self.cfg.total_chunk_horizon, self.transition_dim)
        return self.diffusion.conditional_sample(
            shape,
            cond,
            guided=self.cfg.guided,
            return_chain=return_chain,
            **kwargs,
        )

    def generate_full_trajectory(
        self,
        initial_states: torch.Tensor,
        *,
        max_length: Optional[int] = None,
        guided: bool = False,
        verbose: bool = False,
        **guidance_kw: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Autoregressively generate a full unnormalized state-action trajectory.

        NOTE:
        1. `initial_states` are expected in the same unnormalized state space used by
           the dataset / rollout latents.
        2. To build the first FiLM condition, the code concatenates each initial state
           with a zero action, applies the transition-level normalizer, and keeps only
           the normalized state slice. This matches the state marginal implied by the
           full `(state, action)` normalization statistics.
        3. Diffusion denoising and chunk sampling operate entirely in normalized
           transition space.
        4. Returned rollout states and actions are unnormalized by applying
           `self.unnormalizer(...)` to the sampled transitions before splitting the
           state and action channels.
        5. Autoregressive reconditioning uses the normalized state slice from
           `sample.trajectories[:, -frame_stack:, :state_dim]` directly. It does not
           renormalize states from the unnormalized rollout buffers.
        6. Guidance prefixes are passed as strict-past windows of length
           `frame_stack`. The robomimic score adapter consumes `prefix_states`
           directly as `[s_{t-To}, ..., s_{t-1}]`, while `prefix_actions` uses
           the same strict-past contract except for the explicit rollout-start
           empty-prefix boundary case handled inside `policy.py`.

        When `normalization_stats` is `None`, `self.normalizer` and
        `self.unnormalizer` are identity maps, so the same logic still applies with
        no numerical change.
        """
        if self.cfg.conditioning_mode == "none":
            raise NotImplementedError(
                "generate_full_trajectory does not support conditioning_mode='none'. "
                "Use chunk-level evaluation for the EEF-only debug path."
            )

        batch_size = int(initial_states.shape[0])
        if max_length is None:
            max_length = int(self.cfg.trajectory_horizon)
        else:
            max_length = int(max_length)
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}.")
        chunk_horizon = int(self.cfg.chunk_horizon)
        frame_stack = int(self.cfg.frame_stack)
        total_horizon = int(self.cfg.total_chunk_horizon)

        all_states = np.zeros((batch_size, max_length, self.state_dim), dtype=np.float32)
        all_actions = np.zeros((batch_size, max_length, self.action_dim), dtype=np.float32)

        initial_states = initial_states.to(self.device)
        dummy_actions = torch.zeros(
            batch_size,
            self.action_dim,
            device=self.device,
            dtype=initial_states.dtype,
        )
        initial_transition = torch.cat([initial_states, dummy_actions], dim=-1)
        initial_state_norm = self.normalizer(initial_transition)[:, : self.state_dim]
        cond_states = initial_state_norm.unsqueeze(1).expand(-1, frame_stack, -1).clone()
        # The robomimic score adapter consumes this as the strict-past state
        # prefix `[s_{t-To}, ..., s_{t-1}]`.
        prefix_states = initial_states.unsqueeze(1).expand(-1, frame_stack, -1).clone()
        # The first chunk has no real past actions yet, so the exact scorer owns
        # the only remaining start-boundary padding special case.
        prefix_actions = initial_states.new_empty((batch_size, 0, self.action_dim))

        def roll_prefix_buffer(
            buffer: Optional[torch.Tensor],
            new_steps: torch.Tensor,
            *,
            window: int,
        ) -> torch.Tensor:
            if buffer is None:
                combined = new_steps
            else:
                combined = torch.cat([buffer, new_steps], dim=1)
            return combined[:, -window:, :].clone()

        total_generated = 0
        self.diffusion.eval()
        with torch.no_grad():
            while total_generated < max_length:
                steps_to_add = min(chunk_horizon, max_length - total_generated)
                cond = cond_states.reshape(batch_size, -1)
                guidance_kwargs = dict(guidance_kw)
                guidance_kwargs["prefix_states"] = prefix_states
                guidance_kwargs["prefix_actions"] = prefix_actions
                sample = self.diffusion.conditional_sample(
                    shape=(batch_size, total_horizon, self.transition_dim),
                    cond=cond,
                    guided=guided,
                    verbose=verbose,
                    **guidance_kwargs,
                )
                chunk = self.unnormalizer(sample.trajectories)
                gen_states = chunk[:, :, : self.state_dim]
                gen_actions = chunk[:, :, self.state_dim :]

                t_end = total_generated + steps_to_add
                all_states[:, total_generated:t_end, :] = (
                    gen_states[:, :steps_to_add, :].detach().cpu().numpy()
                )
                all_actions[:, total_generated:t_end, :] = (
                    gen_actions[:, :steps_to_add, :].detach().cpu().numpy()
                )
                total_generated = t_end

                if total_generated >= max_length:
                    break

                # Chunk diffusion re-conditioning
                cond_states = roll_prefix_buffer(
                    cond_states,
                    sample.trajectories[:, :steps_to_add, : self.state_dim],
                    window=frame_stack,
                )
                
                # Policy score re-conditioning keeps a strict-past window of
                # length `frame_stack` for the off-by-one robomimic adapter.
                prefix_states = roll_prefix_buffer(
                    prefix_states,
                    gen_states[:, :steps_to_add, :],
                    window=frame_stack,
                )
                prefix_actions = roll_prefix_buffer(
                    prefix_actions,
                    gen_actions[:, :steps_to_add, :],
                    window=frame_stack,
                )

        return all_states, all_actions

    def ope_estimate(
        self,
        initial_states: torch.Tensor,
        reward_predictor: RewardPredictor | Any,
        max_length: Optional[int] = None,
        guided: bool = False,
        verbose: bool = False,
        **guidance_kw: Any,
    ) -> float:
        """Estimate mean discounted return for generated trajectories with a reward model."""
        if not hasattr(reward_predictor, "predict"):
            raise TypeError(
                "reward_predictor must expose a predict(states, actions) method."
            )

        ope_gamma = float(self.cfg.ope_gamma)
        if not (0.0 < ope_gamma <= 1.0):
            raise ValueError(
                f"ope_gamma must satisfy 0 < ope_gamma <= 1, got {self.cfg.ope_gamma}."
            )

        predictor_state_dim = getattr(reward_predictor, "state_dim", None)
        if predictor_state_dim is not None and int(predictor_state_dim) != self.state_dim:
            raise ValueError(
                "Reward predictor state_dim must match SopeDiffuser state_dim "
                f"({predictor_state_dim} != {self.state_dim})."
            )

        predictor_action_dim = getattr(reward_predictor, "action_dim", None)
        if predictor_action_dim is not None and int(predictor_action_dim) != self.action_dim:
            raise ValueError(
                "Reward predictor action_dim must match SopeDiffuser action_dim "
                f"({predictor_action_dim} != {self.action_dim})."
            )

        states, actions = self.generate_full_trajectory(
            initial_states,
            max_length=max_length,
            guided=guided,
            verbose=verbose,
            **guidance_kw,
        )
        if states.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "Generated states and actions must share batch and horizon dimensions; "
                f"got {states.shape} vs {actions.shape}."
            )
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"Generated state dim mismatch: expected {self.state_dim}, got {states.shape[-1]}."
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Generated action dim mismatch: expected {self.action_dim}, got {actions.shape[-1]}."
            )

        reward_preds = reward_predictor.predict(states, actions)
        reward_preds_t = (
            reward_preds
            if torch.is_tensor(reward_preds)
            else torch.as_tensor(reward_preds, dtype=torch.float32)
        )
        reward_preds_t = reward_preds_t.to(dtype=torch.float32)
        if reward_preds_t.ndim == 3 and reward_preds_t.shape[-1] == 1:
            reward_preds_t = reward_preds_t.squeeze(-1)
        expected_shape = (states.shape[0], states.shape[1])
        if tuple(reward_preds_t.shape) != expected_shape:
            raise ValueError(
                "Reward predictor outputs must have shape (batch, horizon); "
                f"expected {expected_shape}, got {tuple(reward_preds_t.shape)}."
            )

        discounts = torch.pow(
            torch.full(
                (states.shape[1],),
                ope_gamma,
                dtype=reward_preds_t.dtype,
                device=reward_preds_t.device,
            ),
            torch.arange(states.shape[1], device=reward_preds_t.device),
        )
        trajectory_returns = torch.sum(reward_preds_t * discounts.unsqueeze(0), dim=1)
        return float(trajectory_returns.mean().item())


def cross_validate_configs(cfg_dataset: Any, cfg_diffusion: SopeDiffusionConfig) -> None:
    """Validate that dataset and diffusion configs agree on dimensions and conditioning."""
    if not isinstance(cfg_diffusion, SopeDiffusionConfig):
        raise TypeError(
            f"cfg_diffusion must be a SopeDiffusionConfig, got {type(cfg_diffusion)}"
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
            "Config mismatch: dataset state_dim must equal SopeDiffusionConfig.state_dim "
            f"({dataset_state_dim} != {cfg_diffusion.state_dim})."
        )
    if int(dataset_action_dim) != cfg_diffusion.action_dim:
        raise ValueError(
            "Config mismatch: dataset action_dim must equal SopeDiffusionConfig.action_dim "
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
