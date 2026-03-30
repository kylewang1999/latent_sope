from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from third_party.robomimic.robomimic.algo import RolloutPolicy


@dataclass(frozen=True)
class DiffusionPolicyScoreConfig:
    score_timestep: int = 1
    repeat_single_state_to_horizon: bool = True


class DiffusionPolicy(RolloutPolicy):
    """Robomimic diffusion-policy adapter with a SOPE-compatible score API.

    The current SOPE guidance path only supplies per-step `(state, action)` pairs
    and does not pass the sampler timestep into `grad_log_prob`. For now we use a
    fixed diffusion timestep and treat the incoming `state` tensor as an already
    encoded observation feature.
    """

    def __init__(
        self,
        policy: Any,
        obs_normalization_stats: Optional[dict[str, Any]] = None,
        action_normalization_stats: Optional[dict[str, Any]] = None,
        *,
        config: Optional[DiffusionPolicyScoreConfig] = None,
    ) -> None:
        super().__init__(
            policy,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
        )
        self.score_config = config or DiffusionPolicyScoreConfig()

    @property
    def obs_feature_dim(self) -> int:
        nets = self._policy_nets()
        return int(nets["policy"]["obs_encoder"].output_shape()[0])

    @property
    def observation_horizon(self) -> int:
        return int(self.policy.algo_config.horizon.observation_horizon)

    @property
    def prediction_horizon(self) -> int:
        return int(self.policy.algo_config.horizon.prediction_horizon)

    @property
    def action_dim(self) -> int:
        return int(self.policy.ac_dim)

    @property
    def action_start_index(self) -> int:
        return int(self.observation_horizon - 1)

    def _policy_nets(self) -> Any:
        if getattr(self.policy, "ema", None) is not None:
            return self.policy.ema.averaged_model
        return self.policy.nets

    def _prepare_obs_cond(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            obs_features = state
        elif state.ndim == 2:
            if state.shape[-1] == self.obs_feature_dim * self.observation_horizon:
                obs_features = state.view(state.shape[0], self.observation_horizon, self.obs_feature_dim)
            elif state.shape[-1] == self.obs_feature_dim:
                if not self.score_config.repeat_single_state_to_horizon:
                    raise ValueError(
                        "Received single-step encoded observations but repeating to the "
                        "robomimic observation horizon is disabled."
                    )
                obs_features = state.unsqueeze(1).expand(-1, self.observation_horizon, -1)
            else:
                raise ValueError(
                    "State feature dimension does not match robomimic obs encoder output: "
                    f"got {state.shape[-1]}, expected {self.obs_feature_dim} or "
                    f"{self.obs_feature_dim * self.observation_horizon}."
                )
        else:
            raise ValueError(f"Expected state rank 2 or 3, got {state.ndim}.")

        if obs_features.shape[1] != self.observation_horizon:
            raise ValueError(
                "Observation feature horizon mismatch: "
                f"got {obs_features.shape[1]}, expected {self.observation_horizon}."
            )
        return obs_features.flatten(start_dim=1)

    def _prepare_action_sequence(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 3:
            if action.shape[1] != self.prediction_horizon:
                raise ValueError(
                    f"Action horizon mismatch: got {action.shape[1]}, expected {self.prediction_horizon}."
                )
            return action
        if action.ndim != 2:
            raise ValueError(f"Expected action rank 2 or 3, got {action.ndim}.")
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch: got {action.shape[-1]}, expected {self.action_dim}."
            )
        return action.unsqueeze(1).expand(-1, self.prediction_horizon, -1).contiguous()

    @torch.no_grad()
    def sample_tensor(self, state: torch.Tensor) -> torch.Tensor:
        """Sample the first executable robomimic action for already-encoded observations."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.policy.device)
        obs_cond = self._prepare_obs_cond(state_t)
        nets = self._policy_nets()

        if self.policy.algo_config.ddpm.enabled:
            num_inference_timesteps = int(self.policy.algo_config.ddpm.num_inference_timesteps)
        elif self.policy.algo_config.ddim.enabled:
            num_inference_timesteps = int(self.policy.algo_config.ddim.num_inference_timesteps)
        else:
            raise ValueError("Expected a DDPM or DDIM robomimic diffusion policy.")

        self.policy.noise_scheduler.set_timesteps(num_inference_timesteps)
        sample = torch.randn(
            (state_t.shape[0], self.prediction_horizon, self.action_dim),
            device=self.policy.device,
            dtype=state_t.dtype,
        )
        for timestep in self.policy.noise_scheduler.timesteps:
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=sample,
                timestep=timestep,
                global_cond=obs_cond,
            )
            sample = self.policy.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=sample,
            ).prev_sample
        return sample[:, self.action_start_index, :]

    def grad_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return a score-like action gradient for SOPE guidance."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.policy.device)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.policy.device)

        obs_cond = self._prepare_obs_cond(state_t)
        action_seq = self._prepare_action_sequence(action_t)
        nets = self._policy_nets()

        timestep = max(
            0,
            min(
                int(self.score_config.score_timestep),
                int(self.policy.noise_scheduler.config.num_train_timesteps) - 1,
            ),
        )
        timestep_t = torch.full(
            (action_seq.shape[0],),
            timestep,
            device=self.policy.device,
            dtype=torch.long,
        )
        noise_pred = nets["policy"]["noise_pred_net"](
            sample=action_seq,
            timestep=timestep_t,
            global_cond=obs_cond,
        )

        alpha_bar = self.policy.noise_scheduler.alphas_cumprod[timestep_t]
        score_scale = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-6)).view(-1, 1, 1)
        score_seq = -noise_pred / score_scale
        return score_seq[:, self.action_start_index, :]
