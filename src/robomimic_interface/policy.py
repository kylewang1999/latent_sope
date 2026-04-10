from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from einops import rearrange

from third_party.robomimic.robomimic.algo import RolloutPolicy

if TYPE_CHECKING:
    from third_party.robomimic.robomimic.algo.diffusion_policy import (
        DiffusionPolicyUNet as RobomimicDiffusionPolicyUNet,
    )
else:
    RobomimicDiffusionPolicyUNet = Any


@dataclass(frozen=True)
class DiffusionPolicyScoreConfig:
    score_timestep: int = 1
    repeat_single_state_to_horizon: bool = True


class DiffusionPolicy(RolloutPolicy):
    """Robomimic diffusion-policy adapter with a SOPE-compatible score API.

    The current SOPE guidance path supplies chunk-shaped `(state, action)`
    tensors and does not pass the sampler timestep into `grad_log_prob`. For
    now we use a fixed diffusion timestep and treat each incoming `state`
    vector as an already encoded observation feature.
    """

    def __init__(
        self,
        policy: RobomimicDiffusionPolicyUNet,
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
                obs_features = rearrange(
                    state,
                    "b (to d) -> b to d",
                    to=self.observation_horizon,
                    d=self.obs_feature_dim,
                )
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
        return rearrange(obs_features, "b to d -> b (to d)")

    def _prepare_action_sequence(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2:
            raise ValueError(
                "DiffusionPolicy.grad_log_prob expects single-step actions with "
                f"shape [B, Da], got rank {action.ndim}."
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch: got {action.shape[-1]}, expected {self.action_dim}."
            )
        return action.unsqueeze(1).expand(-1, self.prediction_horizon, -1).contiguous()

    def _build_chunk_window_indices(
        self,
        *,
        chunk_horizon: int,
        window_horizon: int,
        anchor_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        if chunk_horizon <= 0:
            raise ValueError(f"chunk_horizon must be positive, got {chunk_horizon}.")
        chunk_steps = torch.arange(chunk_horizon, device=device, dtype=torch.long)
        window_offsets = torch.arange(window_horizon, device=device, dtype=torch.long) - int(anchor_index)
        return torch.clamp(
            chunk_steps[:, None] + window_offsets[None, :],
            min=0,
            max=chunk_horizon - 1,
        )

    def _build_chunk_obs_windows(self, state: torch.Tensor) -> torch.Tensor:
        """Build causal observation windows ending at each chunk step.

        Input shape:
        - `state`: [B, H, Dobs_feat]

        Output shape:
        - [B, H, To, Dobs_feat]

        The current chunk step `i` is placed at the final observation slot, and
        missing prefix history is filled by replicating the earliest in-chunk
        state.
        """
        if state.ndim != 3:
            raise ValueError(f"Expected chunk states with shape [B, H, Dobs_feat], got {tuple(state.shape)}.")
        if state.shape[-1] != self.obs_feature_dim:
            raise ValueError(
                "State feature dimension does not match robomimic obs encoder output: "
                f"got {state.shape[-1]}, expected {self.obs_feature_dim}."
            )
        indices = self._build_chunk_window_indices(
            chunk_horizon=state.shape[1],
            window_horizon=self.observation_horizon,
            anchor_index=self.observation_horizon - 1,
            device=state.device,
        )
        return state[:, indices, :]

    def _build_chunk_action_windows(self, action: torch.Tensor) -> torch.Tensor:
        """Build aligned action windows for chunk-conditioned score queries.

        Input shape:
        - `action`: [B, H, Da]

        Output shape:
        - [B, H, Tp, Da]

        The queried chunk action at step `i` is placed at
        `action_start_index = observation_horizon - 1`, with both prefix and
        suffix context filled by in-chunk edge replication when needed.
        """
        if action.ndim != 3:
            raise ValueError(f"Expected chunk actions with shape [B, H, Da], got {tuple(action.shape)}.")
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch: got {action.shape[-1]}, expected {self.action_dim}."
            )
        indices = self._build_chunk_window_indices(
            chunk_horizon=action.shape[1],
            window_horizon=self.prediction_horizon,
            anchor_index=self.action_start_index,
            device=action.device,
        )
        return action[:, indices, :]

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
        r"""Return a chunk-shaped action score for the local SOPE guidance contract.

        Shape contract:
        - `state` must be `[B, H, obs_feature_dim]`.
        - `action` must be `[B, H, action_dim]`.
        - the returned score always has shape `[B, H, action_dim]`.

        Robomimic's diffusion policy is a chunk model over action sequences of
        length `prediction_horizon`, while the local sampler now calls
        `grad_log_prob(state, action)` on full chunk tensors.
        This adapter bridges that mismatch as follows:

        - chunk-derived causal observation windows are built for each chunk step,
          with in-chunk left-edge replication when the full observation horizon
          is unavailable
        - chunk-derived action windows are built for each chunk step so the
          queried action lands at `action_start_index = observation_horizon - 1`,
          with in-chunk edge replication at both boundaries
        - those windows are flattened to `(B * H)` robomimic denoiser queries
        - the robomimic denoiser is evaluated at the fixed diffusion timestep
          `score_timestep`, converted to the usual `predict-epsilon` score
          estimate `-\hat{\epsilon}/\sqrt{1-\bar{\alpha}_t}`, and then
          collapsed back to one action score per chunk step
        - the flattened scores are finally reshaped to `[B, H, action_dim]`

        So the returned tensor matches the input `action` chunk shape expected
        by the local sampler, but it is still a surrogate extracted from a
        chunk-conditioned model rather than the exact score of a joint action
        chunk density: the adapter still uses a fixed diffusion timestep and
        does not receive the sampler's true prefix history outside the current
        chunk.
        """
        if state.ndim != 3:
            raise ValueError(
                "DiffusionPolicy.grad_log_prob expects chunk states with shape "
                f"[B, H, Dobs_feat], got {tuple(state.shape)}."
            )
        if action.ndim != 3:
            raise ValueError(
                "DiffusionPolicy.grad_log_prob expects chunk actions with shape "
                f"[B, H, Da], got {tuple(action.shape)}."
            )
        if state.shape[:2] != action.shape[:2]:
            raise ValueError(
                "State and action chunk leading dimensions must match: "
                f"got state {tuple(state.shape)} vs action {tuple(action.shape)}."
            )

        B, H, _ = state.shape
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.policy.device)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.policy.device)

        obs_windows = self._build_chunk_obs_windows(state_t)
        action_windows = self._build_chunk_action_windows(action_t)

        obs_cond = rearrange(obs_windows, "b h to d -> (b h) (to d)")
        action_seq = rearrange(action_windows, "b h tp da -> (b h) tp da")
        nets = self._policy_nets()

        timestep = max(
            0,
            min(
                int(self.score_config.score_timestep),
                int(self.policy.noise_scheduler.config.num_train_timesteps) - 1,
            ),
        )
        timestep_t = torch.full(
            (B * H,),
            timestep,
            device=self.policy.device,
            dtype=torch.long,
        )
        # Predicted epsilon over the repeated action chunk: [(B * H), Tp, Da].
        noise_pred = nets["policy"]["noise_pred_net"](
            sample=action_seq,
            timestep=timestep_t,
            global_cond=obs_cond,
        )

        alpha_bar = self.policy.noise_scheduler.alphas_cumprod[timestep_t]
        score_scale = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-6)).view(-1, 1, 1)
        # Per-step chunk score surrogate extracted from epsilon prediction: [(B * H), Tp, Da].
        score_seq = -noise_pred / score_scale
        # Collapse back to one action score per flattened step, then restore [B, H, Da].
        score = rearrange(
            score_seq[:, self.action_start_index, :],
            "(b h) da -> b h da",
            b=B,
            h=H,
        )
        assert score.shape == action_t.shape, (
            f"DiffusionPolicy.grad_log_prob must return shape {tuple(action_t.shape)}, "
            f"got {tuple(score.shape)}."
        )
        return score
