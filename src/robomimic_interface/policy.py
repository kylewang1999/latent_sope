from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from einops import rearrange

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from third_party.robomimic.robomimic.algo import RolloutPolicy

if TYPE_CHECKING:
    from third_party.robomimic.robomimic.algo.diffusion_policy import (
        DiffusionPolicyUNet as RobomimicDiffusionPolicyUNet,
    )
    from third_party.robomimic.robomimic.models.diffusion_policy_nets import (
        ConditionalUnet1D as RobomimicConditionalUnet1D,
    )
else:
    RobomimicDiffusionPolicyUNet = Any
    RobomimicConditionalUnet1D = Any


@dataclass(frozen=True)
class DiffusionPolicyScoreConfig:
    score_timestep: int = 1


class DiffusionPolicy(RolloutPolicy):
    """Robomimic diffusion-policy adapter with a SOPE-compatible score API.

    The current SOPE guidance path supplies chunk-shaped `(state, action)`
    tensors and does not pass the sampler timestep into `grad_log_prob`. For
    now we use a fixed diffusion timestep and treat each incoming `state`
    vector as an already encoded observation feature.
    """

    policy: RobomimicDiffusionPolicyUNet

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
        """Index of the current chunk action inside a robomimic action window.

        Robomimic aligns the executable "current" action with the final
        observation in the `To = observation_horizon` conditioning window, so
        the aligned action slot is `To - 1`.

        Example: Let current trajectory time be `i` and 
        - `To=2` (observation_horizon) 
        - `Tp=4` (prediction_horizon)
        then the observation window is `[s_{i-1}, s_i]`, so the current action
        `a_i` must align with the last observation slot and therefore sits at
        index `To - 1 = 1`. Since `Tp=4`, the full predicted action window has
        `Tp = 4` slots total, giving `[a_{i-1}, a_i, a_{i+1}, a_{i+2}]`.
        """
        return int(self.observation_horizon - 1)

    def _policy_nets(self) -> nn.ModuleDict:
        if getattr(self.policy, "ema", None) is not None:
            return self.policy.ema.averaged_model
        return self.policy.nets

    def _prepare_obs_cond(self, state: torch.Tensor) -> torch.Tensor:
        expected_flat_obs_dim = self.obs_feature_dim * self.observation_horizon
        assert state.ndim in (2, 3), f"Expected state ndim 2 or 3, got {state.ndim}."
        if state.ndim == 3:
            obs_features = state
        else:
            assert state.shape[-1] in (expected_flat_obs_dim, self.obs_feature_dim), (
                "State feature dimension does not match robomimic obs encoder output: "
                f"got {state.shape[-1]}, expected {self.obs_feature_dim} or "
                f"{expected_flat_obs_dim}."
            )
            if state.shape[-1] == expected_flat_obs_dim:
                obs_features = rearrange(
                    state,
                    "b (to d) -> b to d",
                    to=self.observation_horizon,
                    d=self.obs_feature_dim,
                )
            else:
                obs_features = state.unsqueeze(1).expand(-1, self.observation_horizon, -1)

        assert obs_features.shape[1] == self.observation_horizon, (
            "Observation feature horizon mismatch: "
            f"got {obs_features.shape[1]}, expected {self.observation_horizon}."
        )
        return rearrange(obs_features, "b to d -> b (to d)")

    def _build_chunk_window_indices(
        self,
        *,
        chunk_horizon: int, # num timesteps available in the input traj chunk
        window_horizon: int, # num timesteps to include in each constructed window (e.g. observation_horizon)
        anchor_index: int, # position of the current chunk step in the window
        device: torch.device,
    ) -> torch.Tensor:
        """Return per-step gather indices for fixed-width windows over a chunk.

        Example: let current trajectory time be `i` and set
        - `chunk_horizon=5`
        - `window_horizon=2`
        - `anchor_index=1`
        then the returned indices are
        `indices = [[0, 0], [0, 1], [1, 2], [2, 3], [3, 4]]`.
        """ 
        assert chunk_horizon > 0, f"chunk_horizon must be positive, got {chunk_horizon}."
        chunk_steps = torch.arange(chunk_horizon, device=device, dtype=torch.long)
        window_offsets = torch.arange(window_horizon, device=device, dtype=torch.long) - int(anchor_index)
        indices = torch.clamp(
            chunk_steps[:, None] + window_offsets[None, :],
            min=0,
            max=chunk_horizon - 1,
        )
        return indices # (chunk_horizon, window_horizon)

    def _build_chunk_obs_windows(self, state: torch.Tensor) -> torch.Tensor:
        """Build causal observation windows ending at each chunk step.
        
        Input shape:
        - `state`: [B, H, Dobs], H = chunk_horizon
        
        Output shape:
        - [B, H, To, Dobs], To = observation_horizon

        The current chunk step `i` is placed at the final observation slot, and
        missing prefix history is filled by replicating the earliest in-chunk
        state.
        """
        assert state.ndim == 3, (
            f"Expected chunk states with shape [B, H, Dobs], got {tuple(state.shape)}."
        )
        assert state.shape[-1] == self.obs_feature_dim, (
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
        - `action`: [B, H, Da], H = chunk_horizon

        Output shape:
        - [B, H, Tp, Da], Tp = prediction_horizon

        The queried chunk action at step `i` is placed at
        `action_start_index = observation_horizon - 1`, with both prefix and
        suffix context filled by in-chunk edge replication when needed.
        """
        assert action.ndim == 3, (
            f"Expected chunk actions with shape [B, H, Da], got {tuple(action.shape)}."
        )
        assert action.shape[-1] == self.action_dim, (
            f"Action dimension mismatch: got {action.shape[-1]}, expected {self.action_dim}."
        )
        indices = self._build_chunk_window_indices(
            chunk_horizon=action.shape[1],
            window_horizon=self.prediction_horizon, # IMPORTANT: this is robomimic's prediction horizon, not the local chunk horizon
            anchor_index=self.action_start_index,
            device=action.device,
        )
        return action[:, indices, :]

    def _guidance_timestep_tensor(self, *, batch_size: int) -> torch.Tensor:
        """Return the fixed robomimic diffusion timestep used for score queries."""
        timestep = max(
            0,
            min(
                int(self.score_config.score_timestep),
                int(self.policy.noise_scheduler.config.num_train_timesteps) - 1,
            ),
        )
        return torch.full(
            (batch_size,),
            timestep,
            device=self.policy.device,
            dtype=torch.long,
        )

    @torch.no_grad()
    def sample_tensor(self, state: torch.Tensor) -> torch.Tensor:
        """Sample the first executable robomimic action for already-encoded observations."""
        state = torch.as_tensor(state, dtype=torch.float32, device=self.policy.device)
        obs_cond = self._prepare_obs_cond(state)
        nets = self._policy_nets()

        assert self.policy.algo_config.ddpm.enabled or self.policy.algo_config.ddim.enabled, (
            "Expected a DDPM or DDIM robomimic diffusion policy."
        )
        if self.policy.algo_config.ddpm.enabled:
            num_inference_timesteps = int(self.policy.algo_config.ddpm.num_inference_timesteps)
        else:
            num_inference_timesteps = int(self.policy.algo_config.ddim.num_inference_timesteps)

        self.policy.noise_scheduler.set_timesteps(num_inference_timesteps)
        sample = torch.randn(
            (state.shape[0], self.prediction_horizon, self.action_dim),
            device=self.policy.device,
            dtype=state.dtype,
        )
        noise_pred_net: RobomimicConditionalUnet1D = nets["policy"]["noise_pred_net"]
        for timestep in self.policy.noise_scheduler.timesteps:
            noise_pred = noise_pred_net(
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
        r"""Return a chunk-shaped action score for SOPE chunk diffusion guidance.

        Shape contract:
        - `state` must be `[B, H, obs_feature_dim]`.
        - `action` must be `[B, H, action_dim]`.
        - the returned score always has shape `[B, H, action_dim]`.

        This method bridges two different sequence contracts:

        - robomimic consumes observation windows of shape
          `[N, observation_horizon, obs_feature_dim]` 
        and predicts denoising
          residuals over action sequences of shape
          `[N, prediction_horizon, action_dim]`
        - the local SOPE-style sampler calls `grad_log_prob(state, action)` with
        `[B, H, obs_feature_dim]` and `[B, H, action_dim]`, expecting one action 
        score per chunk step with shape `[B, H, action_dim]`

        This adapter maps the chunk contract into robomimic denoiser queries as
        follows:

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
        assert state.ndim == 3, (
            "DiffusionPolicy.grad_log_prob expects chunk states with shape "
            f"[B, H, Dobs], got {tuple(state.shape)}."
        )
        assert action.ndim == 3, (
            "DiffusionPolicy.grad_log_prob expects chunk actions with shape "
            f"[B, H, Da], got {tuple(action.shape)}."
        )
        assert state.shape[:2] == action.shape[:2], (
            "State and action chunk leading dimensions must match: "
            f"got state {tuple(state.shape)} vs action {tuple(action.shape)}."
        )

        B, H, _ = state.shape
        state = torch.as_tensor(state, dtype=torch.float32, device=self.policy.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.policy.device)

        obs_windows = self._build_chunk_obs_windows(state)
        action_windows = self._build_chunk_action_windows(action)

        obs_cond = rearrange(obs_windows, "b h to d -> (b h) (to d)")
        action_seq = rearrange(action_windows, "b h tp da -> (b h) tp da")
        nets = self._policy_nets()
        noise_pred_net: RobomimicConditionalUnet1D = nets["policy"]["noise_pred_net"]
        timestep_tensor = self._guidance_timestep_tensor(batch_size=B * H)
        # Predicted epsilon over the repeated action chunk: [(B * H), Tp, Da].
        noise_pred = noise_pred_net(
            sample=action_seq,
            timestep=timestep_tensor,
            global_cond=obs_cond,
        )

        # Diffusers schedulers keep coefficient tensors outside the module tree,
        # so they may remain on CPU even when the policy network runs on CUDA.
        alpha_bar = torch.as_tensor(
            self.policy.noise_scheduler.alphas_cumprod,
            device=noise_pred.device,
        )[timestep_tensor.to(device=noise_pred.device)]
        score_scale = torch.sqrt(
            torch.clamp(1.0 - alpha_bar, min=1e-6),
        ).to(dtype=noise_pred.dtype).view(-1, 1, 1)
        # Per-step chunk score surrogate extracted from epsilon prediction: [(B * H), Tp, Da].
        score_seq = -noise_pred / score_scale
        # Collapse back to one action score per flattened step, then restore [B, H, Da].
        score = rearrange(
            score_seq[:, self.action_start_index, :], # (B*H, Tp, Da) -> (B*H, Da)
            "(b h) da -> b h da",
            b=B,
            h=H,
        )
        assert score.shape == action.shape, (
            f"DiffusionPolicy.grad_log_prob must return shape {tuple(action.shape)}, "
            f"got {tuple(score.shape)}."
        )
        return score


if __name__ == "__main__":
    class _ToyObsEncoder(nn.Module):
        def __init__(self, obs_feature_dim: int) -> None:
            super().__init__()
            self._obs_feature_dim = int(obs_feature_dim)

        def output_shape(self) -> tuple[int]:
            return (self._obs_feature_dim,)


    class _ToyNoisePredNet(nn.Module):
        def __init__(
            self,
            *,
            obs_cond_dim: int,
            prediction_horizon: int,
            action_dim: int,
            num_train_timesteps: int,
        ) -> None:
            super().__init__()
            hidden_dim = 64
            self.prediction_horizon = int(prediction_horizon)
            self.action_dim = int(action_dim)
            self.sample_proj = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )
            self.global_cond_proj = nn.Sequential(
                nn.Linear(obs_cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, prediction_horizon * action_dim),
            )
            self.timestep_embed = nn.Embedding(num_train_timesteps, prediction_horizon * action_dim)

        def forward(
            self,
            *,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            global_cond: torch.Tensor,
        ) -> torch.Tensor:
            assert sample.ndim == 3, f"Expected sample ndim 3, got {sample.ndim}."
            assert global_cond.ndim == 2, (
                f"Expected global_cond ndim 2, got {global_cond.ndim}."
            )
            assert sample.shape[1:] == (self.prediction_horizon, self.action_dim), (
                "Toy noise predictor received an unexpected sample shape: "
                f"got {tuple(sample.shape[1:])}, expected "
                f"({self.prediction_horizon}, {self.action_dim})."
            )

            batch_size = sample.shape[0]
            sample_term = self.sample_proj(sample)
            global_term = self.global_cond_proj(global_cond).view(
                batch_size,
                self.prediction_horizon,
                self.action_dim,
            )
            timestep_term = self.timestep_embed(timestep).view(
                batch_size,
                self.prediction_horizon,
                self.action_dim,
            )
            return sample_term + 0.1 * global_term + 0.01 * timestep_term


    class _ToyNoiseScheduler:
        def __init__(self, *, num_train_timesteps: int, device: torch.device) -> None:
            self.config = SimpleNamespace(num_train_timesteps=int(num_train_timesteps))
            self.alphas_cumprod = torch.linspace(
                0.99,
                0.1,
                steps=num_train_timesteps,
                device=device,
                dtype=torch.float32,
            )


    class _ToyRobomimicDiffusionPolicy:
        def __init__(
            self,
            *,
            device: torch.device,
            obs_feature_dim: int,
            observation_horizon: int,
            prediction_horizon: int,
            action_dim: int,
            num_train_timesteps: int,
        ) -> None:
            self.device = device
            self.ac_dim = int(action_dim)
            self.algo_config = SimpleNamespace(
                horizon=SimpleNamespace(
                    observation_horizon=int(observation_horizon),
                    prediction_horizon=int(prediction_horizon),
                ),
            )
            self.noise_scheduler = _ToyNoiseScheduler(
                num_train_timesteps=num_train_timesteps,
                device=device,
            )
            self.nets = nn.ModuleDict(
                {
                    "policy": nn.ModuleDict(
                        {
                            "obs_encoder": _ToyObsEncoder(obs_feature_dim),
                            "noise_pred_net": _ToyNoisePredNet(
                                obs_cond_dim=observation_horizon * obs_feature_dim,
                                prediction_horizon=prediction_horizon,
                                action_dim=action_dim,
                                num_train_timesteps=num_train_timesteps,
                            ),
                        }
                    ),
                }
            )
            self.nets = self.nets.float().to(self.device)
            self.ema = None


    def main() -> None:
        torch.manual_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = 4
        chunk_horizon = 8
        obs_feature_dim = 19
        action_dim = 7
        observation_horizon = 3
        prediction_horizon = 5
        num_train_timesteps = 16

        toy_policy = _ToyRobomimicDiffusionPolicy(
            device=device,
            obs_feature_dim=obs_feature_dim,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            action_dim=action_dim,
            num_train_timesteps=num_train_timesteps,
        )
        policy = DiffusionPolicy(
            toy_policy,
            config=DiffusionPolicyScoreConfig(score_timestep=1),
        )

        state = torch.randn(
            batch_size,
            chunk_horizon,
            obs_feature_dim,
            device=device,
        )
        action = torch.randn(
            batch_size,
            chunk_horizon,
            action_dim,
            device=device,
        )

        with torch.no_grad():
            score = policy.grad_log_prob(state, action)
            state_perturbed_score = policy.grad_log_prob(state + 0.05, action)
            action_perturbed_score = policy.grad_log_prob(state, action + 0.05)

        assert score.shape == (batch_size, chunk_horizon, action_dim), (
            "Toy grad_log_prob smoke test returned an unexpected shape: "
            f"got {tuple(score.shape)}."
        )
        assert torch.isfinite(score).all(), "Toy grad_log_prob smoke test produced non-finite values."
        assert not torch.allclose(score, state_perturbed_score), (
            "Toy grad_log_prob smoke test is unexpectedly insensitive to state perturbations."
        )
        assert not torch.allclose(score, action_perturbed_score), (
            "Toy grad_log_prob smoke test is unexpectedly insensitive to action perturbations."
        )

        print(
            "grad_log_prob smoke test passed:",
            f"device={device}",
            f"state_shape={tuple(state.shape)}",
            f"action_shape={tuple(action.shape)}",
            f"score_shape={tuple(score.shape)}",
        )
        print("score[0, 0] =", score[0, 0].detach().cpu())


    main()
