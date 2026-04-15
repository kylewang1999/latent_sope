from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional
import warnings

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
        self._warning_cache: set[str] = set()

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
        """Index of the first future chunk action in the adapter action window.

        This adapter intentionally uses an off-by-one abstraction so its
        robomimic calls resemble the SOPE chunk diffuser: the conditioning
        prefix is the strict past window of length `To = observation_horizon`,
        and the first returned score or sampled action is the first action
        after that prefix.

        Example: with `To=2`, `H=4`, and `Tp=6`, exact chunk scoring uses
        `obs_window = [s_{t-2}, s_{t-1}]` and
        `action_window = [a_{t-2}, a_{t-1}, a_t, a_{t+1}, a_{t+2}, a_{t+3}]`.
        The returned chunk score slice starts at index `To = 2`, yielding
        `[a_t, a_{t+1}, a_{t+2}, a_{t+3}]`.

        This is an intentional adapter-level choice and does not match native
        robomimic current-action alignment.
        """
        return int(self.observation_horizon)

    def _policy_nets(self) -> nn.ModuleDict:
        if getattr(self.policy, "ema", None) is not None:
            return self.policy.ema.averaged_model
        return self.policy.nets

    def _as_feature_sequence(
        self,
        sequence: Any,
        *,
        feature_name: str,
        batch_size: Optional[int],
        feature_dim: int,
        valid_horizons: tuple[int, ...],
    ) -> torch.Tensor:
        sequence_t = torch.as_tensor(
            sequence,
            dtype=torch.float32,
            device=self.policy.device,
        )
        assert sequence_t.ndim == 3, (
            f"{feature_name} must have shape [B, S, D], got {tuple(sequence_t.shape)}."
        )
        if batch_size is not None:
            assert sequence_t.shape[0] == batch_size, (
                f"{feature_name} batch dimension must match the chunk batch size: "
                f"got {sequence_t.shape[0]} vs {batch_size}."
            )
        assert sequence_t.shape[-1] == feature_dim, (
            f"{feature_name} feature dimension mismatch: got {sequence_t.shape[-1]}, "
            f"expected {feature_dim}."
        )
        assert sequence_t.shape[1] in valid_horizons, (
            f"{feature_name} must have horizon in {valid_horizons}, got "
            f"{sequence_t.shape[1]}."
        )
        return sequence_t

    def _prepare_obs_cond(self, state: torch.Tensor) -> torch.Tensor:
        assert state.ndim in (2, 3), f"Expected state ndim 2 or 3, got {state.ndim}."
        expected_flat_obs_dim = self.obs_feature_dim * self.observation_horizon
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
                # A single encoded observation `[B, Dobs]` has no explicit
                # history axis, so repeat it across the `To` slots expected by
                # robomimic before flattening back to `[B, To * Dobs]`.
                obs_features = state.unsqueeze(1).expand(-1, self.observation_horizon, -1)

        assert obs_features.shape[1] == self.observation_horizon, (
            "Observation feature horizon mismatch: "
            f"got {obs_features.shape[1]}, expected {self.observation_horizon}."
        )
        return rearrange(obs_features, "b to d -> b (to d)")

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warning_cache:
            return
        self._warning_cache.add(key)
        warnings.warn(message, stacklevel=2)

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

    def _score_from_noise_pred(
        self,
        *,
        noise_pred: torch.Tensor,
        timestep_tensor: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = torch.as_tensor(
            self.policy.noise_scheduler.alphas_cumprod,
            device=noise_pred.device,
        )[timestep_tensor.to(device=noise_pred.device)]
        score_scale = torch.sqrt(
            torch.clamp(1.0 - alpha_bar, min=1e-6),
        ).to(dtype=noise_pred.dtype).view(-1, 1, 1)
        return -noise_pred / score_scale

    @torch.no_grad()
    def sample_tensor(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample the first future action from an explicit strict-past prefix.

        Input:
        - state: `[B, To, Dobs]`, the strict past observation prefix ending at
          `t - 1`

        Output:
        - action: `[B, Da]`, the sampled action at slot `To`

        Example with `To=2` and `Tp=6`:
        - caller provides `state = [s_{t-2}, s_{t-1}]`
        - robomimic is queried with that strict-past prefix directly
        - this adapter returns the sampled future action at index `To = 2`,
          corresponding to `a_t`

        This is a breaking API change. The adapter no longer accepts a single
        encoded observation `[B, Dobs]` or a flattened observation prefix.
        """
        state = self._as_feature_sequence(
            state,
            feature_name="state",
            batch_size=None,
            feature_dim=self.obs_feature_dim,
            valid_horizons=(self.observation_horizon,),
        )
        obs_cond = self._prepare_obs_cond(state)
        nets = self._policy_nets()
        assert self.action_start_index < self.prediction_horizon, (
            "The off-by-one adapter requires prediction_horizon > "
            f"observation_horizon, got prediction_horizon={self.prediction_horizon} "
            f"and observation_horizon={self.observation_horizon}."
        )

        assert self.policy.algo_config.ddpm.enabled or self.policy.algo_config.ddim.enabled, (
            "Expected a DDPM or DDIM robomimic diffusion policy."
        )
        if self.policy.algo_config.ddpm.enabled:
            num_inference_timesteps = int(self.policy.algo_config.ddpm.num_inference_timesteps)
        else:
            num_inference_timesteps = int(self.policy.algo_config.ddim.num_inference_timesteps)

        self.policy.noise_scheduler.set_timesteps(num_inference_timesteps)
        if deterministic:
            sample = torch.zeros(
                (state.shape[0], self.prediction_horizon, self.action_dim),
                device=self.policy.device,
                dtype=state.dtype,
            )
        else:
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

    def _single_action_chunk_scoring(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        noise_pred_net: RobomimicConditionalUnet1D,
    ) -> torch.Tensor:
        """Score a chunk via a per-step strict-past sliding-window surrogate.

        Input:
        - state: `[B, H, Dobs]`
        - action: `[B, H, Da]`
        Output:
        - score: `[B, H, Da]`

        Example with `To=2`, `Tp=6`, and `H=4`:
        - SOPE provides the chunk states
          `[s_t, s_{t+1}, s_{t+2}, s_{t+3}]` and
          chunk actions `[a_t, a_{t+1}, a_{t+2}, a_{t+3}]`
        - this helper does not build one full robomimic query for the whole
          chunk; instead it builds one strict-past observation window and one
          action window for each chunk step
        - for the first queried step `t`, edge replication within the chunk
          gives observation window `[s_t, s_t]` and action window
          `[a_t, a_t, a_t, a_{t+1}, a_{t+2}, a_{t+3}]`
        - for the next queried step `t+1`, the windows become
          observation window `[s_t, s_t]` and action window
          `[a_t, a_t, a_{t+1}, a_{t+2}, a_{t+3}, a_{t+3}]`
        - robomimic predicts a score for each length-`Tp` action window, and
          this helper returns the off-by-one-aligned single-action score from
          slot `To = 2`, stacked back into the length-`H` output
          `[a_t, a_{t+1}, a_{t+2}, a_{t+3}]`
        """
        assert self.action_start_index < self.prediction_horizon, (
            "Off-by-one sliding-window scoring requires "
            "prediction_horizon > observation_horizon; got "
            f"prediction_horizon={self.prediction_horizon} and "
            f"observation_horizon={self.observation_horizon}."
        )

        def build_chunk_window_indices(
            *,
            chunk_horizon: int,
            window_horizon: int,
            anchor_index: int,
            device: torch.device,
        ) -> torch.Tensor:
            assert chunk_horizon > 0, (
                f"chunk_horizon must be positive, got {chunk_horizon}."
            )
            chunk_steps = torch.arange(
                chunk_horizon,
                device=device,
                dtype=torch.long,
            )
            window_offsets = (
                torch.arange(window_horizon, device=device, dtype=torch.long)
                - int(anchor_index)
            )
            return torch.clamp(
                chunk_steps[:, None] + window_offsets[None, :],
                min=0,
                max=chunk_horizon - 1,
            )

        def build_chunk_obs_windows(state_chunk: torch.Tensor) -> torch.Tensor:
            assert state_chunk.shape[-1] == self.obs_feature_dim, (
                "State feature dimension does not match robomimic obs encoder output: "
                f"got {state_chunk.shape[-1]}, expected {self.obs_feature_dim}."
            )
            indices = build_chunk_window_indices(
                chunk_horizon=state_chunk.shape[1],
                window_horizon=self.observation_horizon,
                anchor_index=self.observation_horizon,
                device=state_chunk.device,
            )
            return state_chunk[:, indices, :]

        def build_chunk_action_windows(action_chunk: torch.Tensor) -> torch.Tensor:
            assert action_chunk.shape[-1] == self.action_dim, (
                "Action dimension mismatch: "
                f"got {action_chunk.shape[-1]}, expected {self.action_dim}."
            )
            indices = build_chunk_window_indices(
                chunk_horizon=action_chunk.shape[1],
                window_horizon=self.prediction_horizon,
                anchor_index=self.action_start_index,
                device=action_chunk.device,
            )
            return action_chunk[:, indices, :]

        B, H, _ = state.shape
        obs_windows = build_chunk_obs_windows(state)
        action_windows = build_chunk_action_windows(action)
        obs_cond = rearrange(obs_windows, "b h to d -> (b h) (to d)")
        action_seq = rearrange(action_windows, "b h tp da -> (b h) tp da")
        timestep_tensor = self._guidance_timestep_tensor(batch_size=B * H)
        noise_pred = noise_pred_net(
            sample=action_seq,
            timestep=timestep_tensor,
            global_cond=obs_cond,
        )
        score_seq = self._score_from_noise_pred(
            noise_pred=noise_pred,
            timestep_tensor=timestep_tensor,
        )
        return rearrange(
            score_seq[:, self.action_start_index, :],
            "(b h) da -> b h da",
            b=B,
            h=H,
        )

    def _whole_action_chunk_scoring(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        prefix_states: Optional[torch.Tensor],
        prefix_actions: Optional[torch.Tensor],
        noise_pred_net: RobomimicConditionalUnet1D,
    ) -> torch.Tensor:
        """Score a chunk from one exact strict-past robomimic query.

        Input:
        - state: `[B, H, Dobs]`
        - action: `[B, H, Da]`
        - prefix_states: `[B, To, Dobs]`
        - prefix_actions: `[B, To, Da]` or `[B, 0, Da]` at rollout start
        
        Output:
        - score: `[B, H, Da]`

        Example with `To=2`, `Tp=6`, and `H=4`:
        - caller provides strict-past prefixes
          `prefix_states = [s_{t-2}, s_{t-1}]` and
          `prefix_actions = [a_{t-2}, a_{t-1}]`
        - SOPE provides the future chunk states
          `[s_t, s_{t+1}, s_{t+2}, s_{t+3}]` and
          future chunk actions `[a_t, a_{t+1}, a_{t+2}, a_{t+3}]`
        - robomimic is queried with
          `obs_window = [s_{t-2}, s_{t-1}]` and
          `action_window = [a_{t-2}, a_{t-1}, a_t, a_{t+1}, a_{t+2}, a_{t+3}]`
        - at trajectory start, an explicit empty `prefix_actions` tensor is the
          only special case: it is expanded to `[a_t, a_t]` so the full action
          window matches the rollout dataset edge-padding convention
        - robomimic predicts a score for the full length-`Tp` action window, and
          this helper returns the aligned length-`H` slice starting at
          index `To = 2`:
          `[a_t, a_{t+1}, a_{t+2}, a_{t+3}]`
        """
        B, H, _ = state.shape
        assert prefix_states is not None, "Exact chunk scoring requires prefix_states."
        assert prefix_actions is not None, "Exact chunk scoring requires prefix_actions."

        prefix_states_t = self._as_feature_sequence(
            prefix_states,
            feature_name="prefix_states",
            batch_size=B,
            feature_dim=self.obs_feature_dim,
            valid_horizons=(self.observation_horizon,),
        )
        prefix_actions_t = self._as_feature_sequence(
            prefix_actions,
            feature_name="prefix_actions",
            batch_size=B,
            feature_dim=self.action_dim,
            valid_horizons=(0, self.observation_horizon),
        )
        assert prefix_states_t.shape[1] == self.observation_horizon, (
            "Exact robomimic observation window must have length "
            f"{self.observation_horizon}, got {prefix_states_t.shape[1]}."
        )
        if prefix_actions_t.shape[1] == 0:
            self._warn_once(
                "empty_action_prefix",
                "WARNING: repeating the first chunk action across the strict-past "
                "action prefix so the off-by-one exact scorer matches rollout "
                "dataset edge padding at trajectory start.",
            )
            prefix_actions_t = action[:, :1, :].expand(-1, self.observation_horizon, -1)
        assert prefix_actions_t.shape[1] + action.shape[1] == self.prediction_horizon, (
            "Exact robomimic action window must have length "
            f"{self.prediction_horizon}, got {prefix_actions_t.shape[1] + action.shape[1]}."
        )

        obs_window = prefix_states_t  # (B, To, Dobs)
        action_seq = torch.cat([prefix_actions_t, action], dim=1)  # (B, Tp=To+H, Da)
        obs_cond = self._prepare_obs_cond(obs_window)
        timestep_tensor = self._guidance_timestep_tensor(batch_size=B)
        
        noise_pred = noise_pred_net(
            sample=action_seq,
            timestep=timestep_tensor,
            global_cond=obs_cond,
        )
        score_seq = self._score_from_noise_pred(
            noise_pred=noise_pred,
            timestep_tensor=timestep_tensor,
        )
        return score_seq[:, self.action_start_index:self.action_start_index + H, :]

    def grad_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        prefix_states: Optional[torch.Tensor] = None,
        prefix_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Return a chunk-shaped action score for SOPE chunk diffusion guidance.
        Inputs:
        - state: `[B, H, Dobs]`
        - action: `[B, H, Da]`
        - prefix_states: `[B, To, Dobs]`
        - prefix_actions: `[B, To, Da]` or `[B, 0, Da]` at rollout start
        Output:
        - score: `[B, H, Da]`

        `_whole_action_chunk_scoring(...)` is used only when all of the
        following hold:
        - `prediction_horizon == observation_horizon + H`
        - `prefix_states` is the strict-past state window
          `[s_{t-To}, ..., s_{t-1}]`
        - `prefix_actions` is the strict-past action window
          `[a_{t-To}, ..., a_{t-1}]`, except for the rollout-start special
          case where it may be empty and is expanded in-policy

        Example with `To=2`, `H=4`, and `Tp=6`:
        - exact scoring uses
          `obs_window = [s_{t-2}, s_{t-1}]`
          and
          `action_window = [a_{t-2}, a_{t-1}, a_t, a_{t+1}, a_{t+2}, a_{t+3}]`
        - the returned score slice is
          `[a_t, a_{t+1}, a_{t+2}, a_{t+3}]`

        When exact-path criteria fail, `grad_log_prob(...)` falls back to
        `_single_action_chunk_scoring(...)`. That includes missing strict-past
        prefixes and horizon mismatches where
        `prediction_horizon != observation_horizon + H`, as long as the
        surrogate path still has `prediction_horizon > observation_horizon`.
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
        nets = self._policy_nets()
        noise_pred_net: RobomimicConditionalUnet1D = nets["policy"]["noise_pred_net"]

        expected_prediction_horizon = int(self.observation_horizon + H)

        use_whole_action_chunk_scoring = (
            int(self.prediction_horizon) == expected_prediction_horizon
        )

        missing: list[str] = []
        if prefix_states is None: missing.append("prefix_states")
        if prefix_actions is None: missing.append("prefix_actions")
        use_whole_action_chunk_scoring &= not missing
        
        if use_whole_action_chunk_scoring:
            score = self._whole_action_chunk_scoring(
                state,
                action,
                prefix_states=prefix_states,
                prefix_actions=prefix_actions,
                noise_pred_net=noise_pred_net,
            )

        else:
            if int(self.prediction_horizon) != expected_prediction_horizon:
                self._warn_once(
                    "horizon_mismatch",
                    "WARNING: falling back to the off-by-one sliding-window scorer "
                    "because exact full-chunk scoring requires "
                    "prediction_horizon == observation_horizon + chunk_horizon; "
                    f"got prediction_horizon={self.prediction_horizon}, "
                    f"observation_horizon={self.observation_horizon}, "
                    f"chunk_horizon={H}.",
                )
            if missing:
                self._warn_once(
                    f"missing_{'_'.join(missing)}",
                    "WARNING: falling back to the off-by-one sliding-window scorer "
                    "because exact full-chunk scoring requires "
                    f"{', '.join(missing)}.",
                )
            score = self._single_action_chunk_scoring(
                state,
                action,
                noise_pred_net=noise_pred_net,
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
        prediction_horizon = observation_horizon + chunk_horizon
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
        prefix_states = torch.randn(
            batch_size,
            observation_horizon,
            obs_feature_dim,
            device=device,
        )
        prefix_actions = torch.randn(
            batch_size,
            observation_horizon,
            action_dim,
            device=device,
        )

        with torch.no_grad():
            exact_score = policy.grad_log_prob(
                state,
                action,
                prefix_states=prefix_states,
                prefix_actions=prefix_actions,
            )
            prefix_state_perturbed_score = policy.grad_log_prob(
                state,
                action,
                prefix_states=prefix_states + 0.05,
                prefix_actions=prefix_actions,
            )
            action_perturbed_score = policy.grad_log_prob(
                state,
                action + 0.05,
                prefix_states=prefix_states,
                prefix_actions=prefix_actions,
            )
            empty_prefix_score = policy.grad_log_prob(
                state,
                action,
                prefix_states=prefix_states,
                prefix_actions=action.new_empty((batch_size, 0, action_dim)),
            )

        fallback_policy = DiffusionPolicy(
            _ToyRobomimicDiffusionPolicy(
                device=device,
                obs_feature_dim=obs_feature_dim,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon + 2,
                action_dim=action_dim,
                num_train_timesteps=num_train_timesteps,
            ),
            config=DiffusionPolicyScoreConfig(score_timestep=1),
        )
        with torch.no_grad():
            fallback_score = fallback_policy.grad_log_prob(
                state,
                action,
            )
            fallback_state_perturbed_score = fallback_policy.grad_log_prob(
                state + 0.05,
                action,
            )

        assert exact_score.shape == (batch_size, chunk_horizon, action_dim), (
            "Toy grad_log_prob smoke test returned an unexpected shape: "
            f"got {tuple(exact_score.shape)}."
        )
        assert torch.isfinite(exact_score).all(), (
            "Toy grad_log_prob smoke test produced non-finite values on the exact path."
        )
        assert empty_prefix_score.shape == exact_score.shape, (
            "Toy grad_log_prob smoke test returned an unexpected shape for the "
            "rollout-start empty action-prefix case."
        )
        assert torch.isfinite(empty_prefix_score).all(), (
            "Toy grad_log_prob smoke test produced non-finite values for the "
            "rollout-start empty action-prefix case."
        )
        assert not torch.allclose(exact_score, prefix_state_perturbed_score), (
            "Toy grad_log_prob smoke test is unexpectedly insensitive to strict-past "
            "prefix-state perturbations on the exact path."
        )
        assert not torch.allclose(exact_score, action_perturbed_score), (
            "Toy grad_log_prob smoke test is unexpectedly insensitive to action "
            "perturbations on the exact path."
        )
        assert not torch.allclose(fallback_score, fallback_state_perturbed_score), (
            "Toy grad_log_prob smoke test is unexpectedly insensitive to chunk-state "
            "perturbations on the fallback sliding-window path."
        )

        print(
            "grad_log_prob smoke test passed:",
            f"device={device}",
            f"state_shape={tuple(state.shape)}",
            f"action_shape={tuple(action.shape)}",
            f"score_shape={tuple(exact_score.shape)}",
        )
        print("score[0, 0] =", exact_score[0, 0].detach().cpu())


    main()
