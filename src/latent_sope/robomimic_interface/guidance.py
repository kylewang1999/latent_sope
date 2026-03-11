"""Score-based guidance wrapper for robomimic DiffusionPolicyUNet.

Implements `grad_log_prob(states, actions)` by evaluating the policy's noise
prediction network at t=1 (near-clean), matching SOPE's approach for diffusion
policy guidance (see `third_party/sope/opelab/core/policy.py:923-928`).

The key insight: diffusion models don't have tractable log_prob, but guidance
only needs ∇_a log π(a|s). The noise prediction network estimates the score
∇_x log p(x), so we evaluate it at t≈0 and return -ε_pred / σ[t].

Important: robomimic's DiffusionPolicyUNet is a *sequence model* (ConditionalUnet1D)
that predicts noise over a (B, Tp=16, action_dim=7) action sequence, unlike SOPE's
single-step MLP. The scorer must feed coherent multi-step action sequences to get
meaningful scores — tiling a single action across all positions produces OOD input.

Usage:
    from src.latent_sope.robomimic_interface.guidance import RobomimicDiffusionScorer
    from src.latent_sope.robomimic_interface.checkpoints import load_checkpoint, build_algo_from_checkpoint

    ckpt = load_checkpoint(run_dir)
    algo = build_algo_from_checkpoint(ckpt, device="cuda")
    scorer = RobomimicDiffusionScorer(algo, device="cuda")

    # Chunk-level scoring (preferred — single UNet call per batch):
    grad = scorer.grad_log_prob_chunk(states, actions)  # (B, T, action_dim)

    # Single-timestep scoring (delegates to chunk version):
    grad = scorer.grad_log_prob(states, actions)  # (N, action_dim)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.latent_sope.robomimic_interface.encoders import LowDimConcatEncoder


def _squaredcos_cap_v2_alpha_bar(t: float) -> float:
    """Compute alpha_bar for the squaredcos_cap_v2 schedule (from diffusers)."""
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


class RobomimicDiffusionScorer:
    """Wraps a robomimic DiffusionPolicyUNet to provide `grad_log_prob()`.

    This extracts the score function by evaluating the noise prediction network
    at a near-clean timestep, following SOPE's DiffusionPolicy approach.

    The scorer handles:
    - Reconstructing obs dicts from flat state vectors (via LowDimConcatEncoder)
    - Frame stacking for observation conditioning (using actual different states)
    - Building coherent action sequences for the temporal UNet
    - Computing sigma from the DDPM beta schedule

    Key difference from SOPE's DiffusionPolicy: SOPE uses a single-step MLP
    (PearceMlp, input (B, act_dim)), while robomimic uses a temporal UNet
    (ConditionalUnet1D, input (B, Tp=16, act_dim)). We must feed the UNet a
    coherent multi-step action sequence, not tile a single action.
    """

    def __init__(
        self,
        algo: Any,
        device: str = "cuda",
        score_timestep: int = 1,
        obs_keys: Optional[list] = None,
    ):
        """
        Args:
            algo: robomimic PolicyAlgo (DiffusionPolicyUNet) with loaded weights.
            device: torch device.
            score_timestep: diffusion timestep at which to evaluate the score.
                t=1 means near-clean (minimal noise), matching SOPE's approach.
            obs_keys: observation keys in sorted order matching the latent layout.
                If None, uses the default Lift low-dim keys.
        """
        self.device = torch.device(device)
        self.algo = algo
        self.score_timestep = score_timestep

        # Extract key components
        self.obs_encoder = algo.nets["policy"]["obs_encoder"]
        self.noise_pred_net = algo.nets["policy"]["noise_pred_net"]
        self.noise_scheduler = algo.noise_scheduler

        # Policy dimensions
        self.action_dim = algo.ac_dim
        self.prediction_horizon = algo.algo_config.horizon.prediction_horizon
        self.observation_horizon = algo.algo_config.horizon.observation_horizon

        # Set up encoder for state → obs dict conversion
        if obs_keys is None:
            obs_keys = sorted(algo.obs_shapes.keys())
        self.obs_keys = obs_keys
        self.encoder = LowDimConcatEncoder(obs_keys=obs_keys)
        # Initialize obs_shapes/obs_dims from algo
        self.encoder.obs_shapes = {k: tuple(algo.obs_shapes[k]) for k in obs_keys}
        self.encoder.obs_dims = {
            k: int(np.prod(algo.obs_shapes[k])) for k in obs_keys
        }
        self.state_dim = sum(self.encoder.obs_dims.values())

        # Precompute sigma at the score timestep
        self.sigma = self._compute_sigma(score_timestep)

        # Get obs normalization stats from algo if available
        self.obs_norm_stats = getattr(algo, "obs_normalization_stats", None)

        # The first "future" action position in robomimic's prediction horizon.
        # Positions 0..To-2 overlap with the observation window; To-1 onward are
        # future actions. See _get_action_trajectory line 365: start = To - 1.
        self._action_start = self.observation_horizon - 1

    def _compute_sigma(self, t: int) -> float:
        """Compute sigma[t] = sqrt(1 - alpha_bar[t]) from the noise schedule."""
        # The noise scheduler stores alphas_cumprod
        if hasattr(self.noise_scheduler, "alphas_cumprod"):
            alpha_bar_t = self.noise_scheduler.alphas_cumprod[t].item()
        else:
            # Fallback: compute from squaredcos_cap_v2 schedule
            num_steps = self.noise_scheduler.config.num_train_timesteps
            alpha_bar_t = (
                _squaredcos_cap_v2_alpha_bar(t / num_steps)
                / _squaredcos_cap_v2_alpha_bar(0)
            )
        return math.sqrt(1.0 - alpha_bar_t)

    def _states_to_obs_dict_batched(
        self, states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Convert flat state vectors to robomimic obs dict, preserving all dims.

        Args:
            states: (..., state_dim) flat state vectors with arbitrary leading dims.

        Returns:
            obs_dict: {key: (..., *shape)} tensor dict matching leading dims.
        """
        leading_shape = states.shape[:-1]
        D = states.shape[-1]
        # Flatten leading dims → (N, D)
        states_2d = states.reshape(-1, D)

        obs_dict = {}
        offset = 0
        for k in self.obs_keys:
            dim = self.encoder.obs_dims[k]
            shape = self.encoder.obs_shapes[k]
            vals = states_2d[:, offset : offset + dim]
            # Reshape to (N, *shape) then restore leading dims
            vals = vals.reshape(-1, *shape)
            vals = vals.reshape(*leading_shape, *shape)
            obs_dict[k] = vals
            offset += dim

        return obs_dict

    def _encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations through the policy's obs encoder.

        Args:
            obs_dict: {key: (B, To, *shape)} observation tensors with time dim.

        Returns:
            obs_cond: (B, To * obs_features) flattened conditioning vector.
        """
        import robomimic.utils.tensor_utils as TensorUtils

        inputs = {"obs": obs_dict, "goal": None}
        obs_features = TensorUtils.time_distributed(
            inputs, self.obs_encoder, inputs_as_kwargs=True
        )
        # obs_features: (B, To, obs_dim) → flatten to (B, To * obs_dim)
        return obs_features.flatten(start_dim=1)

    def _build_obs_conditioning(
        self, states: torch.Tensor
    ) -> torch.Tensor:
        """Build obs conditioning from chunk states with proper frame stacking.

        Uses the first To states from the chunk as the observation frames,
        giving the encoder two *different* states (previous + current) rather
        than duplicating a single state.

        Args:
            states: (B, T, state_dim) chunk states. T must be >= 1.
                If T >= To, uses states[:, :To].
                If T < To, pads by repeating the first state.

        Returns:
            obs_cond: (B, To * obs_features) flattened conditioning vector.
        """
        B, T, _ = states.shape
        To = self.observation_horizon

        if T >= To:
            obs_states = states[:, :To]  # (B, To, state_dim)
        else:
            # Pad: repeat first state to fill missing frames
            n_pad = To - T
            pad = states[:, :1].expand(-1, n_pad, -1)
            obs_states = torch.cat([pad, states], dim=1)  # (B, To, state_dim)

        # Convert to obs dict with time dimension: {key: (B, To, *shape)}
        obs_dict = self._states_to_obs_dict_batched(obs_states)
        return self._encode_obs(obs_dict)

    def _build_action_sequence(
        self, actions: torch.Tensor
    ) -> torch.Tensor:
        """Build a Tp-length action sequence with chunk actions at correct positions.

        Robomimic's ConditionalUnet1D expects (B, Tp=16, action_dim). The first
        To-1 positions overlap with the observation window; positions To-1 onward
        are future actions. Our chunk actions map to positions
        [action_start, action_start + T_chunk).

        Remaining positions are repeat-padded from the nearest chunk action
        (much better than zero-padding or tiling a single action).

        Args:
            actions: (B, T_chunk, action_dim) chunk actions.

        Returns:
            action_seq: (B, Tp, action_dim) full action sequence.
        """
        B, T_chunk, _ = actions.shape
        Tp = self.prediction_horizon
        start = self._action_start

        action_seq = torch.zeros(B, Tp, self.action_dim, device=self.device)

        # Place chunk actions at the correct positions
        end = min(start + T_chunk, Tp)
        n_placed = end - start
        action_seq[:, start:end] = actions[:, :n_placed]

        # Repeat-pad before start with first chunk action
        if start > 0:
            action_seq[:, :start] = actions[:, :1].expand(-1, start, -1)

        # Repeat-pad after end with last placed chunk action
        if end < Tp:
            action_seq[:, end:] = actions[:, n_placed - 1 : n_placed].expand(
                -1, Tp - end, -1
            )

        return action_seq

    @torch.no_grad()
    def grad_log_prob_chunk(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute ∇_a log π(a|s) for a chunk using a single UNet forward pass.

        This is the primary scoring method. It feeds the UNet a coherent
        multi-step action sequence (the actual chunk actions, properly placed
        in the Tp-length prediction horizon) and extracts per-timestep scores
        at the corresponding positions.

        Args:
            states: (B, T, state_dim) states for each timestep in chunk.
            actions: (B, T, action_dim) actions for each timestep in chunk.

        Returns:
            grad: (B, T, action_dim) per-timestep gradient of log-prob.
        """
        B, T, _ = states.shape
        states = states.to(self.device)
        actions = actions.to(self.device)

        # 1. Build obs conditioning from first To chunk states (proper frame stacking)
        obs_cond = self._build_obs_conditioning(states)  # (B, To * obs_features)

        # 2. Build Tp-length action sequence with chunk actions at correct positions
        action_seq = self._build_action_sequence(actions)  # (B, Tp, action_dim)

        # 3. Single UNet forward pass at score_timestep
        t = torch.full((B,), self.score_timestep, device=self.device, dtype=torch.long)
        noise_pred = self.noise_pred_net(
            sample=action_seq, timestep=t, global_cond=obs_cond
        )
        # noise_pred: (B, Tp, action_dim)

        # 4. Extract scores at chunk positions and convert to score
        start = self._action_start
        end = min(start + T, self.prediction_horizon)
        n_extracted = end - start
        scores = -noise_pred[:, start:end, :] / self.sigma  # (B, n_extracted, action_dim)

        # If chunk is longer than available positions, pad with zeros
        if n_extracted < T:
            zeros = torch.zeros(B, T - n_extracted, self.action_dim, device=self.device)
            scores = torch.cat([scores, zeros], dim=1)

        return scores

    @torch.no_grad()
    def grad_log_prob(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute ∇_a log π(a|s) for single (state, action) pairs.

        Delegates to grad_log_prob_chunk by adding a time dimension.
        For batch scoring of chunks, prefer grad_log_prob_chunk directly.

        Args:
            states: (N, state_dim) flat state vectors.
            actions: (N, action_dim) single-timestep actions.

        Returns:
            grad: (N, action_dim) gradient of log-prob w.r.t. actions.
        """
        # Add time dimension → (N, 1, D), score, remove time dimension
        return self.grad_log_prob_chunk(
            states.unsqueeze(1), actions.unsqueeze(1)
        ).squeeze(1)
