from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import einops

# Ensure `opelab` is importable when using third_party/sope in-place.
_SOPE_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "sope"
if str(_SOPE_ROOT) not in sys.path:
    sys.path.append(str(_SOPE_ROOT))

from third_party.sope.opelab.core.baselines.diffusion.temporal import TemporalUnet  # type: ignore
from third_party.sope.opelab.core.baselines.diffusion.diffusion import GaussianDiffusion  # type: ignore
from third_party.sope.opelab.core.baselines import diffuser

from src.latent_sope.robomimic_interface.dataset import (
    RolloutChunkDataset,
    RolloutChunkDatasetConfig,
)


# TODO: Port over `diffuser.generate_full_trajectory()` chunk-stitching loop (roll forward, update `conditions`, handle termination, track `end_indices`).
# TODO: Wire guidance hyperparameters (`action_scale`, `state_scale`, `use_adaptive`, `use_neg_grad`, `neg_grad_scale`, `normalize_grad`, `k_guide`, `use_action_grad_only`, `clamp`, `l_inf`, `ratio`) into `SopeDiffuser.sample(...)`.
# TODO: Implement `guided=True` end-to-end (configure `GaussianDiffusion` guidance inputs and pass through from config).
# TODO: Add optional action squashing support (`tanh_action`, `TanhBijector` / `gmode`) to match `Diffuser`.
# TODO: Add optional termination predicate handling for full-trajectory rollout.
# TODO: Add optional env-specific initial state generation (or accept initial state tensor input) to match `Diffuser.get_initial_state()`.

# TODO: understand how weighted loss in sope works

""" Utils for normalization / unnormalization """

@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray  # (D,)
    std: np.ndarray  # (D,)


def compute_chunk_stats(x: np.ndarray) -> NormalizationStats:
    """Compute feature-wise mean/std over (N, W, D) data."""
    assert x.ndim == 3, f"Expected (N, W, D), got {x.shape}"
    mean = x.mean(axis=(0, 1)).astype(np.float32)
    std = (x.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    return NormalizationStats(mean=mean, std=std)  # type: ignore

def make_normalizers(stats: Optional[NormalizationStats]) -> Tuple[Callable, Callable]:
    if stats is None:
        return (lambda x: x), (lambda x: x)

    mean = stats.mean
    std = stats.std

    def _norm(x):

        if torch.is_tensor(x):
            mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
            return (x - mean_t) / std_t
        return (x - mean) / std

    def _unnorm(x):
        if torch.is_tensor(x):
            mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
            return x * std_t + mean_t
        return x * std + mean

    return _norm, _unnorm


""" Diffuser (wrapps around TemporalUnet + GaussianDiffusion in third_party/sope)"""

@dataclass(frozen=True)
class SopeDiffusionConfig:
    """Configuration for SOPE-style trajectory chunk diffusion."""

    chunk_horizon: int = 8
    frame_stack: int = 2
    state_dim: int = 19
    action_dim: int = 7
    diffusion_steps: int = 256

    # TemporalUnet backbone
    dim_mults: Tuple[int, ...] = (1, 2)
    attention: bool = False

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
    
    @property
    def total_chunk_horizon(self) -> int:
        return self.chunk_horizon + self.frame_stack


class SopeDiffuser:
    """SOPE trajectory-chunk diffusion wrapper (TemporalUnet + GaussianDiffusion).

    This is intended for chunk diffusion over (z, a) sequences, where z can be
    low-dim concatenated obs or high-dim image embeddings.

    This class references the structure in third_party/sope/opelab/core/baselines/diffuser.py
    """

    def __init__(
        self,
        cfg: SopeDiffusionConfig,
        normalization_stats: Optional[NormalizationStats] = None,
        device: str = "cuda",
        policy: Optional[Any] = None,
        behavior_policy: Optional[Any] = None,
    ):
        """Initialize chunk diffusion components.

        Notes:
            self.model: TemporalUnet denoiser epsilon_theta over (B, W, transition_dim) chunks.
            self.diffusion: GaussianDiffusion wrapper with DDPM logic (q/p sampling, loss, conditioning).
            self.diffusion.policy: optional target policy used to guide sampling via score gradients.
            self.diffusion.behavior_policy: optional behavior policy for negative guidance / contrastive gradients.
        """

        self.cfg: SopeDiffusionConfig = cfg
        self.device = torch.device(device)
        self.state_dim = int(cfg.state_dim)
        self.action_dim = int(cfg.action_dim)
        self.transition_dim = self.state_dim + self.action_dim

        self.normalizer, self.unnormalizer = make_normalizers(normalization_stats)

        self.model = TemporalUnet(
            horizon=cfg.total_chunk_horizon,
            transition_dim=self.transition_dim,
            attention=cfg.attention,
            dim_mults=cfg.dim_mults,
        ).to(self.device)

        self.diffusion = GaussianDiffusion(
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

    def make_optimizer(self):
        return torch.optim.Adam(
            self.diffusion.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def make_cond(self, batch: dict):
        """Create SOPE-style conditioning dict from a batch (B, W, D).
        Inputs:
        - batch: tensor-valued dictionary containing
            - 'states_from': (B, frame_stack, state_dim)
            - 'states_to': (B, chunk_size + 1, state_dim)
            - 'actions_to': (B, chunk_size, action_dim)
        Outputs:
        - cond: dictionary containing
            - 0: (B, state_dim * frame_stack)
        """
        return {int(t): batch['states_from'][:,t,:] for t in range(self.cfg.frame_stack)}

    def loss(self, batch, cond: Optional[dict] = None):
        """Compute the loss for a batch of chunks.
        Inputs:
        - batch: tensor-valued dictionary containing
            - 'states_from': (B, frame_stack, state_dim)
            - 'states_to': (B, chunk_size + 1, state_dim)
            - 'actions_to': (B, chunk_size, action_dim)
        - cond: dictionary containing
            - 0: (B, state_dim * frame_stack)
        """
        cat_states_from = torch.cat([ batch["states_from"], batch["actions_from"]], dim=-1)
        cat_states_to = torch.cat([batch["states_to"][:,:-1,:],  batch["actions_to"]], dim=-1)
        x = torch.cat([cat_states_from, cat_states_to], dim=1) # cat along time axis
        cond = self.make_cond(batch)
        return self.diffusion.loss(x, cond)

    def sample(self, num_samples: int, cond=None, return_chain: bool = False, **kwargs):
        """Sample chunks from the diffusion model."""
        shape = (num_samples, self.cfg.chunk_horizon+self.cfg.frame_stack-1, self.transition_dim)
        return self.diffusion.conditional_sample(
            shape,
            cond,
            guided=self.cfg.guided,
            return_chain=return_chain,
            **kwargs,
        )


def cross_validate_configs(
    cfg_dataset: RolloutChunkDatasetConfig,
    cfg_diffusion: SopeDiffusionConfig,
):
    assert isinstance(cfg_dataset, RolloutChunkDatasetConfig), \
        f"cfg_dataset must be a RolloutChunkDatasetConfig, got {type(cfg_dataset)}"
    assert isinstance(cfg_diffusion, SopeDiffusionConfig), \
        f"cfg_diffusion must be a SopeDiffusionConfig, got {type(cfg_diffusion)}"

    assert (
        cfg_dataset.latents_dim == cfg_diffusion.state_dim
    ), f"Config mismatch: RolloutChunkDatasetConfig.latents_dim must equal SopeDiffusionConfig.latent_dim ({cfg_dataset.latents_dim * cfg_dataset.frame_stack} != {cfg_diffusion.state_dim})."

    assert cfg_dataset.action_dim == cfg_diffusion.action_dim,\
        f"Config mismatch: RolloutChunkDatasetConfig.action_dim must equal action_dim ({cfg_dataset.action_dim} != {cfg_diffusion.action_dim})."

    total_horizon = int(cfg_diffusion.chunk_horizon + cfg_diffusion.frame_stack)
    required_div = 2 ** (len(cfg_diffusion.dim_mults) - 1)
    if total_horizon % required_div != 0:
        raise ValueError(
            f"Invalid dim_mults for total_horizon={total_horizon}: "
            f"len(dim_mults)={len(cfg_diffusion.dim_mults)} requires divisibility by {required_div}. "
            "Adjust dim_mults or change chunk_horizon/frame_stack."
        )
