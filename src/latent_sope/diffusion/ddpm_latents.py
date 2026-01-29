"""DDPM over fixed-length trajectory chunks (latent or low-dim).

Design goals:
- Keep the DDPM logic separate from robomimic (data) concerns.
- Use Hugging Face diffusers when available (recommended).
- Provide a small fallback implementation for environments where diffusers is
  unavailable.

Expected data shape convention:
- Training examples are chunks x0 with shape (W, D).
- We feed the model tensors as (B, C, W) where C == D.
  (Channels-first is standard for diffusers' UNet1DModel.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch.nn as nn


@dataclass(frozen=True)
class DDPMConfig:
    """Configuration for chunk-level DDPM."""

    W: int
    D: int
    num_train_timesteps: int = 1000

    # model arch
    base_channels: int = 128
    layers_per_block: int = 2

    # optimization
    lr: float = 1e-4
    weight_decay: float = 0.0


def _require_torch():
    try:
        import torch

        return torch
    except Exception as e:
        raise ImportError("PyTorch is required for DDPM modules") from e


def _require_sope():
    """Ensure opelab (SOPE) is importable and return needed classes."""
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    sope_root = repo_root / "third_party" / "sope"
    if str(sope_root) not in sys.path:
        sys.path.append(str(sope_root))

    from opelab.core.baselines.diffusion.temporal import TemporalUnet  # type: ignore
    from opelab.core.baselines.diffusion.diffusion import GaussianDiffusion  # type: ignore

    return TemporalUnet, GaussianDiffusion


class ChunkDenoiser:
    """A denoiser epsilon_theta(x_t, t) for (B, C, W) chunk tensors."""

    def __init__(self, cfg: DDPMConfig):
        torch = _require_torch()
        

        self.cfg = cfg
        self._torch = torch
        self._nn = nn

        # Prefer diffusers UNet1DModel if installed.
        self._use_diffusers = False
        self.unet = None

        try:
            from diffusers import UNet1DModel  # type: ignore

            # UNet1DModel expects:
            #   sample: (B, in_channels, sample_size)
            #   timestep: int or (B,)
            # and returns an object with field `.sample`.
            self.unet = UNet1DModel(
                sample_size=cfg.W,
                in_channels=cfg.D,
                out_channels=cfg.D,
                layers_per_block=cfg.layers_per_block,
                block_out_channels=(cfg.base_channels, cfg.base_channels * 2, cfg.base_channels * 2),
                down_block_types=("DownBlock1D", "DownBlock1D", "AttnDownBlock1D"),
                up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1D"),
            )
            self._use_diffusers = True
        except Exception:
            # Fallback: small Conv1D network + timestep embedding.
            self.unet = None
            self._use_diffusers = False

            class _Fallback(nn.Module):
                def __init__(self, W: int, C: int, hidden: int):
                    super().__init__()
                    self.time_mlp = nn.Sequential(
                        nn.Embedding(2048, hidden),
                        nn.SiLU(),
                        nn.Linear(hidden, hidden),
                    )
                    self.net = nn.Sequential(
                        nn.Conv1d(C, hidden, kernel_size=3, padding=1),
                        nn.SiLU(),
                        nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                        nn.SiLU(),
                        nn.Conv1d(hidden, C, kernel_size=3, padding=1),
                    )

                def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                    # x: (B, C, W), t: (B,)
                    # Add timestep embedding as a channel-wise bias.
                    emb = self.time_mlp(t.clamp(0, 2047))  # (B, hidden)
                    # broadcast to (B, hidden, W)
                    emb = emb[..., None]
                    h = self.net[0](x)
                    h = h + emb
                    # finish remaining layers manually to preserve emb addition semantics
                    for layer in list(self.net)[1:]:
                        h = layer(h)
                    return h

            self.fallback = _Fallback(cfg.W, cfg.D, cfg.base_channels)

    # Make the object behave like an nn.Module without requiring inheritance.
    def parameters(self):
        if self._use_diffusers:
            return self.unet.parameters()
        return self.fallback.parameters()

    def state_dict(self):
        if self._use_diffusers:
            return {"use_diffusers": True, "unet": self.unet.state_dict(), "cfg": self.cfg.__dict__}
        return {"use_diffusers": False, "fallback": self.fallback.state_dict(), "cfg": self.cfg.__dict__}

    def load_state_dict(self, state: dict):
        use_diffusers = bool(state.get("use_diffusers", False))
        if use_diffusers and self._use_diffusers:
            self.unet.load_state_dict(state["unet"])
            return
        if (not use_diffusers) and (not self._use_diffusers):
            self.fallback.load_state_dict(state["fallback"])
            return
        raise ValueError(
            "Checkpoint/model mismatch. If you trained with diffusers, you need diffusers installed to load."
        )

    def to(self, device: str):
        if self._use_diffusers:
            self.unet.to(device)
        else:
            self.fallback.to(device)
        return self

    def train(self):
        if self._use_diffusers:
            self.unet.train()
        else:
            self.fallback.train()

    def eval(self):
        if self._use_diffusers:
            self.unet.eval()
        else:
            self.fallback.eval()

    def __call__(self, x_t, t):
        """Forward pass.

        Args:
            x_t: (B, C, W)
            t: (B,) int64 timesteps

        Returns:
            predicted noise epsilon: (B, C, W)
        """
        torch = self._torch
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x_t.device, dtype=torch.long)
        if t.ndim == 0:
            t = t[None]
        if self._use_diffusers:
            out = self.unet(x_t, t)
            return out.sample
        return self.fallback(x_t, t)


def make_scheduler(cfg: DDPMConfig):
    """Return a noise scheduler compatible with diffusers training loops."""
    try:
        from diffusers import DDPMScheduler  # type: ignore

        return DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
    except Exception as e:
        raise ImportError(
            "diffusers is required for scheduler. Install with `pip install diffusers` (and accelerate if needed)."
        ) from e


def sample_ddpm(
    denoiser: ChunkDenoiser,
    scheduler,
    num_samples: int,
    cfg: DDPMConfig,
    device: str = "cuda",
    num_inference_steps: Optional[int] = None,
) -> np.ndarray:
    """Unconditional sampling from the learned DDPM.

    Returns:
        x0 samples as numpy array of shape (N, W, D).
    """

    torch = _require_torch()
    denoiser.eval()
    denoiser.to(device)

    if num_inference_steps is None:
        num_inference_steps = cfg.num_train_timesteps

    scheduler.set_timesteps(num_inference_steps)

    x = torch.randn((num_samples, cfg.D, cfg.W), device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            eps = denoiser(x, t)
            step_out = scheduler.step(eps, t, x)
            x = step_out.prev_sample

    # (N, C, W) -> (N, W, D)
    x0 = x.permute(0, 2, 1).contiguous().cpu().numpy()
    return x0.astype(np.float32)


def reconstruct_from_noised(
    x0: np.ndarray,
    denoiser: ChunkDenoiser,
    scheduler,
    cfg: DDPMConfig,
    t_start: int,
    device: str = "cuda",
) -> np.ndarray:
    """Denoise a noised version of a ground-truth chunk.

    Useful for evaluation: take x0, add noise at t_start, then run reverse
    process down to 0 and compare reconstructed x0_hat to x0.

    Args:
        x0: (N, W, D) or (W, D)
        t_start: int in [0, cfg.num_train_timesteps-1]

    Returns:
        x0_hat: same shape as x0
    """

    torch = _require_torch()

    x0_np = np.asarray(x0, dtype=np.float32)
    single = False
    if x0_np.ndim == 2:
        x0_np = x0_np[None, ...]
        single = True

    if x0_np.ndim != 3 or x0_np.shape[1] != cfg.W or x0_np.shape[2] != cfg.D:
        raise ValueError(f"Expected (N, {cfg.W}, {cfg.D}), got {x0_np.shape}")

    t_start = int(t_start)
    if not (0 <= t_start < cfg.num_train_timesteps):
        raise ValueError(f"t_start must be in [0, {cfg.num_train_timesteps-1}]")

    denoiser.eval()
    denoiser.to(device)

    # Convert to torch (B, C, W)
    x0_t = torch.from_numpy(x0_np).to(device=device)
    x0_t = x0_t.permute(0, 2, 1).contiguous()

    noise = torch.randn_like(x0_t)
    t = torch.full((x0_t.shape[0],), t_start, device=device, dtype=torch.long)
    x = scheduler.add_noise(x0_t, noise, t)

    # Use full scheduler timesteps but skip those > t_start
    scheduler.set_timesteps(cfg.num_train_timesteps)

    with torch.no_grad():
        for tt in scheduler.timesteps:
            if int(tt) > t_start:
                continue
            eps = denoiser(x, tt)
            step_out = scheduler.step(eps, tt, x)
            x = step_out.prev_sample

    x_hat = x.permute(0, 2, 1).contiguous().cpu().numpy().astype(np.float32)
    if single:
        return x_hat[0]
    return x_hat


@dataclass(frozen=True)
class SopeChunkDiffusionConfig:
    """Configuration for SOPE-style trajectory chunk diffusion."""

    horizon: int
    obs_dim: int
    action_dim: int
    diffusion_steps: int = 256

    # TemporalUnet backbone
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
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


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray  # (D,)
    std: np.ndarray  # (D,)


def compute_chunk_stats(x: np.ndarray) -> NormalizationStats:
    """Compute feature-wise mean/std over (N, W, D) data."""
    if x.ndim != 3:
        raise ValueError(f"Expected (N, W, D), got {x.shape}")
    mean = x.mean(axis=(0, 1)).astype(np.float32)
    std = (x.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    return NormalizationStats(mean=mean, std=std)


def _make_normalizer(stats: Optional[NormalizationStats]) -> Tuple[Callable, Callable]:
    if stats is None:
        return (lambda x: x), (lambda x: x)

    mean = stats.mean
    std = stats.std

    def _norm(x):
        import torch

        if torch.is_tensor(x):
            mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
            return (x - mean_t) / std_t
        return (x - mean) / std

    def _unnorm(x):
        import torch

        if torch.is_tensor(x):
            mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
            return x * std_t + mean_t
        return x * std + mean

    return _norm, _unnorm


class SopeChunkDiffusion:
    """SOPE trajectory-chunk diffusion wrapper (TemporalUnet + GaussianDiffusion).

    This is intended for chunk diffusion over (z, a) sequences, where z can be
    low-dim concatenated obs or high-dim image embeddings.
    """

    def __init__(
        self,
        cfg: SopeChunkDiffusionConfig,
        normalization_stats: Optional[NormalizationStats] = None,
        device: str = "cuda",
        policy: Optional[Any] = None,
        behavior_policy: Optional[Any] = None,
    ):
        torch = _require_torch()
        TemporalUnet, GaussianDiffusion = _require_sope()

        self.cfg = cfg
        self.device = torch.device(device)
        self.obs_dim = int(cfg.obs_dim)
        self.action_dim = int(cfg.action_dim)
        self.transition_dim = self.obs_dim + self.action_dim

        self.normalizer, self.unnormalizer = _make_normalizer(normalization_stats)

        self.model = TemporalUnet(
            horizon=cfg.horizon,
            transition_dim=self.transition_dim,
            attention=cfg.attention,
            dim_mults=cfg.dim_mults,
        ).to(self.device)

        self.diffusion = GaussianDiffusion(
            model=self.model,
            horizon=cfg.horizon,
            observation_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_timesteps=cfg.diffusion_steps,
            normalizer=self.normalizer,
            unnormalizer=self.unnormalizer,
            action_weight=cfg.action_weight,
            loss_discount=cfg.loss_discount,
            loss_type=cfg.loss_type,
            predict_epsilon=cfg.predict_epsilon,
        ).to(self.device)

        if policy is not None:
            self.set_policy(policy, behavior_policy)

    def set_policy(self, policy: Any, behavior_policy: Optional[Any] = None) -> None:
        """Attach policy objects for guidance (if supported by SOPE guidance code)."""
        self.diffusion.policy = policy
        self.diffusion.behavior_policy = behavior_policy

    def make_optimizer(self):
        torch = _require_torch()
        return torch.optim.Adam(
            self.diffusion.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def make_cond(self, batch: "np.ndarray | Any", cond_on_first: bool = True):
        """Create SOPE-style conditioning dict from a batch (B, W, D)."""
        torch = _require_torch()
        if not torch.is_tensor(batch):
            batch = torch.as_tensor(batch, device=self.device, dtype=torch.float32)
        if not cond_on_first:
            return None
        cond = batch[:, 0, : self.obs_dim]
        return {0: cond}

    def loss(self, batch, cond=None):
        torch = _require_torch()
        if not torch.is_tensor(batch):
            batch = torch.as_tensor(batch, device=self.device, dtype=torch.float32)
        return self.diffusion.loss(batch, cond)

    def sample(self, num_samples: int, cond=None, return_chain: bool = False, **kwargs):
        """Sample chunks from the diffusion model."""
        shape = (num_samples, self.cfg.horizon, self.transition_dim)
        return self.diffusion.conditional_sample(
            shape,
            cond,
            guided=self.cfg.guided,
            return_chain=return_chain,
            **kwargs,
        )
