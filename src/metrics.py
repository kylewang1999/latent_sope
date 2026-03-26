"""Evaluation metrics for trajectory chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class L2ChunkError:
    """Summary statistics for chunk reconstruction error."""

    mean_l2: float
    std_l2: float
    rmse_per_dim: np.ndarray  # (D,)


def l2_chunk_error(x_hat: np.ndarray, x_gt: np.ndarray) -> L2ChunkError:
    """Compute per-chunk L2 error and per-dimension RMSE.

    Args:
        x_hat: predicted chunk array of shape (N, W, D) or (W, D)
        x_gt: ground truth chunk array with same shape

    Returns:
        L2ChunkError
    """

    x_hat = np.asarray(x_hat, dtype=np.float32)
    x_gt = np.asarray(x_gt, dtype=np.float32)

    if x_hat.shape != x_gt.shape:
        raise ValueError(f"shape mismatch: x_hat={x_hat.shape}, x_gt={x_gt.shape}")

    # Promote single chunk to batch
    if x_hat.ndim == 2:
        x_hat = x_hat[None, ...]
        x_gt = x_gt[None, ...]

    if x_hat.ndim != 3:
        raise ValueError(f"expected (N, W, D) or (W, D); got {x_hat.shape}")

    diff = x_hat - x_gt
    N, W, D = diff.shape

    per = np.linalg.norm(diff.reshape(N, W * D), axis=1)

    # RMSE per feature-dimension, aggregating over batch and time
    rmse_per_dim = np.sqrt(np.mean(diff ** 2, axis=(0, 1)))

    return L2ChunkError(
        mean_l2=float(per.mean()),
        std_l2=float(per.std()),
        rmse_per_dim=rmse_per_dim.astype(np.float32),
    )
