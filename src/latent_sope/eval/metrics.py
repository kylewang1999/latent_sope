"""Evaluation metrics for trajectory chunks and OPE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as scipy_stats


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


# ─── Trajectory Reconstruction Metrics ───────────────────────────────────────


@dataclass(frozen=True)
class TrajectoryReconstructionError:
    """MSE between real and generated full-length trajectories."""

    state_mse: float          # MSE over state dimensions
    action_mse: float         # MSE over action dimensions
    state_mse_per_dim: np.ndarray   # (state_dim,) per-feature MSE
    action_mse_per_dim: np.ndarray  # (action_dim,) per-feature MSE
    state_mse_per_step: np.ndarray  # (T,) MSE at each timestep (avg over batch & dims)
    action_mse_per_step: np.ndarray # (T,) MSE at each timestep
    num_trajectories: int
    trajectory_length: int


def trajectory_reconstruction_mse(
    real_states: np.ndarray,
    gen_states: np.ndarray,
    real_actions: np.ndarray,
    gen_actions: np.ndarray,
    end_indices: Optional[np.ndarray] = None,
) -> TrajectoryReconstructionError:
    """Compute MSE between real and generated trajectories.

    Compares generated (stitched) trajectories against the real offline
    trajectories they were seeded from, to measure reconstruction quality.

    Args:
        real_states: (B, T, state_dim) real trajectory states.
        gen_states: (B, T, state_dim) generated trajectory states.
        real_actions: (B, T, action_dim) real trajectory actions.
        gen_actions: (B, T, action_dim) generated trajectory actions.
        end_indices: optional (B,) per-trajectory lengths. If provided,
            only steps up to end_indices[i] are used for trajectory i.

    Returns:
        TrajectoryReconstructionError with aggregate and per-dim/per-step MSE.
    """
    real_states = np.asarray(real_states, dtype=np.float32)
    gen_states = np.asarray(gen_states, dtype=np.float32)
    real_actions = np.asarray(real_actions, dtype=np.float32)
    gen_actions = np.asarray(gen_actions, dtype=np.float32)

    if real_states.shape != gen_states.shape:
        raise ValueError(
            f"State shape mismatch: real={real_states.shape}, gen={gen_states.shape}"
        )
    if real_actions.shape != gen_actions.shape:
        raise ValueError(
            f"Action shape mismatch: real={real_actions.shape}, gen={gen_actions.shape}"
        )

    B, T, Ds = real_states.shape
    Da = real_actions.shape[2]

    if end_indices is not None:
        end_indices = np.asarray(end_indices, dtype=np.int64)
        # Build a mask: (B, T) where mask[i, t] = 1 if t < end_indices[i]
        mask = np.arange(T)[None, :] < end_indices[:, None]  # (B, T)
    else:
        mask = np.ones((B, T), dtype=bool)

    mask_expanded_s = mask[..., None]  # (B, T, 1)
    mask_expanded_a = mask[..., None]

    # Masked squared errors
    state_sq_err = (real_states - gen_states) ** 2 * mask_expanded_s
    action_sq_err = (real_actions - gen_actions) ** 2 * mask_expanded_a
    n_valid = mask.sum()

    # Aggregate MSE
    state_mse = float(state_sq_err.sum() / (n_valid * Ds))
    action_mse = float(action_sq_err.sum() / (n_valid * Da))

    # Per-dimension MSE: average over batch and time (masked)
    state_mse_per_dim = state_sq_err.sum(axis=(0, 1)) / n_valid  # (Ds,)
    action_mse_per_dim = action_sq_err.sum(axis=(0, 1)) / n_valid  # (Da,)

    # Per-step MSE: average over batch and dims (masked)
    n_valid_per_step = mask.sum(axis=0)  # (T,)
    n_valid_per_step = np.maximum(n_valid_per_step, 1)  # avoid div by zero
    state_mse_per_step = state_sq_err.mean(axis=2).sum(axis=0) / n_valid_per_step  # (T,)
    action_mse_per_step = action_sq_err.mean(axis=2).sum(axis=0) / n_valid_per_step  # (T,)

    return TrajectoryReconstructionError(
        state_mse=state_mse,
        action_mse=action_mse,
        state_mse_per_dim=state_mse_per_dim.astype(np.float32),
        action_mse_per_dim=action_mse_per_dim.astype(np.float32),
        state_mse_per_step=state_mse_per_step.astype(np.float32),
        action_mse_per_step=action_mse_per_step.astype(np.float32),
        num_trajectories=B,
        trajectory_length=T,
    )


# ─── OPE Evaluation Metrics ──────────────────────────────────────────────────


@dataclass(frozen=True)
class OPEResult:
    """Result of comparing an OPE estimate to oracle ground truth."""

    oracle_value: float
    ope_estimate: float
    mse: float
    relative_error: float
    ope_std: float  # std across synthetic trajectory returns


def ope_eval(
    oracle_value: float,
    synthetic_returns: np.ndarray,
) -> OPEResult:
    """Evaluate an OPE estimate against oracle ground truth.

    Args:
        oracle_value: true policy value from Step 0.
        synthetic_returns: (N,) array of returns from scored synthetic trajectories.

    Returns:
        OPEResult with MSE, relative error, etc.
    """
    synthetic_returns = np.asarray(synthetic_returns, dtype=np.float64)
    ope_estimate = float(synthetic_returns.mean())
    mse = float((ope_estimate - oracle_value) ** 2)
    relative_error = float(abs(ope_estimate - oracle_value) / max(abs(oracle_value), 1e-8))
    return OPEResult(
        oracle_value=float(oracle_value),
        ope_estimate=ope_estimate,
        mse=mse,
        relative_error=relative_error,
        ope_std=float(synthetic_returns.std()),
    )


# ─── Multi-Policy OPE Metrics ───────────────────────────────────────────────


@dataclass(frozen=True)
class MultiPolicyOPEResult:
    """Result of evaluating OPE estimates across multiple policies."""

    oracle_values: np.ndarray        # (P,) true values per policy
    ope_estimates: np.ndarray        # (P,) OPE estimates per policy
    per_policy: List[OPEResult]      # individual OPEResult per policy

    # Aggregate metrics
    mean_mse: float                  # mean MSE across policies
    mean_relative_error: float       # mean relative error across policies

    # Rank correlation
    spearman_rho: float              # Spearman rank correlation
    spearman_pvalue: float           # p-value for Spearman test

    # Regret metrics
    regret_at_1: float               # V*(best true) - V*(best estimated)
    regret_at_k: Dict[int, float]    # k -> regret for top-k selection


def spearman_rank_correlation(
    oracle_values: Sequence[float],
    ope_estimates: Sequence[float],
) -> tuple[float, float]:
    """Spearman rank correlation between oracle values and OPE estimates.

    Args:
        oracle_values: (P,) true policy values.
        ope_estimates: (P,) estimated policy values.

    Returns:
        (rho, p_value). rho in [-1, 1]; 1 = perfect rank agreement.
    """
    oracle_values = np.asarray(oracle_values, dtype=np.float64)
    ope_estimates = np.asarray(ope_estimates, dtype=np.float64)
    if len(oracle_values) < 3:
        return float("nan"), float("nan")
    result = scipy_stats.spearmanr(oracle_values, ope_estimates)
    return float(result.correlation), float(result.pvalue)


def regret_at_k(
    oracle_values: Sequence[float],
    ope_estimates: Sequence[float],
    k: int = 1,
) -> float:
    """Regret@k: how much worse the top-k estimated policies are vs the true top-k.

    Regret@k = mean(top-k true values) - mean(true values of top-k by estimate).

    Args:
        oracle_values: (P,) true policy values.
        ope_estimates: (P,) estimated policy values.
        k: number of top policies to select.

    Returns:
        Regret (>= 0 if OPE ranking is imperfect, 0 if perfect).
    """
    oracle_values = np.asarray(oracle_values, dtype=np.float64)
    ope_estimates = np.asarray(ope_estimates, dtype=np.float64)
    P = len(oracle_values)
    k = min(k, P)

    # True top-k
    true_top_k_idx = np.argsort(oracle_values)[-k:]
    true_top_k_mean = oracle_values[true_top_k_idx].mean()

    # Top-k by OPE estimate
    est_top_k_idx = np.argsort(ope_estimates)[-k:]
    est_top_k_true_mean = oracle_values[est_top_k_idx].mean()

    return float(true_top_k_mean - est_top_k_true_mean)


def multi_policy_ope_eval(
    oracle_values: Sequence[float],
    ope_estimates: Sequence[float],
    synthetic_returns_per_policy: Optional[Sequence[np.ndarray]] = None,
    k_values: Sequence[int] = (1, 3, 5),
) -> MultiPolicyOPEResult:
    """Evaluate OPE estimates across multiple policies.

    Args:
        oracle_values: (P,) true policy values.
        ope_estimates: (P,) OPE estimates (mean returns from synthetic trajectories).
        synthetic_returns_per_policy: optional list of (N_i,) arrays of per-trajectory
            returns for each policy (used for per-policy std). If None, std is set to 0.
        k_values: tuple of k values for Regret@k computation.

    Returns:
        MultiPolicyOPEResult with all metrics.
    """
    oracle_arr = np.asarray(oracle_values, dtype=np.float64)
    ope_arr = np.asarray(ope_estimates, dtype=np.float64)
    P = len(oracle_arr)

    if len(ope_arr) != P:
        raise ValueError(f"Length mismatch: {P} oracle values vs {len(ope_arr)} OPE estimates")

    # Per-policy results
    per_policy = []
    for i in range(P):
        if synthetic_returns_per_policy is not None:
            returns_i = np.asarray(synthetic_returns_per_policy[i], dtype=np.float64)
        else:
            returns_i = np.array([ope_arr[i]])
        per_policy.append(ope_eval(oracle_arr[i], returns_i))

    # Aggregate
    mses = np.array([r.mse for r in per_policy])
    rel_errors = np.array([r.relative_error for r in per_policy])

    # Spearman
    rho, pval = spearman_rank_correlation(oracle_arr, ope_arr)

    # Regret
    regret_1 = regret_at_k(oracle_arr, ope_arr, k=1)
    regret_dict = {k: regret_at_k(oracle_arr, ope_arr, k=k) for k in k_values}

    return MultiPolicyOPEResult(
        oracle_values=oracle_arr,
        ope_estimates=ope_arr,
        per_policy=per_policy,
        mean_mse=float(mses.mean()),
        mean_relative_error=float(rel_errors.mean()),
        spearman_rho=rho,
        spearman_pvalue=pval,
        regret_at_1=regret_1,
        regret_at_k=regret_dict,
    )
