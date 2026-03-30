from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import torch

from src.robomimic_interface.rollout import RolloutLatentTrajectory, load_rollout_latents
from src.utils import resolve_device

if TYPE_CHECKING:
    from src.sope_diffuser import SopeDiffuser


NP_FLOAT = np.float32


@dataclass
class RMSEMetrics:
    loss: Optional[float] = None
    rmse_eef_pos: Optional[float] = None
    mean_eef_pos: Optional[float] = None
    mean_eef_pos_gt: Optional[float] = None
    rmse_transition: Optional[float] = None
    rmse_state: Optional[float] = None
    rmse_action: Optional[float] = None
    mean_transition: Optional[float] = None
    mean_state: Optional[float] = None
    mean_action: Optional[float] = None
    mean_transition_gt: Optional[float] = None
    mean_state_gt: Optional[float] = None
    mean_action_gt: Optional[float] = None
    num_samples: int = 0
    horizon: Optional[int] = None
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    trajectory_lengths: Optional[np.ndarray] = None
    local_stats: Optional[dict[str, Any]] = None # chunk or trajectory-level stats (mean, var)

@dataclass
class RMSEMetricsReport:
    unguided: dict[str, RMSEMetrics]
    guided: Optional[dict[str, RMSEMetrics]]
    split: Optional[str] = None
    num_rollout_files: Optional[int] = None
    dataset_stats: Optional[dict[str, dict[str, float]]] = None
    value_estimator: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


def _build_eval_summary_metrics(
    report: RMSEMetricsReport,
) -> dict[str, float]:
    summary: dict[str, float] = {
        "eval_metrics:guided/placeholder": float("nan"),
        "eval_diagnostics:guided/placeholder": float("nan"),
    }

    unguided_metrics = report.unguided.get("gen_unnormalized")
    if unguided_metrics is not None:
        for field in ("loss", "rmse_action", "rmse_eef_pos", "rmse_state", "rmse_transition"):
            value = getattr(unguided_metrics, field)
            if value is None:
                continue
            summary[f"eval_metrics:unguided/{field}"] = float(value)
        for field in (
            "mean_action",
            "mean_eef_pos",
            "mean_state",
            "mean_transition",
            "mean_action_gt",
            "mean_eef_pos_gt",
            "mean_state_gt",
            "mean_transition_gt",
        ):
            value = getattr(unguided_metrics, field)
            if value is None:
                continue
            summary[f"eval_diagnostics:unguided/{field}"] = float(value)

    guided_metrics = None if report.guided is None else report.guided.get("gen_unnormalized")
    if guided_metrics is not None:
        for field in ("loss", "rmse_action", "rmse_eef_pos", "rmse_state", "rmse_transition"):
            value = getattr(guided_metrics, field)
            if value is None:
                continue
            summary[f"eval_metrics:guided/{field}"] = float(value)
        for field in (
            "mean_action",
            "mean_eef_pos",
            "mean_state",
            "mean_transition",
            "mean_action_gt",
            "mean_eef_pos_gt",
            "mean_state_gt",
            "mean_transition_gt",
        ):
            value = getattr(guided_metrics, field)
            if value is None:
                continue
            summary[f"eval_diagnostics:guided/{field}"] = float(value)

    return summary


def load_diffusion_checkpoint(
    checkpoint_path: Path,
    *,
    device: Optional[str] = None,
    policy: Optional[Any] = None,
    behavior_policy: Optional[Any] = None,
) -> tuple[SopeDiffuser, dict[str, Any]]:
    from src.sope_diffuser import NormalizationStats, SopeDiffuser, SopeDiffusionConfig

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(str(checkpoint_path), map_location="cpu")

    cfg_diffusion = SopeDiffusionConfig(**payload["diffusion_config"])
    stats_payload = payload.get("normalization_stats")
    stats = None
    if stats_payload is not None:
        stats = NormalizationStats(
            mean=np.asarray(stats_payload["mean"], dtype=NP_FLOAT),
            std=np.asarray(stats_payload["std"], dtype=NP_FLOAT),
        )

    diffuser = SopeDiffuser(
        cfg=cfg_diffusion,
        normalization_stats=stats,
        device=device or resolve_device(prefer_cuda=True),
        policy=policy,
        behavior_policy=behavior_policy,
    )
    diffuser.diffusion.load_state_dict(payload["diffusion_state_dict"])
    diffuser.diffusion.eval()
    return diffuser, payload


def _build_gt_future_chunk(
    diffuser: SopeDiffuser,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    gt_chunk = torch.cat([batch["states_to"][:, :-1, :], batch["actions_to"]], dim=-1)
    return {
        "normalized": gt_chunk,
        "unnormalized": diffuser.unnormalizer(gt_chunk),
    }


def _build_persistence_baseline_chunk(
    diffuser: SopeDiffuser,
    batch: dict[str, torch.Tensor],
    *,
    normalized: bool,
) -> torch.Tensor:
    last_state = batch["states_from"][:, -1:, :].expand(-1, diffuser.cfg.chunk_horizon, -1)
    last_action = batch["actions_from"][:, -1:, :].expand(-1, diffuser.cfg.chunk_horizon, -1)
    baseline = torch.cat([last_state, last_action], dim=-1)
    if normalized:
        return baseline
    return diffuser.unnormalizer(baseline)


def _sample_future_chunk_normalized(
    diffuser: SopeDiffuser,
    batch: dict[str, torch.Tensor],
    *,
    guided: bool = False,
    guidance_kw: Optional[dict[str, Any]] = None,
    verbose: bool = False,
) -> torch.Tensor:
    cond = diffuser.make_cond(batch)
    sample = diffuser.diffusion.conditional_sample(
        shape=(
            int(batch["states_from"].shape[0]),
            int(diffuser.cfg.total_chunk_horizon),
            int(diffuser.transition_dim),
        ),
        cond=cond,
        guided=guided,
        verbose=verbose,
        **(guidance_kw or {}),
    )
    return sample.trajectories[:, diffuser.cfg.frame_stack:, :]


def _finalize_chunk_metrics(metrics: RMSEMetrics) -> RMSEMetrics:
    if metrics.num_samples <= 0:
        raise ValueError("No evaluation chunks were processed.")
    if metrics.horizon is None or metrics.state_dim is None or metrics.action_dim is None:
        raise ValueError("Chunk metric accumulator is missing shape metadata.")

    total_steps = max(metrics.num_samples * metrics.horizon, 1)
    transition_dim = metrics.state_dim + metrics.action_dim
    denom_transition = max(total_steps * transition_dim, 1)
    denom_state = max(total_steps * metrics.state_dim, 1)
    denom_action = max(total_steps * metrics.action_dim, 1)
    eef_pos_dim = None
    if metrics.rmse_eef_pos is not None:
        eef_pos_dim = 3

    metrics.mean_transition = float(metrics.mean_transition / denom_transition)
    metrics.mean_state = float(metrics.mean_state / denom_state)
    metrics.mean_action = float(metrics.mean_action / denom_action)
    metrics.mean_transition_gt = float(metrics.mean_transition_gt / denom_transition)
    metrics.mean_state_gt = float(metrics.mean_state_gt / denom_state)
    metrics.mean_action_gt = float(metrics.mean_action_gt / denom_action)
    metrics.rmse_transition = float(np.sqrt(metrics.rmse_transition / denom_transition))
    metrics.rmse_state = float(np.sqrt(metrics.rmse_state / denom_state))
    metrics.rmse_action = float(np.sqrt(metrics.rmse_action / denom_action))
    if eef_pos_dim is not None:
        denom_eef_pos = max(total_steps * eef_pos_dim, 1)
        metrics.mean_eef_pos = float(metrics.mean_eef_pos / denom_eef_pos)
        metrics.mean_eef_pos_gt = float(metrics.mean_eef_pos_gt / denom_eef_pos)
        metrics.rmse_eef_pos = float(np.sqrt(metrics.rmse_eef_pos / denom_eef_pos))
    return metrics


def evaluate_diffusion_chunk_mse(
    diffuser: SopeDiffuser,
    loader: torch.utils.data.DataLoader,
    *,
    device: Optional[str] = None,
    evaluate_guided: bool = True,
    guidance_kw: Optional[dict[str, Any]] = None,
    max_chunks: Optional[int] = None,
) -> tuple[dict[str, RMSEMetrics], Optional[dict[str, RMSEMetrics]]]:
    eval_device = torch.device(device or diffuser.device)
    diffuser.diffusion.eval()
    eef_pos_slice: Optional[slice] = None
    if diffuser.state_dim >= 13:
        resolve_eef_pos_slice = getattr(diffuser, "_resolve_eef_pos_slice", None)
        if callable(resolve_eef_pos_slice):
            try:
                eef_pos_slice = resolve_eef_pos_slice()
            except ValueError:
                eef_pos_slice = None

    def _init_accumulator() -> RMSEMetrics:
        metrics = RMSEMetrics(
            rmse_transition=0.0,
            rmse_state=0.0,
            rmse_action=0.0,
            mean_transition=0.0,
            mean_state=0.0,
            mean_action=0.0,
            mean_transition_gt=0.0,
            mean_state_gt=0.0,
            mean_action_gt=0.0,
            num_samples=0,
            horizon=int(diffuser.cfg.chunk_horizon),
            state_dim=int(diffuser.state_dim),
            action_dim=int(diffuser.action_dim),
        )
        if eef_pos_slice is not None:
            metrics.rmse_eef_pos = 0.0
            metrics.mean_eef_pos = 0.0
            metrics.mean_eef_pos_gt = 0.0
        return metrics

    def _update_accumulator(
        accumulator: RMSEMetrics,
        pred_chunk: torch.Tensor,
        gt_chunk: torch.Tensor,
    ) -> None:
        sq_err = (pred_chunk - gt_chunk) ** 2
        sq_err_state = sq_err[..., : diffuser.state_dim]
        sq_err_action = sq_err[..., diffuser.state_dim :]
        if eef_pos_slice is not None:
            sq_err_eef_pos = sq_err_state[..., eef_pos_slice]
        pred_chunk = pred_chunk ** 2 
        gt_chunk = gt_chunk ** 2
        pred_chunk_state = pred_chunk[..., : diffuser.state_dim]
        pred_chunk_action = pred_chunk[..., diffuser.state_dim :]
        gt_chunk_state = gt_chunk[..., : diffuser.state_dim]
        gt_chunk_action = gt_chunk[..., diffuser.state_dim :]
        if eef_pos_slice is not None:
            pred_chunk_eef_pos = pred_chunk_state[..., eef_pos_slice]
            gt_chunk_eef_pos = gt_chunk_state[..., eef_pos_slice]

        batch_chunks = int(pred_chunk.shape[0])
        accumulator.num_samples += batch_chunks
        accumulator.rmse_transition += float(sq_err.sum().item())
        accumulator.rmse_state += float(sq_err_state.sum().item())
        accumulator.rmse_action += float(sq_err_action.sum().item())
        if eef_pos_slice is not None:
            accumulator.rmse_eef_pos += float(sq_err_eef_pos.sum().item())
        accumulator.mean_transition += float(pred_chunk.sum().item())
        accumulator.mean_state += float(pred_chunk_state.sum().item())
        accumulator.mean_action += float(pred_chunk_action.sum().item())
        if eef_pos_slice is not None:
            accumulator.mean_eef_pos += float(pred_chunk_eef_pos.sum().item())
        accumulator.mean_transition_gt += float(gt_chunk.sum().item())
        accumulator.mean_state_gt += float(gt_chunk_state.sum().item())
        accumulator.mean_action_gt += float(gt_chunk_action.sum().item())
        if eef_pos_slice is not None:
            accumulator.mean_eef_pos_gt += float(gt_chunk_eef_pos.sum().item())

    def _accumulate(guided: bool) -> dict[str, RMSEMetrics]:
        sample_raw_acc = _init_accumulator()
        sample_norm_acc = _init_accumulator()
        baseline_raw_acc = _init_accumulator()
        baseline_norm_acc = _init_accumulator()

        for batch in loader:
            batch_t = {
                key: value.to(eval_device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            sample_norm = _sample_future_chunk_normalized(
                diffuser,
                batch_t,
                guided=guided,
                guidance_kw=guidance_kw,
                verbose=False,
            )
            sample_raw = diffuser.unnormalizer(sample_norm)
            gt_chunk = _build_gt_future_chunk(diffuser, batch_t)
            gt_norm = gt_chunk["normalized"]
            gt_raw = gt_chunk["unnormalized"]
            baseline_norm = _build_persistence_baseline_chunk(
                diffuser,
                batch_t,
                normalized=True,
            )
            baseline_raw = diffuser.unnormalizer(baseline_norm)

            if max_chunks is not None:
                remaining = max_chunks - sample_raw_acc.num_samples
                if remaining <= 0:
                    break
                sample_norm = sample_norm[:remaining]
                sample_raw = sample_raw[:remaining]
                gt_norm = gt_norm[:remaining]
                gt_raw = gt_raw[:remaining]
                baseline_norm = baseline_norm[:remaining]
                baseline_raw = baseline_raw[:remaining]

            _update_accumulator(sample_norm_acc, sample_norm, gt_norm)
            _update_accumulator(sample_raw_acc, sample_raw, gt_raw)
            _update_accumulator(baseline_norm_acc, baseline_norm, gt_norm)
            _update_accumulator(baseline_raw_acc, baseline_raw, gt_raw)

            if max_chunks is not None and sample_raw_acc.num_samples >= max_chunks:
                break

        return {
            "gen_unnormalized": _finalize_chunk_metrics(sample_raw_acc), # Primary metric
            "gen_normalized": _finalize_chunk_metrics(sample_norm_acc),
            "baseline_unnormalized": _finalize_chunk_metrics(baseline_raw_acc),
            "baseline_normalized": _finalize_chunk_metrics(baseline_norm_acc),
        }

    with torch.no_grad():
        unguided = _accumulate(guided=False)
        guided_metrics = None
        if evaluate_guided and getattr(diffuser.diffusion, "policy", None) is not None:
            guided_metrics = _accumulate(guided=True)

    return unguided, guided_metrics


""" For evaluating sope chunk diffusion during training """

def evaluate_sope(
    diffuser: SopeDiffuser,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    primary_metric_key: str = "gen_unnormalized",
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    run: Optional[Any] = None,
    epoch_iterator: Optional[Any] = None,
    avg_epoch_loss: Optional[float] = None,
    current_lr: Optional[float] = None,
) -> RMSEMetricsReport:
    diffuser.diffusion.eval()
    loss_sum = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            loss_out = diffuser.loss(batch_t)
            loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
            loss_sum += float(loss.item())
            batches += 1

    unguided_chunk_eval, _ = evaluate_diffusion_chunk_mse(
        diffuser,
        loader,
        device=str(device),
        evaluate_guided=False,
    )
    eval_loss = loss_sum / max(batches, 1)
    if primary_metric_key not in unguided_chunk_eval:
        available_keys = ", ".join(sorted(unguided_chunk_eval))
        raise ValueError(
            f"Unknown primary_metric_key={primary_metric_key!r}. "
            f"Available keys: {available_keys}."
        )
    unguided_chunk_eval[primary_metric_key].loss = float(eval_loss)
    report = RMSEMetricsReport(
        unguided=unguided_chunk_eval,
        guided=None,
        metadata={
            "evaluation_type": "chunk",
            "primary_metric_key": primary_metric_key,
        },
    )
    diffuser.diffusion.train()
    
    # update epoch tqdm iterator postfix
    if epoch_iterator is not None:
        postfix: dict[str, str] = {}
        if avg_epoch_loss is not None:
            postfix["loss"] = f"{avg_epoch_loss:.4f}"
        postfix["eval"] = f"{eval_loss:.4f}"
        if current_lr is not None:
            postfix["lr"] = f"{current_lr:.2e}"
        epoch_iterator.set_postfix(**postfix)

    # update wandb run with eval metrics
    if run is not None:
        summary_metrics = _build_eval_summary_metrics(report)
        summary_metrics["eval/epoch"] = float(epoch) if epoch is not None else float("nan")
        summary_metrics["eval/step"] = float(step) if step is not None else float("nan")
        run.log(summary_metrics, step=step)

    return report


""" For evaluating loaded sope chunk diffusion checkpoint """

def _load_guidance_policy(
    *,
    policy_run_dir: Optional[Path] = None,
    policy_ckpt_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> Optional[Any]:
    if policy_run_dir is None:
        return None

    from src.robomimic_interface.checkpoints import build_algo_from_checkpoint, load_checkpoint
    from src.robomimic_interface.policy import DiffusionPolicy

    resolved_device = device or resolve_device(prefer_cuda=True)
    ckpt = load_checkpoint(policy_run_dir, ckpt_path=policy_ckpt_path)
    return DiffusionPolicy(
        policy=build_algo_from_checkpoint(ckpt, device=resolved_device),
        obs_normalization_stats=ckpt.ckpt_dict.get("obs_normalization_stats"),
        action_normalization_stats=ckpt.ckpt_dict.get("action_normalization_stats"),
    )


def _build_split_loader(
    checkpoint_payload: dict[str, Any],
    *,
    data_path: Path,
    split: str,
    seed: int,
    batch_size: int,
    device: Optional[str] = None,
) -> tuple[torch.utils.data.DataLoader, list[Path], dict[str, dict[str, float]]]:
    from src.robomimic_interface.dataset import RolloutChunkDatasetConfig, make_rollout_chunk_dataloader
    from src.robomimic_interface.dataset import summarize_dataset_feature_stats
    from src.train import _assign_dataset_stats, _split_rollout_paths

    cfg_dataset = RolloutChunkDatasetConfig(**checkpoint_payload["dataset_config"])
    training_payload = checkpoint_payload.get("training_config") or {}
    train_fraction = float(training_payload.get("train_fraction", 0.8))
    train_paths, eval_paths = _split_rollout_paths(
        [data_path],
        seed=seed,
        train_fraction=train_fraction,
    )

    if split == "train":
        selected_paths = train_paths
    elif split == "eval":
        if not eval_paths:
            raise ValueError("Requested eval split, but the rollout corpus does not produce a held-out split.")
        selected_paths = eval_paths
    elif split == "all":
        selected_paths = train_paths + eval_paths
    else:
        raise ValueError(f"Unknown split={split}. Expected one of: train, eval, all.")

    loader_device = device or resolve_device(prefer_cuda=True)
    train_loader, train_stats = make_rollout_chunk_dataloader(
        paths=train_paths,
        config=cfg_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        encoder=None,
        obs_keys=None,
        encoder_device=loader_device,
    )
    if split == "train":
        return train_loader, selected_paths, summarize_dataset_feature_stats(train_loader.dataset)

    selected_loader, _ = make_rollout_chunk_dataloader(
        paths=selected_paths,
        config=cfg_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        encoder=None,
        obs_keys=None,
        encoder_device=loader_device,
    )
    _assign_dataset_stats(selected_loader.dataset, train_stats if cfg_dataset.normalize else None)
    return (
        selected_loader,
        selected_paths,
        summarize_dataset_feature_stats(selected_loader.dataset),
    )


def evaluate_saved_diffusion_chunk_mse(
    diffusion_checkpoint_path: Path,
    *,
    data_path: Path,
    split: str = "eval",
    split_seed: int = 0,
    batch_size: int = 256,
    policy_run_dir: Optional[Path] = None,
    policy_ckpt_path: Optional[Path] = None,
    device: Optional[str] = None,
    guidance_kw: Optional[dict[str, Any]] = None,
    max_chunks: Optional[int] = None,
) -> RMSEMetricsReport:
    resolved_device = device or resolve_device(prefer_cuda=True)
    guidance_policy = _load_guidance_policy(
        policy_run_dir=policy_run_dir,
        policy_ckpt_path=policy_ckpt_path,
        device=resolved_device,
    )
    diffuser, payload = load_diffusion_checkpoint(
        diffusion_checkpoint_path,
        device=resolved_device,
        policy=guidance_policy,
    )
    loader, selected_paths, dataset_stats = _build_split_loader(
        payload,
        data_path=data_path,
        split=split,
        seed=split_seed,
        batch_size=batch_size,
        device=resolved_device,
    )
    unguided, guided = evaluate_diffusion_chunk_mse(
        diffuser,
        loader,
        device=resolved_device,
        evaluate_guided=guidance_policy is not None,
        guidance_kw=guidance_kw,
        max_chunks=max_chunks,
    )
    return RMSEMetricsReport(
        unguided=unguided,
        guided=guided,
        split=split,
        num_rollout_files=len(selected_paths),
        dataset_stats=dataset_stats,
        metadata={
            "evaluation_type": "chunk",
            "max_chunks": max_chunks,
        },
    )


""" For autoregressively generating full trajectories from sope chunk diffusion """
def generate_full_trajectory(
    diffuser: SopeDiffuser,
    initial_states: torch.Tensor,
    *,
    max_length: int,
    guided: bool = False,
    verbose: bool = False,
    **guidance_kw: Any,
) -> tuple[np.ndarray, np.ndarray]:
    diffuser.diffusion.eval()

    batch_size = int(initial_states.shape[0])
    chunk_horizon = int(diffuser.cfg.chunk_horizon)
    frame_stack = int(diffuser.cfg.frame_stack)
    total_horizon = int(diffuser.cfg.total_chunk_horizon)

    all_states = np.zeros((batch_size, max_length, diffuser.state_dim), dtype=NP_FLOAT)
    all_actions = np.zeros((batch_size, max_length, diffuser.action_dim), dtype=NP_FLOAT)

    init_dev = initial_states.to(diffuser.device)
    dummy_actions = torch.zeros(
        batch_size,
        diffuser.action_dim,
        device=diffuser.device,
        dtype=init_dev.dtype,
    )
    init_padded = torch.cat([init_dev, dummy_actions], dim=-1)
    init_norm = diffuser.normalizer(init_padded)[:, : diffuser.state_dim]
    cond_states = init_norm.unsqueeze(1).expand(-1, frame_stack, -1).clone()

    total_generated = 0
    while total_generated < max_length:
        steps_to_add = min(chunk_horizon, max_length - total_generated)
        cond = {t: cond_states[:, t, :] for t in range(frame_stack)}
        sample = diffuser.diffusion.conditional_sample(
            shape=(batch_size, total_horizon, diffuser.transition_dim),
            cond=cond,
            guided=guided,
            verbose=verbose,
            **guidance_kw,
        )
        chunk = diffuser.unnormalizer(sample.trajectories)
        gen_states = chunk[:, frame_stack:, : diffuser.state_dim]
        gen_actions = chunk[:, frame_stack:, diffuser.state_dim :]

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

        cond_states = sample.trajectories[:, -frame_stack:, : diffuser.state_dim].clone()

    return all_states, all_actions


def trajectory_state_error(
    real_states: np.ndarray,
    generated_states: np.ndarray,
    *,
    trajectory_lengths: Optional[np.ndarray] = None,
) -> RMSEMetrics:
    real_states = np.asarray(real_states, dtype=NP_FLOAT)
    generated_states = np.asarray(generated_states, dtype=NP_FLOAT)
    if real_states.shape != generated_states.shape:
        raise ValueError(
            f"State shape mismatch: real={real_states.shape}, generated={generated_states.shape}"
        )

    batch_size, max_length, state_dim = real_states.shape
    if trajectory_lengths is None:
        lengths = np.full(batch_size, max_length, dtype=np.int64)
        mask = np.ones((batch_size, max_length), dtype=bool)
    else:
        lengths = np.asarray(trajectory_lengths, dtype=np.int64)
        if lengths.shape != (batch_size,):
            raise ValueError(
                f"trajectory_lengths must have shape {(batch_size,)}, got {lengths.shape}"
            )
        mask = np.arange(max_length)[None, :] < lengths[:, None]

    sq_err = (real_states - generated_states) ** 2 * mask[..., None]
    n_valid = max(int(mask.sum()), 1)
    state_mse = float(sq_err.sum() / (n_valid * state_dim))

    return RMSEMetrics(
        rmse_state=float(np.sqrt(state_mse)),
        mean_state=float(real_states[mask].mean()) if int(mask.sum()) > 0 else 0.0,
        num_samples=batch_size,
        horizon=max_length,
        state_dim=state_dim,
        trajectory_lengths=lengths.astype(np.int64),
    )


def evaluate_diffusion_trajectory_state_error(
    diffuser: SopeDiffuser,
    trajectories: Sequence[RolloutLatentTrajectory],
    *,
    evaluate_guided: bool = True,
    guidance_kw: Optional[dict[str, Any]] = None,
) -> RMSEMetricsReport:
    if not trajectories:
        raise ValueError("At least one trajectory is required for evaluation.")

    max_length = max(int(traj.latents.shape[0]) for traj in trajectories)
    batch_size = len(trajectories)
    state_dim = diffuser.state_dim

    real_states = np.zeros((batch_size, max_length, state_dim), dtype=NP_FLOAT)
    initial_states = np.zeros((batch_size, state_dim), dtype=NP_FLOAT)
    lengths = np.zeros(batch_size, dtype=np.int64)

    for i, traj in enumerate(trajectories):
        traj_states = np.asarray(traj.latents, dtype=NP_FLOAT)
        if traj_states.ndim == 3:
            traj_states = traj_states[:, 0, :]
        if traj_states.shape[-1] != state_dim:
            raise ValueError(
                f"Trajectory {i} state_dim mismatch: expected {state_dim}, got {traj_states.shape[-1]}"
            )
        length = int(traj_states.shape[0])
        real_states[i, :length, :] = traj_states
        initial_states[i, :] = traj_states[0]
        lengths[i] = length

    initial_states_t = torch.from_numpy(initial_states).to(diffuser.device)
    unguided_states, _ = generate_full_trajectory(
        diffuser,
        initial_states_t,
        max_length=max_length,
        guided=False,
    )
    unguided_error = trajectory_state_error(
        real_states,
        unguided_states,
        trajectory_lengths=lengths,
    )

    guided_error = None
    if evaluate_guided and getattr(diffuser.diffusion, "policy", None) is not None:
        guided_states, _ = generate_full_trajectory(
            diffuser,
            initial_states_t,
            max_length=max_length,
            guided=True,
            **(guidance_kw or {}),
        )
        guided_error = trajectory_state_error(
            real_states,
            guided_states,
            trajectory_lengths=lengths,
        )

    return RMSEMetricsReport(
        unguided={"gen_unnormalized": unguided_error},
        guided=None if guided_error is None else {"gen_unnormalized": guided_error},
        metadata={
            "evaluation_type": "trajectory",
            "num_trajectories": batch_size,
            "max_trajectory_length": max_length,
        },
    )


def evaluate_saved_diffusion_trajectory_state_error(
    diffusion_checkpoint_path: Path,
    rollout_paths: Sequence[Path],
    *,
    policy_run_dir: Optional[Path] = None,
    policy_ckpt_path: Optional[Path] = None,
    max_trajectories: Optional[int] = None,
    device: Optional[str] = None,
    guidance_kw: Optional[dict[str, Any]] = None,
) -> RMSEMetricsReport:
    resolved_device = device or resolve_device(prefer_cuda=True)
    policy = _load_guidance_policy(
        policy_run_dir=policy_run_dir,
        policy_ckpt_path=policy_ckpt_path,
        device=resolved_device,
    )
    diffuser, _ = load_diffusion_checkpoint(
        diffusion_checkpoint_path,
        device=resolved_device,
        policy=policy,
    )

    rollout_paths = [Path(p) for p in rollout_paths]
    if max_trajectories is not None:
        rollout_paths = rollout_paths[:max_trajectories]

    trajectories = [load_rollout_latents(p) for p in rollout_paths]
    return evaluate_diffusion_trajectory_state_error(
        diffuser,
        trajectories,
        evaluate_guided=policy is not None,
        guidance_kw=guidance_kw,
    )
