from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
import gc
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Optional

import h5py
import numpy as np
from scipy.stats import spearmanr
import torch
from tqdm import tqdm

from src.diffusion import (
    NormalizationStats,
    RewardPredictor,
    RewardPredictorConfig,
    SopeDiffuser,
    SopeDiffusionConfig,
)
from src.robomimic_interface.rollout import load_rollout_latents
from src.utils import resolve_device


NP_FLOAT = np.float32
DEFAULT_GUIDED_SUCCESS_HEIGHT_THRESHOLD = 0.84
DEFAULT_GUIDED_SUCCESS_OBJECT_Z_INDEX: Optional[int] = None
_LIFT_OBJECT_Z_WITHIN_OBJECT = 2
_EPOCH_PATTERN = re.compile(r"model_epoch_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class GuidanceConfig:
    action_score_scale: float = 0.2
    use_adaptive: bool = True
    use_neg_grad: bool = True
    action_score_postprocess: str = "l2"
    num_guidance_iters: int = 2
    clamp_linf: float = 1.0
    action_neg_score_weight: float = 1.0

    def to_sampling_kwargs(self) -> dict[str, Any]:
        """Return this config in the keyword format expected by guided sampling."""
        return {
            "action_score_scale": float(self.action_score_scale),
            "use_adaptive": bool(self.use_adaptive),
            "use_neg_grad": bool(self.use_neg_grad),
            "action_score_postprocess": str(self.action_score_postprocess),
            "num_guidance_iters": int(self.num_guidance_iters),
            "clamp_linf": float(self.clamp_linf),
            "action_neg_score_weight": float(self.action_neg_score_weight),
        }


@dataclass
class ChunkMetrics:
    loss: Optional[float] = None
    rmse_transition: Optional[float] = None
    rmse_state: Optional[float] = None
    rmse_action: Optional[float] = None
    num_samples: int = 0
    horizon: Optional[int] = None
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None


@dataclass
class ChunkMetricsReport:
    unguided: dict[str, ChunkMetrics]
    guided: Optional[dict[str, ChunkMetrics]]
    split: Optional[str] = None
    num_rollout_files: Optional[int] = None
    dataset_stats: Optional[dict[str, dict[str, float]]] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class SavedRolloutOPEReport:
    run_dir: Path
    diffusion_checkpoint: Path
    reward_checkpoint: Path
    data: Path
    split: str
    training_seed: int
    train_fraction: float
    num_rollout_files: int
    rollout_batch_size: int
    max_trajectories: Optional[int]
    rollout_horizon: int
    ope_gamma: float
    reward_transform: str
    device: str
    guided: bool
    autoregressive_rollout_mse: float
    return_gt_transformed: float
    return_ope_estimate: float
    target_policy_checkpoint: Optional[Path] = None
    behavior_policy_checkpoint: Optional[Path] = None
    target_score_timestep: Optional[int] = None
    behavior_score_timestep: Optional[int] = None
    guidance_config: Optional[GuidanceConfig] = None


@dataclass
class MultipolicyOPEPolicyReport:
    target_policy_checkpoint: Path
    target_policy_epoch: int
    num_rollout_files: int
    rollout_horizon: int
    mean_predicted_transformed_return: float
    std_predicted_transformed_return: float
    mean_true_transformed_return: float
    std_true_transformed_return: float
    mean_true_raw_return: float
    std_true_raw_return: float
    true_rollout_success_rate: float
    guided_rollout_success_rate: float
    guided_rollout_success_height_threshold: float
    guided_target_rollout_mse: float
    guided_target_rollout_rmse: float
    mean_true_rollout_horizon: float
    min_true_rollout_horizon: int
    max_true_rollout_horizon: int
    target_score_timestep: int
    behavior_score_timestep: int


@dataclass
class MultipolicyOPEReport:
    run_dir: Path
    diffusion_checkpoint: Path
    reward_checkpoint: Path
    data: Path
    source_dataset_path: Path
    split: str
    training_seed: int
    train_fraction: float
    num_rollout_files: int
    rollout_batch_size: int
    max_trajectories: Optional[int]
    target_policy_dir: Path
    max_target_policies: Optional[int]
    num_target_policies: int
    configured_rollout_horizon: int
    rollout_horizon: int
    ope_gamma: float
    reward_transform: str
    device: str
    guided: bool
    behavior_policy_checkpoint: Path
    rollout_env_dataset_path: Path
    rollout_env_wrapper_checkpoint: Path
    guidance_config: GuidanceConfig
    guided_success_state_index: int
    guided_success_height_threshold: float
    spearman_correlation_transformed: Optional[float]
    policy_value_rmse_transformed: float
    policies: list[MultipolicyOPEPolicyReport]
    spearman_correlation_raw: Optional[float] = None
    policy_value_rmse_raw: Optional[float] = None


@dataclass(frozen=True)
class DemoInitialCondition:
    rollout_path: Path
    source_dataset_path: Path
    demo_key: str
    initial_state: dict[str, Any]


@dataclass(frozen=True)
class SavedRolloutTargets:
    states: np.ndarray
    actions: np.ndarray
    transformed_returns: np.ndarray
    raw_returns: np.ndarray
    horizon: int
    reward_transform: str


def to_jsonable(obj: Any) -> Any:
    """Recursively convert reports, arrays, and paths into JSON-serializable values."""
    if is_dataclass(obj):
        return {
            field.name: to_jsonable(getattr(obj, field.name))
            for field in fields(obj)
        }
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def serialize_report(
    report: Any,
    *,
    json_report: Optional[Path] = None,
) -> dict[str, Any]:
    """Serialize a report object to a JSON-ready dict and optionally attach its output path."""
    payload = to_jsonable(report)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected report payload to serialize to a dict, got {type(payload)}.")
    if json_report is not None:
        payload["json_report"] = str(Path(json_report).resolve())
    return payload


def write_report_json(
    report: Any,
    *,
    json_report: Path,
) -> dict[str, Any]:
    """Serialize a report and atomically replace its JSON file on disk."""
    resolved_json_report = Path(json_report).expanduser().resolve()
    payload = serialize_report(report, json_report=resolved_json_report)
    resolved_json_report.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved_json_report.with_name(f".{resolved_json_report.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(resolved_json_report)
    return payload


def _to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value is None or key == "metadata":
            out[key] = value
            continue
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _flatten_scalar_metrics(info: Any) -> dict[str, float]:
    if not isinstance(info, dict):
        return {}

    flat: dict[str, float] = {}
    for key, value in info.items():
        if torch.is_tensor(value) and value.numel() == 1:
            flat[key] = float(value.item())
        elif np.isscalar(value):
            flat[key] = float(value)
    return flat


def _call_loss(
    diffuser: Any,
    batch_t: dict[str, torch.Tensor],
    *,
    compute_batch_rmse: bool = False,
) -> Any:
    try:
        return diffuser.loss(batch_t, compute_batch_rmse=compute_batch_rmse)
    except TypeError as exc:
        if "compute_batch_rmse" not in str(exc):
            raise
        return diffuser.loss(batch_t)


def _build_eval_summary_metrics(
    report: ChunkMetricsReport,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    unguided_metrics = report.unguided.get("gen_unnormalized")
    if unguided_metrics is None:
        return summary

    for field in ("loss", "rmse_action", "rmse_state", "rmse_transition"):
        value = getattr(unguided_metrics, field)
        if value is None:
            continue
        summary[f"eval_metrics/{field}"] = float(value)
    return summary


def _flatten_eval_last_batch_info(info: Any) -> dict[str, float]:
    if not isinstance(info, dict):
        return {}

    flattened: dict[str, float] = {}
    for key, value in info.items():
        if key.startswith("chunk_rmse_"):
            continue
        if torch.is_tensor(value) and value.numel() == 1:
            flattened[f"eval_last_batch/{key}"] = float(value.item())
        elif np.isscalar(value):
            flattened[f"eval_last_batch/{key}"] = float(value)
    return flattened


def load_diffusion_checkpoint(
    checkpoint_path: Path,
    *,
    device: Optional[str] = None,
    policy: Optional[Any] = None,
    behavior_policy: Optional[Any] = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a saved SOPE diffusion checkpoint and instantiate its diffuser wrapper."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
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


def load_reward_checkpoint(
    checkpoint_path: Path,
    *,
    device: Optional[str] = None,
) -> tuple[RewardPredictor, dict[str, Any]]:
    """Load a saved reward predictor checkpoint and return the model plus raw payload."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg_reward = RewardPredictorConfig(**payload["reward_config"])
    predictor = RewardPredictor(
        cfg_reward,
        device=device or resolve_device(prefer_cuda=True),
    )
    predictor.load_state_dict(payload["reward_state_dict"])
    predictor.eval()
    return predictor, payload


def _evaluate_reward_predictor(
    reward_predictor: Any,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    reward_predictor.eval()
    losses: list[float] = []
    baseline_zero_mses: list[float] = []
    pred_means: list[float] = []
    target_means: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch_t = _to_device(batch, device)
            loss, info = reward_predictor.loss(batch_t)
            losses.append(float(loss.item()))
            flat_info = _flatten_scalar_metrics(info)
            if "reward_baseline_zero_mse" in flat_info:
                baseline_zero_mses.append(float(flat_info["reward_baseline_zero_mse"]))
            if "reward_pred_mean" in flat_info:
                pred_means.append(float(flat_info["reward_pred_mean"]))
            if "reward_target_mean" in flat_info:
                target_means.append(float(flat_info["reward_target_mean"]))

    return {
        "reward_eval/loss": float(np.mean(losses)) if losses else float("nan"),
        "reward_eval/baseline_zero_mse": (
            float(np.mean(baseline_zero_mses)) if baseline_zero_mses else float("nan")
        ),
        "reward_eval/pred_mean": float(np.mean(pred_means)) if pred_means else float("nan"),
        "reward_eval/target_mean": float(np.mean(target_means)) if target_means else float("nan"),
    }


def _build_gt_future_chunk(
    diffuser: Any,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if getattr(diffuser.cfg, "conditioning_mode", "prefix_states") == "none":
        zeros = torch.zeros_like(batch["actions_to"])
        gt_chunk = torch.cat([batch["states_to"][:, :-1, :], zeros], dim=-1)
    else:
        gt_chunk = torch.cat([batch["states_to"][:, :-1, :], batch["actions_to"]], dim=-1)
    return {
        "normalized": gt_chunk,
        "unnormalized": diffuser.unnormalizer(gt_chunk),
    }


def _build_persistence_baseline_chunk(
    diffuser: Any,
    batch: dict[str, torch.Tensor],
    *,
    normalized: bool,
) -> torch.Tensor:
    if getattr(diffuser.cfg, "conditioning_mode", "prefix_states") == "none":
        last_state = batch["states_to"][:, :1, :].expand(-1, diffuser.cfg.chunk_horizon, -1)
        last_action = torch.zeros_like(batch["actions_to"])
    else:
        last_state = batch["states_from"][:, -1:, :].expand(-1, diffuser.cfg.chunk_horizon, -1)
        last_action = batch["actions_from"][:, -1:, :].expand(-1, diffuser.cfg.chunk_horizon, -1)
    baseline = torch.cat([last_state, last_action], dim=-1)
    if normalized:
        return baseline
    return diffuser.unnormalizer(baseline)


def _build_guidance_prefix_context(
    diffuser: Any,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if getattr(diffuser.cfg, "conditioning_mode", "prefix_states") == "none":
        return {}
    prefix_chunk = torch.cat([batch["states_from"], batch["actions_from"]], dim=-1)
    prefix_chunk = diffuser.unnormalizer(prefix_chunk)
    return {
        "prefix_states": prefix_chunk[..., : diffuser.state_dim],
        "prefix_actions": prefix_chunk[..., diffuser.state_dim :],
    }


def _sample_future_chunk_normalized(
    diffuser: Any,
    batch: dict[str, torch.Tensor],
    *,
    guided: bool = False,
    guidance_kw: Optional[dict[str, Any]] = None,
    verbose: bool = False,
) -> torch.Tensor:
    cond = diffuser.make_cond(batch)
    guidance_kwargs = dict(guidance_kw or {})
    guidance_kwargs.update(_build_guidance_prefix_context(diffuser, batch))
    sample = diffuser.diffusion.conditional_sample(
        shape=(
            int(batch["states_from"].shape[0]),
            int(diffuser.cfg.total_chunk_horizon),
            int(diffuser.transition_dim),
        ),
        cond=cond,
        guided=guided,
        verbose=verbose,
        **guidance_kwargs,
    )
    return sample.trajectories


def _finalize_chunk_metrics(metrics: ChunkMetrics) -> ChunkMetrics:
    if metrics.num_samples <= 0:
        raise ValueError("No evaluation chunks were processed.")
    if metrics.horizon is None or metrics.state_dim is None or metrics.action_dim is None:
        raise ValueError("Chunk metric accumulator is missing shape metadata.")

    total_steps = max(metrics.num_samples * metrics.horizon, 1)
    transition_dim = metrics.state_dim + metrics.action_dim
    denom_transition = max(total_steps * transition_dim, 1)
    denom_state = max(total_steps * metrics.state_dim, 1)
    denom_action = max(total_steps * metrics.action_dim, 1)

    if metrics.rmse_transition is not None:
        metrics.rmse_transition = float(np.sqrt(metrics.rmse_transition / denom_transition))
    if metrics.rmse_state is not None:
        metrics.rmse_state = float(np.sqrt(metrics.rmse_state / denom_state))
    if metrics.rmse_action is not None:
        metrics.rmse_action = float(np.sqrt(metrics.rmse_action / denom_action))
    return metrics


def evaluate_diffusion_chunk_mse(
    diffuser: Any,
    loader: torch.utils.data.DataLoader,
    *,
    device: Optional[str] = None,
    evaluate_guided: bool = True,
    guidance_kw: Optional[dict[str, Any]] = None,
    max_chunks: Optional[int] = None,
) -> tuple[dict[str, ChunkMetrics], Optional[dict[str, ChunkMetrics]]]:
    """Evaluate chunk-level transition, state, and action RMSE for sampled diffusion outputs."""
    eval_device = torch.device(device or diffuser.device)
    diffuser.diffusion.eval()

    def _init_accumulator() -> ChunkMetrics:
        return ChunkMetrics(
            rmse_transition=0.0,
            rmse_state=0.0,
            rmse_action=0.0,
            num_samples=0,
            horizon=int(diffuser.cfg.chunk_horizon),
            state_dim=int(diffuser.state_dim),
            action_dim=int(diffuser.action_dim),
        )

    def _update_accumulator(
        accumulator: ChunkMetrics,
        pred_chunk: torch.Tensor,
        gt_chunk: torch.Tensor,
    ) -> None:
        sq_err = (pred_chunk - gt_chunk) ** 2
        sq_err_state = sq_err[..., : diffuser.state_dim]
        sq_err_action = sq_err[..., diffuser.state_dim :]

        accumulator.num_samples += int(pred_chunk.shape[0])
        accumulator.rmse_transition += float(sq_err.sum().item())
        accumulator.rmse_state += float(sq_err_state.sum().item())
        accumulator.rmse_action += float(sq_err_action.sum().item())

    def _accumulate(guided: bool) -> dict[str, ChunkMetrics]:
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
            "gen_unnormalized": _finalize_chunk_metrics(sample_raw_acc),
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


def evaluate_sope(
    diffuser: Any,
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
) -> ChunkMetricsReport:
    """Run the standard held-out chunk evaluation used during SOPE training."""
    diffuser.diffusion.eval()
    loss_sum = 0.0
    batches = 0
    last_batch_info: dict[str, float] = {}
    with torch.no_grad():
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            batch_t = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            loss_out = _call_loss(
                diffuser,
                batch_t,
                compute_batch_rmse=(batch_idx == total_batches - 1),
            )
            if isinstance(loss_out, tuple):
                loss, info = loss_out
            else:
                loss, info = loss_out, {}
            loss_sum += float(loss.item())
            batches += 1
            if batch_idx == total_batches - 1:
                last_batch_info = _flatten_eval_last_batch_info(info)

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
    report = ChunkMetricsReport(
        unguided=unguided_chunk_eval,
        guided=None,
        metadata={
            "evaluation_type": "chunk",
            "primary_metric_key": primary_metric_key,
            "eval_last_batch": last_batch_info or None,
        },
    )
    diffuser.diffusion.train()

    if epoch_iterator is not None:
        postfix: dict[str, str] = {}
        if avg_epoch_loss is not None:
            postfix["loss"] = f"{avg_epoch_loss:.4f}"
        postfix["eval"] = f"{eval_loss:.4f}"
        if current_lr is not None:
            postfix["lr"] = f"{current_lr:.2e}"
        epoch_iterator.set_postfix(**postfix)

    if run is not None:
        summary_metrics = _build_eval_summary_metrics(report)
        summary_metrics["eval/epoch"] = float(epoch) if epoch is not None else float("nan")
        summary_metrics["eval/step"] = float(step) if step is not None else float("nan")
        summary_metrics.update(last_batch_info)
        run.log(summary_metrics, step=step)

    return report


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
) -> tuple[torch.utils.data.DataLoader, list[Any], dict[str, dict[str, float]]]:
    from src.robomimic_interface.dataset import summarize_dataset_feature_stats
    from src.train import _assign_dataset_stats, _split_rollout_paths

    training_payload = checkpoint_payload.get("training_config") or {}
    data_kind = str(training_payload.get("data_kind", "rollout"))
    train_fraction = float(training_payload.get("train_fraction", 0.8))

    if data_kind == "sope_gym":
        from src.sope_interface.dataset import (
            SopeGymChunkDatasetConfig,
            make_sope_gym_chunk_dataloader,
            split_sope_gym_episodes,
            train_eval_split_sope_gym_episodes,
        )

        cfg_dataset = SopeGymChunkDatasetConfig(**checkpoint_payload["dataset_config"])
        episodes = split_sope_gym_episodes(data_path)
        train_episodes, eval_episodes = train_eval_split_sope_gym_episodes(
            episodes,
            seed=seed,
            train_fraction=train_fraction,
        )

        if split == "train":
            selected_episodes = train_episodes
        elif split == "eval":
            if not eval_episodes:
                raise ValueError(
                    "Requested eval split, but the SOPE Gym dataset does not produce a held-out split."
                )
            selected_episodes = eval_episodes
        elif split == "all":
            selected_episodes = train_episodes + eval_episodes
        else:
            raise ValueError(f"Unknown split={split}. Expected one of: train, eval, all.")

        train_loader, train_stats = make_sope_gym_chunk_dataloader(
            episodes=train_episodes,
            config=cfg_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        )
        if split == "train":
            return (
                train_loader,
                [episode.episode_id for episode in selected_episodes],
                summarize_dataset_feature_stats(train_loader.dataset),
            )

        selected_loader, _ = make_sope_gym_chunk_dataloader(
            episodes=selected_episodes,
            config=cfg_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        )
        _assign_dataset_stats(
            selected_loader.dataset,
            train_stats if cfg_dataset.normalize else None,
        )
        return (
            selected_loader,
            [episode.episode_id for episode in selected_episodes],
            summarize_dataset_feature_stats(selected_loader.dataset),
        )

    from src.robomimic_interface.dataset import (
        RolloutChunkDatasetConfig,
        make_rollout_chunk_dataloader,
    )

    cfg_dataset = RolloutChunkDatasetConfig(**checkpoint_payload["dataset_config"])
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
) -> ChunkMetricsReport:
    """Load a saved diffusion checkpoint and evaluate chunk RMSE on a selected dataset split."""
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
    return ChunkMetricsReport(
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


def _resolve_data_path(
    *,
    data_arg: Optional[Path],
    diffusion_payload: dict[str, Any],
) -> Path:
    if data_arg is not None:
        return data_arg.resolve()

    training_payload = diffusion_payload.get("training_config") or {}
    data_entries = training_payload.get("data") or []
    if not data_entries:
        raise ValueError(
            "Could not infer rollout data path from diffusion checkpoint training_config['data']; "
            "pass --data explicitly."
        )
    return Path(data_entries[0]).resolve()


def _select_rollout_paths(
    *,
    data_path: Path,
    split: str,
    diffusion_payload: dict[str, Any],
) -> tuple[list[Path], int, float]:
    from src.train import _split_rollout_paths

    training_payload = diffusion_payload.get("training_config") or {}
    split_seed = int(training_payload.get("seed", 0))
    train_fraction = float(training_payload.get("train_fraction", 0.8))
    train_paths, eval_paths = _split_rollout_paths(
        [data_path],
        seed=split_seed,
        train_fraction=train_fraction,
    )

    if split == "train":
        selected_paths = train_paths
    elif split == "eval":
        if not eval_paths:
            raise ValueError(
                "Requested eval split, but the rollout corpus does not produce a held-out split."
            )
        selected_paths = eval_paths
    elif split == "all":
        selected_paths = train_paths + eval_paths
    else:
        raise ValueError(f"Unknown split={split!r}.")

    return selected_paths, split_seed, train_fraction


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_numpy_array(value: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _as_torch_tensor(
    value: Any,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    if torch.is_tensor(value):
        value_t = value.detach()
        return value_t.to(device=device if device is not None else value_t.device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _build_initial_states(
    rollout_paths: list[Path],
    *,
    dataset_config_payload: dict[str, Any],
) -> np.ndarray:
    from src.robomimic_interface.dataset import RolloutChunkDataset, RolloutChunkDatasetConfig

    cfg_dataset = RolloutChunkDatasetConfig(**dataset_config_payload)
    initial_states: list[np.ndarray] = []
    for rollout_path in rollout_paths:
        traj = load_rollout_latents(rollout_path)
        dataset = RolloutChunkDataset(
            traj=traj,
            config=cfg_dataset,
            encoder=None,
            obs_keys=None,
            encoder_device="cpu",
            demo_id=rollout_path.stem,
        )
        if dataset.latents.shape[0] <= 0:
            raise ValueError(f"Rollout {rollout_path} has no latent states.")
        initial_states.append(np.asarray(dataset.latents[0], dtype=np.float32))

    if not initial_states:
        raise ValueError("No rollout initial states were collected for OPE evaluation.")
    return np.stack(initial_states, axis=0).astype(np.float32)


def _discounted_returns(
    rewards: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim == 1:
        rewards = rewards[None, :]
    assert rewards.ndim == 2, f"Expected rewards to have shape [B, T], got {rewards.shape}."
    discounts = np.power(np.float32(gamma), np.arange(rewards.shape[1], dtype=np.float32))
    return np.sum(rewards * discounts[None, :], axis=1, dtype=np.float64).astype(np.float32)


def _discounted_returns_torch(
    rewards: Any,
    *,
    gamma: float,
) -> torch.Tensor:
    rewards_t = _as_torch_tensor(rewards, dtype=torch.float32)
    if rewards_t.ndim == 1:
        rewards_t = rewards_t.unsqueeze(0)
    assert rewards_t.ndim == 2, f"Expected rewards to have shape [B, T], got {tuple(rewards_t.shape)}."
    discounts = torch.pow(
        torch.full(
            (rewards_t.shape[1],),
            gamma,
            dtype=rewards_t.dtype,
            device=rewards_t.device,
        ),
        torch.arange(
            rewards_t.shape[1],
            dtype=rewards_t.dtype,
            device=rewards_t.device,
        ),
    )
    return torch.sum(rewards_t * discounts.unsqueeze(0), dim=1, dtype=torch.float64).to(dtype=torch.float32)


def _load_saved_rollout_targets(
    rollout_paths: list[Path],
    *,
    dataset_config_payload: dict[str, Any],
    ope_gamma: float,
    max_length: int,
) -> SavedRolloutTargets:
    from src.robomimic_interface.dataset import (
        RolloutChunkDataset,
        RolloutChunkDatasetConfig,
        resolve_reward_transform,
    )

    cfg_dataset = RolloutChunkDatasetConfig(**dataset_config_payload)
    reward_transform = resolve_reward_transform(cfg_dataset.reward_transform)

    true_states: list[np.ndarray] = []
    true_actions: list[np.ndarray] = []
    true_rewards_raw: list[np.ndarray] = []
    true_rewards_transformed: list[np.ndarray] = []
    effective_horizon: Optional[int] = None

    for rollout_path in rollout_paths:
        traj = load_rollout_latents(rollout_path)
        dataset = RolloutChunkDataset(
            traj=traj,
            config=cfg_dataset,
            encoder=None,
            obs_keys=None,
            encoder_device="cpu",
            demo_id=rollout_path.stem,
        )
        states = np.asarray(dataset.latents, dtype=np.float32)
        actions = np.asarray(dataset.actions, dtype=np.float32)
        rewards_raw = np.asarray(dataset.rewards, dtype=np.float32)
        rewards_transformed = np.asarray(
            reward_transform(states, rewards_raw),
            dtype=np.float32,
        )

        rollout_horizon = min(
            int(max_length),
            int(states.shape[0]),
            int(actions.shape[0]),
            int(rewards_raw.shape[0]),
            int(rewards_transformed.shape[0]),
        )
        if rollout_horizon <= 0:
            raise ValueError(f"Rollout {rollout_path} has no valid steps for OPE evaluation.")

        effective_horizon = rollout_horizon if effective_horizon is None else min(effective_horizon, rollout_horizon)
        true_states.append(states[:rollout_horizon])
        true_actions.append(actions[:rollout_horizon])
        true_rewards_raw.append(rewards_raw[:rollout_horizon])
        true_rewards_transformed.append(rewards_transformed[:rollout_horizon])

    if effective_horizon is None:
        raise ValueError("No true rollout targets were collected for OPE evaluation.")

    aligned_states = np.stack([states[:effective_horizon] for states in true_states], axis=0).astype(np.float32)
    aligned_actions = np.stack([actions[:effective_horizon] for actions in true_actions], axis=0).astype(np.float32)
    raw_returns = np.asarray(
        [
            float(_discounted_returns(rewards[:effective_horizon], gamma=ope_gamma)[0])
            for rewards in true_rewards_raw
        ],
        dtype=np.float32,
    )
    transformed_returns = np.asarray(
        [
            float(_discounted_returns(rewards[:effective_horizon], gamma=ope_gamma)[0])
            for rewards in true_rewards_transformed
        ],
        dtype=np.float32,
    )
    return SavedRolloutTargets(
        states=aligned_states,
        actions=aligned_actions,
        transformed_returns=transformed_returns,
        raw_returns=raw_returns,
        horizon=int(effective_horizon),
        reward_transform=str(cfg_dataset.reward_transform),
    )


def _reward_predictions_to_numpy(predictions: Any) -> np.ndarray:
    return _as_numpy_array(_reward_predictions_to_tensor(predictions), dtype=np.float32)


def _reward_predictions_to_tensor(
    predictions: Any,
    *,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    rewards = _as_torch_tensor(predictions, dtype=torch.float32, device=device)
    if rewards.ndim == 3 and rewards.shape[-1] == 1:
        rewards = rewards.squeeze(-1)
    if rewards.ndim != 2:
        raise ValueError(f"Expected reward predictions to have shape [B, T], got {tuple(rewards.shape)}.")
    return rewards


def _transition_error_totals(
    pred_states: np.ndarray,
    pred_actions: np.ndarray,
    true_states: np.ndarray,
    true_actions: np.ndarray,
) -> tuple[float, int]:
    pred_transition = np.concatenate([pred_states, pred_actions], axis=-1)
    true_transition = np.concatenate([true_states, true_actions], axis=-1)
    return (
        float(np.square(pred_transition - true_transition).sum(dtype=np.float64)),
        int(pred_transition.size),
    )


def _transition_error_totals_torch(
    pred_states: Any,
    pred_actions: Any,
    true_states: Any,
    true_actions: Any,
) -> tuple[float, int]:
    pred_states_t = _as_torch_tensor(pred_states, dtype=torch.float32)
    pred_actions_t = _as_torch_tensor(pred_actions, dtype=torch.float32, device=pred_states_t.device)
    true_states_t = _as_torch_tensor(true_states, dtype=torch.float32, device=pred_states_t.device)
    true_actions_t = _as_torch_tensor(true_actions, dtype=torch.float32, device=pred_states_t.device)
    pred_transition = torch.cat([pred_states_t, pred_actions_t], dim=-1)
    true_transition = torch.cat([true_states_t, true_actions_t], dim=-1)
    if pred_transition.shape != true_transition.shape:
        raise ValueError(
            "Transition tensors must share the same shape; "
            f"got {tuple(pred_transition.shape)} vs {tuple(true_transition.shape)}."
        )
    return (
        float(torch.square(pred_transition - true_transition).sum(dtype=torch.float64).item()),
        int(pred_transition.numel()),
    )


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"RMSE inputs must share the same shape, got {x.shape} vs {y.shape}.")
    return float(np.sqrt(np.mean(np.square(x - y), dtype=np.float64)))


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Spearman inputs must share the same shape, got {x.shape} vs {y.shape}.")
    if x.size < 2:
        return None
    coefficient = spearmanr(x, y).statistic
    if coefficient is None or not np.isfinite(float(coefficient)):
        return None
    return float(coefficient)


def _maybe_free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_policy_checkpoint(
    policy_checkpoint: Path,
) -> tuple[Path, Path, Path]:
    resolved_checkpoint = policy_checkpoint.expanduser().resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {resolved_checkpoint}")

    for run_dir in resolved_checkpoint.parents:
        if (run_dir / "config.json").is_file():
            return resolved_checkpoint, run_dir, resolved_checkpoint.relative_to(run_dir)

    raise ValueError(
        "Could not locate a robomimic run directory for checkpoint "
        f"{resolved_checkpoint}. Expected an ancestor containing config.json."
    )


def _load_diffusion_policy(
    *,
    policy_checkpoint: Path,
    device: Optional[str],
    score_timestep: int,
) -> tuple[Any, Path]:
    from src.robomimic_interface.checkpoints import build_algo_from_checkpoint, load_checkpoint
    from src.robomimic_interface.policy import DiffusionPolicy, DiffusionPolicyScoreConfig

    resolved_checkpoint, run_dir, relative_checkpoint = _resolve_policy_checkpoint(policy_checkpoint)
    checkpoint = load_checkpoint(run_dir, ckpt_path=relative_checkpoint)
    resolved_device = device or resolve_device(prefer_cuda=True)
    policy = DiffusionPolicy(
        policy=build_algo_from_checkpoint(checkpoint, device=resolved_device),
        obs_normalization_stats=checkpoint.ckpt_dict.get("obs_normalization_stats"),
        action_normalization_stats=checkpoint.ckpt_dict.get("action_normalization_stats"),
        config=DiffusionPolicyScoreConfig(score_timestep=score_timestep),
    )
    return policy, resolved_checkpoint


def _load_rollout_policy(
    *,
    policy_checkpoint: Path,
    device: Optional[str],
) -> tuple[Any, Path]:
    from src.robomimic_interface.checkpoints import build_rollout_policy_from_checkpoint, load_checkpoint

    resolved_checkpoint, run_dir, relative_checkpoint = _resolve_policy_checkpoint(policy_checkpoint)
    checkpoint = load_checkpoint(run_dir, ckpt_path=relative_checkpoint)
    resolved_device = device or resolve_device(prefer_cuda=True)
    policy = build_rollout_policy_from_checkpoint(
        checkpoint,
        device=resolved_device,
        verbose=False,
    )
    return policy, resolved_checkpoint


def _validate_guidance_policy_horizon(
    *,
    diffuser: Any,
    policy: Any,
    role: str,
) -> None:
    assert int(policy.observation_horizon) == int(diffuser.cfg.frame_stack), (
        "SOPE frame_stack must match the robomimic observation_horizon when "
        f"using {role} guidance: got frame_stack={diffuser.cfg.frame_stack}, "
        f"{role}.observation_horizon={policy.observation_horizon}."
    )


def evaluate_saved_rollout_ope(
    *,
    diffusion_checkpoint_path: Path,
    reward_checkpoint_path: Path,
    data_path: Optional[Path] = None,
    split: str = "eval",
    rollout_batch_size: int = 32,
    max_trajectories: Optional[int] = None,
    target_policy_checkpoint: Optional[Path] = None,
    behavior_policy_checkpoint: Optional[Path] = None,
    target_score_timestep: int = 1,
    behavior_score_timestep: int = 1,
    guidance_config: Optional[GuidanceConfig] = None,
    device: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> SavedRolloutOPEReport:
    """Evaluate unguided or guided saved-rollout OPE from rollout initial states."""
    resolved_device = device or resolve_device(prefer_cuda=True)
    diffusion_checkpoint_path = Path(diffusion_checkpoint_path).resolve()
    reward_checkpoint_path = Path(reward_checkpoint_path).resolve()
    resolved_run_dir = Path(run_dir).resolve() if run_dir is not None else diffusion_checkpoint_path.parent

    guided = target_policy_checkpoint is not None
    resolved_target_policy_checkpoint: Optional[Path] = None
    resolved_behavior_policy_checkpoint: Optional[Path] = None
    active_guidance_config: Optional[GuidanceConfig] = None
    target_policy = None
    behavior_policy = None

    if guided:
        active_guidance_config = guidance_config or GuidanceConfig()
        target_policy, resolved_target_policy_checkpoint = _load_diffusion_policy(
            policy_checkpoint=Path(target_policy_checkpoint),
            device=resolved_device,
            score_timestep=target_score_timestep,
        )
        if active_guidance_config.use_neg_grad:
            if behavior_policy_checkpoint is None:
                raise ValueError(
                    "behavior_policy_checkpoint is required when guidance_config.use_neg_grad=True."
                )
            behavior_policy, resolved_behavior_policy_checkpoint = _load_diffusion_policy(
                policy_checkpoint=Path(behavior_policy_checkpoint),
                device=resolved_device,
                score_timestep=behavior_score_timestep,
            )
        elif behavior_policy_checkpoint is not None:
            resolved_behavior_policy_checkpoint = Path(behavior_policy_checkpoint).expanduser().resolve()

    diffuser, diffusion_payload = load_diffusion_checkpoint(
        diffusion_checkpoint_path,
        device=resolved_device,
        policy=target_policy,
        behavior_policy=behavior_policy if active_guidance_config and active_guidance_config.use_neg_grad else None,
    )
    reward_predictor, _ = load_reward_checkpoint(
        reward_checkpoint_path,
        device=resolved_device,
    )

    if guided and target_policy is not None:
        _validate_guidance_policy_horizon(
            diffuser=diffuser,
            policy=target_policy,
            role="target policy",
        )
    if behavior_policy is not None:
        _validate_guidance_policy_horizon(
            diffuser=diffuser,
            policy=behavior_policy,
            role="behavior policy",
        )

    resolved_data_path = _resolve_data_path(
        data_arg=data_path,
        diffusion_payload=diffusion_payload,
    )
    selected_paths, split_seed, train_fraction = _select_rollout_paths(
        data_path=resolved_data_path,
        split=split,
        diffusion_payload=diffusion_payload,
    )
    if max_trajectories is not None:
        selected_paths = selected_paths[:max_trajectories]
    if not selected_paths:
        raise ValueError("No rollout files selected for OPE evaluation.")

    _seed_everything(split_seed)

    initial_states = _build_initial_states(
        selected_paths,
        dataset_config_payload=diffusion_payload["dataset_config"],
    )
    targets = _load_saved_rollout_targets(
        selected_paths,
        dataset_config_payload=diffusion_payload["dataset_config"],
        ope_gamma=float(diffuser.cfg.ope_gamma),
        max_length=int(diffuser.cfg.trajectory_horizon),
    )

    total_weighted_estimate = 0.0
    total_count = 0
    total_squared_error = 0.0
    total_transition_elements = 0
    sampling_kwargs = {} if active_guidance_config is None else active_guidance_config.to_sampling_kwargs()

    batch_starts = range(0, len(initial_states), rollout_batch_size)
    batch_iterator = tqdm(
        batch_starts,
        desc="Guided OPE" if guided else "OPE",
        unit="batch",
        leave=False,
    )
    with torch.no_grad():
        for start in batch_iterator:
            end = min(start + rollout_batch_size, len(initial_states))
            batch_initial_states = torch.from_numpy(initial_states[start:end]).to(
                diffuser.device,
                dtype=torch.float32,
            )
            batch_states, batch_actions = diffuser.generate_full_trajectory(
                batch_initial_states,
                max_length=targets.horizon,
                guided=guided,
                **sampling_kwargs,
            )
            sq_error, num_elements = _transition_error_totals_torch(
                batch_states,
                batch_actions,
                targets.states[start:end],
                targets.actions[start:end],
            )
            total_squared_error += sq_error
            total_transition_elements += num_elements

            batch_reward_preds_t = _reward_predictions_to_tensor(
                reward_predictor.predict(batch_states, batch_actions),
                device=batch_states.device,
            )
            batch_returns = _discounted_returns_torch(
                batch_reward_preds_t,
                gamma=float(diffuser.cfg.ope_gamma),
            )
            total_weighted_estimate += float(batch_returns.sum(dtype=torch.float64).item())
            total_count += end - start
            batch_iterator.set_postfix(
                ope_return=f"{(total_weighted_estimate / max(total_count, 1)):.4f}",
                rollout_mse=f"{(total_squared_error / max(total_transition_elements, 1)):.4e}",
            )

    return SavedRolloutOPEReport(
        run_dir=resolved_run_dir,
        diffusion_checkpoint=diffusion_checkpoint_path,
        reward_checkpoint=reward_checkpoint_path,
        data=resolved_data_path,
        split=split,
        training_seed=split_seed,
        train_fraction=train_fraction,
        num_rollout_files=len(selected_paths),
        rollout_batch_size=int(rollout_batch_size),
        max_trajectories=max_trajectories,
        rollout_horizon=int(targets.horizon),
        ope_gamma=float(diffuser.cfg.ope_gamma),
        reward_transform=targets.reward_transform,
        device=str(diffuser.device),
        guided=guided,
        autoregressive_rollout_mse=float(total_squared_error / max(total_transition_elements, 1)),
        return_gt_transformed=float(np.mean(targets.transformed_returns)),
        return_ope_estimate=float(total_weighted_estimate / max(total_count, 1)),
        target_policy_checkpoint=resolved_target_policy_checkpoint,
        behavior_policy_checkpoint=resolved_behavior_policy_checkpoint,
        target_score_timestep=int(target_score_timestep) if guided else None,
        behavior_score_timestep=(
            int(behavior_score_timestep)
            if active_guidance_config is not None and active_guidance_config.use_neg_grad
            else None
        ),
        guidance_config=active_guidance_config,
    )


def _resolve_rollout_eval_horizon(max_length: int) -> int:
    resolved_horizon = int(max_length)
    if resolved_horizon <= 0:
        raise ValueError(f"rollout horizon must be positive, got {resolved_horizon}.")
    return resolved_horizon


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return value


def _load_demo_initial_conditions(
    rollout_paths: list[Path],
) -> tuple[list[DemoInitialCondition], Path]:
    demo_specs: list[tuple[Path, Path, str]] = []
    dataset_paths: set[Path] = set()

    for rollout_path in rollout_paths:
        with h5py.File(rollout_path, "r") as handle:
            source_dataset_path = handle.attrs.get("source_dataset_path", None)
            demo_key = handle.attrs.get("demo_key", None)
        if source_dataset_path is None or demo_key is None:
            raise ValueError(
                f"Rollout file {rollout_path} is missing source_dataset_path or demo_key attrs "
                "needed to reconstruct the source-demo initial condition."
            )
        resolved_dataset_path = Path(_decode_attr(source_dataset_path)).expanduser().resolve()
        resolved_demo_key = str(_decode_attr(demo_key))
        demo_specs.append((rollout_path.resolve(), resolved_dataset_path, resolved_demo_key))
        dataset_paths.add(resolved_dataset_path)

    if not demo_specs:
        raise ValueError("No rollout paths were selected for target-policy evaluation.")
    if len(dataset_paths) != 1:
        raise ValueError(
            "The selected rollout files point to multiple source datasets. "
            f"Expected one compatible source dataset, got {sorted(str(path) for path in dataset_paths)}."
        )

    source_dataset_path = next(iter(dataset_paths))
    if not source_dataset_path.is_file():
        raise FileNotFoundError(f"Source robomimic dataset not found: {source_dataset_path}")

    demo_refs: list[DemoInitialCondition] = []
    with h5py.File(source_dataset_path, "r") as handle:
        for rollout_path, _, demo_key in demo_specs:
            demo_group = handle[f"data/{demo_key}"]
            states = np.asarray(demo_group["states"])
            if states.shape[0] <= 0:
                raise ValueError(f"Demo {demo_key!r} in {source_dataset_path} has no simulator states.")

            initial_state: dict[str, Any] = {"states": np.asarray(states[0]).copy()}
            if "model_file" in demo_group.attrs:
                initial_state["model"] = str(_decode_attr(demo_group.attrs["model_file"]))
            ep_meta = demo_group.attrs.get("ep_meta", None)
            if ep_meta is not None:
                initial_state["ep_meta"] = _decode_attr(ep_meta)

            demo_refs.append(
                DemoInitialCondition(
                    rollout_path=rollout_path,
                    source_dataset_path=source_dataset_path,
                    demo_key=demo_key,
                    initial_state=initial_state,
                )
            )

    return demo_refs, source_dataset_path


def _find_import_package_root(package_name: str) -> Path:
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / package_name
        if candidate.is_dir() and (candidate / "__init__.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate package root for {package_name!r} on sys.path.")


def _ensure_writable_mujoco_py_shadow() -> None:
    shadow_root = Path("/tmp/codex_mujoco_py_shadow")
    shadow_package_root = shadow_root / "mujoco_py"
    source_package_root = _find_import_package_root("mujoco_py")
    if not shadow_package_root.is_dir():
        shutil.copytree(source_package_root, shadow_package_root, dirs_exist_ok=True)
    shadow_root_str = str(shadow_root)
    if shadow_root_str not in sys.path:
        sys.path.insert(0, shadow_root_str)


def _build_rollout_env_from_dataset(
    source_dataset_path: Path,
    *,
    policy_checkpoint: Path,
) -> tuple[Any, Path, Path]:
    _ensure_writable_mujoco_py_shadow()
    import third_party.robomimic.robomimic.utils.env_utils as EnvUtils
    import third_party.robomimic.robomimic.utils.file_utils as FileUtils
    from src.robomimic_interface.checkpoints import load_checkpoint

    resolved_dataset_path = source_dataset_path.expanduser().resolve()
    if not resolved_dataset_path.is_file():
        raise FileNotFoundError(f"Source robomimic dataset not found: {resolved_dataset_path}")

    resolved_policy_checkpoint, run_dir, relative_checkpoint = _resolve_policy_checkpoint(policy_checkpoint)
    checkpoint = load_checkpoint(run_dir, ckpt_path=relative_checkpoint)
    env_meta = FileUtils.get_env_metadata_from_dataset(str(resolved_dataset_path))
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=bool(checkpoint.ckpt_dict.get("shape_metadata", {}).get("use_images", False)),
        use_depth_obs=bool(checkpoint.ckpt_dict.get("shape_metadata", {}).get("use_depths", False)),
    )
    config, _ = FileUtils.config_from_checkpoint(
        algo_name=checkpoint.ckpt_dict["algo_name"],
        ckpt_dict=checkpoint.ckpt_dict,
        verbose=False,
    )
    env = EnvUtils.wrap_env_from_config(env, config=config)
    return env, resolved_dataset_path, resolved_policy_checkpoint


def _resolve_feature_dim_from_hook(feature_hook: Any) -> int:
    output_shape = getattr(feature_hook._obs_encoder, "output_shape", None)
    if not callable(output_shape):
        raise RuntimeError("Resolved behavior-policy obs encoder does not expose output_shape().")
    shape = tuple(output_shape())
    if len(shape) != 1:
        raise RuntimeError(
            f"Expected obs_encoder.output_shape() to return a 1D feature width, got {shape}."
        )
    return int(shape[0])


def _filter_obs_dict(
    obs: dict[str, Any],
    *,
    expected_keys: set[str],
    role: str,
) -> dict[str, Any]:
    missing_keys = sorted(key for key in expected_keys if key not in obs)
    if missing_keys:
        raise KeyError(f"{role} observation is missing required keys: {missing_keys}")
    return {key: obs[key] for key in expected_keys}


def _resolve_policy_obs_keys(policy: Any) -> set[str]:
    algo = getattr(policy, "policy", policy)
    obs_shapes = getattr(algo, "obs_shapes", None)
    if obs_shapes is None:
        raise RuntimeError("Could not resolve expected observation keys from rollout policy.")
    return set(obs_shapes.keys())


def _resolve_feature_hook_obs_keys(feature_hook: Any) -> set[str]:
    obs_shapes = getattr(feature_hook._obs_encoder, "obs_shapes", None)
    if obs_shapes is None:
        raise RuntimeError("Could not resolve expected observation keys from behavior feature hook.")
    return set(obs_shapes.keys())


def _rollout_from_initial_state(
    *,
    policy: Any,
    env: Any,
    initial_state: dict[str, Any],
    horizon: int,
    recorder: Any,
) -> Any:
    from src.robomimic_interface.rollout import RolloutStats

    policy_obs_keys = _resolve_policy_obs_keys(policy)
    recorder_obs_keys = _resolve_feature_hook_obs_keys(recorder.feature_hook)

    policy.start_episode()
    obs_raw = env.reset_to(initial_state)
    if obs_raw is None:
        raise RuntimeError("env.reset_to(initial_state) returned None; expected an observation dict.")
    recorder.start_episode(
        _filter_obs_dict(
            obs_raw,
            expected_keys=recorder_obs_keys,
            role="Recorder",
        )
    )

    total_reward = 0.0
    success = False
    step_i = -1
    try:
        for step_i in range(horizon):
            act = policy(
                ob=_filter_obs_dict(
                    obs_raw,
                    expected_keys=policy_obs_keys,
                    role="Policy",
                )
            )
            next_obs_raw, reward, done, info = env.step(act)
            total_reward += float(reward)
            success = bool(env.is_success()["task"])
            recorder.record_step(
                obs=_filter_obs_dict(
                    obs_raw,
                    expected_keys=recorder_obs_keys,
                    role="Recorder",
                ),
                action=act,
                reward=float(reward),
                done=bool(done),
                info=info,
                next_obs=_filter_obs_dict(
                    next_obs_raw,
                    expected_keys=recorder_obs_keys,
                    role="Recorder",
                ),
            )
            if done or success:
                break
            obs_raw = deepcopy(next_obs_raw)
    except env.rollout_exceptions as exc:
        print(f"WARNING: got rollout exception {exc}")
    if step_i < 0:
        raise RuntimeError("Target-policy rollout terminated before recording any steps.")

    return recorder.finalize(
        RolloutStats(
            total_reward=float(total_reward),
            horizon=int(step_i + 1),
            success_rate=float(success),
        )
    )


def _guided_success_mask(
    guided_states: np.ndarray,
    *,
    object_z_index: int,
    height_threshold: float = DEFAULT_GUIDED_SUCCESS_HEIGHT_THRESHOLD,
) -> np.ndarray:
    guided_states = np.asarray(guided_states, dtype=np.float32)
    assert guided_states.ndim == 3, f"Expected guided_states shape [B, T, D], got {guided_states.shape}."
    if guided_states.shape[-1] <= object_z_index:
        raise ValueError(
            "Guided success heuristic requires the Lift cube-height coordinate to remain in the "
            f"state prefix at index {object_z_index}, but state_dim={guided_states.shape[-1]}."
        )
    return np.max(guided_states[:, :, object_z_index], axis=1) > float(height_threshold)


def _obs_encoder_feature_width(
    obs_encoder: Any,
    *,
    key: str,
) -> tuple[int, bool]:
    feat_shape = tuple(obs_encoder.obs_shapes[key])
    randomizers = obs_encoder.obs_randomizers[key]
    for randomizer in randomizers:
        if randomizer is not None:
            feat_shape = tuple(randomizer.output_shape_in(feat_shape))
    obs_nets = obs_encoder.obs_nets
    obs_net = obs_nets[key] if hasattr(obs_nets, "__getitem__") else getattr(obs_nets, key)
    if obs_net is not None:
        feat_shape = tuple(obs_net.output_shape(feat_shape))
    for randomizer in randomizers:
        if randomizer is not None:
            feat_shape = tuple(randomizer.output_shape_out(feat_shape))
    return int(np.prod(feat_shape)), obs_net is None


def _resolve_guided_success_object_z_index(
    feature_hook: Any,
) -> int:
    feature_type = getattr(feature_hook, "feature_type", None)
    if feature_type == "low_dim_concat":
        obs_shapes = feature_hook._get_obs_shapes()
        offset = 0
        for key in feature_hook.low_dim_keys:
            width = int(np.prod(obs_shapes[key]))
            if key == "object":
                if width <= _LIFT_OBJECT_Z_WITHIN_OBJECT:
                    raise ValueError(
                        "Lift guided-success heuristic requires an object observation with at least "
                        f"{_LIFT_OBJECT_Z_WITHIN_OBJECT + 1} coordinates, got width {width}."
                    )
                return offset + _LIFT_OBJECT_Z_WITHIN_OBJECT
            offset += width
        raise ValueError(
            "Lift guided-success heuristic requires the raw 'object' key to be present in the "
            f"low-dimensional feature order, got keys {feature_hook.low_dim_keys}."
        )

    if feature_type == "both":
        obs_encoder = getattr(feature_hook, "_obs_encoder", None)
        if obs_encoder is None or not hasattr(obs_encoder, "obs_shapes"):
            raise RuntimeError(
                "Could not resolve the policy observation encoder needed to infer the guided-success state index."
            )
        offset = 0
        for key in obs_encoder.obs_shapes:
            width, preserves_raw_coordinates = _obs_encoder_feature_width(obs_encoder, key=key)
            if key == "object":
                if not preserves_raw_coordinates:
                    raise ValueError(
                        "Lift guided-success heuristic requires the raw 'object' observation to remain directly "
                        "addressable in feat_type='both', but the policy observation encoder transforms it."
                    )
                if width <= _LIFT_OBJECT_Z_WITHIN_OBJECT:
                    raise ValueError(
                        "Lift guided-success heuristic requires an object observation with at least "
                        f"{_LIFT_OBJECT_Z_WITHIN_OBJECT + 1} coordinates, got width {width}."
                    )
                return offset + _LIFT_OBJECT_Z_WITHIN_OBJECT
            offset += width
        raise ValueError(
            "Lift guided-success heuristic requires the raw 'object' key to be present in the "
            f"obs-encoder feature order, got keys {list(obs_encoder.obs_shapes.keys())}."
        )

    raise ValueError(
        "Lift guided-success heuristic is only defined when the feature space preserves raw object coordinates. "
        f"Unsupported feature_type={feature_type!r}."
    )


def _parse_policy_epoch(path: Path) -> int:
    match = _EPOCH_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not parse target-policy epoch from checkpoint name: {path.name}")
    return int(match.group(1))


def _resolve_target_policy_paths(
    target_policy_dir: Path,
    *,
    max_target_policies: Optional[int],
) -> list[Path]:
    if max_target_policies is not None and max_target_policies <= 0:
        raise ValueError(
            f"max_target_policies must be positive when provided, got {max_target_policies}."
        )

    resolved_dir = target_policy_dir.expanduser().resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Target policy directory not found: {resolved_dir}")

    candidates = sorted({*resolved_dir.glob("model_epoch*.pth"), *resolved_dir.glob("model_epoch*.pt")})
    if not candidates:
        raise FileNotFoundError(
            f"No target-policy checkpoints matching model_epoch*.pth or model_epoch*.pt found under {resolved_dir}"
        )

    scored = [(_parse_policy_epoch(path), path.name, path.resolve()) for path in candidates]
    scored.sort(key=lambda item: (item[0], item[1]))
    resolved_paths = [path for _, _, path in scored]
    if max_target_policies is not None:
        resolved_paths = resolved_paths[:max_target_policies]
    return resolved_paths


def _build_guided_multipolicy_ope_report(
    *,
    run_dir: Path,
    diffusion_checkpoint_path: Path,
    reward_checkpoint_path: Path,
    data_path: Path,
    source_dataset_path: Path,
    split: str,
    training_seed: int,
    train_fraction: float,
    num_rollout_files: int,
    rollout_batch_size: int,
    max_trajectories: Optional[int],
    target_policy_dir: Path,
    max_target_policies: Optional[int],
    configured_rollout_horizon: int,
    rollout_horizon: int,
    ope_gamma: float,
    reward_transform: str,
    device: str,
    behavior_policy_checkpoint: Path,
    rollout_env_dataset_path: Path,
    rollout_env_wrapper_checkpoint: Path,
    guidance_config: GuidanceConfig,
    guided_success_state_index: int,
    guided_success_height_threshold: float,
    policy_reports: list[MultipolicyOPEPolicyReport],
) -> MultipolicyOPEReport:
    assert policy_reports, "Multipolicy OPE report requires at least one policy report."

    pred_means = np.asarray(
        [policy_report.mean_predicted_transformed_return for policy_report in policy_reports],
        dtype=np.float64,
    )
    true_means = np.asarray(
        [policy_report.mean_true_transformed_return for policy_report in policy_reports],
        dtype=np.float64,
    )
    raw_true_means = np.asarray(
        [policy_report.mean_true_raw_return for policy_report in policy_reports],
        dtype=np.float64,
    )

    report = MultipolicyOPEReport(
        run_dir=run_dir,
        diffusion_checkpoint=diffusion_checkpoint_path,
        reward_checkpoint=reward_checkpoint_path,
        data=data_path,
        source_dataset_path=source_dataset_path,
        split=split,
        training_seed=training_seed,
        train_fraction=train_fraction,
        num_rollout_files=num_rollout_files,
        rollout_batch_size=int(rollout_batch_size),
        max_trajectories=max_trajectories,
        target_policy_dir=target_policy_dir,
        max_target_policies=max_target_policies,
        num_target_policies=len(policy_reports),
        configured_rollout_horizon=int(configured_rollout_horizon),
        rollout_horizon=int(rollout_horizon),
        ope_gamma=ope_gamma,
        reward_transform=reward_transform,
        device=device,
        guided=True,
        behavior_policy_checkpoint=behavior_policy_checkpoint,
        rollout_env_dataset_path=rollout_env_dataset_path,
        rollout_env_wrapper_checkpoint=rollout_env_wrapper_checkpoint,
        guidance_config=guidance_config,
        guided_success_state_index=int(guided_success_state_index),
        guided_success_height_threshold=float(guided_success_height_threshold),
        spearman_correlation_transformed=_spearman(pred_means, true_means),
        policy_value_rmse_transformed=_rmse(pred_means, true_means),
        policies=list(policy_reports),
    )
    if reward_transform == "identity":
        report.spearman_correlation_raw = _spearman(pred_means, raw_true_means)
        report.policy_value_rmse_raw = _rmse(pred_means, raw_true_means)
    return report


def evaluate_guided_multipolicy_ope(
    *,
    diffusion_checkpoint_path: Path,
    reward_checkpoint_path: Path,
    target_policy_dir: Path,
    behavior_policy_checkpoint: Path,
    data_path: Optional[Path] = None,
    split: str = "eval",
    rollout_batch_size: int = 32,
    max_trajectories: Optional[int] = None,
    max_target_policies: Optional[int] = None,
    rollout_horizon: int = 80,
    target_score_timestep: int = 1,
    behavior_score_timestep: int = 1,
    guidance_config: Optional[GuidanceConfig] = None,
    guided_success_object_z_index: Optional[int] = DEFAULT_GUIDED_SUCCESS_OBJECT_Z_INDEX,
    guided_success_height_threshold: float = DEFAULT_GUIDED_SUCCESS_HEIGHT_THRESHOLD,
    device: Optional[str] = None,
    run_dir: Optional[Path] = None,
    json_report: Optional[Path] = None,
) -> MultipolicyOPEReport:
    """Evaluate guided multipolicy OPE by comparing generated trajectories to online target rollouts."""
    from src.robomimic_interface.dataset import RolloutChunkDatasetConfig, resolve_reward_transform
    from src.robomimic_interface.rollout import PolicyFeatureHook, RolloutLatentRecorder

    resolved_device = device or resolve_device(prefer_cuda=True)
    diffusion_checkpoint_path = Path(diffusion_checkpoint_path).resolve()
    reward_checkpoint_path = Path(reward_checkpoint_path).resolve()
    resolved_run_dir = Path(run_dir).resolve() if run_dir is not None else diffusion_checkpoint_path.parent
    resolved_json_report = (
        Path(json_report).expanduser().resolve() if json_report is not None else None
    )
    resolved_target_policy_dir = Path(target_policy_dir).expanduser().resolve()
    active_guidance_config = guidance_config or GuidanceConfig()
    resolved_rollout_horizon = _resolve_rollout_eval_horizon(rollout_horizon)

    target_policy_paths = _resolve_target_policy_paths(
        resolved_target_policy_dir,
        max_target_policies=max_target_policies,
    )
    resolved_behavior_policy_checkpoint = Path(behavior_policy_checkpoint).expanduser().resolve()

    behavior_guidance_policy = None
    if active_guidance_config.use_neg_grad:
        behavior_guidance_policy, _ = _load_diffusion_policy(
            policy_checkpoint=resolved_behavior_policy_checkpoint,
            device=resolved_device,
            score_timestep=behavior_score_timestep,
        )

    reward_predictor, reward_payload = load_reward_checkpoint(
        reward_checkpoint_path,
        device=resolved_device,
    )
    reward_dataset_cfg = RolloutChunkDatasetConfig(**reward_payload["dataset_config"])
    reward_transform = resolve_reward_transform(reward_dataset_cfg.reward_transform)

    bootstrap_target_policy, _ = _load_diffusion_policy(
        policy_checkpoint=target_policy_paths[0],
        device=resolved_device,
        score_timestep=target_score_timestep,
    )
    diffuser, diffusion_payload = load_diffusion_checkpoint(
        diffusion_checkpoint_path,
        device=resolved_device,
        policy=bootstrap_target_policy,
        behavior_policy=behavior_guidance_policy if active_guidance_config.use_neg_grad else None,
    )
    _validate_guidance_policy_horizon(
        diffuser=diffuser,
        policy=bootstrap_target_policy,
        role="target policy",
    )
    if behavior_guidance_policy is not None:
        _validate_guidance_policy_horizon(
            diffuser=diffuser,
            policy=behavior_guidance_policy,
            role="behavior policy",
        )

    resolved_data_path = _resolve_data_path(
        data_arg=data_path,
        diffusion_payload=diffusion_payload,
    )
    selected_paths, split_seed, train_fraction = _select_rollout_paths(
        data_path=resolved_data_path,
        split=split,
        diffusion_payload=diffusion_payload,
    )
    if max_trajectories is not None:
        selected_paths = selected_paths[:max_trajectories]
    if not selected_paths:
        raise ValueError("No rollout files selected for OPE evaluation.")

    _seed_everything(split_seed)

    initial_states = _build_initial_states(
        selected_paths,
        dataset_config_payload=diffusion_payload["dataset_config"],
    )
    demo_refs, source_dataset_path = _load_demo_initial_conditions(selected_paths)
    behavior_rollout_policy, resolved_behavior_rollout_checkpoint = _load_rollout_policy(
        policy_checkpoint=resolved_behavior_policy_checkpoint,
        device=resolved_device,
    )
    behavior_feature_hook = PolicyFeatureHook(behavior_rollout_policy, feat_type="both")
    env, rollout_env_dataset_path, rollout_env_wrapper_checkpoint = _build_rollout_env_from_dataset(
        source_dataset_path,
        policy_checkpoint=target_policy_paths[0],
    )

    behavior_feature_dim = _resolve_feature_dim_from_hook(behavior_feature_hook)
    if behavior_feature_dim != int(diffuser.cfg.state_dim):
        raise ValueError(
            "Behavior-policy feature width must match the SOPE diffusion state_dim for target-rollout "
            f"re-encoding. Got behavior_feature_dim={behavior_feature_dim} vs state_dim={diffuser.cfg.state_dim}."
        )
    resolved_guided_success_object_z_index = (
        _resolve_guided_success_object_z_index(behavior_feature_hook)
        if guided_success_object_z_index is None
        else int(guided_success_object_z_index)
    )
    if resolved_guided_success_object_z_index < 0:
        raise ValueError(
            "guided_success_object_z_index must be non-negative, got "
            f"{resolved_guided_success_object_z_index}."
        )

    ope_gamma = float(diffuser.cfg.ope_gamma)
    sampling_kwargs = active_guidance_config.to_sampling_kwargs()
    policy_reports: list[MultipolicyOPEPolicyReport] = []

    for policy_idx, target_policy_path in enumerate(target_policy_paths, start=1):
        target_policy_epoch = _parse_policy_epoch(target_policy_path)
        target_guidance_policy, _ = _load_diffusion_policy(
            policy_checkpoint=target_policy_path,
            device=resolved_device,
            score_timestep=target_score_timestep,
        )
        diffuser, _ = load_diffusion_checkpoint(
            diffusion_checkpoint_path,
            device=resolved_device,
            policy=target_guidance_policy,
            behavior_policy=behavior_guidance_policy if active_guidance_config.use_neg_grad else None,
        )
        _validate_guidance_policy_horizon(
            diffuser=diffuser,
            policy=target_guidance_policy,
            role="target policy",
        )

        guided_states = np.zeros(
            (len(initial_states), resolved_rollout_horizon, int(diffuser.cfg.state_dim)),
            dtype=np.float32,
        )
        guided_actions = np.zeros(
            (len(initial_states), resolved_rollout_horizon, int(diffuser.cfg.action_dim)),
            dtype=np.float32,
        )
        predicted_transformed_returns = np.zeros((len(initial_states),), dtype=np.float32)

        batch_starts = range(0, len(initial_states), rollout_batch_size)
        batch_iterator = tqdm(
            batch_starts,
            desc=f"Guided epoch {target_policy_epoch} ({policy_idx}/{len(target_policy_paths)})",
            unit="batch",
            leave=False,
        )
        with torch.no_grad():
            for start in batch_iterator:
                end = min(start + rollout_batch_size, len(initial_states))
                batch_initial_states = torch.from_numpy(initial_states[start:end]).to(
                    diffuser.device,
                    dtype=torch.float32,
                )
                batch_states, batch_actions = diffuser.generate_full_trajectory(
                    batch_initial_states,
                    max_length=resolved_rollout_horizon,
                    guided=True,
                    **sampling_kwargs,
                )
                batch_reward_preds_t = _reward_predictions_to_tensor(
                    reward_predictor.predict(batch_states, batch_actions),
                    device=batch_states.device,
                )
                batch_states_np = _as_numpy_array(batch_states, dtype=np.float32)
                batch_actions_np = _as_numpy_array(batch_actions, dtype=np.float32)
                guided_states[start:end] = batch_states_np
                guided_actions[start:end] = batch_actions_np
                predicted_transformed_returns[start:end] = _as_numpy_array(
                    _discounted_returns_torch(
                        batch_reward_preds_t,
                        gamma=ope_gamma,
                    ),
                    dtype=np.float32,
                )

        del diffuser
        del target_guidance_policy
        _maybe_free_cuda()

        target_rollout_policy, resolved_target_rollout_checkpoint = _load_rollout_policy(
            policy_checkpoint=target_policy_path,
            device=resolved_device,
        )

        true_transformed_returns: list[float] = []
        true_raw_returns: list[float] = []
        true_successes: list[float] = []
        true_rollout_horizons: list[int] = []
        total_squared_error = 0.0
        total_transition_elements = 0

        rollout_iterator = tqdm(
            enumerate(demo_refs),
            total=len(demo_refs),
            desc=f"True rollout epoch {target_policy_epoch}",
            unit="traj",
            leave=False,
        )
        for traj_idx, demo_ref in rollout_iterator:
            recorder = RolloutLatentRecorder(
                feature_hook=behavior_feature_hook,
                store_obs=False,
                store_next_obs=False,
            )
            target_traj = _rollout_from_initial_state(
                policy=target_rollout_policy,
                env=env,
                initial_state=demo_ref.initial_state,
                horizon=resolved_rollout_horizon,
                recorder=recorder,
            )
            if target_traj.latents is None:
                raise RuntimeError(
                    f"Target rollout for demo {demo_ref.demo_key!r} under {target_policy_path.name} "
                    "did not produce re-encoded latent features."
                )

            aligned_horizon = min(
                int(resolved_rollout_horizon),
                int(target_traj.latents.shape[0]),
                int(target_traj.actions.shape[0]),
                int(target_traj.rewards.shape[0]),
            )
            if aligned_horizon <= 0:
                raise ValueError(
                    f"Target rollout for demo {demo_ref.demo_key!r} under {target_policy_path.name} "
                    "has no valid aligned steps."
                )

            transformed_rewards = np.asarray(
                reward_transform(
                    np.asarray(target_traj.latents[:aligned_horizon], dtype=np.float32),
                    np.asarray(target_traj.rewards[:aligned_horizon], dtype=np.float32),
                ),
                dtype=np.float32,
            )
            true_transformed_returns.append(
                float(_discounted_returns(transformed_rewards, gamma=ope_gamma)[0])
            )
            true_raw_returns.append(float(target_traj.total_reward))
            true_successes.append(float(target_traj.success))
            true_rollout_horizons.append(int(target_traj.horizon))

            sq_error, num_elements = _transition_error_totals(
                guided_states[traj_idx, :aligned_horizon],
                guided_actions[traj_idx, :aligned_horizon],
                np.asarray(target_traj.latents[:aligned_horizon], dtype=np.float32),
                np.asarray(target_traj.actions[:aligned_horizon], dtype=np.float32),
            )
            total_squared_error += sq_error
            total_transition_elements += num_elements

        del target_rollout_policy
        _maybe_free_cuda()

        guided_success_rate = float(
            np.mean(
                _guided_success_mask(
                    guided_states,
                    object_z_index=resolved_guided_success_object_z_index,
                    height_threshold=guided_success_height_threshold,
                )
            )
        )
        rollout_mse = total_squared_error / max(total_transition_elements, 1)
        policy_reports.append(
            MultipolicyOPEPolicyReport(
                target_policy_checkpoint=resolved_target_rollout_checkpoint,
                target_policy_epoch=int(target_policy_epoch),
                num_rollout_files=len(demo_refs),
                rollout_horizon=int(resolved_rollout_horizon),
                mean_predicted_transformed_return=float(np.mean(predicted_transformed_returns)),
                std_predicted_transformed_return=float(np.std(predicted_transformed_returns, dtype=np.float64)),
                mean_true_transformed_return=float(np.mean(true_transformed_returns)),
                std_true_transformed_return=float(np.std(true_transformed_returns, dtype=np.float64)),
                mean_true_raw_return=float(np.mean(true_raw_returns)),
                std_true_raw_return=float(np.std(true_raw_returns, dtype=np.float64)),
                true_rollout_success_rate=float(np.mean(true_successes)),
                guided_rollout_success_rate=guided_success_rate,
                guided_rollout_success_height_threshold=float(guided_success_height_threshold),
                guided_target_rollout_mse=float(rollout_mse),
                guided_target_rollout_rmse=float(np.sqrt(rollout_mse)),
                mean_true_rollout_horizon=float(np.mean(true_rollout_horizons)),
                min_true_rollout_horizon=int(min(true_rollout_horizons)),
                max_true_rollout_horizon=int(max(true_rollout_horizons)),
                target_score_timestep=int(target_score_timestep),
                behavior_score_timestep=int(behavior_score_timestep),
            )
        )
        if resolved_json_report is not None:
            partial_report = _build_guided_multipolicy_ope_report(
                run_dir=resolved_run_dir,
                diffusion_checkpoint_path=diffusion_checkpoint_path,
                reward_checkpoint_path=reward_checkpoint_path,
                data_path=resolved_data_path,
                source_dataset_path=source_dataset_path,
                split=split,
                training_seed=split_seed,
                train_fraction=train_fraction,
                num_rollout_files=len(selected_paths),
                rollout_batch_size=rollout_batch_size,
                max_trajectories=max_trajectories,
                target_policy_dir=resolved_target_policy_dir,
                max_target_policies=max_target_policies,
                configured_rollout_horizon=rollout_horizon,
                rollout_horizon=resolved_rollout_horizon,
                ope_gamma=ope_gamma,
                reward_transform=str(reward_dataset_cfg.reward_transform),
                device=str(resolved_device),
                behavior_policy_checkpoint=resolved_behavior_rollout_checkpoint,
                rollout_env_dataset_path=rollout_env_dataset_path,
                rollout_env_wrapper_checkpoint=rollout_env_wrapper_checkpoint,
                guidance_config=active_guidance_config,
                guided_success_state_index=resolved_guided_success_object_z_index,
                guided_success_height_threshold=guided_success_height_threshold,
                policy_reports=policy_reports,
            )
            # Replacing the same JSON file after each completed policy preserves finished work if
            # a later target-policy rollout fails or the job is interrupted.
            write_report_json(partial_report, json_report=resolved_json_report)

    report = _build_guided_multipolicy_ope_report(
        run_dir=resolved_run_dir,
        diffusion_checkpoint_path=diffusion_checkpoint_path,
        reward_checkpoint_path=reward_checkpoint_path,
        data_path=resolved_data_path,
        source_dataset_path=source_dataset_path,
        split=split,
        training_seed=split_seed,
        train_fraction=train_fraction,
        num_rollout_files=len(selected_paths),
        rollout_batch_size=rollout_batch_size,
        max_trajectories=max_trajectories,
        target_policy_dir=resolved_target_policy_dir,
        max_target_policies=max_target_policies,
        configured_rollout_horizon=rollout_horizon,
        rollout_horizon=resolved_rollout_horizon,
        ope_gamma=ope_gamma,
        reward_transform=str(reward_dataset_cfg.reward_transform),
        device=str(resolved_device),
        behavior_policy_checkpoint=resolved_behavior_rollout_checkpoint,
        rollout_env_dataset_path=rollout_env_dataset_path,
        rollout_env_wrapper_checkpoint=rollout_env_wrapper_checkpoint,
        guidance_config=active_guidance_config,
        guided_success_state_index=resolved_guided_success_object_z_index,
        guided_success_height_threshold=guided_success_height_threshold,
        policy_reports=policy_reports,
    )
    return report
