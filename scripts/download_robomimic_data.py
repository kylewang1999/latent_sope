#!/usr/bin/env python3
"""Download released robomimic datasets into this repository's data tree.

[robomimic 0.5 documentation](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)

Examples:
    # Download Lift proficient-human low-dimensional data.
    python scripts/download_robomimic_data.py --task lift --quality ph --hdf5-type low_dim

    # Download Lift mixed-human data with the image HDF5 type.
    python scripts/download_robomimic_data.py --task lift --quality mh --hdf5-type raw
    
    # Post-process the downloaded raw dataset to create image observations.
    python third_party/robomimic/robomimic/scripts/dataset_states_to_obs.py \
        --dataset data/robomimic/lift/mh/demo_v15.hdf5 \
        --output_name image_v15.hdf5 \
        --done_mode 2 \
        --camera_names agentview robot0_eye_in_hand \
        --camera_height 84 \
        --camera_width 84 \
        --compress

Robomimic HDF5 type names combine two axes: observation storage and reward
labeling.

- ``raw``: source replay data, usually simulator states, actions, metadata, and
  enough information to regenerate observations. This is not normally the
  train-ready observation dataset.
- ``low_dim``: train-ready low-dimensional observations such as robot
  proprioception, end-effector pose, gripper state, and object state. This is
  compact and fast to load.
- ``image``: train-ready RGB camera observations such as ``agentview_image`` or
  ``robot0_eye_in_hand_image``, usually alongside relevant low-dimensional
  keys. This is much larger and slower to load.
- ``*_sparse``: sparse task-completion rewards.
- ``*_dense``: dense or shaped environment rewards.

For example, ``low_dim_sparse`` means low-dimensional observations with sparse
rewards, and ``image_dense`` means RGB observations with dense rewards.
Simulated image datasets are often registered by robomimic but not directly
downloadable; they are generated from ``raw`` datasets with robomimic's
observation extraction scripts.

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOMIMIC_ROOT = REPO_ROOT / "third_party" / "robomimic"
for import_root in (REPO_ROOT, ROBOMIMIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.utils import PATHS

from third_party.robomimic.robomimic import DATASET_REGISTRY, HF_REPO_ID
import third_party.robomimic.robomimic.utils.file_utils as FileUtils


TASK_ALIASES = {
    "cube_lift": "lift",
    "cube-lift": "lift",
    "lift_cube": "lift",
    "lift-cube": "lift",
}
QUALITY_ALIASES = {
    "mixed-human": "mh",
    "mixed_human": "mh",
    "multi-human": "mh",
    "multi_human": "mh",
    "proficient-human": "ph",
    "proficient_human": "ph",
    "machine-generated": "mg",
    "machine_generated": "mg",
}
HDF5_ALIASES = {
    "low-dim": "low_dim",
    "lowdim": "low_dim",
    "low-dim-sparse": "low_dim_sparse",
    "lowdim-sparse": "low_dim_sparse",
    "low_dim-sparse": "low_dim_sparse",
    "low-dim-dense": "low_dim_dense",
    "lowdim-dense": "low_dim_dense",
    "low_dim-dense": "low_dim_dense",
    "image-sparse": "image_sparse",
    "image-dense": "image_dense",
    "demo": "raw",
}


@dataclass(frozen=True)
class DatasetSelection:
    task: str
    quality: str
    hdf5_type: str
    url: str | None
    horizon: int | None
    output_dir: Path
    output_path: Path | None


def _normalize_token(token: str, aliases: dict[str, str]) -> str:
    normalized = token.strip().lower()
    return aliases.get(normalized, normalized)


def _basename_from_url(url: str | None) -> str | None:
    if url is None:
        return None
    return Path(url).name


def _format_available(values: Iterable[str]) -> str:
    return ", ".join(sorted(values))


def _expand_tasks(tasks: Sequence[str]) -> list[str]:
    normalized = [_normalize_token(task, TASK_ALIASES) for task in tasks]
    special = {"all", "sim", "real"}.intersection(normalized)
    if special:
        if len(normalized) != 1:
            raise ValueError("Use only one special task selector at a time: all, sim, or real.")
        selector = normalized[0]
        all_tasks = sorted(DATASET_REGISTRY)
        if selector == "all":
            return all_tasks
        if selector == "sim":
            return [task for task in all_tasks if "real" not in task]
        return [task for task in all_tasks if "real" in task]

    unknown = [task for task in normalized if task not in DATASET_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown task(s): {_format_available(unknown)}. "
            f"Available tasks: {_format_available(DATASET_REGISTRY)}."
        )
    return normalized


def _expand_qualities(task: str, qualities: Sequence[str]) -> list[str]:
    available = DATASET_REGISTRY[task]
    if len(qualities) == 1 and qualities[0].lower() == "all":
        return sorted(available)

    normalized = [_normalize_token(quality, QUALITY_ALIASES) for quality in qualities]
    unknown = [quality for quality in normalized if quality not in available]
    if unknown:
        raise ValueError(
            f"Task {task!r} does not have quality/dataset type(s): "
            f"{_format_available(unknown)}. Available for this task: "
            f"{_format_available(available)}."
        )
    return normalized


def _expand_hdf5_types(task: str, quality: str, hdf5_types: Sequence[str]) -> list[str]:
    available = DATASET_REGISTRY[task][quality]
    if len(hdf5_types) == 1 and hdf5_types[0].lower() == "all":
        return sorted(available)

    normalized = [_normalize_token(hdf5_type, HDF5_ALIASES) for hdf5_type in hdf5_types]
    unknown = [hdf5_type for hdf5_type in normalized if hdf5_type not in available]
    if unknown:
        raise ValueError(
            f"Dataset {task}/{quality} does not have HDF5 type(s): "
            f"{_format_available(unknown)}. Available for this dataset: "
            f"{_format_available(available)}."
        )
    return normalized


def resolve_selections(
    *,
    tasks: Sequence[str],
    qualities: Sequence[str],
    hdf5_types: Sequence[str],
    output_root: Path,
) -> list[DatasetSelection]:
    selections: list[DatasetSelection] = []
    for task in _expand_tasks(tasks):
        for quality in _expand_qualities(task, qualities):
            for hdf5_type in _expand_hdf5_types(task, quality, hdf5_types):
                metadata = DATASET_REGISTRY[task][quality][hdf5_type]
                url = metadata.get("url")
                output_dir = output_root / task / quality
                basename = _basename_from_url(url)
                selections.append(
                    DatasetSelection(
                        task=task,
                        quality=quality,
                        hdf5_type=hdf5_type,
                        url=url,
                        horizon=metadata.get("horizon"),
                        output_dir=output_dir,
                        output_path=(output_dir / basename) if basename is not None else None,
                    )
                )
    return selections


def print_registry() -> None:
    for task in sorted(DATASET_REGISTRY):
        print(task)
        for quality in sorted(DATASET_REGISTRY[task]):
            hdf5_types = _format_available(DATASET_REGISTRY[task][quality])
            print(f"  {quality}: {hdf5_types}")


def _print_selection(selection: DatasetSelection, *, dry_run: bool) -> None:
    mode = "Would download" if dry_run else "Downloading"
    print(
        f"{mode}: task={selection.task} quality={selection.quality} "
        f"hdf5_type={selection.hdf5_type}"
    )
    if selection.url is not None:
        print(f"  source: {selection.url}")
        print(f"  output: {selection.output_path}")
    else:
        print("  source: unavailable in the robomimic registry")
        print(f"  output dir: {selection.output_dir}")


def download_selection(
    selection: DatasetSelection,
    *,
    dry_run: bool,
    overwrite: bool,
    skip_unavailable: bool,
) -> str:
    _print_selection(selection, dry_run=dry_run)

    if selection.url is None:
        message = (
            "No direct download URL is registered for this HDF5 type. "
            "For simulated image datasets, download the raw dataset and generate "
            "image observations with robomimic's dataset_states_to_obs script."
        )
        if skip_unavailable:
            print(f"  skipped: {message}")
            return "unavailable"
        raise ValueError(message)

    assert selection.output_path is not None
    if selection.output_path.exists():
        if not overwrite:
            print(f"  skipped: file already exists, use --overwrite to replace it")
            return "exists"
        if dry_run:
            print("  dry run: would overwrite existing file")
            return "dry_run"
        selection.output_path.unlink()

    if dry_run:
        print("  dry run: download skipped")
        return "dry_run"

    selection.output_dir.mkdir(parents=True, exist_ok=True)
    

    if "real" in selection.task:
        FileUtils.download_url(
            url=selection.url,
            download_dir=str(selection.output_dir),
            check_overwrite=False,
        )
    else:
        FileUtils.download_file_from_hf(
            repo_id=HF_REPO_ID,
            filename=selection.url,
            download_dir=str(selection.output_dir),
            check_overwrite=False,
        )
    print("  done")
    return "downloaded"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download released robomimic datasets into data/robomimic/<task>/<quality>."
        )
    )
    parser.add_argument(
        "--task",
        "--tasks",
        dest="tasks",
        nargs="+",
        default=["lift"],
        help=(
            "Task(s) to download. Defaults to lift. Use all, sim, real, or names "
            "such as lift, can, square, transport, tool_hang."
        ),
    )
    parser.add_argument(
        "--quality",
        "--qualities",
        "--dataset-type",
        "--dataset-types",
        dest="qualities",
        nargs="+",
        default=["mh"],
        help=(
            "Dataset quality/type(s). Defaults to mh. Common values: ph, mh, mg, paired. "
            "Aliases such as mixed-human and proficient-human are accepted."
        ),
    )
    parser.add_argument(
        "--hdf5-type",
        "--hdf5-types",
        dest="hdf5_types",
        nargs="+",
        default=["low_dim"],
        help=(
            "HDF5 observation type(s). Defaults to low_dim. Common values: raw, "
            "low_dim, image, low_dim_sparse, low_dim_dense."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PATHS.data_root / "robomimic",
        help="Root directory for downloaded data. Defaults to data/robomimic.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace an existing destination file.",
    )
    parser.add_argument(
        "--skip-unavailable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip registry entries without direct download URLs instead of raising.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Print the selected datasets and output paths without downloading. "
            "Enabled by default; pass --no-dry-run to download."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available task / quality / HDF5-type combinations and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_registry()
        return 0

    output_root = args.output_root.expanduser().resolve()
    selections = resolve_selections(
        tasks=args.tasks,
        qualities=args.qualities,
        hdf5_types=args.hdf5_types,
        output_root=output_root,
    )
    if not selections:
        raise RuntimeError("No robomimic datasets matched the requested selection.")

    counts = {"downloaded": 0, "exists": 0, "unavailable": 0, "dry_run": 0}
    for selection in selections:
        status = download_selection(
            selection,
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
            skip_unavailable=bool(args.skip_unavailable),
        )
        counts[status] = counts.get(status, 0) + 1
        print("")

    print(
        "Summary: "
        f"downloaded={counts.get('downloaded', 0)} "
        f"exists={counts.get('exists', 0)} "
        f"unavailable={counts.get('unavailable', 0)} "
        f"dry_run={counts.get('dry_run', 0)}"
    )
    print(f"Dataset root: {output_root}")
    print("Use ROBOMIMIC_DATASET_DIR to point other scripts at this root when needed.")
    if args.dry_run:
        print("Dry run is enabled by default. Pass --no-dry-run to download files.")

    no_action = counts.get("downloaded", 0) == 0 and counts.get("dry_run", 0) == 0
    all_unavailable = counts.get("unavailable", 0) == len(selections)
    return 1 if not args.dry_run and no_action and all_unavailable else 0


if __name__ == "__main__":
    raise SystemExit(main())
