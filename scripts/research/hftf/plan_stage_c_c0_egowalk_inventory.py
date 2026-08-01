#!/usr/bin/env python3
"""Recompute the frozen HFTF Stage C C0 EgoWalk metadata cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from huggingface_hub import HfApi


SCHEMA = "blindassist_hftf_stage_c_c0_egowalk_inventory"
PROTOCOL_SCHEMA = "blindassist_hftf_stage_c_source_feasibility_c0"
PROTOCOL_STATUS = "FROZEN_BEFORE_C0_MEDIA_CONTENT_OR_GEOMETRY_OUTCOME"
POSE_COLUMNS = (
    "cart_x",
    "cart_y",
    "cart_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_protocol(
    protocol: dict[str, Any], protocol_path: Path
) -> None:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("Stage C C0 protocol is not frozen")
    parent_path = protocol_path.parent / str(
        protocol["parent_result_path"]
    )
    if _sha256(parent_path) != protocol["parent_result_sha256"]:
        raise ValueError("Stage C C0 parent-result hash mismatch")


def _repo_file_inventory(source: dict[str, Any]) -> dict[str, Any]:
    api = HfApi()
    files: dict[str, Any] = {}
    for entry in api.list_repo_tree(
        source["dataset_repo"],
        repo_type=source["repo_type"],
        revision=source["revision"],
        recursive=True,
        expand=True,
    ):
        path = getattr(entry, "path", None)
        if not path or not (
            path.endswith(".parquet")
            or path.endswith(".mp4")
            or path.endswith(".mkv")
        ):
            continue
        lfs = getattr(entry, "lfs", None)
        files[path] = {
            "size_bytes": int(getattr(entry, "size", 0)),
            "sha256": getattr(lfs, "sha256", None) if lfs else None,
        }
    return files


def _expected_paths(trajectory: str) -> dict[str, str]:
    return {
        "pose": f"data/{trajectory}.parquet",
        "rgb": f"video/rgb/{trajectory}__rgb.mp4",
        "depth": f"video/depth/{trajectory}__depth.mkv",
    }


def _metadata_metrics(
    trajectory: str,
    parquet_path: Path,
    height: Any,
    files: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    frame = pl.read_parquet(parquet_path).sort("frame")
    missing_columns = sorted(
        set(("timestamp", "trajectory", "frame", *POSE_COLUMNS))
        - set(frame.columns)
    )
    if missing_columns:
        raise ValueError(
            f"{trajectory}: missing parquet columns {missing_columns}"
        )
    rows = frame.height
    frames = frame["frame"].to_list()
    timestamps = frame["timestamp"].to_list()
    deltas = [
        int(later) - int(earlier)
        for earlier, later in zip(timestamps, timestamps[1:])
    ]
    trajectory_values = frame["trajectory"].unique().to_list()
    null_frames = int(
        frame.select(
            pl.any_horizontal(
                [pl.col(column).is_null() for column in POSE_COLUMNS]
            ).sum()
        ).item()
    )
    quaternion_error: float | None = None
    max_distance: float | None = None
    max_step: float | None = None
    reset_count: int | None = None
    if null_frames == 0 and rows > 0:
        quaternion_norm = (
            frame["quat_x"] ** 2
            + frame["quat_y"] ** 2
            + frame["quat_z"] ** 2
            + frame["quat_w"] ** 2
        ).sqrt()
        quaternion_error = float((quaternion_norm - 1.0).abs().max())
        xyz = frame.select(["cart_x", "cart_y", "cart_z"]).to_numpy()
        origin = xyz[0]
        distance = np.sqrt(np.sum((xyz - origin) ** 2, axis=1))
        max_distance = float(np.max(distance))
        steps = np.sqrt(np.sum((xyz[1:] - xyz[:-1]) ** 2, axis=1))
        max_step = float(np.max(steps)) if len(steps) else 0.0
        reset_count = sum(
            1
            for index in range(1, len(distance))
            if distance[index - 1] > 3.0
            and distance[index] < 0.25
            and steps[index - 1] > 0.5
        )
    paths = _expected_paths(trajectory)
    file_records = {name: files.get(path) for name, path in paths.items()}
    reasons: list[str] = []
    if rows < int(gates["minimum_rows"]):
        reasons.append("too_few_rows")
    if trajectory_values != [trajectory]:
        reasons.append("trajectory_column_mismatch")
    if frames != list(range(rows)):
        reasons.append("frame_sequence_not_zero_contiguous")
    if not deltas or not all(delta > 0 for delta in deltas):
        reasons.append("timestamps_not_strict")
    else:
        median_bounds = gates["median_timestamp_delta_ms_inclusive"]
        if not (
            float(median_bounds[0])
            <= float(statistics.median(deltas))
            <= float(median_bounds[1])
        ):
            reasons.append("median_timestamp_delta_out_of_bounds")
        every_bounds = gates["every_timestamp_delta_ms_inclusive"]
        if not all(
            int(every_bounds[0]) <= delta <= int(every_bounds[1])
            for delta in deltas
        ):
            reasons.append("timestamp_delta_out_of_bounds")
    if null_frames:
        reasons.append("pose_null_frames")
    if (
        quaternion_error is None
        or not math.isfinite(quaternion_error)
        or quaternion_error
        > float(gates["maximum_quaternion_abs_norm_error"])
    ):
        reasons.append("quaternion_norm_invalid")
    if (
        max_distance is None
        or max_distance < float(gates["minimum_max_distance_from_start_m"])
    ):
        reasons.append("insufficient_motion")
    if (
        max_step is None
        or max_step > float(gates["maximum_single_step_translation_m"])
    ):
        reasons.append("single_step_translation_too_large")
    if reset_count is None or reset_count:
        reasons.append("odometry_reinitialization_candidate")
    if height is None or not math.isfinite(float(height)):
        reasons.append("camera_height_missing")
    for name, record in file_records.items():
        if (
            not isinstance(record, dict)
            or not record.get("sha256")
            or int(record.get("size_bytes", 0)) <= 0
        ):
            reasons.append(f"{name}_lfs_binding_missing")
    if file_records["pose"] and (
        _sha256(parquet_path) != file_records["pose"]["sha256"]
    ):
        reasons.append("local_pose_sha256_mismatch")
    total_bytes = sum(
        int(record["size_bytes"])
        for record in file_records.values()
        if isinstance(record, dict)
    )
    return {
        "trajectory": trajectory,
        "recording_date": trajectory[:10],
        "metadata_healthy": not reasons,
        "rejection_reasons": reasons,
        "rows": rows,
        "null_pose_frame_count": null_frames,
        "timestamp_delta_ms": {
            "minimum": min(deltas) if deltas else None,
            "median": statistics.median(deltas) if deltas else None,
            "maximum": max(deltas) if deltas else None,
        },
        "quaternion_max_abs_norm_error": quaternion_error,
        "maximum_distance_from_start_m": max_distance,
        "maximum_single_step_translation_m": max_step,
        "reinitialization_candidate_count": reset_count,
        "camera_height_m": float(height) if height is not None else None,
        "repo_paths": paths,
        "files": file_records,
        "total_bytes": total_bytes,
    }


def plan(
    protocol_path: Path,
    metadata_root: Path,
    files: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    _validate_protocol(protocol, protocol_path)
    source = protocol["source_roles"]["egowalk"]
    bindings = source["metadata_bindings"]
    meta_root = metadata_root / "meta"
    binding_paths = {
        "info_json_sha256": meta_root / "info.json",
        "camera_rgb_json_sha256": meta_root / "camera_rgb.json",
        "heights_json_sha256": meta_root / "heights.json",
        "trajectories_json_sha256": meta_root / "trajectories.json",
    }
    for name, path in binding_paths.items():
        if _sha256(path) != bindings[name]:
            raise ValueError(f"EgoWalk metadata binding mismatch: {name}")
    info = _load_json(meta_root / "info.json")
    if int(info["fps"]) != int(
        protocol["canonical_temporal_contract"]["timeline_hz"]
    ):
        raise ValueError("EgoWalk metadata FPS does not match C0 timeline")
    heights = _load_json(meta_root / "heights.json")
    with (meta_root / "trajectories.json").open(
        "r", encoding="utf-8"
    ) as handle:
        trajectories = json.load(handle)
    if (
        not isinstance(trajectories, list)
        or len(trajectories) != int(bindings["trajectory_count"])
        or len(trajectories) != len(set(trajectories))
    ):
        raise ValueError("EgoWalk trajectory inventory count drift")
    parquet_names = sorted(
        path.stem for path in (metadata_root / "data").glob("*.parquet")
    )
    if parquet_names != sorted(trajectories):
        raise ValueError("Local EgoWalk parquet set does not match metadata")
    files = files if files is not None else _repo_file_inventory(source)
    gates = source["metadata_health_gates"]
    ledger = [
        _metadata_metrics(
            str(trajectory),
            metadata_root / "data" / f"{trajectory}.parquet",
            heights.get(str(trajectory)),
            files,
            gates,
        )
        for trajectory in sorted(trajectories)
    ]
    healthy = sorted(
        (item for item in ledger if item["metadata_healthy"]),
        key=lambda item: (item["total_bytes"], item["trajectory"]),
    )
    selected: list[dict[str, Any]] = []
    used_dates: set[str] = set()
    required = int(source["selection_rule"]["required_trajectories"])
    for item in healthy:
        if item["recording_date"] in used_dates:
            continue
        selected.append(item)
        used_dates.add(item["recording_date"])
        if len(selected) == required:
            break
    selected_ids = [item["trajectory"] for item in selected]
    expected = source["expected_metadata_only_selected_cohort"]
    terminal = (
        "C0_EGOWALK_METADATA_COHORT_LOCKED"
        if selected_ids == expected and len(selected) == required
        else "C0_EGOWALK_METADATA_SELECTION_NOT_EVALUABLE"
    )
    reason_counts = Counter(
        reason for item in ledger for reason in item["rejection_reasons"]
    )
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": protocol["workflow_profile"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "dataset_repo": source["dataset_repo"],
        "dataset_revision": source["revision"],
        "metadata_root": str(metadata_root.resolve()),
        "trajectory_count": len(ledger),
        "metadata_healthy_count": len(healthy),
        "metadata_rejection_reason_counts": dict(sorted(reason_counts.items())),
        "selection_order": source["selection_rule"]["order"],
        "selected_trajectories": selected,
        "expected_selected_trajectory_ids": expected,
        "selection_matches_frozen_expected_cohort": selected_ids == expected,
        "inventory_ledger": ledger,
        "rgb_or_depth_media_content_read": False,
        "annotation_outcome_read": False,
        "teacher_label_outcome_read": False,
        "student_output_read": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = plan(
            args.protocol.resolve(),
            args.metadata_root.resolve(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "trajectory_count": report["trajectory_count"],
                    "metadata_healthy_count": report[
                        "metadata_healthy_count"
                    ],
                    "selected_trajectory_ids": [
                        item["trajectory"]
                        for item in report["selected_trajectories"]
                    ],
                    "output": str(output),
                }
            )
        )
        return 0 if report["terminal"].endswith("_LOCKED") else 3
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
