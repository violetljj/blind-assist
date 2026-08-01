#!/usr/bin/env python3
"""Replay HFTF Stage C C0.1 with parquet-authoritative time."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_stage_c_c0_egowalk_transport import (  # noqa: E402
    _audit_trajectory,
    _load_json,
    _sha256,
)


SCHEMA = "blindassist_hftf_stage_c_c0_1_timebase_repair_audit"
PROTOCOL_SCHEMA = "blindassist_hftf_stage_c_source_feasibility_c0_1"
PROTOCOL_STATUS = "FROZEN_BEFORE_C0_1_SCHEMA_REPAIR_REPLAY"
BASE_PROTOCOL_SCHEMA = "blindassist_hftf_stage_c_source_feasibility_c0"
INVENTORY_SCHEMA = "blindassist_hftf_stage_c_c0_egowalk_inventory"
C0_AUDIT_SCHEMA = "blindassist_hftf_stage_c_c0_egowalk_transport_audit"
ALLOWED_C0_FAILURES = {"rgb_rate_mismatch", "depth_rate_mismatch"}


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = (repo_root / value).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"Frozen replay path leaves repo: {path}") from error
    return path


def _validate_replay_bindings(
    protocol: dict[str, Any],
    protocol_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("Stage C C0.1 protocol is not frozen")
    parent_path = protocol_path.parent / str(
        protocol["parent_result_path"]
    )
    if _sha256(parent_path) != protocol["parent_result_sha256"]:
        raise ValueError("C0.1 parent-result hash mismatch")
    base_binding = protocol["base_c0_protocol"]
    base_path = protocol_path.parent / str(base_binding["path"])
    if _sha256(base_path) != base_binding["sha256"]:
        raise ValueError("C0.1 base C0 protocol hash mismatch")
    base = _load_json(base_path)
    if base.get("schema") != BASE_PROTOCOL_SCHEMA:
        raise ValueError("C0.1 base protocol schema mismatch")
    frozen = protocol["frozen_replay_inputs"]
    inventory_path = _resolve_repo_path(
        repo_root, str(frozen["inventory_path"])
    )
    c0_audit_path = _resolve_repo_path(
        repo_root, str(frozen["c0_transport_audit_path"])
    )
    if _sha256(inventory_path) != frozen["inventory_sha256"]:
        raise ValueError("C0.1 frozen inventory hash mismatch")
    if (
        _sha256(c0_audit_path)
        != frozen["c0_transport_audit_sha256"]
    ):
        raise ValueError("C0.1 frozen C0 audit hash mismatch")
    inventory = _load_json(inventory_path)
    old_audit = _load_json(c0_audit_path)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("C0.1 frozen inventory schema mismatch")
    if (
        old_audit.get("schema") != C0_AUDIT_SCHEMA
        or old_audit.get("terminal")
        != "C0_EGOWALK_MEDIA_TRANSPORT_NOT_EVALUABLE"
    ):
        raise ValueError("C0.1 frozen predecessor audit mismatch")
    selected_ids = [
        item["trajectory"] for item in inventory["selected_trajectories"]
    ]
    if selected_ids != frozen["trajectory_ids"]:
        raise ValueError("C0.1 frozen trajectory set mismatch")
    old_by_id = {
        item["trajectory"]: item for item in old_audit["trajectory_reports"]
    }
    if set(old_by_id) != set(selected_ids):
        raise ValueError("C0.1 predecessor trajectory reports mismatch")
    for trajectory in selected_ids:
        failures = set(old_by_id[trajectory]["gate_failures"])
        if failures != ALLOWED_C0_FAILURES:
            raise ValueError(
                f"C0.1 predecessor has non-timebase failures: "
                f"{trajectory}: {sorted(failures)}"
            )
    return base, inventory, old_audit


def _parquet_timeline_metrics(path: Path) -> dict[str, Any]:
    frame = pl.read_parquet(path).sort("frame")
    rows = frame.height
    frames = frame["frame"].to_list()
    timestamps = frame["timestamp"].to_list()
    deltas = [
        int(later) - int(earlier)
        for earlier, later in zip(timestamps, timestamps[1:])
    ]
    median = float(statistics.median(deltas)) if deltas else None
    return {
        "rows": rows,
        "frame_zero_contiguous": frames == list(range(rows)),
        "timestamps_strictly_increasing": bool(deltas)
        and all(delta > 0 for delta in deltas),
        "minimum_timestamp_delta_ms": min(deltas) if deltas else None,
        "median_timestamp_delta_ms": median,
        "maximum_timestamp_delta_ms": max(deltas) if deltas else None,
        "effective_rate_hz": 1000.0 / median if median else None,
    }


def audit(
    protocol_path: Path,
    media_root: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    base, inventory, old_audit = _validate_replay_bindings(
        protocol, protocol_path, repo_root
    )
    selected = inventory["selected_trajectories"]
    repair_gates = protocol[
        "c0_1_transport_gates_each_selected_trajectory"
    ]
    meta_info = _load_json(media_root / "meta/info.json")
    meta_fps_pass = int(meta_info["fps"]) == int(
        repair_gates["meta_info_fps_must_equal"]
    )
    reports: list[dict[str, Any]] = []
    for item in selected:
        replay = _audit_trajectory(item, media_root, base)
        trajectory = item["trajectory"]
        parquet = media_root / item["repo_paths"]["pose"]
        timeline = _parquet_timeline_metrics(parquet)
        failures: list[str] = []
        if set(replay["gate_failures"]) != ALLOWED_C0_FAILURES:
            failures.append("replay_has_non_timebase_base_c0_failure")
        if not meta_fps_pass:
            failures.append("meta_info_fps_mismatch")
        if (
            timeline["rows"] != int(item["rows"])
            or not timeline["frame_zero_contiguous"]
            or not timeline["timestamps_strictly_increasing"]
        ):
            failures.append("parquet_frame_or_timestamp_alignment_failure")
        median_bounds = repair_gates[
            "parquet_median_timestamp_delta_ms_inclusive"
        ]
        if not (
            float(median_bounds[0])
            <= float(timeline["median_timestamp_delta_ms"])
            <= float(median_bounds[1])
        ):
            failures.append("parquet_median_timestamp_delta_failure")
        every_bounds = repair_gates[
            "every_parquet_timestamp_delta_ms_inclusive"
        ]
        if not (
            int(every_bounds[0])
            <= int(timeline["minimum_timestamp_delta_ms"])
            and int(timeline["maximum_timestamp_delta_ms"])
            <= int(every_bounds[1])
        ):
            failures.append("parquet_timestamp_delta_failure")
        rate_bounds = repair_gates[
            "effective_parquet_rate_hz_inclusive"
        ]
        if not (
            float(rate_bounds[0])
            <= float(timeline["effective_rate_hz"])
            <= float(rate_bounds[1])
        ):
            failures.append("effective_parquet_rate_failure")
        reports.append(
            {
                "trajectory": trajectory,
                "parquet_timeline": timeline,
                "container_nominal_rate_hz_record_only": {
                    "rgb": replay["rgb_stream"]["rate_hz"],
                    "depth": replay["depth_stream"]["rate_hz"],
                },
                "frame_counts": {
                    "pose": timeline["rows"],
                    "rgb": replay["rgb_stream"]["decoded_frame_count"],
                    "depth": replay["depth_stream"]["decoded_frame_count"],
                },
                "rgb_pts_strict_constant": (
                    replay["rgb_stream"]["pts_strictly_increasing"]
                    and replay["rgb_stream"]["pts_constant_step"]
                ),
                "depth_pts_strict_constant": (
                    replay["depth_stream"]["pts_strictly_increasing"]
                    and replay["depth_stream"]["pts_constant_step"]
                ),
                "surface_observability_pass_unchanged": replay[
                    "surface_observability_pass"
                ],
                "base_c0_replay_failures": replay["gate_failures"],
                "c0_1_gate_failures": failures,
                "c0_1_pass": not failures
                and replay["surface_observability_pass"],
            }
        )
    transport_pass = meta_fps_pass and all(
        item["c0_1_pass"] for item in reports
    )
    surface_pass = all(
        item["surface_observability_pass_unchanged"] for item in reports
    )
    if not transport_pass:
        terminal = "C0_1_FRAME_INDEX_TIMEBASE_REPAIR_NOT_EVALUABLE"
    elif not surface_pass:
        terminal = "C0_1_NATURAL_SURFACE_OBSERVABILITY_NOT_EVALUABLE"
    else:
        terminal = (
            "C0_1_STAGE_C_SOURCE_TRANSPORT_FEASIBILITY_SUPPORTED"
        )
    frozen = protocol["frozen_replay_inputs"]
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "claim_ceiling": protocol["claim_ceiling"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "base_c0_protocol_sha256": protocol["base_c0_protocol"]["sha256"],
        "inventory_sha256": frozen["inventory_sha256"],
        "predecessor_c0_audit_sha256": frozen[
            "c0_transport_audit_sha256"
        ],
        "meta_info_fps": int(meta_info["fps"]),
        "meta_info_fps_pass": meta_fps_pass,
        "trajectory_reports": reports,
        "all_repaired_transport_gates_pass": transport_pass,
        "all_unchanged_surface_observability_gates_pass": surface_pass,
        "container_nominal_rate_used_as_physical_timeline": False,
        "semantic_class_input_read": False,
        "annotation_input_read": False,
        "teacher_label_outcome_computed": False,
        "student_training_or_output_computed": False,
        "hazard_or_safe_truth_claimed": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
        "success_authority": protocol["success_authority"],
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
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = audit(
            args.protocol.resolve(),
            args.media_root.resolve(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "all_repaired_transport_gates_pass": report[
                        "all_repaired_transport_gates_pass"
                    ],
                    "all_unchanged_surface_observability_gates_pass": report[
                        "all_unchanged_surface_observability_gates_pass"
                    ],
                    "output": str(output),
                }
            )
        )
        return 0 if report["terminal"].endswith("_SUPPORTED") else 3
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
