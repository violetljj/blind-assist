#!/usr/bin/env python3
"""Audit the frozen HFTF Stage C C0 EgoWalk media transports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


SCHEMA = "blindassist_hftf_stage_c_c0_egowalk_transport_audit"
PROTOCOL_SCHEMA = "blindassist_hftf_stage_c_source_feasibility_c0"
PROTOCOL_STATUS = "FROZEN_BEFORE_C0_MEDIA_CONTENT_OR_GEOMETRY_OUTCOME"
INVENTORY_SCHEMA = "blindassist_hftf_stage_c_c0_egowalk_inventory"
INVENTORY_TERMINAL = "C0_EGOWALK_METADATA_COHORT_LOCKED"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sample_indices(rows: int, count: int = 32) -> list[int]:
    if rows <= 0 or count <= 1:
        raise ValueError("Sample index dimensions must be positive")
    return sorted(
        {
            math.floor(index * (rows - 1) / (count - 1))
            for index in range(count)
        }
    )


def _validate_inputs(
    protocol: dict[str, Any],
    protocol_path: Path,
    inventory: dict[str, Any],
) -> list[dict[str, Any]]:
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
    if (
        inventory.get("schema") != INVENTORY_SCHEMA
        or inventory.get("terminal") != INVENTORY_TERMINAL
        or inventory.get("protocol_sha256") != _sha256(protocol_path)
    ):
        raise ValueError("C0 inventory is not locked to current protocol")
    selected = inventory.get("selected_trajectories")
    if not isinstance(selected, list):
        raise ValueError("C0 selected trajectory list is missing")
    selected_ids = [item.get("trajectory") for item in selected]
    expected = protocol["source_roles"]["egowalk"][
        "expected_metadata_only_selected_cohort"
    ]
    if (
        selected_ids != expected
        or not inventory.get("selection_matches_frozen_expected_cohort")
    ):
        raise ValueError("C0 inventory cohort differs from frozen cohort")
    if (
        inventory.get("rgb_or_depth_media_content_read")
        or inventory.get("annotation_outcome_read")
        or inventory.get("teacher_label_outcome_read")
        or inventory.get("student_output_read")
    ):
        raise ValueError("C0 inventory selection firewall violated")
    return selected


def _decode_stream(
    path: Path,
    role: str,
    sample_indices: list[int],
) -> dict[str, Any]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required for C0 media audit") from error
    sample_set = set(sample_indices)
    samples: dict[int, np.ndarray] = {}
    pts: list[int] = []
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(f"{path}: expected exactly one video stream")
        stream = streams[0]
        rate = stream.average_rate or stream.base_rate
        rate_hz = float(rate) if rate is not None else None
        width = int(stream.width)
        height = int(stream.height)
        source_pixel_format = (
            stream.format.name if stream.format is not None else None
        )
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"{path}: decoded frame without PTS")
            pts.append(int(frame.pts))
            if index not in sample_set:
                continue
            if role == "rgb":
                samples[index] = frame.to_ndarray(format="rgb24")
            elif role == "depth":
                raw = frame.to_ndarray(format="gray16le")
                depth = raw.astype(np.float32) / 1000.0
                depth[raw == 0] = np.nan
                samples[index] = depth
            else:
                raise ValueError(f"Unsupported media role: {role}")
        time_base = (
            [stream.time_base.numerator, stream.time_base.denominator]
            if stream.time_base is not None
            else None
        )
        base_rate = (
            [stream.base_rate.numerator, stream.base_rate.denominator]
            if stream.base_rate is not None
            else None
        )
    pts_strict = all(later > earlier for earlier, later in zip(pts, pts[1:]))
    pts_step_constant = (
        len(pts) > 1
        and len(
            {
                later - earlier
                for earlier, later in zip(pts, pts[1:])
            }
        )
        == 1
    )
    return {
        "path": str(path.resolve()),
        "role": role,
        "rate_hz": rate_hz,
        "width": width,
        "height": height,
        "source_pixel_format": source_pixel_format,
        "decoded_frame_count": len(pts),
        "first_pts": pts[0] if pts else None,
        "last_pts": pts[-1] if pts else None,
        "pts_strictly_increasing": pts_strict,
        "pts_constant_step": pts_step_constant,
        "time_base": time_base,
        "base_rate": base_rate,
        "decoded_sample_indices": sorted(samples),
        "_samples": samples,
    }


def _surface_metrics(
    depth_samples: dict[int, np.ndarray],
    minimum_fraction: float,
    common_fraction: float,
) -> dict[str, Any]:
    ordered = sorted(depth_samples)
    finite_masks: dict[int, np.ndarray] = {}
    per_frame: list[dict[str, Any]] = []
    for index in ordered:
        depth = depth_samples[index]
        valid = np.isfinite(depth) & (depth > 0)
        finite_masks[index] = valid
        bottom = valid[valid.shape[0] // 2 :, :]
        per_frame.append(
            {
                "frame_index": index,
                "positive_finite_depth_fraction": float(np.mean(valid)),
                "bottom_half_positive_finite_depth_fraction": float(
                    np.mean(bottom)
                ),
                "positive_depth_min_m": (
                    float(np.min(depth[valid])) if np.any(valid) else None
                ),
                "positive_depth_median_m": (
                    float(np.median(depth[valid])) if np.any(valid) else None
                ),
                "positive_depth_max_m": (
                    float(np.max(depth[valid])) if np.any(valid) else None
                ),
            }
        )
    adjacent: list[dict[str, Any]] = []
    for earlier, later in zip(ordered, ordered[1:]):
        if finite_masks[earlier].shape != finite_masks[later].shape:
            raise ValueError("Depth sample shapes differ within trajectory")
        fraction = float(
            np.mean(finite_masks[earlier] & finite_masks[later])
        )
        adjacent.append(
            {
                "earlier_frame_index": earlier,
                "later_frame_index": later,
                "common_positive_finite_depth_fraction": fraction,
                "passes": fraction >= common_fraction,
            }
        )
    return {
        "sample_count": len(ordered),
        "per_frame": per_frame,
        "frames_passing_global_depth_fraction": sum(
            item["positive_finite_depth_fraction"] >= minimum_fraction
            for item in per_frame
        ),
        "frames_passing_bottom_half_depth_fraction_0_25": sum(
            item["bottom_half_positive_finite_depth_fraction"] >= 0.25
            for item in per_frame
        ),
        "adjacent_pairs": adjacent,
        "adjacent_pairs_passing_common_support": sum(
            item["passes"] for item in adjacent
        ),
    }


def _audit_trajectory(
    selected: dict[str, Any],
    media_root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    trajectory = selected["trajectory"]
    rows = int(selected["rows"])
    samples = _sample_indices(rows)
    local_paths = {
        role: media_root / selected["repo_paths"][role]
        for role in ("pose", "rgb", "depth")
    }
    hashes: dict[str, Any] = {}
    for role, path in local_paths.items():
        expected = selected["files"][role]
        actual_size = path.stat().st_size
        actual_sha = _sha256(path)
        hashes[role] = {
            "path": str(path.resolve()),
            "expected_size_bytes": int(expected["size_bytes"]),
            "actual_size_bytes": actual_size,
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual_sha,
            "matches": (
                actual_size == int(expected["size_bytes"])
                and actual_sha == expected["sha256"]
            ),
        }
    pose_rows = pl.read_parquet(local_paths["pose"]).height
    rgb = _decode_stream(local_paths["rgb"], "rgb", samples)
    depth = _decode_stream(local_paths["depth"], "depth", samples)
    rgb_samples = rgb.pop("_samples")
    depth_samples = depth.pop("_samples")
    media_gates = protocol["media_transport_gates_each_selected_trajectory"]
    surface_gates = protocol["surface_observability_canary"]
    surface = _surface_metrics(
        depth_samples,
        float(
            media_gates[
                "minimum_positive_finite_depth_fraction_per_sampled_frame"
            ]
        ),
        float(
            surface_gates[
                "common_finite_depth_support_fraction_at_least"
            ]
        ),
    )
    failures: list[str] = []
    if not all(item["matches"] for item in hashes.values()):
        failures.append("local_file_binding_mismatch")
    if pose_rows != rows:
        failures.append("pose_row_count_mismatch")
    expected_rate = float(media_gates["rgb_and_depth_reported_rate_hz"])
    for role, stream in (("rgb", rgb), ("depth", depth)):
        if (
            stream["rate_hz"] is None
            or abs(float(stream["rate_hz"]) - expected_rate) > 1e-6
        ):
            failures.append(f"{role}_rate_mismatch")
        if stream["decoded_frame_count"] != rows:
            failures.append(f"{role}_frame_count_mismatch")
        if (
            not stream["pts_strictly_increasing"]
            or not stream["pts_constant_step"]
        ):
            failures.append(f"{role}_pts_not_strict_constant_step")
        if stream["decoded_sample_indices"] != samples:
            failures.append(f"{role}_sample_decode_incomplete")
    if any(
        item["positive_finite_depth_fraction"]
        < float(
            media_gates[
                "minimum_positive_finite_depth_fraction_per_sampled_frame"
            ]
        )
        for item in surface["per_frame"]
    ):
        failures.append("sampled_depth_fraction_below_transport_gate")
    if surface["frames_passing_global_depth_fraction"] < int(
        surface_gates["minimum_frames_with_finite_depth_fraction_gate"]
    ):
        failures.append("global_depth_observability_below_gate")
    if surface[
        "frames_passing_bottom_half_depth_fraction_0_25"
    ] < int(
        surface_gates[
            "minimum_frames_with_bottom_half_depth_support_fraction_at_least_0_25"
        ]
    ):
        failures.append("bottom_half_depth_observability_below_gate")
    if surface["adjacent_pairs_passing_common_support"] < int(
        surface_gates[
            "minimum_adjacent_sample_pairs_with_common_finite_depth_support"
        ]
    ):
        failures.append("common_depth_support_below_gate")
    if any(
        sample.ndim != 3 or sample.shape[2] != 3
        for sample in rgb_samples.values()
    ):
        failures.append("rgb24_sample_shape_invalid")
    if any(sample.ndim != 2 for sample in depth_samples.values()):
        failures.append("gray16_depth_sample_shape_invalid")
    return {
        "trajectory": trajectory,
        "rows": rows,
        "camera_height_m": selected["camera_height_m"],
        "sample_indices": samples,
        "file_bindings": hashes,
        "rgb_stream": rgb,
        "depth_stream": depth,
        "surface_observability": surface,
        "gate_failures": failures,
        "media_transport_pass": not any(
            failure
            for failure in failures
            if failure
            not in {
                "global_depth_observability_below_gate",
                "bottom_half_depth_observability_below_gate",
                "common_depth_support_below_gate",
            }
        ),
        "surface_observability_pass": not any(
            failure
            in {
                "global_depth_observability_below_gate",
                "bottom_half_depth_observability_below_gate",
                "common_depth_support_below_gate",
            }
            for failure in failures
        ),
    }


def audit(
    protocol_path: Path,
    inventory_path: Path,
    media_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    inventory = _load_json(inventory_path)
    selected = _validate_inputs(protocol, protocol_path, inventory)
    trajectory_reports = [
        _audit_trajectory(item, media_root, protocol) for item in selected
    ]
    media_pass = all(
        item["media_transport_pass"] for item in trajectory_reports
    )
    surface_pass = all(
        item["surface_observability_pass"] for item in trajectory_reports
    )
    if not media_pass:
        terminal = "C0_EGOWALK_MEDIA_TRANSPORT_NOT_EVALUABLE"
    elif not surface_pass:
        terminal = "C0_NATURAL_SURFACE_OBSERVABILITY_NOT_EVALUABLE"
    else:
        terminal = "C0_STAGE_C_SOURCE_TRANSPORT_FEASIBILITY_SUPPORTED"
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "claim_ceiling": (
            "NATURAL_DEPTH_SURFACE_OBSERVABILITY_ONLY"
            if terminal.endswith("_SUPPORTED")
            else "NOT_EVALUABLE"
        ),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": _sha256(inventory_path),
        "media_root": str(media_root.resolve()),
        "trajectory_reports": trajectory_reports,
        "all_media_transport_gates_pass": media_pass,
        "all_surface_observability_gates_pass": surface_pass,
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
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = audit(
            args.protocol.resolve(),
            args.inventory.resolve(),
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
                    "all_media_transport_gates_pass": report[
                        "all_media_transport_gates_pass"
                    ],
                    "all_surface_observability_gates_pass": report[
                        "all_surface_observability_gates_pass"
                    ],
                    "output": str(output),
                }
            )
        )
        return 0 if report["terminal"].endswith("_SUPPORTED") else 3
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
