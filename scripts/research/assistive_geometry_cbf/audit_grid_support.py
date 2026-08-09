#!/usr/bin/env python3
"""Audit TRAIN-only source geometry support for AG-CBF R0 before any oracle run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.assistive_geometry.arkitscenes_truth_reader import unproject_depth


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELATIVE = Path(
    "docs/research/assistive-geometry-cbf/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0_DATA_SUPPORT_AUDIT_PROTOCOL_2026-08-09.json"
)
REQUIRED_TARGET_FIELDS = (
    "depth_m_source",
    "depth_valid_source",
    "ground_probability_source",
    "ground_label_valid_source",
    "intrinsics_source",
    "up_camera",
    "camera_height_m",
    "ground_plane_valid",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def flatten_manifest(manifest: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    parents: list[str] = []
    frames: list[dict[str, Any]] = []
    for video in manifest.get("videos", []):
        parent = str(video["video_id"])
        require(str(video.get("role")) == "TRAIN", f"non-TRAIN parent: {parent}")
        parents.append(parent)
        frames.extend({**frame, "video_id": parent} for frame in video.get("frames", []))
    return parents, frames


def select_parent_frames(
    frames: list[dict[str, Any]],
    parent_ids: list[str],
    frames_per_parent: int,
) -> list[dict[str, Any]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_parent[str(frame["video_id"])].append(frame)
    selected: list[dict[str, Any]] = []
    for parent in parent_ids:
        values = by_parent[parent]
        require(len(values) >= frames_per_parent, f"insufficient frames for {parent}")
        indices = np.linspace(0, len(values) - 1, frames_per_parent, dtype=np.int64)
        require(len(set(int(value) for value in indices)) == frames_per_parent, "duplicate frame")
        selected.extend(values[int(index)] for index in indices)
    return selected


def ground_axes(up_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.asarray(up_camera, dtype=np.float64)
    require(up.shape == (3,) and np.all(np.isfinite(up)), "up_camera must be finite length three")
    norm = float(np.linalg.norm(up))
    require(norm > 1e-9, "up_camera is degenerate")
    up /= norm
    optical = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    heading = optical - float(np.dot(optical, up)) * up
    heading_norm = float(np.linalg.norm(heading))
    require(heading_norm > 1e-6, "optical axis has no ground projection")
    heading /= heading_norm
    if float(np.dot(heading, optical)) < 0.0:
        heading = -heading
    lateral = np.cross(heading, up)
    lateral_norm = float(np.linalg.norm(lateral))
    require(lateral_norm > 1e-9, "ground lateral axis is degenerate")
    return heading, lateral / lateral_norm


def _cell_mask(
    row: np.ndarray,
    column: np.ndarray,
    mask: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    grid = np.zeros((rows, columns), dtype=bool)
    if np.any(mask):
        grid[row[mask], column[mask]] = True
    return grid


def frame_grid_metrics(target: dict[str, np.ndarray], policy: dict[str, Any]) -> dict[str, Any]:
    depth = np.asarray(target["depth_m_source"], dtype=np.float64)
    depth_valid = np.asarray(target["depth_valid_source"], dtype=bool)
    ground_probability = np.asarray(target["ground_probability_source"], dtype=np.float64)
    ground_label_valid = np.asarray(target["ground_label_valid_source"], dtype=bool)
    intrinsics = np.asarray(target["intrinsics_source"], dtype=np.float64)
    up_camera = np.asarray(target["up_camera"], dtype=np.float64)
    camera_height = float(np.asarray(target["camera_height_m"]).item())
    ground_plane_valid = bool(np.asarray(target["ground_plane_valid"]).item())

    finite_contract = (
        depth.ndim == 2
        and depth_valid.shape == depth.shape
        and ground_probability.shape == depth.shape
        and ground_label_valid.shape == depth.shape
        and intrinsics.shape == (3, 3)
        and np.all(np.isfinite(intrinsics))
        and float(intrinsics[0, 0]) > 0.0
        and float(intrinsics[1, 1]) > 0.0
        and up_camera.shape == (3,)
        and np.all(np.isfinite(up_camera))
        and np.isfinite(camera_height)
        and np.all(np.isfinite(ground_probability))
    )
    empty = {
        "ground_plane_valid": ground_plane_valid,
        "finite_geometry_contract": bool(finite_contract),
        "in_grid_point_count": 0,
        "observed_cell_count": 0,
        "ground_cell_count": 0,
        "obstacle_cell_count": 0,
        "ground_cells_by_forward_quartile": [0, 0, 0, 0],
        "observed_cells_by_lateral_third": [0, 0, 0],
    }
    if not ground_plane_valid or not finite_contract:
        return empty

    heading, lateral_axis = ground_axes(up_camera)
    valid = (
        depth_valid
        & ground_label_valid
        & np.isfinite(depth)
        & (depth >= float(policy["depth_min_m"]))
        & (depth <= float(policy["depth_max_m"]))
    )
    points, pixels = unproject_depth(depth, intrinsics, valid, int(policy["point_stride"]))
    if not len(points):
        return empty
    up = up_camera / np.linalg.norm(up_camera)
    forward = points @ heading
    lateral = points @ lateral_axis
    heights = points @ up + camera_height
    forward_min, forward_max = (float(value) for value in policy["forward_range_m"])
    lateral_min, lateral_max = (float(value) for value in policy["lateral_range_m"])
    rows, columns = (int(value) for value in policy["grid_shape"])
    in_grid = (
        (forward >= forward_min)
        & (forward < forward_max)
        & (lateral >= lateral_min)
        & (lateral < lateral_max)
    )
    row = np.clip(
        ((forward - forward_min) / (forward_max - forward_min) * rows).astype(np.int64),
        0,
        rows - 1,
    )
    column = np.clip(
        ((lateral - lateral_min) / (lateral_max - lateral_min) * columns).astype(np.int64),
        0,
        columns - 1,
    )
    pixel_ground = ground_probability[pixels[:, 1], pixels[:, 0]] >= float(
        policy["ground_probability_threshold"]
    )
    obstacle = (
        (heights >= float(policy["obstacle_height_range_m"][0]))
        & (heights <= float(policy["obstacle_height_range_m"][1]))
        & ~pixel_ground
    )
    observed_grid = _cell_mask(row, column, in_grid, rows, columns)
    ground_grid = _cell_mask(row, column, in_grid & pixel_ground, rows, columns)
    obstacle_grid = _cell_mask(row, column, in_grid & obstacle, rows, columns)
    quartiles = [
        int(np.count_nonzero(part))
        for part in np.array_split(ground_grid, 4, axis=0)
    ]
    thirds = [
        int(np.count_nonzero(part))
        for part in np.array_split(observed_grid, 3, axis=1)
    ]
    return {
        "ground_plane_valid": True,
        "finite_geometry_contract": True,
        "in_grid_point_count": int(np.count_nonzero(in_grid)),
        "observed_cell_count": int(np.count_nonzero(observed_grid)),
        "ground_cell_count": int(np.count_nonzero(ground_grid)),
        "obstacle_cell_count": int(np.count_nonzero(obstacle_grid)),
        "ground_cells_by_forward_quartile": quartiles,
        "observed_cells_by_lateral_third": thirds,
    }


def evaluate_frame_metrics(metrics: dict[str, Any], policy: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not metrics["ground_plane_valid"]:
        reasons.append("UNKNOWN_GROUND_PLANE")
    if not metrics["finite_geometry_contract"]:
        reasons.append("UNKNOWN_GEOMETRY_CONTRACT")
    thresholds = policy["frame_gate"]
    for key, reason in (
        ("in_grid_point_count", "UNKNOWN_IN_GRID_POINTS"),
        ("observed_cell_count", "UNKNOWN_OBSERVED_GRID_CELLS"),
        ("ground_cell_count", "UNKNOWN_GROUND_GRID_CELLS"),
    ):
        if int(metrics[key]) < int(thresholds[f"minimum_{key}"]):
            reasons.append(reason)
    if min(int(value) for value in metrics["ground_cells_by_forward_quartile"]) < int(
        thresholds["minimum_ground_cells_per_forward_quartile"]
    ):
        reasons.append("UNKNOWN_LONGITUDINAL_GROUND_SUPPORT")
    if min(int(value) for value in metrics["observed_cells_by_lateral_third"]) < int(
        thresholds["minimum_observed_cells_per_lateral_third"]
    ):
        reasons.append("UNKNOWN_LATERAL_OBSERVATION_SUPPORT")
    return not reasons, reasons


def evaluate_route_gate(
    per_parent_evaluable: dict[str, int],
    orientation_evaluable: dict[str, int],
    policy: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    gate = policy["route_gate"]
    passing_parents = [
        parent
        for parent, count in per_parent_evaluable.items()
        if count >= int(gate["minimum_evaluable_frames_per_passing_parent"])
    ]
    total = int(sum(per_parent_evaluable.values()))
    portrait = int(orientation_evaluable.get("portrait", 0))
    landscape = int(orientation_evaluable.get("landscape", 0))
    qualified = (
        total >= int(gate["minimum_total_evaluable_frames"])
        and len(passing_parents) >= int(gate["minimum_passing_parents"])
        and portrait >= int(gate["minimum_evaluable_frames_per_orientation"])
        and landscape >= int(gate["minimum_evaluable_frames_per_orientation"])
    )
    return qualified, {
        "total_evaluable_frames": total,
        "passing_parent_count": len(passing_parents),
        "passing_parents": passing_parents,
        "orientation_evaluable_frames": {"portrait": portrait, "landscape": landscape},
    }


def _validate_target_path(path: Path) -> None:
    normalized = str(path.resolve()).replace("/", "\\").lower()
    require(
        "assistive-geometry-b1-train-targets-20260809-r0\\targets\\" in normalized,
        f"target escaped frozen TRAIN target root: {path}",
    )
    require("development" not in normalized and "confirmation" not in normalized, "protected target path")


def validate_protocol(protocol: dict[str, Any]) -> None:
    require(
        protocol.get("schema") == "blindassist.assistive_geometry_cbf.r0_data_support_audit_protocol.v1",
        "protocol schema drift",
    )
    require(protocol.get("status") == "TRAIN_DATA_SUPPORT_AUDIT_LOCKED_NOT_RUN", "status drift")
    authority = protocol.get("authority", {})
    for key in ("model_access", "feature_access", "development_access", "confirmation_access", "training_authorized"):
        require(authority.get(key) is False, f"authority leaked: {key}")
    require(protocol.get("frames_per_parent") == 64, "frame count drift")
    require(protocol.get("frame_selection") == "SOURCE_ORDER_EVENLY_SPACED", "selection drift")
    require(tuple(protocol.get("target_field_access", [])) == REQUIRED_TARGET_FIELDS, "target fields drift")
    source = protocol.get("input", {})
    require(source.get("data_role") == "TRAIN", "input role drift")
    require(source.get("outcome_access") == "SOURCE_GEOMETRY_CONTENT_ONLY", "outcome access drift")
    require(source.get("claim_use") == "TRAIN_GRID_DATA_SUPPORT_AUDIT_ONLY", "claim-use drift")
    source_path = REPO_ROOT / str(source.get("path"))
    require(source_path.is_file(), "input manifest missing")
    require(sha256_file(source_path) == source.get("sha256"), "input manifest SHA drift")
    implementation = protocol.get("implementation", {})
    expected = {
        "scripts/research/assistive_geometry_cbf/audit_grid_support.py",
        "scripts/research/assistive_geometry_cbf/test_audit_grid_support.py",
        "scripts/research/assistive_geometry/arkitscenes_truth_reader.py",
    }
    require(set(implementation) == expected, "implementation path set drift")
    for logical, expected_sha in implementation.items():
        require(sha256_file(REPO_ROOT / logical) == expected_sha, f"implementation SHA drift: {logical}")


def execute(protocol_path: Path, output: Path) -> int:
    canonical_protocol = (REPO_ROOT / PROTOCOL_RELATIVE).resolve()
    require(protocol_path.resolve() == canonical_protocol, "custom protocol is not authorized")
    protocol = load_json(canonical_protocol)
    validate_protocol(protocol)
    require(output.resolve() == (REPO_ROOT / protocol["output"]).resolve(), "output path drift")
    require(not output.exists(), "output collision")
    manifest = load_json(REPO_ROOT / protocol["input"]["path"])
    parent_order, frames = flatten_manifest(manifest)
    require(parent_order == protocol["parent_order"], "parent order drift")
    selected = select_parent_frames(frames, parent_order, int(protocol["frames_per_parent"]))
    started = time.perf_counter()
    per_parent_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_parent_evaluable = {parent: 0 for parent in parent_order}
    orientation_evaluable = {"portrait": 0, "landscape": 0}
    for frame in selected:
        receipt = frame.get("target", {})
        path = Path(str(receipt.get("path", "")))
        frame_id = f"{frame['video_id']}/{frame['frame_stem']}"
        _validate_target_path(path)
        require(path.is_file(), f"missing target: {frame_id}")
        require(path.stat().st_size == int(receipt.get("bytes", -1)), f"target bytes drift: {frame_id}")
        require(sha256_file(path) == str(receipt.get("sha256", "")).upper(), f"target SHA drift: {frame_id}")
        with np.load(path, allow_pickle=False) as loaded:
            require(all(field in loaded.files for field in REQUIRED_TARGET_FIELDS), f"target field missing: {frame_id}")
            target = {field: np.asarray(loaded[field]) for field in REQUIRED_TARGET_FIELDS}
        metrics = frame_grid_metrics(target, protocol["grid_policy"])
        evaluable, reasons = evaluate_frame_metrics(metrics, protocol["grid_policy"])
        orientation = str(frame.get("orientation_family"))
        require(orientation in orientation_evaluable, f"orientation drift: {frame_id}")
        if evaluable:
            per_parent_evaluable[str(frame["video_id"])] += 1
            orientation_evaluable[orientation] += 1
        per_parent_details[str(frame["video_id"])].append(
            {
                "frame_stem": str(frame["frame_stem"]),
                "orientation_family": orientation,
                "target_sha256": str(receipt["sha256"]).upper(),
                "evaluable": evaluable,
                "unknown_reasons": reasons,
                "metrics": metrics,
            }
        )
    qualified, route_metrics = evaluate_route_gate(
        per_parent_evaluable,
        orientation_evaluable,
        protocol["grid_policy"],
    )
    elapsed = time.perf_counter() - started
    result = {
        "schema": "blindassist.assistive_geometry_cbf.r0_data_support_audit_result.v1",
        "protocol_sha256": sha256_file(canonical_protocol),
        "input_manifest_sha256": protocol["input"]["sha256"],
        "authority": protocol["authority"],
        "parent_order": parent_order,
        "selected_frame_count": len(selected),
        "per_parent_evaluable_frames": per_parent_evaluable,
        "per_parent_frames": dict(per_parent_details),
        "route_gate_metrics": route_metrics,
        "elapsed_seconds": elapsed,
        "terminal": (
            "AG_CBF_R0_DATA_SUPPORT_PASS_ORACLE_LOCK_ALLOWED"
            if qualified
            else "AG_CBF_R0_DATA_SUPPORT_NOT_EVALUABLE_ROUTE_CLOSE"
        ),
        "oracle_authorized": qualified,
        "model_or_training_authorized": False,
        "claim_ceiling": (
            "TRAIN-only source-geometry grid support and integrity. UNKNOWN is not negative; no corridor "
            "oracle value, learnability, Development, Confirmation, device, product, production, or safety authority."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "route_gate_metrics": route_metrics,
                "elapsed_seconds": elapsed,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if qualified else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / PROTOCOL_RELATIVE)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts.local/evidence/assistive-geometry-cbf/r0-data-support-audit/result.json",
    )
    args = parser.parse_args()
    return execute(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
