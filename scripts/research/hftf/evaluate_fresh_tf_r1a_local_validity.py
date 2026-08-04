#!/usr/bin/env python3
"""R1-A mechanics canary for motion-compensated local geometric validity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA = "blindassist_fresh_tf_r1a_local_validity_canary_v1"
PROTOCOL_SCHEMA = "blindassist_fresh_tf_r1a_local_validity_protocol_v1"


@dataclass(frozen=True)
class Frame:
    timestamp: float
    rgb_path: Path
    depth_path: Path
    pose: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _data_lines(path: Path) -> list[list[str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not math.isfinite(float(parts[0])):
            raise ValueError(f"{path}:{line_number}: non-finite timestamp")
        rows.append(parts)
    if not rows:
        raise ValueError(f"no data rows: {path}")
    return rows


def pose_matrix(values: list[str] | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (7,):
        raise ValueError("pose requires tx ty tz qx qy qz qw")
    tx, ty, tz, qx, qy, qz, qw = values
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = [tx, ty, tz]
    return result


def _nearest(rows: list[list[str]], timestamp: float) -> tuple[list[str], float]:
    times = np.fromiter((float(row[0]) for row in rows), dtype=np.float64)
    index = int(np.searchsorted(times, timestamp))
    candidates = [max(0, index - 1), min(len(rows) - 1, index)]
    best = min(candidates, key=lambda item: (abs(times[item] - timestamp), item))
    return rows[best], abs(float(times[best]) - timestamp)


def admitted_frames(root: Path, max_delta_s: float, rate_hz: float) -> list[Frame]:
    rgb_rows = _data_lines(root / "rgb.txt")
    depth_rows = _data_lines(root / "depth.txt")
    pose_rows = _data_lines(root / "groundtruth.txt")
    associated: list[Frame] = []
    for rgb in rgb_rows:
        timestamp = float(rgb[0])
        depth, depth_delta = _nearest(depth_rows, timestamp)
        pose, pose_delta = _nearest(pose_rows, timestamp)
        if depth_delta <= max_delta_s and pose_delta <= max_delta_s:
            associated.append(
                Frame(
                    timestamp=timestamp,
                    rgb_path=root / rgb[1],
                    depth_path=root / depth[1],
                    pose=pose_matrix(pose[1:8]),
                )
            )
    if not associated:
        raise ValueError(f"no admitted frames: {root}")
    period = 1.0 / rate_hz
    start = associated[0].timestamp
    sampled = []
    lattice_index = 0
    for frame in associated:
        if frame.timestamp + 1e-12 >= start + lattice_index * period:
            sampled.append(frame)
            lattice_index += 1
    return sampled


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read RGB: {path}")
    return image


def read_depth(path: Path, scale: float) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise OSError(f"cannot read single-channel depth: {path}")
    return depth.astype(np.float32) / scale


def farneback(first: np.ndarray, second: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        first,
        second,
        None,
        float(config["pyr_scale"]),
        int(config["levels"]),
        int(config["winsize"]),
        int(config["iterations"]),
        int(config["poly_n"]),
        float(config["poly_sigma"]),
        int(config["flags"]),
    )


def zbuffer_winners(linear_pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
    if linear_pixels.shape != depths.shape:
        raise ValueError("z-buffer inputs must have identical shapes")
    if linear_pixels.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((np.arange(linear_pixels.size), depths, linear_pixels))
    sorted_pixels = linear_pixels[order]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    return order[first]


def _project_depth(
    depth: np.ndarray,
    transform: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth.shape
    fx, fy, cx, cy = intrinsics
    vv, uu = np.indices((height, width), dtype=np.float32)
    valid = depth > 0
    source_linear = np.flatnonzero(valid)
    z = depth.ravel()[source_linear].astype(np.float64)
    u = uu.ravel()[source_linear].astype(np.float64)
    v = vv.ravel()[source_linear].astype(np.float64)
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    target_z = transformed[:, 2]
    target_u = fx * transformed[:, 0] / np.maximum(target_z, 1e-9) + cx
    target_v = fy * transformed[:, 1] / np.maximum(target_z, 1e-9) + cy
    return source_linear, u, v, target_u, target_v, target_z


def classify_cell(
    *,
    age_ms: float,
    denominator: int,
    out_of_frame: int,
    occluded: int,
    visible: int,
    projected: int,
    flow_pass: int,
    median_warp_residual_px: float | None,
    config: dict[str, Any],
) -> str:
    if age_ms > float(config["hard_ttl_ms"]):
        return "STALE"
    if denominator and out_of_frame / denominator >= 0.40:
        return "OUT_OF_FRAME"
    minimum = int(config["minimum_valid_projected_points_per_cell"])
    if occluded >= int(config["minimum_occlusion_evidence_points_per_cell"]):
        return "OCCLUDED"
    support_fraction = visible / denominator if denominator else 0.0
    if support_fraction < float(config["minimum_supported_area_fraction"]):
        return "NEWLY_EXPOSED"
    if projected < minimum or flow_pass < minimum or flow_pass / projected < 0.60:
        return "LOW_FLOW_SUPPORT"
    if median_warp_residual_px is None or median_warp_residual_px > float(
        config["geometry_flow_warp_residual_px_max"]
    ):
        return "HIGH_WARP_RESIDUAL"
    return "SUPPORTED"


def evaluate_pair(
    anchor_rgb: np.ndarray,
    anchor_depth: np.ndarray,
    anchor_pose: np.ndarray,
    current_rgb: np.ndarray,
    current_depth: np.ndarray,
    current_pose: np.ndarray,
    age_ms: float,
    intrinsics: tuple[float, float, float, float],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if anchor_depth.shape != current_depth.shape or anchor_rgb.shape != current_rgb.shape:
        raise ValueError("anchor/current shapes differ")
    height, width = current_depth.shape
    if anchor_rgb.shape != (height, width):
        raise ValueError("RGB/depth shapes differ")
    margin = int(config["boundary_margin_px"])
    anchor_to_current = np.linalg.inv(current_pose) @ anchor_pose
    current_to_anchor = np.linalg.inv(anchor_pose) @ current_pose
    source_linear, source_u, source_v, target_u, target_v, target_z = _project_depth(
        anchor_depth, anchor_to_current, intrinsics
    )
    inside = (
        (target_z > 0)
        & (target_u >= margin)
        & (target_u < width - margin)
        & (target_v >= margin)
        & (target_v < height - margin)
    )
    source_linear = source_linear[inside]
    source_u = source_u[inside]
    source_v = source_v[inside]
    target_u = target_u[inside]
    target_v = target_v[inside]
    target_z = target_z[inside]
    rounded_u = np.rint(target_u).astype(np.int32)
    rounded_v = np.rint(target_v).astype(np.int32)
    linear_target = rounded_v.astype(np.int64) * width + rounded_u
    winners = zbuffer_winners(linear_target, target_z)
    source_linear = source_linear[winners]
    source_u = source_u[winners]
    source_v = source_v[winners]
    target_u = target_u[winners]
    target_v = target_v[winners]
    target_z = target_z[winners]
    rounded_u = rounded_u[winners]
    rounded_v = rounded_v[winners]
    linear_target = linear_target[winners]

    current_flat = current_depth.ravel()
    observed = current_flat[linear_target]
    current_valid_at_projection = observed > 0
    tolerance = np.maximum(0.10, 0.05 * observed)
    occluded = current_valid_at_projection & (target_z > observed + tolerance)
    visible = current_valid_at_projection & (np.abs(target_z - observed) <= tolerance)

    forward = farneback(anchor_rgb, current_rgb, config["optical_flow"])
    backward = farneback(current_rgb, anchor_rgb, config["optical_flow"])
    source_ui = source_linear % width
    source_vi = source_linear // width
    flow_xy = forward[source_vi, source_ui]
    flow_u = source_u + flow_xy[:, 0]
    flow_v = source_v + flow_xy[:, 1]
    flow_inside = (
        (flow_u >= margin)
        & (flow_u < width - margin)
        & (flow_v >= margin)
        & (flow_v < height - margin)
    )
    sample_u = np.clip(np.rint(flow_u).astype(np.int32), 0, width - 1)
    sample_v = np.clip(np.rint(flow_v).astype(np.int32), 0, height - 1)
    backward_xy = backward[sample_v, sample_u]
    fb_residual = np.hypot(
        flow_u + backward_xy[:, 0] - source_u,
        flow_v + backward_xy[:, 1] - source_v,
    )
    flow_pass = flow_inside & (fb_residual <= float(config["forward_backward_residual_px_max"]))
    warp_residual = np.hypot(flow_u - target_u, flow_v - target_v)

    _, _, _, back_u, back_v, back_z = _project_depth(
        current_depth, current_to_anchor, intrinsics
    )
    current_valid_linear = np.flatnonzero(current_depth.ravel() > 0)
    back_out = (
        (back_z <= 0)
        | (back_u < margin)
        | (back_u >= width - margin)
        | (back_v < margin)
        | (back_v >= height - margin)
    )
    back_out_mask = np.zeros(height * width, dtype=bool)
    back_out_mask[current_valid_linear] = back_out

    columns = int(config["image_grid_columns"])
    rows = int(config["image_grid_rows"])
    results = []
    current_valid = current_depth > 0
    projected_map = np.zeros((height, width), dtype=np.int32)
    visible_map = np.zeros((height, width), dtype=np.int32)
    occluded_map = np.zeros((height, width), dtype=np.int32)
    flow_map = np.zeros((height, width), dtype=np.int32)
    projected_map[rounded_v, rounded_u] = 1
    visible_map[rounded_v[visible], rounded_u[visible]] = 1
    occluded_map[rounded_v[occluded], rounded_u[occluded]] = 1
    flow_map[rounded_v[flow_pass], rounded_u[flow_pass]] = 1
    cell_x = np.minimum(rounded_u * columns // width, columns - 1)
    cell_y = np.minimum(rounded_v * rows // height, rows - 1)
    target_cell = cell_y * columns + cell_x
    for row in range(rows):
        y0, y1 = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            x0, x1 = column * width // columns, (column + 1) * width // columns
            cell_index = row * columns + column
            denominator = int(np.count_nonzero(current_valid[y0:y1, x0:x1]))
            out_count = int(np.count_nonzero(back_out_mask.reshape(height, width)[y0:y1, x0:x1]))
            projected_count = int(np.count_nonzero(projected_map[y0:y1, x0:x1]))
            visible_count = int(np.count_nonzero(visible_map[y0:y1, x0:x1]))
            occluded_count = int(np.count_nonzero(occluded_map[y0:y1, x0:x1]))
            flow_count = int(np.count_nonzero(flow_map[y0:y1, x0:x1]))
            residuals = warp_residual[(target_cell == cell_index) & flow_pass]
            median_residual = float(np.median(residuals)) if residuals.size else None
            state = classify_cell(
                age_ms=age_ms,
                denominator=denominator,
                out_of_frame=out_count,
                occluded=occluded_count,
                visible=visible_count,
                projected=projected_count,
                flow_pass=flow_count,
                median_warp_residual_px=median_residual,
                config=config,
            )
            results.append(
                {
                    "row": row,
                    "column": column,
                    "state": state,
                    "current_valid_depth_pixels": denominator,
                    "out_of_frame_pixels": out_count,
                    "projected_pixels": projected_count,
                    "visible_pixels": visible_count,
                    "occluded_pixels": occluded_count,
                    "flow_supported_pixels": flow_count,
                    "support_fraction": visible_count / denominator if denominator else 0.0,
                    "median_warp_residual_px": median_residual,
                }
            )
    return results


def evaluate_sequence(root: Path, protocol: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = protocol["mechanics_lock"]
    frames = admitted_frames(
        root,
        float(protocol["association_and_admission"]["rgb_depth_timestamp_delta_ms_max"]) / 1000,
        float(config["evaluation_rate_hz"]),
    )
    family = "freiburg1" if "freiburg1" in root.name else "freiburg3"
    intrinsics = tuple(float(value) for value in config["intrinsics_fx_fy_cx_cy"][family])
    anchor_index = 0
    trace = []
    states: Counter[str] = Counter()
    ages = []
    for index, frame in enumerate(frames):
        if (frame.timestamp - frames[anchor_index].timestamp) * 1000 >= float(config["anchor_period_ms"]):
            anchor_index = index
        anchor = frames[anchor_index]
        anchor_rgb = read_rgb(anchor.rgb_path)
        anchor_depth = read_depth(anchor.depth_path, float(config["depth_scale_uint16_per_m"]))
        if index == anchor_index:
            current_rgb, current_depth = anchor_rgb, anchor_depth
        else:
            current_rgb = read_rgb(frame.rgb_path)
            current_depth = read_depth(frame.depth_path, float(config["depth_scale_uint16_per_m"]))
        age_ms = (frame.timestamp - anchor.timestamp) * 1000
        cells = evaluate_pair(
            anchor_rgb,
            anchor_depth,
            anchor.pose,
            current_rgb,
            current_depth,
            frame.pose,
            age_ms,
            intrinsics,
            config,
        )
        frame_states = Counter(cell["state"] for cell in cells)
        states.update(frame_states)
        ages.append(age_ms)
        trace.append(
            {
                "frame_index": index,
                "timestamp": frame.timestamp,
                "anchor_frame_index": anchor_index,
                "age_ms": age_ms,
                "states": dict(sorted(frame_states.items())),
                "cells": cells,
            }
        )
    total = sum(states.values())
    summary = {
        "sequence": root.name,
        "sampled_frames": len(frames),
        "evaluated_cells": total,
        "state_counts": dict(sorted(states.items())),
        "cell_support_coverage": states["SUPPORTED"] / total if total else 0.0,
        "invalid_cell_fraction": 1 - states["SUPPORTED"] / total if total else 1.0,
        "occlusion_opportunity_cells": states["OCCLUDED"],
        "newly_exposed_opportunity_cells": states["NEWLY_EXPOSED"],
        "out_of_frame_opportunity_cells": states["OUT_OF_FRAME"],
        "maximum_anchor_age_ms": max(ages),
    }
    return summary, trace


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    receipt = load_json(args.source_receipt)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected protocol schema")
    protocol_hash = sha256(args.protocol)
    if receipt.get("protocol_sha256") != protocol_hash:
        raise ValueError("source receipt does not bind current protocol")
    roots = []
    for archive in receipt["archives"]:
        archive_path = args.source_root / archive["file"]
        if sha256(archive_path) != archive["sha256"]:
            raise ValueError(f"archive SHA-256 mismatch: {archive_path}")
        root = args.source_root / archive["sequence"]
        for required in protocol["source_lock"]["required_files"]:
            if not (root / required).is_file():
                raise FileNotFoundError(root / required)
        roots.append(root)
    summaries = []
    traces = []
    for root in roots:
        summary, trace = evaluate_sequence(root, protocol)
        summaries.append(summary)
        traces.append({"sequence": root.name, "frames": trace})
    coverages = [row["cell_support_coverage"] for row in summaries]
    opportunities = {
        key: sum(int(row[key]) for row in summaries)
        for key in (
            "occlusion_opportunity_cells",
            "newly_exposed_opportunity_cells",
            "out_of_frame_opportunity_cells",
        )
    }
    result = {
        "schema": SCHEMA,
        "date": protocol["date"],
        "terminal": "FRESH_TF_R1A_MECHANICS_CANARY_COMPLETE_FORMAL_EFFECT_NOT_ADMISSIBLE",
        "authority": "MECHANICS_OPPORTUNITY_CANARY_ONLY",
        "protocol_sha256": protocol_hash,
        "source_receipt_sha256": sha256(args.source_receipt),
        "implementation_sha256": sha256(Path(__file__)),
        "sessions": summaries,
        "aggregate": {
            "macro_session_cell_support_coverage": float(np.mean(coverages)),
            "worst_session_cell_support_coverage": min(coverages),
            **opportunities,
            "newly_exposed_inheritance_count": 0,
            "occluded_inheritance_count": 0,
        },
        "formal_gate_results": "NOT_RUN",
        "formal_effect_evaluation_admissible": False,
        "reason": "one locked session per mechanism role; direction/traversability truth and the required second sessions are absent",
        "claim_ceiling": protocol["claim_ceiling"],
    }
    write_new(args.output, result)
    write_new(args.trace_output, {"schema": SCHEMA + "_trace", "sessions": traces})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
