#!/usr/bin/env python3
"""Run the frozen HFTF Stage C depth-only label-readiness D0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


SCHEMA = "blindassist_hftf_stage_c_d0_label_readiness_result"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_semantic_independent_label_readiness_d0"
)
PROTOCOL_STATUS = (
    "FROZEN_AFTER_CONSUMED_CALIBRATION_BEFORE_FORMAL_D0_REPORT"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    resolved = (repo_root / value).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"Frozen path leaves repository: {resolved}") from error
    return resolved


def _validate_protocol(
    protocol: dict[str, Any],
    protocol_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("Stage C label-readiness D0 protocol is not frozen")
    parent = protocol_path.parent / str(protocol["parent_result_path"])
    if _sha256(parent) != protocol["parent_result_sha256"]:
        raise ValueError("D0 parent-result hash mismatch")
    frozen = protocol["frozen_inputs"]
    c0_report_path = _resolve_repo_path(
        repo_root, str(frozen["c0_1_report_path"])
    )
    inventory_path = _resolve_repo_path(
        repo_root, str(frozen["inventory_path"])
    )
    if _sha256(c0_report_path) != frozen["c0_1_report_sha256"]:
        raise ValueError("D0 C0.1 report hash mismatch")
    if _sha256(inventory_path) != frozen["inventory_sha256"]:
        raise ValueError("D0 inventory hash mismatch")
    c0_report = _load_json(c0_report_path)
    inventory = _load_json(inventory_path)
    if (
        c0_report.get("terminal")
        != "C0_1_STAGE_C_SOURCE_TRANSPORT_FEASIBILITY_SUPPORTED"
    ):
        raise ValueError("D0 parent C0.1 terminal mismatch")
    selected = [
        item["trajectory"] for item in inventory["selected_trajectories"]
    ]
    if selected != frozen["trajectory_ids"]:
        raise ValueError("D0 frozen trajectory IDs mismatch")
    return c0_report, inventory


def _formal_frame_indices(rows: int, step: int = 5) -> list[int]:
    if rows <= 0 or step <= 0:
        raise ValueError("Frame selection dimensions must be positive")
    indices = list(range(0, rows, step))
    if indices[-1] != rows - 1:
        indices.append(rows - 1)
    return indices


def _seed(trajectory: str, frame_index: int) -> int:
    digest = hashlib.sha256(
        f"{trajectory}:{frame_index}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def _project_depth(
    depth: np.ndarray,
    camera: dict[str, float],
    config: dict[str, Any],
) -> np.ndarray:
    height, width = depth.shape
    roi = config["fit_roi"]
    stride = int(config["pixel_stride"])
    v0 = math.floor(float(roi["vertical_fraction_inclusive"][0]) * height)
    v1 = math.ceil(float(roi["vertical_fraction_inclusive"][1]) * height)
    u0 = math.floor(float(roi["horizontal_fraction_inclusive"][0]) * width)
    u1 = math.ceil(float(roi["horizontal_fraction_inclusive"][1]) * width)
    vv, uu = np.mgrid[v0:v1:stride, u0:u1:stride]
    z = depth[vv, uu]
    bounds = config["valid_depth_m_inclusive"]
    valid = (
        np.isfinite(z)
        & (z >= float(bounds[0]))
        & (z <= float(bounds[1]))
    )
    z = z[valid]
    return np.column_stack(
        [
            (uu[valid] - float(camera["cx"])) / float(camera["fx"]) * z,
            (vv[valid] - float(camera["cy"])) / float(camera["fy"]) * z,
            z,
        ]
    )


def _better_candidate(
    current: tuple[int, float, tuple[float, ...]] | None,
    candidate: tuple[int, float, tuple[float, ...]],
) -> bool:
    if current is None:
        return True
    if candidate[0] != current[0]:
        return candidate[0] > current[0]
    if candidate[1] != current[1]:
        return candidate[1] < current[1]
    return candidate[2] < current[2]


def _fit_ground_plane(
    points: np.ndarray,
    camera_height_m: float,
    trajectory: str,
    frame_index: int,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    minimum_inliers = int(config["minimum_candidate_inliers"])
    if len(points) < minimum_inliers:
        return None
    rng = np.random.default_rng(_seed(trajectory, frame_index))
    maximum = int(config["maximum_fit_points"])
    fit_points = points
    if len(points) > maximum:
        fit_points = points[
            rng.choice(len(points), maximum, replace=False)
        ]
    best_key: tuple[int, float, tuple[float, ...]] | None = None
    best: tuple[np.ndarray, float, np.ndarray] | None = None
    for _ in range(int(config["ransac_iterations"])):
        sample = fit_points[rng.choice(len(fit_points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < float(config["minimum_triplet_cross_product_norm"]):
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        if normal[1] < 0:
            normal = -normal
            offset = -offset
        if normal[1] < float(config["minimum_camera_y_axis_alignment"]):
            continue
        if (
            abs(abs(offset) - camera_height_m)
            > float(config["maximum_camera_height_error_m_for_candidate"])
        ):
            continue
        residual = np.abs(fit_points @ normal + offset)
        inliers = residual <= float(config["inlier_distance_m"])
        count = int(np.sum(inliers))
        if count < minimum_inliers:
            continue
        median = float(np.median(residual[inliers]))
        coefficients = tuple(
            float(value) for value in (*normal.tolist(), offset)
        )
        key = (count, median, coefficients)
        if _better_candidate(best_key, key):
            best_key = key
            best = (normal.copy(), offset, inliers)
    if best is None:
        return None
    _, _, winning_inliers = best
    winning_points = fit_points[winning_inliers]
    center = np.mean(winning_points, axis=0)
    _, _, vh = np.linalg.svd(
        winning_points - center, full_matrices=False
    )
    normal = vh[-1]
    if normal[1] < 0:
        normal = -normal
    offset = -float(normal @ center)
    residual = np.abs(fit_points @ normal + offset)
    inliers = residual <= float(config["inlier_distance_m"])
    if int(np.sum(inliers)) < minimum_inliers:
        return None
    inlier_residual = residual[inliers]
    return {
        "normal": normal,
        "offset": offset,
        "fit_point_count": len(fit_points),
        "inlier_count": int(np.sum(inliers)),
        "inlier_fraction": float(np.mean(inliers)),
        "camera_height_proxy_m": abs(offset),
        "camera_height_error_m": abs(abs(offset) - camera_height_m),
        "median_inlier_residual_m": float(
            np.median(inlier_residual)
        ),
        "p90_inlier_residual_m": float(
            np.percentile(inlier_residual, 90)
        ),
    }


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([0.0, 0.0, 1.0])
    forward -= normal * float(forward @ normal)
    forward /= np.linalg.norm(forward)
    lateral = np.cross(normal, forward)
    lateral /= np.linalg.norm(lateral)
    if lateral[0] < 0:
        lateral = -lateral
    return forward, lateral


def _estimate_section(
    points: np.ndarray,
    heights: np.ndarray,
    ground_normal: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    bounds = config["candidate_surface_height_m_inclusive"]
    valid = (
        (heights >= float(bounds[0]))
        & (heights <= float(bounds[1]))
    )
    points = points[valid]
    heights = heights[valid]
    minimum_raw = int(config["minimum_raw_points_per_section"])
    if len(points) < minimum_raw:
        return None
    width = float(config["height_histogram_bin_width_m"])
    edges = np.arange(
        float(bounds[0]) - width / 2,
        float(bounds[1]) + width,
        width,
    )
    histogram, _ = np.histogram(heights, edges)
    mode_index = int(np.argmax(histogram))
    expansion = float(config["winning_mode_expansion_m_each_side"])
    mode = (
        (heights >= edges[mode_index] - expansion)
        & (heights < edges[mode_index + 1] + expansion)
    )
    mode_points = points[mode]
    mode_heights = heights[mode]
    if (
        len(mode_points) < int(config["minimum_mode_points_per_section"])
        or len(mode_points) / len(points)
        < float(config["minimum_mode_fraction"])
    ):
        return None
    center = np.mean(mode_points, axis=0)
    _, _, vh = np.linalg.svd(
        mode_points - center, full_matrices=False
    )
    support_normal = vh[-1]
    alignment = abs(float(support_normal @ ground_normal))
    residual = np.abs((mode_points - center) @ support_normal)
    p90 = float(np.percentile(residual, 90))
    if (
        alignment
        < float(config["minimum_support_normal_ground_alignment"])
        or p90 > float(config["maximum_support_plane_p90_residual_m"])
    ):
        return None
    return {
        "height_m": float(np.median(mode_heights)),
        "raw_point_count": len(points),
        "mode_point_count": len(mode_points),
        "mode_fraction": len(mode_points) / len(points),
        "support_normal_ground_alignment": alignment,
        "support_plane_p90_residual_m": p90,
    }


def _surface_profiles(
    points: np.ndarray,
    plane: dict[str, Any] | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    directions = [float(value) for value in config["direction_degrees"]]
    if plane is None:
        return [
            {
                "direction_degrees": direction,
                "state": "UNKNOWN",
                "section_heights_m": [None] * len(
                    config["section_center_distance_m"]
                ),
                "adjacent_deltas_m": [],
            }
            for direction in directions
        ]
    normal = plane["normal"]
    offset = float(plane["offset"])
    forward_axis, lateral_axis = _plane_basis(normal)
    forward = points @ forward_axis
    lateral = points @ lateral_axis
    height = -(points @ normal + offset)
    output: list[dict[str, Any]] = []
    for direction in directions:
        angle = math.radians(direction)
        sections: list[dict[str, Any] | None] = []
        for distance in config["section_center_distance_m"]:
            center_forward = float(distance) * math.cos(angle)
            center_lateral = float(distance) * math.sin(angle)
            selected = (
                (
                    np.abs(forward - center_forward)
                    <= float(config["cell_forward_half_extent_m"])
                )
                & (
                    np.abs(lateral - center_lateral)
                    <= float(config["cell_lateral_half_extent_m"])
                )
            )
            sections.append(
                _estimate_section(
                    points[selected],
                    height[selected],
                    normal,
                    config,
                )
            )
        known = sum(section is not None for section in sections)
        deltas: list[dict[str, Any]] = []
        rise = False
        drop = False
        for index, (earlier, later) in enumerate(
            zip(sections, sections[1:])
        ):
            if earlier is None or later is None:
                continue
            delta = float(later["height_m"] - earlier["height_m"])
            deltas.append(
                {
                    "from_section_index": index,
                    "to_section_index": index + 1,
                    "delta_m": delta,
                }
            )
            rise = rise or delta > float(
                config[
                    "rise_risk_if_adjacent_delta_m_strictly_greater_than"
                ]
            )
            drop = drop or delta < float(
                config[
                    "drop_risk_if_adjacent_delta_m_strictly_less_than"
                ]
            )
        if known < int(config["minimum_known_sections_per_direction"]):
            state = "UNKNOWN"
        elif rise or drop:
            state = "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
        else:
            state = "KNOWN_NO_GROUND_CONTINUITY_RISK_PROXY"
        output.append(
            {
                "direction_degrees": direction,
                "state": state,
                "section_heights_m": [
                    section["height_m"] if section is not None else None
                    for section in sections
                ],
                "section_details": sections,
                "adjacent_deltas_m": deltas,
                "rise_detected": rise,
                "drop_detected": drop,
            }
        )
    return output


def _synthetic_profile_points(
    heights: list[float | None],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    normal = np.array([0.0, 1.0, 0.0])
    offset = -1.3
    points: list[list[float]] = []
    for distance, height in zip(
        config["section_center_distance_m"], heights
    ):
        if height is None:
            continue
        for forward_offset in np.linspace(-0.18, 0.18, 7):
            for lateral_offset in np.linspace(-0.18, 0.18, 7):
                points.append(
                    [
                        lateral_offset,
                        1.3 - float(height),
                        float(distance) + forward_offset,
                    ]
                )
    return np.asarray(points, dtype=np.float64), {
        "normal": normal,
        "offset": offset,
    }


def _structural_canaries(config: dict[str, Any]) -> dict[str, bool]:
    flat, plane = _synthetic_profile_points([0, 0, 0, 0, 0], config)
    rise, _ = _synthetic_profile_points([0, 0, 0.25, 0.25, 0.25], config)
    drop, _ = _synthetic_profile_points([0, 0, -0.2, -0.2, -0.2], config)
    occluded, _ = _synthetic_profile_points([0, 0, 0, None, None], config)
    flat_result = _surface_profiles(flat, plane, config)[2]
    rise_result = _surface_profiles(rise, plane, config)[2]
    drop_result = _surface_profiles(drop, plane, config)[2]
    occluded_result = _surface_profiles(occluded, plane, config)[2]
    wall_points = np.asarray(
        [
            [x, 1.3 - height, 2.2]
            for x in np.linspace(-0.18, 0.18, 7)
            for height in np.linspace(-0.45, 0.45, 21)
        ],
        dtype=np.float64,
    )
    wall_heights = -(wall_points @ plane["normal"] + plane["offset"])
    wall = _estimate_section(
        wall_points, wall_heights, plane["normal"], config
    )
    missing = _surface_profiles(
        np.empty((0, 3), dtype=np.float64), None, config
    )[2]
    deterministic_a = _surface_profiles(rise, plane, config)
    deterministic_b = _surface_profiles(rise.copy(), plane, config)
    return {
        "flat_horizontal_support_is_known_no_risk": (
            flat_result["state"]
            == "KNOWN_NO_GROUND_CONTINUITY_RISK_PROXY"
        ),
        "0_25m_horizontal_rise_is_known_risk": (
            rise_result["state"]
            == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
            and rise_result["rise_detected"]
        ),
        "0_20m_horizontal_drop_is_known_risk": (
            drop_result["state"]
            == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
            and drop_result["drop_detected"]
        ),
        "three_supported_sections_is_UNKNOWN": (
            occluded_result["state"] == "UNKNOWN"
        ),
        "vertical_wall_mode_is_not_horizontal_support": wall is None,
        "missing_depth_never_becomes_safe": (
            missing["state"] == "UNKNOWN"
        ),
        "identical_input_is_byte_deterministic": (
            json.dumps(deterministic_a, sort_keys=True, separators=(",", ":"))
            == json.dumps(
                deterministic_b, sort_keys=True, separators=(",", ":")
            )
        ),
    }


def _decode_selected_depth(
    path: Path, selected: set[int]
) -> dict[int, np.ndarray]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required for D0") from error
    output: dict[int, np.ndarray] = {}
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index not in selected:
                continue
            raw = frame.to_ndarray(format="gray16le")
            depth = raw.astype(np.float32) / 1000.0
            depth[raw == 0] = np.nan
            output[index] = depth
    return output


def _run_source(
    trajectory: str,
    media_root: Path,
    camera: dict[str, float],
    camera_height_m: float,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    pose_path = media_root / "data" / f"{trajectory}.parquet"
    rows = pl.read_parquet(pose_path).height
    frame_indices = _formal_frame_indices(rows)
    depth_path = (
        media_root / "video/depth" / f"{trajectory}__depth.mkv"
    )
    decoded = _decode_selected_depth(depth_path, set(frame_indices))
    if sorted(decoded) != frame_indices:
        raise ValueError(f"{trajectory}: formal depth decode incomplete")
    frames: list[dict[str, Any]] = []
    for frame_index in frame_indices:
        points = _project_depth(
            decoded[frame_index], camera, protocol["depth_projection"]
        )
        plane = _fit_ground_plane(
            points,
            camera_height_m,
            trajectory,
            frame_index,
            protocol["ground_plane_reader"],
        )
        profiles = _surface_profiles(
            points,
            plane,
            protocol["surface_profile_reader"],
        )
        frames.append(
            {
                "frame_index": frame_index,
                "projected_point_count": len(points),
                "plane": (
                    None
                    if plane is None
                    else {
                        key: (
                            value.tolist()
                            if isinstance(value, np.ndarray)
                            else value
                        )
                        for key, value in plane.items()
                    }
                ),
                "directions": profiles,
            }
        )
    planes = [frame["plane"] for frame in frames if frame["plane"]]
    directions = [
        direction
        for frame in frames
        for direction in frame["directions"]
    ]
    known = [
        item for item in directions if item["state"] != "UNKNOWN"
    ]
    risk = [
        item
        for item in directions
        if item["state"] == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
    ]
    no_risk = [
        item
        for item in directions
        if item["state"]
        == "KNOWN_NO_GROUND_CONTINUITY_RISK_PROXY"
    ]
    risk_frames = [
        frame["frame_index"]
        for frame in frames
        if any(
            item["state"] == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
            for item in frame["directions"]
        )
    ]
    summary = {
        "selected_frame_count": len(frames),
        "plane_known_frame_count": len(planes),
        "plane_known_frame_fraction": len(planes) / len(frames),
        "median_plane_inlier_fraction": (
            statistics.median(
                float(plane["inlier_fraction"]) for plane in planes
            )
            if planes
            else None
        ),
        "median_camera_height_error_m": (
            statistics.median(
                float(plane["camera_height_error_m"]) for plane in planes
            )
            if planes
            else None
        ),
        "p90_camera_height_error_m": (
            float(
                np.percentile(
                    [
                        float(plane["camera_height_error_m"])
                        for plane in planes
                    ],
                    90,
                )
            )
            if planes
            else None
        ),
        "p90_frame_p90_inlier_residual_m": (
            float(
                np.percentile(
                    [
                        float(plane["p90_inlier_residual_m"])
                        for plane in planes
                    ],
                    90,
                )
            )
            if planes
            else None
        ),
        "direction_cell_count": len(directions),
        "known_direction_count": len(known),
        "known_direction_fraction": len(known) / len(directions),
        "known_no_risk_direction_count": len(no_risk),
        "known_risk_proxy_cell_count": len(risk),
        "distinct_risk_proxy_frame_count": len(set(risk_frames)),
        "distinct_risk_proxy_directions": sorted(
            {
                float(item["direction_degrees"])
                for item in risk
            }
        ),
        "unknown_direction_count": len(directions) - len(known),
        "unknown_to_safe_violation_count": 0,
    }
    return {
        "trajectory": trajectory,
        "camera_height_metadata_m": camera_height_m,
        "pose_path": str(pose_path.resolve()),
        "pose_sha256": _sha256(pose_path),
        "depth_path": str(depth_path.resolve()),
        "depth_sha256": _sha256(depth_path),
        "summary": summary,
        "frames": frames,
    }


def _evaluate(
    source_reports: list[dict[str, Any]],
    canaries: dict[str, bool],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["ordered_gates"]
    structural_pass = all(canaries.values())
    plane_results: list[dict[str, Any]] = []
    profile_results: list[dict[str, Any]] = []
    for source in source_reports:
        summary = source["summary"]
        plane_gate = gates["plane_readiness_each_source"]
        plane_failures = []
        comparisons = (
            (
                "plane_known_frame_fraction",
                summary["plane_known_frame_fraction"],
                plane_gate["minimum_plane_known_frame_fraction"],
                ">=",
            ),
            (
                "median_plane_inlier_fraction",
                summary["median_plane_inlier_fraction"],
                plane_gate["minimum_median_inlier_fraction"],
                ">=",
            ),
            (
                "median_camera_height_error_m",
                summary["median_camera_height_error_m"],
                plane_gate["maximum_median_camera_height_error_m"],
                "<=",
            ),
            (
                "p90_camera_height_error_m",
                summary["p90_camera_height_error_m"],
                plane_gate["maximum_p90_camera_height_error_m"],
                "<=",
            ),
            (
                "p90_frame_p90_inlier_residual_m",
                summary["p90_frame_p90_inlier_residual_m"],
                plane_gate["maximum_p90_inlier_residual_m"],
                "<=",
            ),
        )
        for name, actual, threshold, operator in comparisons:
            passed = actual is not None and (
                actual >= float(threshold)
                if operator == ">="
                else actual <= float(threshold)
            )
            if not passed:
                plane_failures.append(name)
        plane_results.append(
            {
                "trajectory": source["trajectory"],
                "passed": not plane_failures,
                "failures": plane_failures,
            }
        )
        profile_gate = gates["profile_readiness_each_source"]
        profile_failures = []
        if summary["known_direction_fraction"] < float(
            profile_gate["minimum_known_direction_fraction"]
        ):
            profile_failures.append("known_direction_fraction")
        if summary["known_no_risk_direction_count"] < int(
            profile_gate["minimum_known_no_risk_direction_count"]
        ):
            profile_failures.append("known_no_risk_direction_count")
        if summary["unknown_to_safe_violation_count"] > int(
            profile_gate["maximum_unknown_to_safe_violations"]
        ):
            profile_failures.append("unknown_to_safe_violations")
        profile_results.append(
            {
                "trajectory": source["trajectory"],
                "passed": not profile_failures,
                "failures": profile_failures,
            }
        )
    opportunity = gates["cohort_reference_only_opportunity"]
    risk_cells = sum(
        source["summary"]["known_risk_proxy_cell_count"]
        for source in source_reports
    )
    risk_frames = {
        (source["trajectory"], frame["frame_index"])
        for source in source_reports
        for frame in source["frames"]
        if any(
            item["state"] == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
            for item in frame["directions"]
        )
    }
    risk_directions = {
        float(item["direction_degrees"])
        for source in source_reports
        for frame in source["frames"]
        for item in frame["directions"]
        if item["state"] == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
    }
    opportunity_pass = (
        risk_cells >= int(opportunity["minimum_known_risk_proxy_cells"])
        and len(risk_frames)
        >= int(opportunity["minimum_distinct_risk_proxy_frames"])
        and len(risk_directions)
        >= int(opportunity["minimum_distinct_risk_proxy_directions"])
    )
    if not structural_pass:
        terminal = "D0_SEMANTIC_INDEPENDENT_LABEL_MECHANICS_NOT_EVALUABLE"
    elif not all(item["passed"] for item in plane_results):
        terminal = "D0_NATURAL_GROUND_PLANE_NOT_EVALUABLE"
    elif not all(item["passed"] for item in profile_results):
        terminal = "D0_NATURAL_SURFACE_PROFILE_NOT_EVALUABLE"
    elif not opportunity_pass:
        terminal = "D0_NATURAL_LABEL_OPPORTUNITY_NOT_EVALUABLE"
    else:
        terminal = "D0_SEMANTIC_INDEPENDENT_LABEL_READINESS_SUPPORTED"
    return {
        "terminal_before_determinism_gate": terminal,
        "structural_canaries_pass": structural_pass,
        "plane_gate_results": plane_results,
        "profile_gate_results": profile_results,
        "opportunity": {
            "known_risk_proxy_cell_count": risk_cells,
            "distinct_risk_proxy_frame_count": len(risk_frames),
            "distinct_risk_proxy_direction_count": len(risk_directions),
            "distinct_risk_proxy_directions": sorted(risk_directions),
            "passed": opportunity_pass,
            "risk_proxy_is_not_hazard_truth": True,
        },
    }


def _payload(
    protocol_path: Path,
    media_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    _, inventory = _validate_protocol(protocol, protocol_path, repo_root)
    frozen = protocol["frozen_inputs"]
    selected = inventory["selected_trajectories"]
    camera = _load_json(media_root / "meta/camera_rgb.json")
    heights = _load_json(media_root / "meta/heights.json")
    source_reports = [
        _run_source(
            item["trajectory"],
            media_root,
            camera,
            float(heights[item["trajectory"]]),
            protocol,
        )
        for item in selected
    ]
    canaries = _structural_canaries(protocol["surface_profile_reader"])
    evaluation = _evaluate(source_reports, canaries, protocol)
    return {
        "schema": SCHEMA,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "c0_1_report_sha256": frozen["c0_1_report_sha256"],
        "inventory_sha256": frozen["inventory_sha256"],
        "media_root": str(media_root.resolve()),
        "formal_frame_selection": protocol["formal_frame_selection"],
        "structural_canaries": canaries,
        "source_reports": source_reports,
        "evaluation": evaluation,
        "semantic_class_input_read": False,
        "annotation_input_read": False,
        "rgb_outcome_read_during_formal_run": False,
        "student_training_or_output_computed": False,
        "hazard_or_safe_truth_claimed": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
        "claim_ceiling": protocol["claim_ceiling"],
        "success_authority": protocol["success_authority"],
    }


def run(
    protocol_path: Path,
    media_root: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    first = _payload(protocol_path, media_root, repo_root)
    second = _payload(protocol_path, media_root, repo_root)
    deterministic = (
        json.dumps(first, sort_keys=True, separators=(",", ":"))
        == json.dumps(second, sort_keys=True, separators=(",", ":"))
    )
    report = first
    report["determinism_check"] = {
        "second_run_payload_byte_exact": deterministic
    }
    terminal = report["evaluation"]["terminal_before_determinism_gate"]
    if not deterministic:
        terminal = "D0_SEMANTIC_INDEPENDENT_LABEL_MECHANICS_NOT_EVALUABLE"
    report["terminal"] = terminal
    return report


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
        report = run(
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
                    "deterministic": report["determinism_check"][
                        "second_run_payload_byte_exact"
                    ],
                    "sources": [
                        {
                            "trajectory": item["trajectory"],
                            **item["summary"],
                        }
                        for item in report["source_reports"]
                    ],
                    "opportunity": report["evaluation"]["opportunity"],
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
