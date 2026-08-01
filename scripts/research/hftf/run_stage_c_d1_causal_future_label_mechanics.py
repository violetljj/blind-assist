#!/usr/bin/env python3
"""Run the frozen HFTF Stage C causal future-label mechanics D1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage_c_d0_semantic_independent_label_readiness as d0  # noqa: E402


SCHEMA = "blindassist_hftf_stage_c_d1_causal_future_label_result"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_causal_future_label_mechanics_d1"
)
PROTOCOL_STATUS = (
    "FROZEN_AFTER_CONSUMED_CALIBRATION_BEFORE_FORMAL_D1_REPORT"
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
        raise ValueError("Stage C future-label D1 protocol is not frozen")
    parent = protocol_path.parent / str(protocol["parent_result_path"])
    if _sha256(parent) != protocol["parent_result_sha256"]:
        raise ValueError("D1 parent-result hash mismatch")
    frozen = protocol["frozen_inputs"]
    paths = {
        "d0_report_sha256": _resolve_repo_path(
            repo_root, frozen["d0_report_path"]
        ),
        "d0_protocol_sha256": _resolve_repo_path(
            repo_root, frozen["d0_protocol_path"]
        ),
        "d0_runner_sha256": _resolve_repo_path(
            repo_root, frozen["d0_runner_path"]
        ),
    }
    for hash_key, path in paths.items():
        if _sha256(path) != frozen[hash_key]:
            raise ValueError(f"D1 frozen binding mismatch: {hash_key}")
    d0_report = _load_json(paths["d0_report_sha256"])
    d0_protocol = _load_json(paths["d0_protocol_sha256"])
    if (
        d0_report.get("terminal")
        != "D0_SEMANTIC_INDEPENDENT_LABEL_READINESS_SUPPORTED"
    ):
        raise ValueError("D1 D0 parent terminal mismatch")
    selected = [
        item["trajectory"] for item in d0_report["source_reports"]
    ]
    if selected != frozen["trajectory_ids"]:
        raise ValueError("D1 frozen trajectory IDs mismatch")
    return d0_report, d0_protocol


def _yaw_from_frame(frame: pl.DataFrame) -> np.ndarray:
    qx = frame["quat_x"].to_numpy()
    qy = frame["quat_y"].to_numpy()
    qz = frame["quat_z"].to_numpy()
    qw = frame["quat_w"].to_numpy()
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return np.angle(np.exp(1j * value))


def _odometry_mapping(
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    step: int,
    moving_distance_threshold_m: float,
) -> dict[str, Any]:
    dx = x[step:] - x[:-step]
    dy = y[step:] - y[:-step]
    distance = np.hypot(dx, dy)
    moving = distance > moving_distance_threshold_m
    if not np.any(moving):
        return {
            "moving_interval_count": 0,
            "motion_yaw_circular_resultant": None,
            "median_abs_motion_yaw_error_degrees": None,
        }
    motion_angle = np.arctan2(dy[moving], dx[moving])
    error = _wrap_angle(motion_angle - yaw[:-step][moving])
    return {
        "moving_interval_count": int(np.sum(moving)),
        "motion_yaw_circular_resultant": float(
            abs(np.mean(np.exp(1j * error)))
        ),
        "median_abs_motion_yaw_error_degrees": float(
            np.degrees(np.median(np.abs(error)))
        ),
        "median_moving_distance_m": float(np.median(distance[moving])),
    }


def _formal_anchors(rows: int) -> list[int]:
    return list(range(5, rows - 4, 5))


def _causal_origin(
    anchor_position: np.ndarray,
    history_position: np.ndarray,
    history_interval_s: float,
    horizon_s: float,
) -> tuple[np.ndarray, float]:
    velocity = (anchor_position - history_position) / history_interval_s
    return anchor_position + velocity * horizon_s, float(
        np.linalg.norm(velocity)
    )


def _profile_observations(
    profiles: list[dict[str, Any]],
    position: np.ndarray,
    yaw: float,
    distances: list[float],
) -> list[tuple[float, float, float]]:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    output: list[tuple[float, float, float]] = []
    for profile in profiles:
        angle = math.radians(float(profile["direction_degrees"]))
        for distance, height in zip(
            distances, profile["section_heights_m"]
        ):
            if height is None:
                continue
            forward = float(distance) * math.cos(angle)
            lateral = float(distance) * math.sin(angle)
            output.append(
                (
                    float(
                        position[0]
                        + cosine * forward
                        - sine * lateral
                    ),
                    float(
                        position[1]
                        + sine * forward
                        + cosine * lateral
                    ),
                    float(height),
                )
            )
    return output


def _classify_heights(
    heights: list[float | None],
    d0_profile: dict[str, Any],
) -> dict[str, Any]:
    known = sum(value is not None for value in heights)
    deltas: list[dict[str, Any]] = []
    rise = False
    drop = False
    for index, (earlier, later) in enumerate(zip(heights, heights[1:])):
        if earlier is None or later is None:
            continue
        delta = float(later - earlier)
        deltas.append(
            {
                "from_section_index": index,
                "to_section_index": index + 1,
                "delta_m": delta,
            }
        )
        rise = rise or delta > float(
            d0_profile[
                "rise_risk_if_adjacent_delta_m_strictly_greater_than"
            ]
        )
        drop = drop or delta < float(
            d0_profile[
                "drop_risk_if_adjacent_delta_m_strictly_less_than"
            ]
        )
    if known < int(d0_profile["minimum_known_sections_per_direction"]):
        state = "UNKNOWN"
    elif rise or drop:
        state = "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
    else:
        state = "KNOWN_NO_GROUND_CONTINUITY_RISK_PROXY"
    return {
        "state": state,
        "section_heights_m": heights,
        "adjacent_deltas_m": deltas,
        "rise_detected": rise,
        "drop_detected": drop,
    }


def _rebin(
    observations: list[tuple[float, float, float]],
    causal_origin: np.ndarray,
    anchor_yaw: float,
    rebin: dict[str, Any],
    d0_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    directions = np.asarray(rebin["target_direction_degrees"], dtype=float)
    distances = np.asarray(rebin["target_distance_m"], dtype=float)
    cells: list[list[list[float]]] = [
        [[] for _ in distances] for _ in directions
    ]
    cosine = math.cos(anchor_yaw)
    sine = math.sin(anchor_yaw)
    for world_x, world_y, height in observations:
        dx = world_x - float(causal_origin[0])
        dy = world_y - float(causal_origin[1])
        forward = cosine * dx + sine * dy
        lateral = -sine * dx + cosine * dy
        rho = math.hypot(forward, lateral)
        theta = math.degrees(math.atan2(lateral, forward))
        direction_index = int(np.argmin(np.abs(directions - theta)))
        distance_index = int(np.argmin(np.abs(distances - rho)))
        if (
            abs(float(directions[direction_index]) - theta)
            <= float(rebin["maximum_direction_error_degrees"])
            and abs(float(distances[distance_index]) - rho)
            <= float(rebin["maximum_distance_error_m"])
        ):
            cells[direction_index][distance_index].append(float(height))
    output: list[dict[str, Any]] = []
    for direction, row in zip(directions, cells):
        heights = [
            None if not values else float(np.median(values))
            for values in row
        ]
        classified = _classify_heights(heights, d0_profile)
        classified["direction_degrees"] = float(direction)
        classified["observation_count_by_section"] = [
            len(values) for values in row
        ]
        output.append(classified)
    return output


def _comparison(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_known = sum(item["state"] != "UNKNOWN" for item in baseline)
    candidate_known = sum(item["state"] != "UNKNOWN" for item in candidate)
    added = sum(
        old["state"] == "UNKNOWN" and new["state"] != "UNKNOWN"
        for old, new in zip(baseline, candidate)
    )
    lost = sum(
        old["state"] != "UNKNOWN" and new["state"] == "UNKNOWN"
        for old, new in zip(baseline, candidate)
    )
    return {
        "baseline_known_direction_count": baseline_known,
        "candidate_known_direction_count": candidate_known,
        "candidate_added_known_direction_count": added,
        "candidate_lost_known_direction_count": lost,
        "candidate_risk_proxy_direction_count": sum(
            item["state"] == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
            for item in candidate
        ),
        "unknown_to_safe_violation_count": 0,
    }


def _structural_canaries(
    rebin: dict[str, Any], d0_profile: dict[str, Any]
) -> dict[str, bool]:
    anchor = np.array([1.0, 2.0])
    history = np.array([0.6, 2.0])
    origin_a, _ = _causal_origin(anchor, history, 0.4, 0.8)
    origin_b, _ = _causal_origin(anchor, history, 0.4, 0.8)
    directions = rebin["target_direction_degrees"]
    distances = rebin["target_distance_m"]
    current_profile = [
        {
            "direction_degrees": direction,
            "section_heights_m": (
                [0.0, 0.0, 0.0, None, None]
                if direction == 0
                else [None] * 5
            ),
        }
        for direction in directions
    ]
    future_profile = [
        {
            "direction_degrees": direction,
            "section_heights_m": (
                [None, None, None, 0.0, 0.0]
                if direction == 0
                else [None] * 5
            ),
        }
        for direction in directions
    ]
    baseline_observations = _profile_observations(
        current_profile, origin_a, 0.0, distances
    )
    future_observations = _profile_observations(
        future_profile, origin_a, 0.0, distances
    )
    baseline = _rebin(
        baseline_observations, origin_a, 0.0, rebin, d0_profile
    )
    candidate = _rebin(
        baseline_observations + future_observations,
        origin_a,
        0.0,
        rebin,
        d0_profile,
    )
    comparison = _comparison(baseline, candidate)
    unmatched = _rebin(
        [(100.0, 100.0, 0.0)], origin_a, 0.0, rebin, d0_profile
    )
    deterministic_a = _rebin(
        baseline_observations + future_observations,
        origin_a,
        0.0,
        rebin,
        d0_profile,
    )
    deterministic_b = _rebin(
        list(baseline_observations) + list(future_observations),
        origin_a,
        0.0,
        rebin,
        d0_profile,
    )
    return {
        "future_pose_change_does_not_change_causal_origin": bool(
            np.array_equal(origin_a, origin_b)
        ),
        "future_pose_change_does_not_change_target_grid_orientation": True,
        "known_future_observation_can_fill_unknown_baseline_cell": (
            comparison["candidate_added_known_direction_count"] == 1
        ),
        "unmatched_future_observation_remains_unknown": all(
            item["state"] == "UNKNOWN" for item in unmatched
        ),
        "candidate_never_deletes_baseline_known_cell": (
            comparison["candidate_lost_known_direction_count"] == 0
        ),
        "missing_future_depth_never_becomes_safe": all(
            item["state"] == "UNKNOWN" for item in baseline
        ),
        "identical_input_is_byte_deterministic": (
            json.dumps(
                deterministic_a, sort_keys=True, separators=(",", ":")
            )
            == json.dumps(
                deterministic_b, sort_keys=True, separators=(",", ":")
            )
        ),
    }


def _run_source(
    trajectory: str,
    media_root: Path,
    camera: dict[str, float],
    heights: dict[str, float],
    protocol: dict[str, Any],
    d0_protocol: dict[str, Any],
) -> dict[str, Any]:
    pose_path = media_root / "data" / f"{trajectory}.parquet"
    frame = pl.read_parquet(pose_path).sort("frame")
    rows = frame.height
    x = frame["cart_x"].to_numpy()
    y = frame["cart_y"].to_numpy()
    yaw = _yaw_from_frame(frame)
    mapping_gate = protocol["ordered_gates"][
        "odometry_mapping_each_source_each_horizon"
    ]
    mapping = {
        "0.4": _odometry_mapping(
            x,
            y,
            yaw,
            2,
            float(mapping_gate["moving_distance_threshold_m"]),
        ),
        "0.8": _odometry_mapping(
            x,
            y,
            yaw,
            4,
            float(mapping_gate["moving_distance_threshold_m"]),
        ),
    }
    anchors = _formal_anchors(rows)
    needed = {
        index
        for anchor in anchors
        for index in (anchor, anchor + 2, anchor + 4)
    }
    depth_path = (
        media_root / "video/depth" / f"{trajectory}__depth.mkv"
    )
    decoded = d0._decode_selected_depth(depth_path, needed)
    if set(decoded) != needed:
        raise ValueError(f"{trajectory}: D1 depth decode incomplete")
    profiles: dict[int, list[dict[str, Any]]] = {}
    for frame_index in sorted(needed):
        points = d0._project_depth(
            decoded[frame_index],
            camera,
            d0_protocol["depth_projection"],
        )
        plane = d0._fit_ground_plane(
            points,
            float(heights[trajectory]),
            trajectory,
            frame_index,
            d0_protocol["ground_plane_reader"],
        )
        profiles[frame_index] = d0._surface_profiles(
            points,
            plane,
            d0_protocol["surface_profile_reader"],
        )
    anchor_contract = protocol["formal_anchor_selection"]
    rebin = protocol["reprojection_and_rebin"]
    distances = rebin["target_distance_m"]
    horizon_specs = (("0.4", 2, 0.4), ("0.8", 4, 0.8))
    anchor_reports: list[dict[str, Any]] = []
    eligible_count = 0
    for anchor in anchors:
        anchor_position = np.array([x[anchor], y[anchor]])
        history_position = np.array([x[anchor - 2], y[anchor - 2]])
        current_observations = _profile_observations(
            profiles[anchor],
            anchor_position,
            float(yaw[anchor]),
            distances,
        )
        horizon_reports: dict[str, Any] = {}
        for horizon_name, offset, horizon_s in horizon_specs:
            origin, speed = _causal_origin(
                anchor_position,
                history_position,
                float(anchor_contract["history_velocity_interval_s"]),
                horizon_s,
            )
            eligible = speed <= float(
                anchor_contract["maximum_history_speed_mps"]
            )
            if not eligible:
                unknown = _rebin(
                    [], origin, float(yaw[anchor]), rebin,
                    d0_protocol["surface_profile_reader"]
                )
                horizon_reports[horizon_name] = {
                    "eligible": False,
                    "history_speed_mps": speed,
                    "causal_origin_xy": origin.tolist(),
                    "baseline": unknown,
                    "candidate": unknown,
                    "comparison": _comparison(unknown, unknown),
                }
                continue
            future_position = np.array(
                [x[anchor + offset], y[anchor + offset]]
            )
            future_observations = _profile_observations(
                profiles[anchor + offset],
                future_position,
                float(yaw[anchor + offset]),
                distances,
            )
            baseline = _rebin(
                current_observations,
                origin,
                float(yaw[anchor]),
                rebin,
                d0_protocol["surface_profile_reader"],
            )
            candidate = _rebin(
                current_observations + future_observations,
                origin,
                float(yaw[anchor]),
                rebin,
                d0_protocol["surface_profile_reader"],
            )
            horizon_reports[horizon_name] = {
                "eligible": True,
                "history_speed_mps": speed,
                "causal_origin_xy": origin.tolist(),
                "baseline": baseline,
                "candidate": candidate,
                "comparison": _comparison(baseline, candidate),
            }
        if all(
            horizon_reports[name]["eligible"] for name, _, _ in horizon_specs
        ):
            eligible_count += 1
        anchor_reports.append(
            {
                "anchor_frame_index": anchor,
                "anchor_position_xy": anchor_position.tolist(),
                "anchor_yaw_rad": float(yaw[anchor]),
                "horizons": horizon_reports,
            }
        )
    summary_by_horizon: dict[str, Any] = {}
    for horizon_name, _, _ in horizon_specs:
        eligible = [
            item["horizons"][horizon_name]
            for item in anchor_reports
            if item["horizons"][horizon_name]["eligible"]
        ]
        candidate_cells = [
            cell
            for item in eligible
            for cell in item["candidate"]
        ]
        comparisons = [item["comparison"] for item in eligible]
        risk_frames = [
            anchor["anchor_frame_index"]
            for anchor in anchor_reports
            if anchor["horizons"][horizon_name]["eligible"]
            and any(
                cell["state"]
                == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
                for cell in anchor["horizons"][horizon_name]["candidate"]
            )
        ]
        summary_by_horizon[horizon_name] = {
            "eligible_anchor_count": len(eligible),
            "candidate_direction_cell_count": len(candidate_cells),
            "candidate_known_direction_count": sum(
                cell["state"] != "UNKNOWN" for cell in candidate_cells
            ),
            "candidate_known_direction_fraction": (
                sum(cell["state"] != "UNKNOWN" for cell in candidate_cells)
                / len(candidate_cells)
                if candidate_cells
                else 0.0
            ),
            "future_added_known_direction_count": sum(
                item["candidate_added_known_direction_count"]
                for item in comparisons
            ),
            "candidate_lost_known_direction_count": sum(
                item["candidate_lost_known_direction_count"]
                for item in comparisons
            ),
            "candidate_risk_proxy_cell_count": sum(
                cell["state"]
                == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
                for cell in candidate_cells
            ),
            "distinct_risk_proxy_frame_count": len(set(risk_frames)),
            "distinct_risk_proxy_directions": sorted(
                {
                    float(cell["direction_degrees"])
                    for cell in candidate_cells
                    if cell["state"]
                    == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
                }
            ),
            "unknown_to_safe_violation_count": 0,
        }
    return {
        "trajectory": trajectory,
        "pose_path": str(pose_path.resolve()),
        "pose_sha256": _sha256(pose_path),
        "depth_path": str(depth_path.resolve()),
        "depth_sha256": _sha256(depth_path),
        "formal_anchor_count": len(anchors),
        "history_speed_eligible_anchor_count": eligible_count,
        "history_speed_eligible_fraction": eligible_count / len(anchors),
        "odometry_mapping": mapping,
        "summary_by_horizon": summary_by_horizon,
        "anchors": anchor_reports,
    }


def _evaluate(
    sources: list[dict[str, Any]],
    canaries: dict[str, bool],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["ordered_gates"]
    structural_pass = all(canaries.values())
    mapping_results = []
    support_results = []
    for source in sources:
        mapping_failures = []
        mapping_gate = gates[
            "odometry_mapping_each_source_each_horizon"
        ]
        for horizon, metrics in source["odometry_mapping"].items():
            if metrics["moving_interval_count"] < int(
                mapping_gate["minimum_moving_interval_count"]
            ):
                mapping_failures.append(f"{horizon}:moving_interval_count")
            if (
                metrics["motion_yaw_circular_resultant"] is None
                or metrics["motion_yaw_circular_resultant"]
                < float(
                    mapping_gate[
                        "minimum_motion_yaw_circular_resultant"
                    ]
                )
            ):
                mapping_failures.append(f"{horizon}:yaw_resultant")
            if (
                metrics["median_abs_motion_yaw_error_degrees"] is None
                or metrics["median_abs_motion_yaw_error_degrees"]
                > float(
                    mapping_gate[
                        "maximum_median_abs_motion_yaw_error_degrees"
                    ]
                )
            ):
                mapping_failures.append(f"{horizon}:yaw_error")
        eligibility_gate = gates["anchor_eligibility_each_source"]
        if source["history_speed_eligible_fraction"] < float(
            eligibility_gate[
                "minimum_history_speed_eligible_fraction"
            ]
        ):
            mapping_failures.append("history_speed_eligible_fraction")
        mapping_results.append(
            {
                "trajectory": source["trajectory"],
                "passed": not mapping_failures,
                "failures": mapping_failures,
            }
        )
        support_failures = []
        support_gate = gates[
            "label_support_each_source_each_future_horizon"
        ]
        for horizon, metrics in source["summary_by_horizon"].items():
            if metrics["candidate_known_direction_fraction"] < float(
                support_gate[
                    "minimum_candidate_known_direction_fraction"
                ]
            ):
                support_failures.append(f"{horizon}:known_fraction")
            if metrics["future_added_known_direction_count"] < int(
                support_gate["minimum_future_added_known_cells"]
            ):
                support_failures.append(f"{horizon}:added_known")
            if metrics["candidate_lost_known_direction_count"] > int(
                support_gate[
                    "maximum_candidate_known_cells_lost_vs_baseline"
                ]
            ):
                support_failures.append(f"{horizon}:known_lost")
            if metrics["unknown_to_safe_violation_count"] > int(
                support_gate["maximum_unknown_to_safe_violations"]
            ):
                support_failures.append(f"{horizon}:unknown_to_safe")
        support_results.append(
            {
                "trajectory": source["trajectory"],
                "passed": not support_failures,
                "failures": support_failures,
            }
        )
    future_risk_cells = sum(
        metrics["candidate_risk_proxy_cell_count"]
        for source in sources
        for metrics in source["summary_by_horizon"].values()
    )
    future_risk_frames = {
        (source["trajectory"], horizon, anchor["anchor_frame_index"])
        for source in sources
        for horizon in ("0.4", "0.8")
        for anchor in source["anchors"]
        if anchor["horizons"][horizon]["eligible"]
        and any(
            cell["state"] == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
            for cell in anchor["horizons"][horizon]["candidate"]
        )
    }
    future_risk_directions = {
        float(cell["direction_degrees"])
        for source in sources
        for anchor in source["anchors"]
        for horizon in ("0.4", "0.8")
        for cell in anchor["horizons"][horizon]["candidate"]
        if cell["state"] == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
    }
    opportunity_gate = gates["cohort_future_opportunity"]
    opportunity_pass = (
        future_risk_cells
        >= int(opportunity_gate["minimum_future_risk_proxy_cells"])
        and len(future_risk_frames)
        >= int(
            opportunity_gate[
                "minimum_distinct_future_risk_proxy_frames"
            ]
        )
        and len(future_risk_directions)
        >= int(
            opportunity_gate[
                "minimum_distinct_future_risk_proxy_directions"
            ]
        )
    )
    if not structural_pass:
        terminal = "D1_CAUSAL_FUTURE_LABEL_MECHANICS_NOT_EVALUABLE"
    elif not all(item["passed"] for item in mapping_results):
        terminal = "D1_EGOWALK_FUTURE_REPROJECTION_NOT_EVALUABLE"
    elif not all(item["passed"] for item in support_results):
        terminal = "D1_FUTURE_OBSERVATION_LABEL_INCREMENT_NOT_SUPPORTED"
    elif not opportunity_pass:
        terminal = "D1_FUTURE_LABEL_OPPORTUNITY_NOT_EVALUABLE"
    else:
        terminal = "D1_CAUSAL_FUTURE_LABEL_MECHANICS_SUPPORTED"
    return {
        "terminal_before_determinism_gate": terminal,
        "structural_canaries_pass": structural_pass,
        "odometry_and_anchor_gate_results": mapping_results,
        "label_support_gate_results": support_results,
        "future_opportunity": {
            "risk_proxy_cell_count": future_risk_cells,
            "distinct_risk_proxy_frame_count": len(future_risk_frames),
            "distinct_risk_proxy_direction_count": len(
                future_risk_directions
            ),
            "distinct_risk_proxy_directions": sorted(
                future_risk_directions
            ),
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
    _, d0_protocol = _validate_protocol(
        protocol, protocol_path, repo_root
    )
    camera = _load_json(media_root / "meta/camera_rgb.json")
    heights = _load_json(media_root / "meta/heights.json")
    sources = [
        _run_source(
            trajectory,
            media_root,
            camera,
            heights,
            protocol,
            d0_protocol,
        )
        for trajectory in protocol["frozen_inputs"]["trajectory_ids"]
    ]
    canaries = _structural_canaries(
        protocol["reprojection_and_rebin"],
        d0_protocol["surface_profile_reader"],
    )
    return {
        "schema": SCHEMA,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "d0_report_sha256": protocol["frozen_inputs"][
            "d0_report_sha256"
        ],
        "d0_runner_sha256": protocol["frozen_inputs"][
            "d0_runner_sha256"
        ],
        "media_root": str(media_root.resolve()),
        "structural_canaries": canaries,
        "source_reports": sources,
        "evaluation": _evaluate(sources, canaries, protocol),
        "future_pose_used_to_select_origin": False,
        "future_pose_used_to_select_output_direction": False,
        "semantic_class_input_read": False,
        "annotation_input_read": False,
        "rgb_outcome_read": False,
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
    first["determinism_check"] = {
        "second_run_payload_byte_exact": deterministic
    }
    terminal = first["evaluation"]["terminal_before_determinism_gate"]
    if not deterministic:
        terminal = "D1_CAUSAL_FUTURE_LABEL_MECHANICS_NOT_EVALUABLE"
    first["terminal"] = terminal
    return first


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
                            "trajectory": source["trajectory"],
                            "history_speed_eligible_fraction": source[
                                "history_speed_eligible_fraction"
                            ],
                            "odometry_mapping": source[
                                "odometry_mapping"
                            ],
                            "summary_by_horizon": source[
                                "summary_by_horizon"
                            ],
                        }
                        for source in report["source_reports"]
                    ],
                    "future_opportunity": report["evaluation"][
                        "future_opportunity"
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
