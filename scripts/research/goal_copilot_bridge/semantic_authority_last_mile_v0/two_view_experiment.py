"""Run the three SAGE-LM V1-B source-pose two-view diagnostic arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.research.assistive_geometry.arkitscenes_truth_reader import interpolate_camera_to_world, parse_trajectory

from .experiment import _aggregate
from .rgb_experiment import (
    ALIGNMENT_TOLERANCE_M,
    GEOMETRY_CONFIDENCE_THRESHOLD,
    NEAR_THRESHOLD_M,
    EvaluatorEpisode,
    _baseline,
    _decode_input,
    _decode_truth,
    _sage_lm,
    _v1_criteria,
)
from .two_view_observation import SourceCameraPose, SourcePoseTwoViewBoundaryProvider


SCHEMA_VERSION = "sage_lm_v1b_source_pose_two_view_boundary_geometry"
ARMS = ("b0", "b1", "b2")


def _source_poses(materialized: dict) -> tuple[SourceCameraPose, SourceCameraPose, dict]:
    first_rgb = Path(materialized["source"]["first_rgb"])
    rgb_dir = first_rgb.parent
    frames = sorted(rgb_dir.glob("*.png"))
    try:
        start = frames.index(first_rgb)
    except ValueError as error:
        raise ValueError(f"frozen first source frame is missing from sequence: {first_rgb}") from error
    active_index = int(materialized["input"]["active_parallax_frame_index"])
    window = frames[start : start + len(materialized["input"]["rgb_frames"])]
    selected = (window[0], window[active_index])
    trajectory = parse_trajectory(rgb_dir.parent / "lowres_wide.traj")
    window_poses = []
    window_interpolation = []
    for frame in window:
        timestamp = float(frame.stem.rsplit("_", 1)[1])
        transform, receipt = interpolate_camera_to_world(trajectory, timestamp, maximum_gap_seconds=0.25)
        window_poses.append(
            SourceCameraPose(
                tuple(float(value) for value in transform[:3, 3]),
                tuple(tuple(float(value) for value in row) for row in transform[:3, :3]),
            )
        )
        window_interpolation.append(receipt)
    poses = (window_poses[0], window_poses[active_index])
    interpolation = (window_interpolation[0], window_interpolation[active_index])
    relative_translation_a = poses[0].rotation.T @ (poses[1].position - poses[0].position)
    actual_lateral = float(abs(relative_translation_a[0]))
    actual_forward = float(abs(relative_translation_a[2]))
    stored_first = np.asarray(materialized["truth"]["camera_positions_m"][0], dtype=np.float64)
    stored_active = np.asarray(materialized["truth"]["camera_positions_m"][active_index], dtype=np.float64)
    valid_alternatives = []
    for index, candidate in enumerate(window_poses[2:], start=2):
        relative = poses[0].rotation.T @ (candidate.position - poses[0].position)
        lateral = float(abs(relative[0]))
        forward = float(abs(relative[2]))
        if 0.18 <= lateral <= 0.30 and forward <= 0.45:
            valid_alternatives.append(
                {"frame_index": index, "lateral_baseline_m": lateral, "forward_delta_m": forward}
            )
    return poses[0], poses[1], {
        "source_frame_paths": [str(path) for path in selected],
        "interpolation": interpolation,
        "actual_relative_translation_camera_a_m": [float(value) for value in relative_translation_a],
        "actual_lateral_baseline_m": actual_lateral,
        "actual_forward_delta_m": actual_forward,
        "intended_active_pair_gate_pass": 0.18 <= actual_lateral <= 0.30 and actual_forward <= 0.45,
        "valid_alternative_active_pairs_in_same_window": valid_alternatives,
        "frozen_camera_positions_field_error_m": [
            float(np.linalg.norm(stored_first - poses[0].position)),
            float(np.linalg.norm(stored_active - poses[1].position)),
        ],
    }


def _evaluator_episode(materialized: dict) -> tuple[EvaluatorEpisode, object, object]:
    episode_input = _decode_input(materialized["input"])
    truth = _decode_truth(materialized["truth"])
    evaluator = EvaluatorEpisode(
        episode_id=truth.episode_id,
        kind=episode_input.kind,
        aperture_x_m=truth.aperture_center_x_m,
        aperture_width_m=truth.aperture_width_m,
        anchor_x_m=float(materialized["baseline_anchor_x_m"]),
        start_x_m=0.0,
        start_range_m=truth.start_range_m,
        anchor_height_m=0.16 if episode_input.kind == "QR_ENTRANCE" else 0.22,
        occlusion_steps=tuple(materialized["occlusion_frame_indices"]),
    )
    return evaluator, episode_input, truth


def _arm_diagnostics(rows: list[dict], arm: str) -> dict:
    observations = [row[arm]["observation"] for row in rows]
    center_errors = sorted(
        abs(observation["center_x_m"] - row["truth"]["aperture_center_x_m"])
        for observation, row in zip(observations, rows)
        if observation["center_x_m"] is not None
    )
    range_errors = sorted(
        abs(observation["range_m"] - row["truth"]["start_range_m"])
        for observation, row in zip(observations, rows)
        if observation["range_m"] is not None
    )

    def median(values: list[float]) -> float | None:
        if not values:
            return None
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) * 0.5

    return {
        "geometry_output_count": sum(observation["center_x_m"] is not None for observation in observations),
        "geometry_confidence_pass_count": sum(observation["geometry_confidence"] >= GEOMETRY_CONFIDENCE_THRESHOLD for observation in observations),
        "center_absolute_error_mean_m": sum(center_errors) / len(center_errors) if center_errors else None,
        "center_absolute_error_median_m": median(center_errors),
        "range_absolute_error_mean_m": sum(range_errors) / len(range_errors) if range_errors else None,
        "range_absolute_error_median_m": median(range_errors),
        "uses_lk": False,
        "uses_metric_depth": False,
    }


def run(cohort_path: Path) -> dict:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if len(cohort["episodes"]) != 24:
        raise ValueError("V1-B requires the frozen 24-episode V1 cohort")
    rows = []
    controls_retained = {arm: 0 for arm in ARMS}
    for materialized in cohort["episodes"]:
        evaluator, episode_input, truth = _evaluator_episode(materialized)
        pose_a, pose_b, pose_audit = _source_poses(materialized)
        baseline = _baseline(evaluator)
        arms = {}
        for arm in ARMS:
            provider = SourcePoseTwoViewBoundaryProvider(
                episode_input,
                truth if arm in {"b0", "b1"} else None,
                pose_a,
                pose_b,
                arm,
            )
            result = _sage_lm(evaluator, provider)
            arms[arm] = result
            if materialized["control"] and result["true_arrival"]:
                controls_retained[arm] += 1
        rows.append(
            {
                "episode_id": evaluator.episode_id,
                "kind": evaluator.kind,
                "control": materialized["control"],
                "source": materialized["source"],
                "truth": materialized["truth"],
                "source_pose_audit": pose_audit,
                "baseline": baseline,
                **arms,
            }
        )
    baseline_metrics = _aggregate(row["baseline"] for row in rows)
    arm_metrics = {arm: _aggregate(row[arm] for row in rows) for arm in ARMS}
    criteria = {arm: _v1_criteria(baseline_metrics, arm_metrics[arm], controls_retained[arm]) for arm in ARMS}
    baseline_values = [row["source_pose_audit"]["actual_lateral_baseline_m"] for row in rows]
    forward_values = [row["source_pose_audit"]["actual_forward_delta_m"] for row in rows]
    source_pose_audit = {
        "authority": "ORIGINAL_ARKITSCENES_TRAJECTORY_OFFICIAL_WORLD_TO_CAMERA_INVERSION",
        "frozen_camera_positions_m_field_valid": False,
        "failure": "MATERIALIZER_INTERPRETED_ROTATION_VECTOR_COLUMNS_AS_CAMERA_POSITION",
        "intended_active_pair_gate_pass_count": sum(row["source_pose_audit"]["intended_active_pair_gate_pass"] for row in rows),
        "same_window_valid_alternative_count": sum(bool(row["source_pose_audit"]["valid_alternative_active_pairs_in_same_window"]) for row in rows),
        "actual_lateral_baseline_min_m": min(baseline_values),
        "actual_lateral_baseline_max_m": max(baseline_values),
        "actual_lateral_baseline_mean_m": sum(baseline_values) / len(baseline_values),
        "actual_forward_delta_max_m": max(forward_values),
    }
    evaluable = source_pose_audit["intended_active_pair_gate_pass_count"] == 24
    raw_passed = {arm: all(criteria[arm].values()) for arm in ARMS}
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "REVERSIBLE_EXPLORATION_DEVELOPMENT_STANDARD",
        "experiment_label": "V1_B_SOURCE_POSE_TWO_VIEW_BOUNDARY_GEOMETRY",
        "identity_contract": "EXACT_SEMANTIC_AUTHORITY_FIXED_GEOMETRY_CANNOT_REBIND",
        "frozen_policy": {
            "near_threshold_m": NEAR_THRESHOLD_M,
            "alignment_tolerance_m": ALIGNMENT_TOLERANCE_M,
            "geometry_confidence_threshold": GEOMETRY_CONFIDENCE_THRESHOLD,
            "movement_step_m": 0.28,
            "forward_step_m": 0.54,
            "arrival_support_frames": 2,
        },
        "observation_contract": {
            "source_pose": "ARKITSCENES_NATIVE_CAMERA_TO_WORLD",
            "optical_flow": "NOT_RUN",
            "monocular_metric_depth": "NOT_RUN",
            "b0": "EVALUATOR_BOUNDARY_PIXELS_PLUS_SOURCE_POSE",
            "b1": "RGB_BOUNDARY_CANDIDATES_PLUS_EVALUATOR_ASSOCIATION_PLUS_SOURCE_POSE",
            "b2": "RGB_BOUNDARY_CANDIDATES_PLUS_AUTOMATIC_POSE_CONSTRAINED_ASSOCIATION",
        },
        "cohort": {"episode_count": len(rows), "kinds": cohort["kinds"], "control_count": 6},
        "metrics": {"bbox_center_scale": baseline_metrics, **arm_metrics, "controls_retained": controls_retained},
        "observation_diagnostics": {arm: _arm_diagnostics(rows, arm) for arm in ARMS},
        "source_pose_audit": source_pose_audit,
        "criteria": criteria,
        "raw_criteria_passed": raw_passed,
        "passed": {"b0": raw_passed["b0"], "b1": raw_passed["b1"] if evaluable else None, "b2": raw_passed["b2"] if evaluable else None},
        "evaluable": evaluable,
        "adjudication": (
            "V1_B_EVALUABLE"
            if evaluable
            else "NOT_EVALUABLE_SOURCE_POSE_PAIR_CONTRACT_INVALID"
        ),
        "rows": rows,
        "claim_ceiling": "SOURCE_POSE_ASSISTED_CONTROLLED_REAL_RGB_GEOMETRY_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cohort)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"adjudication": report["adjudication"], "evaluable": report["evaluable"], "metrics": report["metrics"], "criteria": report["criteria"], "passed": report["passed"], "observation_diagnostics": report["observation_diagnostics"], "source_pose_audit": report["source_pose_audit"]}, indent=2))


if __name__ == "__main__":
    main()
