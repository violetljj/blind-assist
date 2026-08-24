"""Paired controlled-loop evaluation for the SAGE-LM V1 RGB adapter."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .experiment import Pose, _advance, _aggregate, _direction
from .observation import CameraIntrinsics, ExactAnchorObservation, RgbEpisodeInput, RgbEpisodeTruth
from .rgb_observation import RgbObservationProvider


SCHEMA_VERSION = "sage_lm_v1_controlled_real_rgb_observation"
ALIGNMENT_TOLERANCE_M = 0.22
NEAR_THRESHOLD_M = 0.82
GEOMETRY_CONFIDENCE_THRESHOLD = 0.35


@dataclass(frozen=True)
class EvaluatorEpisode:
    episode_id: str
    kind: str
    aperture_x_m: float
    aperture_width_m: float
    anchor_x_m: float
    start_x_m: float
    start_range_m: float
    anchor_height_m: float
    occlusion_steps: tuple[int, ...]


def _decode_input(value: dict) -> RgbEpisodeInput:
    return RgbEpisodeInput(
        episode_id=value["episode_id"],
        kind=value["kind"],
        rgb_frames=tuple(Path(path) for path in value["rgb_frames"]),
        intrinsics=CameraIntrinsics(**value["intrinsics"]),
        commanded_baseline_m=float(value["commanded_baseline_m"]),
        active_parallax_frame_index=int(value["active_parallax_frame_index"]),
        exact_anchor_observations=tuple(
            ExactAnchorObservation(
                frame_index=int(row["frame_index"]),
                referent_id=row["referent_id"],
                bbox_xyxy=tuple(row["bbox_xyxy"]) if row["bbox_xyxy"] is not None else None,
            )
            for row in value["exact_anchor_observations"]
        ),
    )


def _decode_truth(value: dict) -> RgbEpisodeTruth:
    return RgbEpisodeTruth(
        episode_id=value["episode_id"],
        aperture_center_x_m=float(value["aperture_center_x_m"]),
        aperture_width_m=float(value["aperture_width_m"]),
        start_range_m=float(value["start_range_m"]),
        camera_positions_m=tuple(tuple(row) for row in value["camera_positions_m"]),
        endpoint_center_x_m=float(value["endpoint_center_x_m"]),
    )


def _truth_arrived(episode: EvaluatorEpisode, pose: Pose) -> bool:
    usable_half_width = episode.aperture_width_m / 2.0 - 0.12
    return pose.range_m <= NEAR_THRESHOLD_M and abs(pose.x_m - episode.aperture_x_m) <= usable_half_width


def _baseline(episode: EvaluatorEpisode) -> dict:
    pose = Pose(episode.start_x_m, episode.start_range_m)
    path = [[pose.x_m, pose.range_m]]
    correct = count = completion = movement_lost = lost = reacquired = 0
    completion_step = None
    was_lost = False
    for step in range(22):
        visible = step not in episode.occlusion_steps
        if not visible:
            if not was_lost:
                lost += 1
            was_lost = True
            movement_lost += 1
        else:
            if was_lost:
                reacquired += 1
            was_lost = False
        target_x = episode.anchor_x_m
        correct += int(_direction(target_x - pose.x_m) == _direction(episode.aperture_x_m - pose.x_m))
        count += 1
        _advance(pose, target_x)
        path.append([pose.x_m, pose.range_m])
        apparent_anchor_height = episode.anchor_height_m / max(pose.range_m, 0.1)
        if apparent_anchor_height >= 0.19:
            completion = 1
            completion_step = step
            break
    arrived = _truth_arrived(episode, pose)
    return {
        "arm": "BBOX_CENTER_SCALE",
        "completion": bool(completion),
        "completion_step": completion_step,
        "true_arrival": arrived,
        "premature_arrival": bool(completion and not arrived),
        "endpoint_lateral_error_m": abs(pose.x_m - episode.aperture_x_m),
        "direction_correct": correct,
        "direction_count": count,
        "lost_events": lost,
        "reacquisitions": reacquired,
        "movement_during_lost": movement_lost,
        "path": path,
    }


def _sage_lm(episode: EvaluatorEpisode, provider: RgbObservationProvider) -> dict:
    observation = provider.observe()
    physical_pose = Pose(episode.start_x_m, episode.start_range_m)
    policy_range_m = observation.range_m if observation.range_m is not None else float("inf")
    path = [[physical_pose.x_m, physical_pose.range_m]]
    target_x = observation.center_x_m if observation.center_x_m is not None else physical_pose.x_m
    if observation.geometry_confidence >= GEOMETRY_CONFIDENCE_THRESHOLD:
        physical_pose.x_m += 0.12 if physical_pose.x_m <= target_x else -0.12
    path.append([physical_pose.x_m, physical_pose.range_m])
    correct = count = completion = movement_lost = lost = reacquired = arrival_support = 0
    completion_step = None
    was_lost = False
    visible_steps = {row.frame_index for row in provider.input.exact_anchor_observations if row.visible}
    for step in range(1, 28):
        visible = step in visible_steps or step >= len(provider.input.exact_anchor_observations)
        if not visible:
            if not was_lost:
                lost += 1
            was_lost = True
            arrival_support = 0
            path.append([physical_pose.x_m, physical_pose.range_m])
            continue
        if was_lost:
            reacquired += 1
        was_lost = False
        if observation.geometry_confidence < GEOMETRY_CONFIDENCE_THRESHOLD:
            path.append([physical_pose.x_m, physical_pose.range_m])
            continue
        correct += int(_direction(target_x - physical_pose.x_m) == _direction(episode.aperture_x_m - physical_pose.x_m))
        count += 1
        _advance(physical_pose, target_x)
        policy_range_m = max(0.45, policy_range_m - 0.54)
        path.append([physical_pose.x_m, physical_pose.range_m])
        near = policy_range_m <= NEAR_THRESHOLD_M
        aligned = abs(physical_pose.x_m - target_x) <= ALIGNMENT_TOLERANCE_M
        supported = observation.geometry_confidence >= GEOMETRY_CONFIDENCE_THRESHOLD
        arrival_support = arrival_support + 1 if near and aligned and supported else 0
        if arrival_support >= 2:
            completion = 1
            completion_step = step
            break
    arrived = _truth_arrived(episode, physical_pose)
    return {
        "arm": "SAGE_LM_RGB_APERTURE_PROGRESS",
        "completion": bool(completion),
        "completion_step": completion_step,
        "true_arrival": arrived,
        "premature_arrival": bool(completion and not arrived),
        "endpoint_lateral_error_m": abs(physical_pose.x_m - episode.aperture_x_m),
        "direction_correct": correct,
        "direction_count": count,
        "lost_events": lost,
        "reacquisitions": reacquired,
        "movement_during_lost": movement_lost,
        "geometry_confidence": observation.geometry_confidence,
        "observation": {
            "visible": observation.visible,
            "center_x_m": observation.center_x_m,
            "width_m": observation.width_m,
            "range_m": observation.range_m,
            "boundary_confidence": observation.boundary_confidence,
            "flow_confidence": observation.flow_confidence,
            "depth_consistency": observation.depth_consistency,
            "geometry_confidence": observation.geometry_confidence,
        },
        "diagnostics": provider.diagnostics,
        "path": path,
    }


def run(cohort_path: Path) -> dict:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    rows = []
    controls_retained = 0
    for materialized in cohort["episodes"]:
        input_value = _decode_input(materialized["input"])
        truth = _decode_truth(materialized["truth"])
        if input_value.episode_id != truth.episode_id:
            raise ValueError("input/truth episode mismatch")
        evaluator = EvaluatorEpisode(
            episode_id=truth.episode_id,
            kind=input_value.kind,
            aperture_x_m=truth.aperture_center_x_m,
            aperture_width_m=truth.aperture_width_m,
            anchor_x_m=float(materialized["baseline_anchor_x_m"]),
            start_x_m=0.0,
            start_range_m=truth.start_range_m,
            anchor_height_m=0.16 if input_value.kind == "QR_ENTRANCE" else 0.22,
            occlusion_steps=tuple(materialized["occlusion_frame_indices"]),
        )
        baseline = _baseline(evaluator)
        provider = RgbObservationProvider(input_value)
        sage = _sage_lm(evaluator, provider)
        if materialized["control"] and sage["true_arrival"]:
            controls_retained += 1
        rows.append(
            {
                "episode_id": evaluator.episode_id,
                "kind": evaluator.kind,
                "control": materialized["control"],
                "source": materialized["source"],
                "truth": materialized["truth"],
                "baseline": baseline,
                "sage_lm": sage,
            }
        )
    baseline_metrics = _aggregate(row["baseline"] for row in rows)
    sage_metrics = _aggregate(row["sage_lm"] for row in rows)
    center_errors = sorted(
        abs(row["sage_lm"]["observation"]["center_x_m"] - row["truth"]["aperture_center_x_m"])
        for row in rows
        if row["sage_lm"]["observation"]["center_x_m"] is not None
    )
    range_errors = sorted(
        abs(row["sage_lm"]["observation"]["range_m"] - row["truth"]["start_range_m"])
        for row in rows
        if row["sage_lm"]["observation"]["range_m"] is not None
    )

    def median(values: list[float]) -> float | None:
        if not values:
            return None
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0

    observation_diagnostics = {
        "boundary_pair_output_count": sum(row["sage_lm"]["observation"]["center_x_m"] is not None for row in rows),
        "geometry_confidence_pass_count": sum(row["sage_lm"]["observation"]["geometry_confidence"] >= GEOMETRY_CONFIDENCE_THRESHOLD for row in rows),
        "reciprocal_flow_confidence_ge_0_5_count": sum(row["sage_lm"]["observation"]["flow_confidence"] >= 0.5 for row in rows),
        "depth_consistency_ge_0_5_count": sum(row["sage_lm"]["observation"]["depth_consistency"] >= 0.5 for row in rows),
        "aperture_center_absolute_error_mean_m": sum(center_errors) / len(center_errors) if center_errors else None,
        "aperture_center_absolute_error_median_m": median(center_errors),
        "metric_range_absolute_error_mean_m": sum(range_errors) / len(range_errors) if range_errors else None,
        "metric_range_absolute_error_median_m": median(range_errors),
        "primary_failure_layer": "RECIPROCAL_FLOW_SURVIVAL_THEN_BOUNDARY_ASSOCIATION_AND_METRIC_RANGE",
    }
    criteria = {
        "target_front_arrival_at_least_18": int(round(sage_metrics["target_front_arrival_rate"] * 24)) >= 18,
        "net_success_gain_at_least_8": int(round((sage_metrics["target_front_arrival_rate"] - baseline_metrics["target_front_arrival_rate"]) * 24)) >= 8,
        "median_lateral_error_at_most_0_20": sage_metrics["median_endpoint_lateral_error_m"] <= 0.20,
        "median_lateral_error_reduction_at_least_50pct": sage_metrics["median_endpoint_lateral_error_m"] <= baseline_metrics["median_endpoint_lateral_error_m"] * 0.50,
        "completion_precision_at_least_85pct": (sage_metrics["completion_precision"] or 0.0) >= 0.85,
        "premature_arrival_at_most_3": sage_metrics["premature_arrival_count"] <= 3,
        "controls_retained_at_least_5_of_6": controls_retained >= 5,
        "movement_while_lost_zero": sage_metrics["movement_steps_while_lost"] == 0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "REVERSIBLE_EXPLORATION_DEVELOPMENT_STANDARD",
        "experiment_label": "CONTROLLED_REAL_RGB_OBSERVATION_IN_SIMULATED_GEOMETRY_LOOP",
        "identity_contract": "EXACT_SEMANTIC_AUTHORITY_FIXED_GEOMETRY_CANNOT_REBIND",
        "semantic_anchor_provenance": cohort["semantic_anchor_provenance"],
        "real_capture_scope": cohort["real_capture_scope"],
        "selection_scope": cohort["selection_scope"],
        "frozen_policy": {
            "near_threshold_m": NEAR_THRESHOLD_M,
            "alignment_tolerance_m": ALIGNMENT_TOLERANCE_M,
            "geometry_confidence_threshold": GEOMETRY_CONFIDENCE_THRESHOLD,
            "movement_step_m": 0.28,
            "forward_step_m": 0.54,
            "arrival_support_frames": 2,
        },
        "cohort": {"episode_count": len(rows), "kinds": cohort["kinds"], "control_count": 6},
        "metrics": {"bbox_center_scale": baseline_metrics, "sage_lm": sage_metrics, "controls_retained": controls_retained},
        "observation_diagnostics": observation_diagnostics,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "rows": rows,
        "claim_ceiling": "CONTROLLED_REAL_RGB_OBSERVATION_IN_SIMULATED_GEOMETRY_LOOP_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cohort)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "criteria": report["criteria"], "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
