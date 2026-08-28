from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scenefun3d_action_ready_pose import _angle_degrees, _predict, _round_vector, _unit
from scenefun3d_functional_handoff_ceiling import (
    FunctionalProposal,
    _load_json,
    _load_parent_boxes,
    _load_ply_xyz,
    _match_parent,
    _sha256,
    _transform_points,
)


PROBE_TRANSLATION_M = 0.02
PROBE_ROTATION_DEGREES = 5.0
MEASUREMENT_NOISE_STD_M = 0.0015
MAX_PAIRED_POINTS = 128
MIN_PAIRED_POINTS = 20
ROTATION_MIN_DEGREES = 2.0
ROTATION_RESIDUAL_GAIN_M = 0.0015
DIRECTION_HIT_DEGREES = 15.0
PIVOT_LINE_HIT_M = 0.05
MIN_EVALUABLE_ESTIMATES = 5
MIN_TRANSLATIONAL_ESTIMATES = 4
MIN_DIRECTION_HIT_GAIN = 0.20


def _sha256_payload(payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _rodrigues(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    axis = _unit(axis)
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    identity = np.eye(3)
    return identity + math.sin(angle_radians) * skew + (1.0 - math.cos(angle_radians)) * (skew @ skew)


def _apply_rotation(
    points: np.ndarray, axis: np.ndarray, origin: np.ndarray, angle_radians: float
) -> np.ndarray:
    rotation = _rodrigues(axis, angle_radians)
    return (points - origin) @ rotation.T + origin


def _paired_sample(points: np.ndarray) -> np.ndarray | None:
    if len(points) < MIN_PAIRED_POINTS:
        return None
    count = min(len(points), MAX_PAIRED_POINTS)
    indices = np.linspace(0, len(points) - 1, num=count, dtype=np.int64)
    return points[indices]


def build_probe_observations(scene_dir: Path, video_id: str) -> dict[str, Any]:
    visit_id = scene_dir.name
    paths = {
        "annotations": scene_dir / f"{visit_id}_annotations.json",
        "descriptions": scene_dir / f"{visit_id}_descriptions.json",
        "motions": scene_dir / f"{visit_id}_motions.json",
        "laser_scan": scene_dir / f"{visit_id}_laser_scan.ply",
        "transform": scene_dir / video_id / f"{video_id}_transform.npy",
        "object_boxes": scene_dir / video_id / f"{video_id}_3dod_annotation.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SceneFun3D probe inputs: {missing}")

    annotations = {
        row["annot_id"]: row for row in _load_json(paths["annotations"])["annotations"]
    }
    descriptions = _load_json(paths["descriptions"])["descriptions"]
    motions = {row["annot_id"]: row for row in _load_json(paths["motions"])["motions"]}
    xyz = _load_ply_xyz(paths["laser_scan"])
    transform = np.load(paths["transform"])
    rotation_to_arkit = transform[:3, :3]
    parents = _load_parent_boxes(paths["object_boxes"])

    proposals: dict[str, FunctionalProposal] = {}
    authorized_ids = {
        target_id for description in descriptions for target_id in description["annot_id"]
    }
    for target_id in authorized_ids:
        annotation = annotations.get(target_id)
        if annotation is None:
            continue
        points = _transform_points(
            xyz[np.asarray(annotation["indices"], dtype=np.int64)], transform
        )
        matched = _match_parent(points, parents)
        if matched is None:
            continue
        parent, coverage = matched
        proposals[target_id] = FunctionalProposal(
            candidate_id=target_id,
            points=points,
            center=points.mean(axis=0),
            parent=parent,
            parent_coverage=coverage,
        )

    observations: list[dict[str, Any]] = []
    not_evaluable: list[dict[str, Any]] = []
    for description in descriptions:
        target_ids = tuple(description["annot_id"])
        target_proposals = [proposals[target_id] for target_id in target_ids if target_id in proposals]
        if len(target_proposals) != len(target_ids):
            not_evaluable.append(
                {
                    "desc_id": description["desc_id"],
                    "reason": "NOT_EVALUABLE_PARENT_BINDING",
                }
            )
            continue
        if len({proposal.parent.binding_id for proposal in target_proposals}) != 1:
            not_evaluable.append(
                {
                    "desc_id": description["desc_id"],
                    "reason": "NOT_EVALUABLE_MULTI_PARENT_TASK",
                }
            )
            continue
        for proposal in target_proposals:
            motion = motions.get(proposal.candidate_id)
            sample = _paired_sample(proposal.points)
            static_prediction = _predict(description["description"], proposal)
            if motion is None or sample is None or static_prediction is None:
                not_evaluable.append(
                    {
                        "desc_id": description["desc_id"],
                        "functional_seed_id": proposal.candidate_id,
                        "reason": "NOT_EVALUABLE_CAUSAL_PROBE_SOURCE",
                    }
                )
                continue
            truth_axis = _unit(
                rotation_to_arkit @ np.asarray(motion["motion_dir"], dtype=np.float64)
            )
            if motion["motion_type"] == "trans":
                signed_axis = truth_axis
                if motion.get("motion_viz_orient") == "inwards":
                    signed_axis = -signed_axis
                moved = sample + signed_axis * PROBE_TRANSLATION_M
            else:
                truth_origin = _transform_points(
                    xyz[np.asarray([motion["motion_origin_idx"]], dtype=np.int64)], transform
                )[0]
                moved = _apply_rotation(
                    sample,
                    truth_axis,
                    truth_origin,
                    math.radians(PROBE_ROTATION_DEGREES),
                )
            seed = int(hashlib.sha256(proposal.candidate_id.encode()).hexdigest()[:8], 16)
            generator = np.random.default_rng(seed)
            before = sample + generator.normal(0.0, MEASUREMENT_NOISE_STD_M, sample.shape)
            after = moved + generator.normal(0.0, MEASUREMENT_NOISE_STD_M, moved.shape)
            observations.append(
                {
                    "observation_id": f"{description['desc_id']}::{proposal.candidate_id}",
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "functional_seed_id": proposal.candidate_id,
                    "parent_binding_id": proposal.parent.binding_id,
                    "before_points": np.round(before, 6).tolist(),
                    "after_points": np.round(after, 6).tolist(),
                    "static_baseline": {
                        "motion_type": static_prediction["motion_type"],
                        "action_axis": static_prediction["action_axis"],
                        "action_origin": static_prediction["action_origin"],
                    },
                }
            )

    return {
        "schema_version": 1,
        "simulator": "SCENEFUN3D-EVALUATOR-MOTION-MICRO-PROBE-V1",
        "source": {
            "dataset": "SceneFun3D v1 train",
            "visit_id": visit_id,
            "video_id": video_id,
            "sha256": {name: _sha256(path) for name, path in paths.items()},
        },
        "simulation_contract": {
            "translation_m": PROBE_TRANSLATION_M,
            "rotation_degrees": PROBE_ROTATION_DEGREES,
            "measurement_noise_std_m": MEASUREMENT_NOISE_STD_M,
            "max_paired_points": MAX_PAIRED_POINTS,
            "minimum_paired_points": MIN_PAIRED_POINTS,
            "correspondence_authority": "EVALUATOR_PAIRED_FUNCTIONAL_POINTS",
            "safety_authority": "NONE_SIMULATION_ONLY",
        },
        "observations": observations,
        "not_evaluable": not_evaluable,
        "denominators": {
            "descriptions_total": len(descriptions),
            "probe_observations": len(observations),
            "not_evaluable": len(not_evaluable),
        },
    }


def _rigid_estimate(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    before_center = before.mean(axis=0)
    after_center = after.mean(axis=0)
    centered_before = before - before_center
    centered_after = after - after_center
    covariance = centered_before.T @ centered_after
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = after_center - rotation @ before_center
    rigid_aligned = before @ rotation.T + translation
    rigid_rms = float(np.sqrt(np.mean(np.sum((after - rigid_aligned) ** 2, axis=1))))

    translation_only = (after - before).mean(axis=0)
    translation_aligned = before + translation_only
    translation_rms = float(
        np.sqrt(np.mean(np.sum((after - translation_aligned) ** 2, axis=1)))
    )
    angle = math.degrees(
        math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    )
    residual_gain = translation_rms - rigid_rms
    is_rotation = angle >= ROTATION_MIN_DEGREES and residual_gain >= ROTATION_RESIDUAL_GAIN_M
    if not is_rotation:
        return {
            "motion_type": "trans",
            "action_axis": _round_vector(_unit(translation_only)),
            "translation_m": round(float(np.linalg.norm(translation_only)), 6),
            "translation_only_rms_m": round(translation_rms, 6),
            "rigid_rms_m": round(rigid_rms, 6),
            "rotation_evidence_degrees": round(angle, 6),
            "rotation_residual_gain_m": round(residual_gain, 6),
        }

    angle_radians = math.radians(angle)
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle_radians))
    axis = _unit(axis)
    # A rotation axis is a line, so (I - R)c = t is rank deficient along the
    # axis. Fix the representative point to the closest point on that line to
    # the coordinate origin instead of allowing measurement noise to explode
    # through the null space.
    pivot_system = np.vstack((np.eye(3) - rotation, axis.reshape(1, 3)))
    pivot_target = np.concatenate((translation, np.asarray([0.0])))
    pivot_point, *_ = np.linalg.lstsq(pivot_system, pivot_target, rcond=None)
    return {
        "motion_type": "rot",
        "action_axis": _round_vector(axis),
        "pivot_line_point": _round_vector(pivot_point),
        "rotation_degrees": round(angle, 6),
        "translation_only_rms_m": round(translation_rms, 6),
        "rigid_rms_m": round(rigid_rms, 6),
        "rotation_residual_gain_m": round(residual_gain, 6),
    }


def estimate_observations(observation_payload: dict[str, Any]) -> dict[str, Any]:
    estimates: list[dict[str, Any]] = []
    for observation in observation_payload["observations"]:
        before = np.asarray(observation["before_points"], dtype=np.float64)
        after = np.asarray(observation["after_points"], dtype=np.float64)
        estimate = _rigid_estimate(before, after)
        estimates.append(
            {
                "observation_id": observation["observation_id"],
                "desc_id": observation["desc_id"],
                "description": observation["description"],
                "functional_seed_id": observation["functional_seed_id"],
                "parent_binding_id": observation["parent_binding_id"],
                "static_baseline": observation["static_baseline"],
                **estimate,
                "belief_state": "ACTION_MODEL_LOCKED_FROM_CAUSAL_MOTION",
                "handoff_state": "ACTION_GEOMETRY_READY_REACHABILITY_NOT_EVALUABLE",
            }
        )
    return {
        "schema_version": 1,
        "provider": "L10-SC14-PAIRED-MICRO-MOTION-RIGID-ACTION-ESTIMATOR",
        "truth_isolation": (
            "The estimator receives only paired before/after functional points and public binding "
            "metadata. SceneFun3D motion type, direction, origin, and visualization orientation "
            "remain inside the observation simulator/evaluator."
        ),
        "frozen_contract": {
            "rotation_min_degrees": ROTATION_MIN_DEGREES,
            "rotation_residual_gain_m": ROTATION_RESIDUAL_GAIN_M,
            "direction_hit_degrees": DIRECTION_HIT_DEGREES,
            "pivot_line_hit_m": PIVOT_LINE_HIT_M,
            "completion_authority": "EXPLICIT_USER_CONFIRMATION_ONLY",
        },
        "estimates": estimates,
        "provider_payload_sha256": "filled_after_build",
    }


def _point_to_line_distance(point: np.ndarray, line_point: np.ndarray, axis: np.ndarray) -> float:
    axis = _unit(axis)
    return float(np.linalg.norm(np.cross(point - line_point, axis)))


def evaluate_estimates(
    scene_dir: Path,
    video_id: str,
    provider: dict[str, Any],
    provider_file_sha256: str,
) -> dict[str, Any]:
    visit_id = scene_dir.name
    motion_path = scene_dir / f"{visit_id}_motions.json"
    laser_path = scene_dir / f"{visit_id}_laser_scan.ply"
    transform_path = scene_dir / video_id / f"{video_id}_transform.npy"
    motions = {row["annot_id"]: row for row in _load_json(motion_path)["motions"]}
    xyz = _load_ply_xyz(laser_path)
    transform = np.load(transform_path)
    rotation_to_arkit = transform[:3, :3]

    rows: list[dict[str, Any]] = []
    for estimate in provider["estimates"]:
        motion = motions.get(estimate["functional_seed_id"])
        if motion is None:
            continue
        truth_type = motion["motion_type"]
        truth_axis = _unit(
            rotation_to_arkit @ np.asarray(motion["motion_dir"], dtype=np.float64)
        )
        if truth_type == "trans" and motion.get("motion_viz_orient") == "inwards":
            truth_axis = -truth_axis
        predicted_axis = np.asarray(estimate["action_axis"], dtype=np.float64)
        baseline = estimate["static_baseline"]
        baseline_axis = np.asarray(baseline["action_axis"], dtype=np.float64)
        row: dict[str, Any] = {
            "description": estimate["description"],
            "functional_seed_id": estimate["functional_seed_id"],
            "evaluator_motion_type": truth_type,
            "static_motion_type": baseline["motion_type"],
            "causal_motion_type": estimate["motion_type"],
            "static_type_correct": baseline["motion_type"] == truth_type,
            "causal_type_correct": estimate["motion_type"] == truth_type,
        }
        if truth_type == "trans":
            baseline_error = _angle_degrees(baseline_axis, truth_axis)
            causal_error = _angle_degrees(predicted_axis, truth_axis)
            row.update(
                {
                    "static_direction_error_degrees": round(baseline_error, 6),
                    "causal_direction_error_degrees": round(causal_error, 6),
                    "static_direction_hit": baseline_error <= DIRECTION_HIT_DEGREES,
                    "causal_direction_hit": causal_error <= DIRECTION_HIT_DEGREES,
                }
            )
        else:
            baseline_error = _angle_degrees(baseline_axis, truth_axis, sign_invariant=True)
            causal_error = _angle_degrees(predicted_axis, truth_axis, sign_invariant=True)
            truth_origin = _transform_points(
                xyz[np.asarray([motion["motion_origin_idx"]], dtype=np.int64)], transform
            )[0]
            pivot_error = (
                _point_to_line_distance(
                    truth_origin,
                    np.asarray(estimate["pivot_line_point"], dtype=np.float64),
                    predicted_axis,
                )
                if estimate["motion_type"] == "rot"
                else float("inf")
            )
            row.update(
                {
                    "static_axis_error_degrees": round(baseline_error, 6),
                    "causal_axis_error_degrees": round(causal_error, 6),
                    "causal_axis_hit": causal_error <= DIRECTION_HIT_DEGREES,
                    "causal_pivot_line_error_m": round(pivot_error, 6),
                    "causal_pivot_line_hit": pivot_error <= PIVOT_LINE_HIT_M,
                }
            )
        rows.append(row)

    trans_rows = [row for row in rows if row["evaluator_motion_type"] == "trans"]
    rot_rows = [row for row in rows if row["evaluator_motion_type"] == "rot"]
    static_trans_hits = sum(row["static_direction_hit"] for row in trans_rows)
    causal_trans_hits = sum(row["causal_direction_hit"] for row in trans_rows)
    metrics = {
        "evaluable_estimates": len(rows),
        "static_motion_type_correct_count": sum(row["static_type_correct"] for row in rows),
        "causal_motion_type_correct_count": sum(row["causal_type_correct"] for row in rows),
        "causal_motion_type_accuracy": sum(row["causal_type_correct"] for row in rows) / len(rows) if rows else 0.0,
        "translational_count": len(trans_rows),
        "static_direction_hit_count": static_trans_hits,
        "static_direction_hit_rate": static_trans_hits / len(trans_rows) if trans_rows else 0.0,
        "causal_direction_hit_count": causal_trans_hits,
        "causal_direction_hit_rate": causal_trans_hits / len(trans_rows) if trans_rows else 0.0,
        "static_mean_direction_error_degrees": float(np.mean([row["static_direction_error_degrees"] for row in trans_rows])) if trans_rows else 0.0,
        "causal_mean_direction_error_degrees": float(np.mean([row["causal_direction_error_degrees"] for row in trans_rows])) if trans_rows else 0.0,
        "rotational_count": len(rot_rows),
        "causal_rotational_axis_hit_count": sum(row["causal_axis_hit"] for row in rot_rows),
        "causal_pivot_line_hit_count": sum(row["causal_pivot_line_hit"] for row in rot_rows),
    }
    direction_gain = metrics["causal_direction_hit_rate"] - metrics["static_direction_hit_rate"]
    rotation_ok = not rot_rows or (
        metrics["causal_rotational_axis_hit_count"] == len(rot_rows)
        and metrics["causal_pivot_line_hit_count"] == len(rot_rows)
    )
    if len(rows) < MIN_EVALUABLE_ESTIMATES or len(trans_rows) < MIN_TRANSLATIONAL_ESTIMATES:
        decision = "SC14_NOT_EVALUABLE_INSUFFICIENT_CAUSAL_PROBES"
    elif (
        metrics["causal_motion_type_accuracy"] >= 0.90
        and metrics["causal_direction_hit_rate"] >= 0.90
        and direction_gain >= MIN_DIRECTION_HIT_GAIN
        and rotation_ok
    ):
        decision = "SC14_CAUSAL_MICRO_MOTION_ACTION_BELIEF_MECHANICS_SIGNAL"
    else:
        decision = "SC14_CAUSAL_MICRO_MOTION_ACTION_BELIEF_GATE_NOT_MET"
    return {
        "schema_version": 1,
        "experiment": "L10-SC14-CAUSAL-MICRO-MOTION-ACTION-BELIEF",
        "decision": decision,
        "claim_layer": "SIMULATED_PAIRED_MOTION_ACTION_GEOMETRY_MECHANICS",
        "provider_file_sha256": provider_file_sha256,
        "motion_truth_loaded_after_provider_seal": True,
        "frozen_gate": {
            "minimum_evaluable_estimates": MIN_EVALUABLE_ESTIMATES,
            "minimum_translational_estimates": MIN_TRANSLATIONAL_ESTIMATES,
            "direction_hit_degrees": DIRECTION_HIT_DEGREES,
            "minimum_causal_motion_type_accuracy": 0.90,
            "minimum_causal_direction_hit_rate": 0.90,
            "minimum_direction_hit_gain": MIN_DIRECTION_HIT_GAIN,
            "require_all_rotational_axis_and_pivot_hits": True,
        },
        "metrics": metrics,
        "rows": rows,
        "claim_boundary": (
            "This is a paired-point mechanics canary generated from real SceneFun3D functional "
            "geometry and evaluator motion annotations. Correspondences, the micro-motion, and its "
            "safety are privileged; no user action or natural before/after video was executed. A "
            "positive result establishes only causal kinematic identifiability under the named "
            "perturbation. It does not establish RGB tracking, safe probing, reachability, body "
            "orientation, arrival, HANDOFF_READY, user completion, product benefit, or safety."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--observations-output", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene_dir = args.scene_dir.resolve()

    observations = build_probe_observations(scene_dir, args.video_id)
    args.observations_output.parent.mkdir(parents=True, exist_ok=True)
    args.observations_output.write_text(
        json.dumps(observations, indent=2) + "\n", encoding="utf-8"
    )
    provider = estimate_observations(observations)
    provider["provider_payload_sha256"] = _sha256_payload(
        {key: value for key, value in provider.items() if key != "provider_payload_sha256"}
    )
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider_file_sha256 = _sha256(args.provider_output)

    result = evaluate_estimates(scene_dir, args.video_id, provider, provider_file_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], **result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
