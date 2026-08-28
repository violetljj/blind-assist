from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scenefun3d_action_ready_pose import (
    APPROACH_STANDOFF_M,
    DIRECTION_HIT_DEGREES,
    _angle_degrees,
    _outside_parent,
    _parent_face,
    _rotational_axis_and_origin,
    _round_vector,
    _task_motion_type,
    _task_translation_sign,
    _unit,
)
from scenefun3d_functional_handoff_ceiling import (
    FunctionalProposal,
    _load_json,
    _load_parent_boxes,
    _load_ply_xyz,
    _match_parent,
    _sha256,
    _transform_points,
)


LOCAL_SURFACE_RADIUS_M = 0.12
LOCAL_SURFACE_MIN_POINTS = 12
MIN_EVALUABLE_PREDICTIONS = 6
MIN_TRANSLATIONAL_PREDICTIONS = 4
MIN_SIGNED_HIT_GAIN = 0.20
MIN_MEAN_ANGLE_GAIN_DEGREES = 15.0
MIN_OUTSIDE_PARENT_RATE = 0.90


def _sha256_payload(payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _motion_type_hypotheses(task: str) -> tuple[str, ...]:
    text = task.casefold()
    if "door" in text or "window" in text:
        return ("rot", "trans")
    motion_type = _task_motion_type(task)
    return () if motion_type == "unknown" else (motion_type,)


def _local_surface_normal(
    scene_points: np.ndarray,
    contact: np.ndarray,
    parent_outward: np.ndarray,
    parent_center: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    delta = scene_points - contact
    cube_mask = np.all(np.abs(delta) <= LOCAL_SURFACE_RADIUS_M, axis=1)
    neighborhood = scene_points[cube_mask]
    if len(neighborhood) < LOCAL_SURFACE_MIN_POINTS:
        return None
    radial = neighborhood - contact
    neighborhood = neighborhood[
        np.einsum("ij,ij->i", radial, radial) <= LOCAL_SURFACE_RADIUS_M**2
    ]
    if len(neighborhood) < LOCAL_SURFACE_MIN_POINTS:
        return None
    centered = neighborhood - neighborhood.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(centered))
    normal = _unit(eigenvectors[:, 0])
    contact_radial = contact - parent_center
    orientation_reference = contact_radial
    if float(np.linalg.norm(contact_radial)) <= 1e-6:
        orientation_reference = parent_outward
    if float(np.dot(normal, orientation_reference)) < 0.0:
        normal = -normal
    planarity = (
        float((eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[2], 1e-12))
        if len(eigenvalues) == 3
        else 0.0
    )
    return normal, {
        "radius_m": LOCAL_SURFACE_RADIUS_M,
        "point_count": len(neighborhood),
        "eigenvalues": [round(float(value), 9) for value in eigenvalues],
        "planarity": round(planarity, 6),
    }


def _prediction(
    task: str,
    proposal: FunctionalProposal,
    scene_points: np.ndarray,
) -> dict[str, Any] | None:
    parent = proposal.parent
    contact = proposal.center
    face_index, approach_outward, local = _parent_face(contact, parent)
    normal_result = _local_surface_normal(
        scene_points, contact, approach_outward, parent.center
    )
    if normal_result is None:
        return None
    contact_outward, surface_diagnostics = normal_result
    types = _motion_type_hypotheses(task)
    if not types:
        return None
    approach_position = contact + approach_outward * APPROACH_STANDOFF_M
    approach_outside, approach_clearance = _outside_parent(approach_position, parent)

    hypotheses: list[dict[str, Any]] = []
    for motion_type in types:
        if motion_type == "trans":
            sign = _task_translation_sign(task)
            if sign is None:
                continue
            hypotheses.append(
                {
                    "motion_type": "trans",
                    "action_axis": _round_vector(sign * contact_outward),
                    "action_origin": _round_vector(contact),
                    "model": "TASK_SIGNED_LOCAL_CONTACT_SURFACE_NORMAL",
                }
            )
        else:
            axis, origin, model = _rotational_axis_and_origin(
                task, contact, parent, face_index, approach_outward, local
            )
            hypotheses.append(
                {
                    "motion_type": "rot",
                    "action_axis": _round_vector(axis),
                    "action_origin": _round_vector(origin),
                    "model": model,
                }
            )
    if not hypotheses:
        return None

    baseline_type = _task_motion_type(task)
    baseline_axis = approach_outward
    baseline_sign = _task_translation_sign(task)
    if baseline_type == "trans" and baseline_sign is not None:
        baseline_axis = baseline_sign * baseline_axis

    return {
        "parent_binding_id": parent.binding_id,
        "parent_label": parent.label,
        "contact_point": _round_vector(contact),
        "approach_axis_source": "PARENT_HORIZONTAL_FACE",
        "approach_camera_position": _round_vector(approach_position),
        "approach_camera_facing_axis": _round_vector(-approach_outward),
        "approach_outside_parent": approach_outside,
        "approach_parent_surface_clearance_m": round(approach_clearance, 6),
        "action_axis_source": "LOCAL_3D_CONTACT_SURFACE",
        "local_contact_outward_axis": _round_vector(contact_outward),
        "local_surface_diagnostics": surface_diagnostics,
        "baseline_motion_type": baseline_type,
        "baseline_action_axis": _round_vector(baseline_axis),
        "action_model_state": "UNIQUE" if len(hypotheses) == 1 else "SET_VALUED",
        "hypotheses": hypotheses,
        "handoff_state": "POSE_HYPOTHESES_READY_REACHABILITY_NOT_EVALUABLE",
    }


def build_provider(scene_dir: Path, video_id: str) -> dict[str, Any]:
    visit_id = scene_dir.name
    paths = {
        "annotations": scene_dir / f"{visit_id}_annotations.json",
        "descriptions": scene_dir / f"{visit_id}_descriptions.json",
        "laser_scan": scene_dir / f"{visit_id}_laser_scan.ply",
        "transform": scene_dir / video_id / f"{video_id}_transform.npy",
        "object_boxes": scene_dir / video_id / f"{video_id}_3dod_annotation.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SceneFun3D provider inputs: {missing}")

    annotations = _load_json(paths["annotations"])["annotations"]
    descriptions = _load_json(paths["descriptions"])["descriptions"]
    xyz = _load_ply_xyz(paths["laser_scan"])
    transform = np.load(paths["transform"])
    scene_points = _transform_points(xyz, transform)
    parents = _load_parent_boxes(paths["object_boxes"])
    authorized_target_ids = {
        target_id
        for description in descriptions
        for target_id in description["annot_id"]
    }

    proposals: dict[str, FunctionalProposal] = {}
    for annotation in annotations:
        if annotation["annot_id"] not in authorized_target_ids:
            continue
        points = scene_points[np.asarray(annotation["indices"], dtype=np.int64)]
        matched = _match_parent(points, parents)
        if matched is None:
            continue
        parent, coverage = matched
        proposals[annotation["annot_id"]] = FunctionalProposal(
            candidate_id=annotation["annot_id"],
            points=points,
            center=points.mean(axis=0),
            parent=parent,
            parent_coverage=coverage,
        )

    predictions: list[dict[str, Any]] = []
    not_evaluable: list[dict[str, Any]] = []
    for description in descriptions:
        target_ids = tuple(description["annot_id"])
        target_proposals = [proposals[target_id] for target_id in target_ids if target_id in proposals]
        if len(target_proposals) != len(target_ids):
            not_evaluable.append(
                {
                    "desc_id": description["desc_id"],
                    "reason": "NOT_EVALUABLE_PARENT_BINDING",
                    "target_count": len(target_ids),
                    "parent_bound_target_count": len(target_proposals),
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
            predicted = _prediction(description["description"], proposal, scene_points)
            if predicted is None:
                not_evaluable.append(
                    {
                        "desc_id": description["desc_id"],
                        "functional_seed_id": proposal.candidate_id,
                        "reason": "NOT_EVALUABLE_LOCAL_ACTION_FRAME",
                    }
                )
                continue
            predictions.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "functional_seed_id": proposal.candidate_id,
                    **predicted,
                }
            )

    return {
        "schema_version": 1,
        "provider": "L10-SC13-DECOUPLED-APPROACH-AND-LOCAL-ACTION-FRAMES",
        "source": {
            "dataset": "SceneFun3D v1 train",
            "visit_id": visit_id,
            "video_id": video_id,
            "sha256": {name: _sha256(path) for name, path in paths.items()},
        },
        "truth_isolation": (
            "This is a functional-grounding-authorized action-frame ceiling. "
            "The provider uses public task text, parent OBBs, scene geometry, and authorized "
            "functional contact seeds. It ignores affordance labels and does not load motions.json."
        ),
        "frozen_contract": {
            "local_surface_radius_m": LOCAL_SURFACE_RADIUS_M,
            "local_surface_min_points": LOCAL_SURFACE_MIN_POINTS,
            "approach_standoff_m": APPROACH_STANDOFF_M,
            "direction_hit_degrees": DIRECTION_HIT_DEGREES,
            "algorithm_inputs": [
                "public task description",
                "opaque exact-parent binding and OBB geometry",
                "authorized functional contact seed geometry",
                "local unlabeled 3D scene points",
            ],
            "forbidden_algorithm_inputs": [
                "functional affordance label",
                "motion type, direction, origin, or visualization orientation",
            ],
            "uncertainty_rule": (
                "door and window tasks remain a set of translational and rotational hypotheses"
            ),
            "completion_authority": "EXPLICIT_USER_CONFIRMATION_ONLY",
        },
        "predictions": predictions,
        "not_evaluable": not_evaluable,
        "denominators": {
            "descriptions_total": len(descriptions),
            "provider_predictions": len(predictions),
            "provider_not_evaluable": len(not_evaluable),
        },
        "provider_payload_sha256": "filled_after_build",
    }


def evaluate_provider(
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
    rotation = transform[:3, :3]

    rows: list[dict[str, Any]] = []
    for prediction in provider["predictions"]:
        motion = motions.get(prediction["functional_seed_id"])
        if motion is None:
            continue
        truth_type = motion["motion_type"]
        truth_axis = rotation @ np.asarray(motion["motion_dir"], dtype=np.float64)
        truth_origin = _transform_points(
            xyz[np.asarray([motion["motion_origin_idx"]], dtype=np.int64)], transform
        )[0]
        hypotheses = {row["motion_type"]: row for row in prediction["hypotheses"]}
        correct_hypothesis = hypotheses.get(truth_type)
        type_set = set(hypotheses)
        row: dict[str, Any] = {
            "desc_id": prediction["desc_id"],
            "description": prediction["description"],
            "functional_seed_id": prediction["functional_seed_id"],
            "evaluator_motion_type": truth_type,
            "baseline_motion_type": prediction["baseline_motion_type"],
            "baseline_type_correct": prediction["baseline_motion_type"] == truth_type,
            "action_model_state": prediction["action_model_state"],
            "hypothesis_types": sorted(type_set),
            "type_set_covers_truth": truth_type in type_set,
            "wrong_singleton": len(type_set) == 1 and truth_type not in type_set,
            "approach_outside_parent": prediction["approach_outside_parent"],
            "local_surface_planarity": prediction["local_surface_diagnostics"]["planarity"],
        }
        if correct_hypothesis is None:
            rows.append(row)
            continue
        predicted_axis = np.asarray(correct_hypothesis["action_axis"], dtype=np.float64)
        baseline_axis = np.asarray(prediction["baseline_action_axis"], dtype=np.float64)
        if truth_type == "trans":
            truth_signed = truth_axis if motion.get("motion_viz_orient") == "outwards" else -truth_axis
            baseline_error = _angle_degrees(baseline_axis, truth_signed)
            successor_error = _angle_degrees(predicted_axis, truth_signed)
            row.update(
                {
                    "evaluator_motion_orientation": motion.get("motion_viz_orient"),
                    "baseline_parent_action_error_degrees": round(baseline_error, 6),
                    "local_contact_action_error_degrees": round(successor_error, 6),
                    "baseline_direction_hit": baseline_error <= DIRECTION_HIT_DEGREES,
                    "local_contact_direction_hit": successor_error <= DIRECTION_HIT_DEGREES,
                }
            )
        else:
            predicted_origin = np.asarray(correct_hypothesis["action_origin"], dtype=np.float64)
            baseline_error = _angle_degrees(
                np.asarray([0.0, 0.0, 1.0]), truth_axis, sign_invariant=True
            )
            successor_error = _angle_degrees(predicted_axis, truth_axis, sign_invariant=True)
            row.update(
                {
                    "baseline_vertical_axis_error_degrees": round(baseline_error, 6),
                    "topology_axis_error_degrees": round(successor_error, 6),
                    "baseline_contact_origin_error_m": round(
                        float(
                            np.linalg.norm(
                                np.asarray(prediction["contact_point"]) - truth_origin
                            )
                        ),
                        6,
                    ),
                    "topology_origin_error_m": round(
                        float(np.linalg.norm(predicted_origin - truth_origin)), 6
                    ),
                    "baseline_axis_hit": baseline_error <= DIRECTION_HIT_DEGREES,
                    "topology_axis_hit": successor_error <= DIRECTION_HIT_DEGREES,
                }
            )
        rows.append(row)

    trans_rows = [
        row
        for row in rows
        if row["evaluator_motion_type"] == "trans"
        and "local_contact_action_error_degrees" in row
    ]
    rot_rows = [
        row
        for row in rows
        if row["evaluator_motion_type"] == "rot" and "topology_axis_error_degrees" in row
    ]
    baseline_trans_hits = sum(row["baseline_direction_hit"] for row in trans_rows)
    successor_trans_hits = sum(row["local_contact_direction_hit"] for row in trans_rows)
    baseline_trans_mean = float(
        np.mean([row["baseline_parent_action_error_degrees"] for row in trans_rows])
    ) if trans_rows else 0.0
    successor_trans_mean = float(
        np.mean([row["local_contact_action_error_degrees"] for row in trans_rows])
    ) if trans_rows else 0.0
    type_coverage = sum(row["type_set_covers_truth"] for row in rows)
    wrong_singletons = sum(row["wrong_singleton"] for row in rows)
    baseline_type_correct = sum(row["baseline_type_correct"] for row in rows)
    outside_count = sum(row["approach_outside_parent"] for row in rows)
    metrics = {
        "evaluable_motion_predictions": len(rows),
        "baseline_motion_type_correct_count": baseline_type_correct,
        "baseline_motion_type_accuracy": baseline_type_correct / len(rows) if rows else 0.0,
        "motion_type_hypothesis_coverage_count": type_coverage,
        "motion_type_hypothesis_coverage": type_coverage / len(rows) if rows else 0.0,
        "set_valued_prediction_count": sum(row["action_model_state"] == "SET_VALUED" for row in rows),
        "wrong_singleton_count": wrong_singletons,
        "translational_count": len(trans_rows),
        "baseline_signed_direction_hit_count": baseline_trans_hits,
        "baseline_signed_direction_hit_rate": baseline_trans_hits / len(trans_rows) if trans_rows else 0.0,
        "local_contact_signed_direction_hit_count": successor_trans_hits,
        "local_contact_signed_direction_hit_rate": successor_trans_hits / len(trans_rows) if trans_rows else 0.0,
        "baseline_mean_signed_angle_error_degrees": baseline_trans_mean,
        "local_contact_mean_signed_angle_error_degrees": successor_trans_mean,
        "rotational_count": len(rot_rows),
        "baseline_rotational_axis_hit_count": sum(row["baseline_axis_hit"] for row in rot_rows),
        "topology_rotational_axis_hit_count": sum(row["topology_axis_hit"] for row in rot_rows),
        "approach_outside_parent_count": outside_count,
        "approach_outside_parent_rate": outside_count / len(rows) if rows else 0.0,
    }
    hit_gain = (
        metrics["local_contact_signed_direction_hit_rate"]
        - metrics["baseline_signed_direction_hit_rate"]
    )
    angle_gain = baseline_trans_mean - successor_trans_mean
    if len(rows) < MIN_EVALUABLE_PREDICTIONS or len(trans_rows) < MIN_TRANSLATIONAL_PREDICTIONS:
        decision = "SC13_NOT_EVALUABLE_INSUFFICIENT_ACTION_MOTIONS"
    elif (
        metrics["motion_type_hypothesis_coverage"] == 1.0
        and wrong_singletons == 0
        and hit_gain >= MIN_SIGNED_HIT_GAIN
        and angle_gain >= MIN_MEAN_ANGLE_GAIN_DEGREES
        and metrics["approach_outside_parent_rate"] >= MIN_OUTSIDE_PARENT_RATE
    ):
        decision = "SC13_DECOUPLED_APPROACH_AND_LOCAL_ACTION_FRAME_DEVELOPMENT_SIGNAL"
    else:
        decision = "SC13_DECOUPLED_APPROACH_AND_LOCAL_ACTION_FRAME_GATE_NOT_MET"

    return {
        "schema_version": 1,
        "experiment": "L10-SC13-DECOUPLED-APPROACH-AND-LOCAL-ACTION-FRAME",
        "decision": decision,
        "claim_layer": "FUNCTIONAL_GROUNDING_CONDITIONAL_ACTION_FRAME_DEVELOPMENT",
        "provider_file_sha256": provider_file_sha256,
        "motion_truth_loaded_after_provider_seal": True,
        "evaluator_source": {
            "motions": str(motion_path),
            "motions_sha256": _sha256(motion_path),
        },
        "frozen_gate": {
            "minimum_evaluable_predictions": MIN_EVALUABLE_PREDICTIONS,
            "minimum_translational_predictions": MIN_TRANSLATIONAL_PREDICTIONS,
            "direction_hit_degrees": DIRECTION_HIT_DEGREES,
            "minimum_signed_hit_gain": MIN_SIGNED_HIT_GAIN,
            "minimum_mean_angle_gain_degrees": MIN_MEAN_ANGLE_GAIN_DEGREES,
            "minimum_approach_outside_parent_rate": MIN_OUTSIDE_PARENT_RATE,
            "require_type_hypothesis_coverage": 1.0,
            "require_wrong_singletons": 0,
        },
        "metrics": metrics,
        "rows": rows,
        "claim_boundary": (
            "This is one fresh SceneFun3D Development scene with evaluator-authorized functional "
            "contact seeds, privileged parent OBBs, and Faro scene geometry. It evaluates whether "
            "approach and manipulation axes must be separate and whether uncertain door/window "
            "kinematics should remain set-valued. It does not establish RGB grounding, collision-free "
            "human reachability, body orientation, arrival, HANDOFF_READY, user completion, product "
            "benefit, or safety. Explicit user confirmation remains the only completion authority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene_dir = args.scene_dir.resolve()
    provider = build_provider(scene_dir, args.video_id)
    provider["provider_payload_sha256"] = _sha256_payload(
        {key: value for key, value in provider.items() if key != "provider_payload_sha256"}
    )
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider_file_sha256 = _sha256(args.provider_output)

    result = evaluate_provider(scene_dir, args.video_id, provider, provider_file_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], **result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
