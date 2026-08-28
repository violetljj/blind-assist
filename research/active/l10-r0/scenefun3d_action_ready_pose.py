from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scenefun3d_functional_handoff_ceiling import (
    FunctionalProposal,
    ParentBox,
    _load_json,
    _load_parent_boxes,
    _load_ply_xyz,
    _match_parent,
    _sha256,
    _transform_points,
)


APPROACH_STANDOFF_M = 0.65
DIRECTION_HIT_DEGREES = 30.0
MIN_EVALUABLE_PREDICTIONS = 8
MIN_TRANSLATIONAL_PREDICTIONS = 8
MIN_SIGNED_HIT_GAIN = 0.25
MIN_MEAN_ANGLE_GAIN_DEGREES = 20.0
MIN_OUTSIDE_PARENT_RATE = 0.90


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("Expected a finite non-zero vector")
    return vector / norm


def _round_vector(vector: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in vector]


def _sha256_payload(payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _contact_local(contact: np.ndarray, parent: ParentBox) -> np.ndarray:
    return (contact - parent.center) @ parent.axes.T


def _parent_face(
    contact: np.ndarray, parent: ParentBox
) -> tuple[int, np.ndarray, np.ndarray]:
    local = _contact_local(contact, parent)
    horizontal_axes = [
        index for index, axis in enumerate(parent.axes) if abs(float(axis[2])) <= 0.65
    ]
    if not horizontal_axes:
        horizontal_axes = list(range(3))
    half_lengths = np.maximum(parent.lengths / 2.0, 1e-6)
    face_index = max(
        horizontal_axes,
        key=lambda index: (
            abs(float(local[index])) / float(half_lengths[index]),
            -index,
        ),
    )
    sign = 1.0 if local[face_index] >= 0.0 else -1.0
    outward = _unit(sign * parent.axes[face_index])
    return face_index, outward, local


def _task_motion_type(task: str) -> str:
    text = task.casefold()
    if any(token in text for token in ("thermostat", "dial", "lock")):
        return "rot"
    if "door" in text or "window" in text:
        return "rot"
    if any(
        token in text
        for token in (
            "drawer",
            "socket",
            "outlet",
            "plug",
            "connect",
            "switch",
            "light",
            "lamp",
            "remote",
            "button",
        )
    ):
        return "trans"
    return "unknown"


def _task_translation_sign(task: str) -> int | None:
    text = task.casefold().strip()
    if text.startswith("open ") or text.startswith("unplug "):
        return 1
    if any(
        text.startswith(prefix)
        for prefix in ("close ", "plug ", "connect ", "turn on ")
    ):
        return -1
    return None


def _rotational_axis_and_origin(
    task: str,
    contact: np.ndarray,
    parent: ParentBox,
    face_index: int,
    outward: np.ndarray,
    local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    text = task.casefold()
    if any(token in text for token in ("thermostat", "dial", "lock")):
        return outward, contact, "CONTACT_NORMAL_ROTATION"

    in_plane = [index for index in range(3) if index != face_index]
    half_lengths = np.maximum(parent.lengths / 2.0, 1e-6)
    hinge_axis_index = min(
        in_plane,
        key=lambda index: (
            abs(float(local[index])) / float(half_lengths[index]),
            index,
        ),
    )
    lever_axis_index = next(index for index in in_plane if index != hinge_axis_index)
    hinge_local = local.copy()
    lever_sign = 1.0 if local[lever_axis_index] >= 0.0 else -1.0
    face_sign = 1.0 if local[face_index] >= 0.0 else -1.0
    hinge_local[lever_axis_index] = -lever_sign * half_lengths[lever_axis_index]
    hinge_local[face_index] = face_sign * half_lengths[face_index]
    hinge_origin = parent.center + hinge_local @ parent.axes
    return (
        _unit(parent.axes[hinge_axis_index]),
        hinge_origin,
        "OPPOSITE_EDGE_HINGE_TOPOLOGY",
    )


def _outside_parent(point: np.ndarray, parent: ParentBox) -> tuple[bool, float]:
    local = _contact_local(point, parent)
    excess = np.abs(local) - parent.lengths / 2.0
    clearance = float(np.max(excess))
    return clearance > 0.0, clearance


def _predict(
    task: str,
    proposal: FunctionalProposal,
) -> dict[str, Any] | None:
    motion_type = _task_motion_type(task)
    if motion_type == "unknown":
        return None
    contact = proposal.center
    parent = proposal.parent
    face_index, outward, local = _parent_face(contact, parent)
    approach_position = contact + outward * APPROACH_STANDOFF_M
    approach_outside, approach_clearance = _outside_parent(approach_position, parent)

    prediction: dict[str, Any] = {
        "motion_type": motion_type,
        "parent_binding_id": parent.binding_id,
        "parent_label": parent.label,
        "contact_point": _round_vector(contact),
        "parent_face_axis_index": face_index,
        "parent_outward_axis": _round_vector(outward),
        "approach_camera_position": _round_vector(approach_position),
        "approach_camera_facing_axis": _round_vector(-outward),
        "approach_outside_parent": approach_outside,
        "approach_parent_surface_clearance_m": round(approach_clearance, 6),
        "approach_state": "POSE_PROPOSED_REACHABILITY_NOT_EVALUABLE",
    }
    if motion_type == "trans":
        sign = _task_translation_sign(task)
        if sign is None:
            return None
        prediction.update(
            {
                "action_axis": _round_vector(sign * outward),
                "action_origin": _round_vector(contact),
                "action_model": (
                    "TASK_SIGNED_PARENT_NORMAL_OUTWARD"
                    if sign > 0
                    else "TASK_SIGNED_PARENT_NORMAL_INWARD"
                ),
            }
        )
    else:
        action_axis, action_origin, model = _rotational_axis_and_origin(
            task, contact, parent, face_index, outward, local
        )
        prediction.update(
            {
                "action_axis": _round_vector(action_axis),
                "action_origin": _round_vector(action_origin),
                "action_model": model,
            }
        )
    return prediction


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
    parents = _load_parent_boxes(paths["object_boxes"])

    proposals: dict[str, FunctionalProposal] = {}
    authorized_target_ids = {
        target_id
        for description in descriptions
        for target_id in description["annot_id"]
    }
    for annotation in annotations:
        if annotation["annot_id"] not in authorized_target_ids:
            continue
        points = _transform_points(
            xyz[np.asarray(annotation["indices"], dtype=np.int64)], transform
        )
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
        parent_ids = {proposal.parent.binding_id for proposal in target_proposals}
        if len(parent_ids) != 1:
            not_evaluable.append(
                {
                    "desc_id": description["desc_id"],
                    "reason": "NOT_EVALUABLE_MULTI_PARENT_TASK",
                    "parent_count": len(parent_ids),
                }
            )
            continue
        for proposal in target_proposals:
            prediction = _predict(description["description"], proposal)
            if prediction is None:
                not_evaluable.append(
                    {
                        "desc_id": description["desc_id"],
                        "annot_id": proposal.candidate_id,
                        "reason": "NOT_EVALUABLE_TASK_ACTION_SEMANTICS",
                    }
                )
                continue
            predictions.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "functional_seed_id": proposal.candidate_id,
                    **prediction,
                }
            )

    return {
        "schema_version": 1,
        "provider": "L10-SC12-TASK-TOPOLOGY-DUAL-AXIS-ACTION-POSE",
        "source": {
            "dataset": "SceneFun3D v1 train",
            "visit_id": visit_id,
            "video_id": video_id,
            "sha256": {name: _sha256(path) for name, path in paths.items()},
        },
        "truth_isolation": (
            "This is a functional-grounding-authorized action-geometry ceiling. "
            "Description-to-functional-part mappings provide upstream contact seeds, but "
            "affordance labels are ignored and motions.json is not loaded by the provider."
        ),
        "frozen_contract": {
            "approach_standoff_m": APPROACH_STANDOFF_M,
            "direction_hit_degrees": DIRECTION_HIT_DEGREES,
            "algorithm_inputs": [
                "public task description",
                "opaque exact-parent binding and OBB geometry",
                "authorized functional contact seed geometry",
            ],
            "forbidden_algorithm_inputs": [
                "functional affordance label",
                "motion type",
                "motion direction",
                "motion origin",
                "motion visualization orientation",
            ],
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


def _angle_degrees(
    predicted: np.ndarray, truth: np.ndarray, *, sign_invariant: bool = False
) -> float:
    dot = float(np.clip(np.dot(_unit(predicted), _unit(truth)), -1.0, 1.0))
    if sign_invariant:
        dot = abs(dot)
    return math.degrees(math.acos(dot))


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
    motions = {
        row["annot_id"]: row for row in _load_json(motion_path)["motions"]
    }
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
        parent_outward = np.asarray(prediction["parent_outward_axis"], dtype=np.float64)
        predicted_axis = np.asarray(prediction["action_axis"], dtype=np.float64)
        predicted_origin = np.asarray(prediction["action_origin"], dtype=np.float64)

        row: dict[str, Any] = {
            "desc_id": prediction["desc_id"],
            "description": prediction["description"],
            "functional_seed_id": prediction["functional_seed_id"],
            "predicted_motion_type": prediction["motion_type"],
            "evaluator_motion_type": truth_type,
            "motion_type_correct": prediction["motion_type"] == truth_type,
            "action_model": prediction["action_model"],
            "approach_outside_parent": prediction["approach_outside_parent"],
        }
        if truth_type == "trans":
            truth_signed = truth_axis.copy()
            if motion.get("motion_viz_orient") == "inwards":
                truth_signed = -truth_signed
            baseline_error = _angle_degrees(parent_outward, truth_signed)
            successor_error = _angle_degrees(predicted_axis, truth_signed)
            row.update(
                {
                    "evaluator_motion_orientation": motion.get("motion_viz_orient"),
                    "baseline_parent_outward_error_degrees": round(baseline_error, 6),
                    "task_signed_action_error_degrees": round(successor_error, 6),
                    "baseline_direction_hit": baseline_error <= DIRECTION_HIT_DEGREES,
                    "task_signed_direction_hit": successor_error <= DIRECTION_HIT_DEGREES,
                }
            )
        else:
            baseline_axis = np.asarray([0.0, 0.0, 1.0])
            baseline_error = _angle_degrees(baseline_axis, truth_axis, sign_invariant=True)
            successor_error = _angle_degrees(predicted_axis, truth_axis, sign_invariant=True)
            baseline_origin_error = float(
                np.linalg.norm(np.asarray(prediction["contact_point"]) - truth_origin)
            )
            successor_origin_error = float(np.linalg.norm(predicted_origin - truth_origin))
            row.update(
                {
                    "baseline_vertical_axis_error_degrees": round(baseline_error, 6),
                    "task_topology_axis_error_degrees": round(successor_error, 6),
                    "baseline_contact_origin_error_m": round(baseline_origin_error, 6),
                    "task_topology_origin_error_m": round(successor_origin_error, 6),
                    "baseline_axis_hit": baseline_error <= DIRECTION_HIT_DEGREES,
                    "task_topology_axis_hit": successor_error <= DIRECTION_HIT_DEGREES,
                }
            )
        rows.append(row)

    trans_rows = [row for row in rows if row["evaluator_motion_type"] == "trans"]
    rot_rows = [row for row in rows if row["evaluator_motion_type"] == "rot"]
    baseline_trans_hits = sum(row["baseline_direction_hit"] for row in trans_rows)
    successor_trans_hits = sum(row["task_signed_direction_hit"] for row in trans_rows)
    baseline_trans_mean = float(
        np.mean([row["baseline_parent_outward_error_degrees"] for row in trans_rows])
    ) if trans_rows else 0.0
    successor_trans_mean = float(
        np.mean([row["task_signed_action_error_degrees"] for row in trans_rows])
    ) if trans_rows else 0.0
    baseline_rot_hits = sum(row["baseline_axis_hit"] for row in rot_rows)
    successor_rot_hits = sum(row["task_topology_axis_hit"] for row in rot_rows)
    outside_count = sum(row["approach_outside_parent"] for row in rows)
    type_correct = sum(row["motion_type_correct"] for row in rows)

    metrics = {
        "evaluable_motion_predictions": len(rows),
        "motion_type_correct_count": type_correct,
        "motion_type_accuracy": type_correct / len(rows) if rows else 0.0,
        "translational_count": len(trans_rows),
        "baseline_signed_direction_hit_count": baseline_trans_hits,
        "baseline_signed_direction_hit_rate": baseline_trans_hits / len(trans_rows) if trans_rows else 0.0,
        "task_signed_direction_hit_count": successor_trans_hits,
        "task_signed_direction_hit_rate": successor_trans_hits / len(trans_rows) if trans_rows else 0.0,
        "baseline_mean_signed_angle_error_degrees": baseline_trans_mean,
        "task_mean_signed_angle_error_degrees": successor_trans_mean,
        "rotational_count": len(rot_rows),
        "baseline_rotational_axis_hit_count": baseline_rot_hits,
        "task_topology_rotational_axis_hit_count": successor_rot_hits,
        "approach_outside_parent_count": outside_count,
        "approach_outside_parent_rate": outside_count / len(rows) if rows else 0.0,
    }
    hit_gain = (
        metrics["task_signed_direction_hit_rate"]
        - metrics["baseline_signed_direction_hit_rate"]
    )
    angle_gain = baseline_trans_mean - successor_trans_mean
    if len(rows) < MIN_EVALUABLE_PREDICTIONS or len(trans_rows) < MIN_TRANSLATIONAL_PREDICTIONS:
        decision = "SC12_NOT_EVALUABLE_INSUFFICIENT_ACTION_MOTIONS"
    elif (
        hit_gain >= MIN_SIGNED_HIT_GAIN
        and angle_gain >= MIN_MEAN_ANGLE_GAIN_DEGREES
        and metrics["approach_outside_parent_rate"] >= MIN_OUTSIDE_PARENT_RATE
    ):
        decision = "SC12_TASK_CONDITIONED_DUAL_AXIS_ACTION_POSE_DEVELOPMENT_SIGNAL"
    else:
        decision = "SC12_TASK_CONDITIONED_DUAL_AXIS_ACTION_POSE_GATE_NOT_MET"

    return {
        "schema_version": 1,
        "experiment": "L10-SC12-TASK-CONDITIONED-DUAL-AXIS-ACTION-POSE",
        "decision": decision,
        "claim_layer": "FUNCTIONAL_GROUNDING_CONDITIONAL_ACTION_AXIS_DEVELOPMENT",
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
        },
        "metrics": metrics,
        "rows": rows,
        "claim_boundary": (
            "This is one source-disjoint SceneFun3D Development scene with evaluator-authorized "
            "functional contact seeds and privileged parent OBBs. It evaluates task-conditioned "
            "manipulation axes and structurally outside-parent camera approach poses. It does not "
            "establish RGB functional grounding, collision-free human reachability, body orientation, "
            "arrival, handoff readiness, user completion, product benefit, or safety. HANDOFF_READY "
            "remains forbidden until position, visibility, grounding, orientation, and reachability "
            "are jointly current and explicit user confirmation remains the only completion authority."
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
