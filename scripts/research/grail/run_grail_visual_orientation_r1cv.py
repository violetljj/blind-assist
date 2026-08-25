#!/usr/bin/env python3
"""Run the frozen deterministic RGB/proposal GRAIL-R1C-V probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from grail_grouping_r1a import predict_groups
from grail_relational_r0 import load_houses, relation_signatures, select_with_projected_relations
from grail_visual_orientation_r1cv import (
    arm_frames,
    oracle_directed_frames,
    ordinals_from_frames,
    predict_visual_frames,
    undirected_angle_degrees,
)
from run_grail_m1 import GrailModel, local_match_features, negative_reference_indices, pose_success, sha256_file


ARMS = ("AXIS_ONLY_DIAGNOSTIC", "SIGN_ONLY_DIAGNOSTIC", "R1C_V_FINAL")
FIELD_GROUPS = ("semantic_type", "sibling_ordinal", "nearby_type")
GATES = {
    "cross_view_slot_agreement_minimum": 70,
    "referent_top1_minimum": 70,
    "complete_pose_minimum": 50,
    "wrong_target_maximum": 1,
    "absence_false_commit_maximum": 1,
    "permutation_consistent": 156,
    "selector_collateral": 0,
    "complete_collateral": 0,
}


def _target_index(candidates: list[dict[str, Any]]) -> int:
    return next(index for index, candidate in enumerate(candidates) if candidate["is_target"])


def _sign_correct(predicted: dict[str, Any], oracle: dict[str, Any]) -> bool:
    return bool(predicted["sign_evaluable"] and oracle["evaluable"]
                and float(np.dot(predicted["directed_axis"], oracle["directed_axis"])) > 0.0)


def _view_analysis(image: np.ndarray, candidates: list[dict[str, Any]],
                   native_coordinates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups = predict_groups(candidates)
    predicted = predict_visual_frames(image, candidates, groups)
    oracle = oracle_directed_frames(candidates, groups, native_coordinates)
    arm_ordinals = {}
    for arm in ARMS:
        frames = arm_frames(image, predicted, oracle, arm)
        arm_ordinals[arm] = ordinals_from_frames(candidates, groups, frames)
    return {"groups": groups, "predicted": predicted, "oracle": oracle, "ordinals": arm_ordinals}


def _signatures_with_ordinals(house: dict[str, Any], candidates: list[dict[str, Any]],
                              ordinals: list[tuple[str, str]]) -> list[dict[str, Any]]:
    signatures = relation_signatures(house, candidates)
    for signature, ordinal in zip(signatures, ordinals):
        signature["part_horizontal"], signature["part_vertical"] = ordinal
    return signatures


@torch.inference_mode()
def run_probe(dataset: Path, collection_path: Path, collection_root: Path, features_path: Path,
              checkpoint_path: Path, development_result_path: Path, r0_result_path: Path,
              r1b_result_path: Path, r1c_o_result_path: Path, reference_supplement_path: Path,
              reference_root: Path, reference_features_path: Path, native_oracle_path: Path) -> dict[str, Any]:
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    features = torch.load(features_path, weights_only=False)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    development = json.loads(development_result_path.read_text(encoding="utf-8"))
    r0 = json.loads(r0_result_path.read_text(encoding="utf-8"))
    r1b = json.loads(r1b_result_path.read_text(encoding="utf-8"))
    r1c_o = json.loads(r1c_o_result_path.read_text(encoding="utf-8"))
    reference_supplement = json.loads(reference_supplement_path.read_text(encoding="utf-8"))
    reference_features = torch.load(reference_features_path, weights_only=False)
    native_oracle = json.loads(native_oracle_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != collection["dataset_sha256"]:
        raise ValueError("R1C-V dataset identity mismatch")
    if sha256_file(collection_path) != features["collection_sha256"]:
        raise ValueError("R1C-V collection/features identity mismatch")
    if sha256_file(checkpoint_path) != development["checkpoint_sha256"]:
        raise ValueError("R1C-V checkpoint/development identity mismatch")
    if r1b["terminal"] != "GRAIL_R1B_REFERENCE_OWNERSHIP_HIGH_BUT_VIEW_LOCAL_ORDINAL_NOT_ALIGNABLE":
        raise ValueError("R1C-V R1B terminal mismatch")
    if r1c_o["terminal"] != "GRAIL_R1C_O_CANONICAL_COORDINATE_CEILING_ESTABLISHED_R1C_V_PROTOCOL_ONLY":
        raise ValueError("R1C-V R1C-O terminal mismatch")
    if reference_features["supplement_sha256"] != sha256_file(reference_supplement_path):
        raise ValueError("R1C-V reference feature/supplement identity mismatch")
    if sha256_file(reference_features_path) != r1b["frozen_inputs"]["reference_features_sha256"]:
        raise ValueError("R1C-V reference feature identity drift")
    if native_oracle["role"] != "EVALUATOR_ONLY_FORBIDDEN_TO_PREDICTOR":
        raise ValueError("R1C-V native oracle role mismatch")
    if native_oracle["dataset_sha256"] != sha256_file(dataset):
        raise ValueError("R1C-V native oracle dataset mismatch")
    if native_oracle["collection_sha256"] != sha256_file(collection_path):
        raise ValueError("R1C-V native oracle collection mismatch")
    if native_oracle["reference_supplement_sha256"] != sha256_file(reference_supplement_path):
        raise ValueError("R1C-V native oracle reference mismatch")

    rows = features["rows"]
    reference_rows = reference_features["rows"]
    if len(rows) != 78 or [row["sample_id"] for row in rows] != [row["sample_id"] for row in reference_rows]:
        raise ValueError("R1C-V frozen 78-case alignment mismatch")
    houses = load_houses(dataset, {int(row["house_index"]) for row in rows})
    query_views, reference_views = [], []
    query_signatures = {arm: [] for arm in ARMS}
    reference_signatures = {arm: [] for arm in ARMS}
    target_view_diagnostics = []
    for row, reference_row in zip(rows, reference_rows):
        house_index = int(row["house_index"])
        native_coordinates = native_oracle["scenes"][str(house_index)]
        query_image = np.asarray(Image.open(collection_root / row["query_image"]).convert("RGB"))
        reference_image = np.asarray(Image.open(reference_root / reference_row["reference_full_image"]).convert("RGB"))
        query_view = _view_analysis(query_image, row["candidates"], native_coordinates)
        reference_view = _view_analysis(reference_image, reference_row["candidates"], native_coordinates)
        query_views.append(query_view)
        reference_views.append(reference_view)
        for arm in ARMS:
            query_signatures[arm].append(_signatures_with_ordinals(
                houses[house_index], row["candidates"], query_view["ordinals"][arm]
            ))
            reference_signatures[arm].append(_signatures_with_ordinals(
                houses[house_index], reference_row["candidates"], reference_view["ordinals"][arm]
            ))
        query_target = _target_index(row["candidates"])
        reference_target = _target_index(reference_row["candidates"])
        query_key = (query_view["groups"][query_target], row["candidates"][query_target]["object_type"])
        reference_key = (
            reference_view["groups"][reference_target], reference_row["candidates"][reference_target]["object_type"]
        )
        query_predicted, reference_predicted = query_view["predicted"][query_key], reference_view["predicted"][reference_key]
        query_oracle, reference_oracle = query_view["oracle"][query_key], reference_view["oracle"][reference_key]
        axis_pair_evaluable = bool(query_oracle["evaluable"] and reference_oracle["evaluable"])
        query_angle = undirected_angle_degrees(query_predicted["undirected_axis"], query_oracle["directed_axis"]) \
            if query_oracle["evaluable"] else None
        reference_angle = undirected_angle_degrees(reference_predicted["undirected_axis"], reference_oracle["directed_axis"]) \
            if reference_oracle["evaluable"] else None
        sign_pair_evaluable = bool(
            axis_pair_evaluable and query_predicted["sign_evaluable"] and reference_predicted["sign_evaluable"]
        )
        target_view_diagnostics.append({
            "sample_id": row["sample_id"],
            "query_key": query_key,
            "reference_key": reference_key,
            "query_target_index": query_target,
            "reference_target_index": reference_target,
            "axis_pair_evaluable": axis_pair_evaluable,
            "query_axis_angle_degrees": query_angle,
            "reference_axis_angle_degrees": reference_angle,
            "axis_pair_within_20_degrees": bool(axis_pair_evaluable and query_angle <= 20.0 and reference_angle <= 20.0),
            "sign_pair_evaluable": sign_pair_evaluable,
            "sign_pair_correct": bool(sign_pair_evaluable and _sign_correct(query_predicted, query_oracle)
                                      and _sign_correct(reference_predicted, reference_oracle)),
            "query_sign_evaluable": query_predicted["sign_evaluable"],
            "reference_sign_evaluable": reference_predicted["sign_evaluable"],
            "query_sign_correct": _sign_correct(query_predicted, query_oracle),
            "reference_sign_correct": _sign_correct(reference_predicted, reference_oracle),
            "slots": {arm: {
                "query": query_view["ordinals"][arm][query_target],
                "reference": reference_view["ordinals"][arm][reference_target],
            } for arm in ARMS},
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GrailModel(checkpoint["dim"]).to(device)
    model.load_state_dict(checkpoint["grail"])
    model.eval()
    negative_indices = negative_reference_indices(rows)
    threshold = float(development["thresholds"]["GRAIL"])
    cached: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        kinds: dict[str, Any] = {}
        for kind, reference_index in (("positive", row_index), ("negative", negative_indices[row_index])):
            scores, poses = [], []
            for candidate in row["candidates"]:
                match_np = candidate["local_match"] if kind == "positive" else local_match_features(
                    candidate["tokens"].astype(np.float32), rows[reference_index]["reference_tokens"].astype(np.float32)
                )
                logit, predicted_pose = model(
                    torch.tensor(row["query_embedding"], device=device),
                    torch.tensor(rows[reference_index]["reference_embedding"], device=device),
                    torch.tensor(candidate["embedding"], device=device),
                    torch.tensor(candidate["geometry"], device=device),
                    torch.tensor(match_np, device=device),
                )
                scores.append(float(torch.sigmoid(logit)))
                poses.append(predicted_pose.cpu().tolist())
            kinds[kind] = {"reference_index": reference_index, "scores": scores, "poses": poses}
        cached.append(kinds)

    r0_records = {record["sample_id"]: record for record in r0["records"]}
    r1b_records = r1b["arms"]["FULL_SCENE_RGB_BBOX"]["records"]
    view_disagreement_failures = {
        record["sample_id"] for record in r1b_records if not record["relation_target"]
        and record["query_target_ordinal_correct"] and record["reference_target_ordinal_correct"]
        and not record["query_reference_oracle_ordinal_agree"]
    }
    if len(view_disagreement_failures) != 23:
        raise ValueError("R1C-V R1B view-disagreement denominator drift")

    arms: dict[str, Any] = {}
    for arm in ARMS:
        records, absence_commits, permutation = [], [], []
        for row_index, row in enumerate(rows):
            candidate_ids = [candidate["object_id"] for candidate in row["candidates"]]
            spatial_keys = [tuple(candidate["bbox"]) for candidate in row["candidates"]]
            decisions: dict[str, Any] = {}
            for kind in ("positive", "negative"):
                values = cached[row_index][kind]
                reference_index = values["reference_index"]
                reference_target = _target_index(reference_rows[reference_index]["candidates"])
                target_signature = reference_signatures[arm][reference_index][reference_target]
                target_slot = (target_signature["part_horizontal"], target_signature["part_vertical"])
                if "NOT_EVALUABLE" in target_slot:
                    selected, confidence, resolution = None, 0.0, "TARGET_SLOT_NOT_EVALUABLE"
                    reverse_selected, reverse_confidence = None, 0.0
                else:
                    selected, confidence, resolution = select_with_projected_relations(
                        target_signature, query_signatures[arm][row_index], values["scores"], spatial_keys, FIELD_GROUPS
                    )
                    reverse_selected, reverse_confidence, _ = select_with_projected_relations(
                        target_signature, list(reversed(query_signatures[arm][row_index])),
                        list(reversed(values["scores"])), list(reversed(spatial_keys)), FIELD_GROUPS,
                    )
                selected_id = None if selected is None else candidate_ids[selected]
                reverse_id = None if reverse_selected is None else list(reversed(candidate_ids))[reverse_selected]
                permutation.append(selected_id == reverse_id and confidence == reverse_confidence)
                decisions[kind] = {"selected": selected, "confidence": confidence, "resolution": resolution}
            positive = decisions["positive"]
            selected = positive["selected"]
            target_index = _target_index(row["candidates"])
            relation_target = selected is not None and bool(row["candidates"][selected]["is_target"])
            target_pose = pose_success(cached[row_index]["positive"]["poses"][target_index], row["truth_local_poses"])
            relation_pose = selected is not None and pose_success(
                cached[row_index]["positive"]["poses"][selected], row["truth_local_poses"]
            )
            committed = positive["confidence"] >= threshold
            same_type = int(row["same_type_visible_candidates"]) >= 2
            baseline = r0_records[row["sample_id"]]
            slots = target_view_diagnostics[row_index]["slots"][arm]
            slot_agree = tuple(slots["query"]) == tuple(slots["reference"]) and "NOT_EVALUABLE" not in slots["query"]
            records.append({
                "sample_id": row["sample_id"],
                "query_target_slot": slots["query"],
                "reference_target_slot": slots["reference"],
                "cross_view_slot_agree": slot_agree,
                "relation_target": relation_target,
                "target_pose_capable": target_pose,
                "relation_complete": committed and relation_target and relation_pose,
                "relation_committed": committed,
                "wrong_target": bool(same_type and committed and not relation_target),
                "resolution": positive["resolution"],
                "baseline_target": baseline["baseline_target"],
                "baseline_complete": baseline["baseline_complete"],
                "was_r1b_view_disagreement_failure": row["sample_id"] in view_disagreement_failures,
            })
            absence_commits.append(decisions["negative"]["confidence"] >= threshold)
        referent = sum(record["relation_target"] for record in records)
        complete = sum(record["relation_complete"] for record in records)
        arms[arm] = {
            "cross_view_canonical_slot_agreement": sum(record["cross_view_slot_agree"] for record in records),
            "referent_top1": referent,
            "referent_and_target_pose": sum(r["relation_target"] and r["target_pose_capable"] for r in records),
            "complete_pose": complete,
            "wrong_target": sum(record["wrong_target"] for record in records),
            "absence_false_commit": sum(absence_commits),
            "r1c_o_uplift_recovery": {"referent": (referent - 44) / 31, "complete": (complete - 22) / 36},
            "r1b_view_disagreement_failure_rescued": sum(
                r["was_r1b_view_disagreement_failure"] and r["relation_target"] for r in records
            ),
            "r1b_view_disagreement_failure_denominator": 23,
            "selector_collateral": sum(r["baseline_target"] and not r["relation_target"] for r in records),
            "complete_collateral": sum(r["baseline_complete"] and not r["relation_complete"] for r in records),
            "permutation_consistent": sum(permutation),
            "permutation_denominator": len(permutation),
            "records": records,
        }

    axis_evaluable = [row for row in target_view_diagnostics if row["axis_pair_evaluable"]]
    sign_evaluable = [row for row in target_view_diagnostics if row["sign_pair_evaluable"]]
    angular_errors = [
        angle for row in target_view_diagnostics
        for angle in (row["query_axis_angle_degrees"], row["reference_axis_angle_degrees"])
        if angle is not None
    ]
    orientation_diagnostics = {
        "axis_evaluable": len(axis_evaluable),
        "axis_denominator": 78,
        "axis_pair_within_20_degrees": sum(row["axis_pair_within_20_degrees"] for row in axis_evaluable),
        "mean_undirected_axis_error_degrees": float(np.mean(angular_errors)) if angular_errors else None,
        "median_undirected_axis_error_degrees": float(np.median(angular_errors)) if angular_errors else None,
        "sign_evaluable": len(sign_evaluable),
        "sign_denominator": 78,
        "sign_pair_correct": sum(row["sign_pair_correct"] for row in sign_evaluable),
        "sign_accuracy_given_evaluable": (
            sum(row["sign_pair_correct"] for row in sign_evaluable) / len(sign_evaluable) if sign_evaluable else None
        ),
        "sign_unknown_rows": sum(not row["query_sign_evaluable"] or not row["reference_sign_evaluable"]
                                 for row in target_view_diagnostics),
        "sign_unknown_target_views": sum(not row[key] for row in target_view_diagnostics
                                         for key in ("query_sign_evaluable", "reference_sign_evaluable")),
    }
    final = arms["R1C_V_FINAL"]
    gate_results = {
        "cross_view_canonical_slot_agreement": final["cross_view_canonical_slot_agreement"] >= GATES["cross_view_slot_agreement_minimum"],
        "referent_top1": final["referent_top1"] >= GATES["referent_top1_minimum"],
        "complete_pose": final["complete_pose"] >= GATES["complete_pose_minimum"],
        "wrong_target": final["wrong_target"] <= GATES["wrong_target_maximum"],
        "absence_false_commit": final["absence_false_commit"] <= GATES["absence_false_commit_maximum"],
        "permutation_consistent": final["permutation_consistent"] == GATES["permutation_consistent"],
        "selector_collateral": final["selector_collateral"] == GATES["selector_collateral"],
        "complete_collateral": final["complete_collateral"] == GATES["complete_collateral"],
    }
    if all(gate_results.values()):
        terminal = "GRAIL_R1C_V_DETERMINISTIC_VISUAL_OWNER_ORIENTATION_ESTABLISHED"
    elif arms["AXIS_ONLY_DIAGNOSTIC"]["cross_view_canonical_slot_agreement"] < 70:
        terminal = "GRAIL_R1C_V_AXIS_NOT_VISUALLY_OBTAINABLE_BY_DETERMINISTIC_PROBE_STOP"
    elif arms["SIGN_ONLY_DIAGNOSTIC"]["cross_view_canonical_slot_agreement"] < 70 \
            or final["cross_view_canonical_slot_agreement"] < 70:
        terminal = "GRAIL_R1C_V_SIGN_NOT_VISUALLY_OBTAINABLE_BY_DETERMINISTIC_PROBE_STOP"
    else:
        terminal = "GRAIL_R1C_V_SLOT_STABLE_BUT_DOWNSTREAM_CEILING_NOT_RECOVERED_STOP"
    return {
        "schema": "blindassist_grail_r1c_v_visual_owner_orientation_probe_v1",
        "mode": "PROJECT_CONSUMED_DEVELOPMENT_DETERMINISTIC_RGB_PROPOSAL_ORIENTATION_PROBE",
        "claim_ceiling": "synthetic consumed Development deterministic RGB/proposal mechanism probe; no natural RGB, learning, formal test, Android, product, or safety authority",
        "predictor_firewall": {
            "allowed": ["full_scene_rgb", "independently_predicted_proposal_group", "proposal_bbox", "semantic_type"],
            "forbidden": ["native_yaw", "native_position", "camera_or_world_pose", "object_id", "sample_order",
                          "reference_query_joint_alignment", "native_coordinate", "evaluator_truth", "outcome"],
        },
        "frozen_inputs": {
            "positive_denominator": 78,
            "wrong_target_denominator": 43,
            "absence_denominator": 78,
            "grail_threshold": threshold,
            "field_groups": list(FIELD_GROUPS),
            "dataset_sha256": sha256_file(dataset),
            "collection_sha256": sha256_file(collection_path),
            "features_sha256": sha256_file(features_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "development_result_sha256": sha256_file(development_result_path),
            "r0_result_sha256": sha256_file(r0_result_path),
            "r1b_result_sha256": sha256_file(r1b_result_path),
            "r1c_o_result_sha256": sha256_file(r1c_o_result_path),
            "reference_supplement_sha256": sha256_file(reference_supplement_path),
            "reference_features_sha256": sha256_file(reference_features_path),
            "native_oracle_sha256": sha256_file(native_oracle_path),
        },
        "historical_slot_agreement": {"r1b_image_frame_privileged": 54, "r1c_o_owner_local_privileged": 78},
        "orientation_diagnostics": orientation_diagnostics,
        "arms": arms,
        "preregistered_gates": GATES,
        "gate_results": gate_results,
        "terminal": terminal,
        "target_view_diagnostics": target_view_diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--r0-result", type=Path, required=True)
    parser.add_argument("--r1b-result", type=Path, required=True)
    parser.add_argument("--r1c-o-result", type=Path, required=True)
    parser.add_argument("--reference-supplement", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--native-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(
        args.dataset, args.collection, args.collection_root, args.features, args.checkpoint,
        args.development_result, args.r0_result, args.r1b_result, args.r1c_o_result,
        args.reference_supplement, args.reference_root, args.reference_features, args.native_oracle,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "orientation_diagnostics": result["orientation_diagnostics"],
        "arms": {name: {key: value for key, value in arm.items() if key != "records"}
                 for name, arm in result["arms"].items()},
        "gate_results": result["gate_results"],
        "terminal": result["terminal"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
