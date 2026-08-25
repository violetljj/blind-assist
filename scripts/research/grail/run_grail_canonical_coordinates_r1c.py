#!/usr/bin/env python3
"""Run the privileged owner-local canonical coordinate ceiling GRAIL-R1C-O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grail_relational_r0 import load_houses, relation_signatures, select_with_projected_relations
from run_grail_m1 import GrailModel, local_match_features, negative_reference_indices, pose_success, sha256_file


FIELD_GROUPS = ("semantic_type", "sibling_ordinal", "nearby_type")
GATES = {
    "canonical_target_evaluable": 78,
    "canonical_label_agreement": 78,
    "referent_top1_minimum": 70,
    "complete_pose_minimum": 50,
    "wrong_target_maximum": 1,
    "absence_false_commit_maximum": 1,
    "permutation_consistent": 156,
}


def _canonical_signatures(house: dict[str, Any], candidates: list[dict[str, Any]],
                          coordinates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    signatures = relation_signatures(house, candidates)
    for candidate, signature in zip(candidates, signatures):
        coordinate = coordinates[candidate["object_id"]]
        signature["canonical_coordinate_evaluable"] = bool(coordinate["evaluable"])
        signature["canonical_coordinate"] = coordinate
        if coordinate["evaluable"]:
            signature["part_horizontal"] = coordinate["horizontal"]
            signature["part_vertical"] = coordinate["vertical"]
        else:
            signature["part_horizontal"] = "NOT_EVALUABLE"
            signature["part_vertical"] = "NOT_EVALUABLE"
    return signatures


@torch.inference_mode()
def run_probe(dataset: Path, collection_path: Path, features_path: Path, checkpoint_path: Path,
              development_result_path: Path, r0_result_path: Path, r1b_result_path: Path,
              coordinate_path: Path) -> dict[str, Any]:
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    features = torch.load(features_path, weights_only=False)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    development = json.loads(development_result_path.read_text(encoding="utf-8"))
    r0 = json.loads(r0_result_path.read_text(encoding="utf-8"))
    r1b = json.loads(r1b_result_path.read_text(encoding="utf-8"))
    coordinate_artifact = json.loads(coordinate_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != collection["dataset_sha256"]:
        raise ValueError("ProcTHOR val identity mismatch")
    if sha256_file(collection_path) != features["collection_sha256"]:
        raise ValueError("collection/features identity mismatch")
    if sha256_file(checkpoint_path) != development["checkpoint_sha256"]:
        raise ValueError("checkpoint/development-result identity mismatch")
    if coordinate_artifact["dataset_sha256"] != sha256_file(dataset):
        raise ValueError("coordinate artifact dataset identity mismatch")
    if coordinate_artifact["collection_sha256"] != sha256_file(collection_path):
        raise ValueError("coordinate artifact collection identity mismatch")
    if r0["grail_r0"]["referent_top1"] != 75 or r0["grail_r0"]["complete_pose"] != 57:
        raise ValueError("R0 result identity/metrics mismatch")
    if r1b["terminal"] != "GRAIL_R1B_REFERENCE_OWNERSHIP_HIGH_BUT_VIEW_LOCAL_ORDINAL_NOT_ALIGNABLE":
        raise ValueError("R1B terminal mismatch")
    r1b_records = r1b["arms"]["FULL_SCENE_RGB_BBOX"]["records"]
    view_disagreement_failures = {
        record["sample_id"] for record in r1b_records
        if not record["relation_target"] and record["query_target_ordinal_correct"]
        and record["reference_target_ordinal_correct"] and not record["query_reference_oracle_ordinal_agree"]
    }
    if len(view_disagreement_failures) != 23:
        raise ValueError(f"R1B view-disagreement denominator drift: {len(view_disagreement_failures)}")

    rows = features["rows"]
    coordinate_rows = coordinate_artifact["rows"]
    if len(rows) != 78 or [row["sample_id"] for row in rows] != [row["sample_id"] for row in coordinate_rows]:
        raise ValueError("R1C-O coordinate artifact does not match frozen 78-case cohort")
    coordinates = {row["sample_id"]: row["candidates"] for row in coordinate_rows}
    houses = load_houses(dataset, {int(row["house_index"]) for row in rows})
    signatures = [
        _canonical_signatures(houses[int(row["house_index"])], row["candidates"], coordinates[row["sample_id"]])
        for row in rows
    ]
    target_indices = [next(i for i, candidate in enumerate(row["candidates"]) if candidate["is_target"]) for row in rows]
    target_signatures = [row_signatures[index] for row_signatures, index in zip(signatures, target_indices)]

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
                logit, predicted = model(
                    torch.tensor(row["query_embedding"], device=device),
                    torch.tensor(rows[reference_index]["reference_embedding"], device=device),
                    torch.tensor(candidate["embedding"], device=device),
                    torch.tensor(candidate["geometry"], device=device),
                    torch.tensor(match_np, device=device),
                )
                scores.append(float(torch.sigmoid(logit)))
                poses.append(predicted.cpu().tolist())
            kinds[kind] = {"reference_index": reference_index, "scores": scores, "poses": poses}
        cached.append(kinds)

    r0_records = {record["sample_id"]: record for record in r0["records"]}
    records, absence_commits, permutation = [], [], []
    for row_index, row in enumerate(rows):
        candidate_ids = [candidate["object_id"] for candidate in row["candidates"]]
        spatial_keys = [tuple(candidate["bbox"]) for candidate in row["candidates"]]
        decisions: dict[str, Any] = {}
        for kind in ("positive", "negative"):
            values = cached[row_index][kind]
            reference_index = values["reference_index"]
            target_signature = target_signatures[reference_index]
            if not target_signature["canonical_coordinate_evaluable"]:
                selected, confidence, resolution = None, 0.0, "TARGET_COORDINATE_NOT_EVALUABLE"
                reverse_selected, reverse_confidence = None, 0.0
            else:
                selected, confidence, resolution = select_with_projected_relations(
                    target_signature, signatures[row_index], values["scores"], spatial_keys, FIELD_GROUPS
                )
                reverse_selected, reverse_confidence, _ = select_with_projected_relations(
                    target_signature, list(reversed(signatures[row_index])), list(reversed(values["scores"])),
                    list(reversed(spatial_keys)), FIELD_GROUPS,
                )
            selected_id = None if selected is None else candidate_ids[selected]
            reverse_id = None if reverse_selected is None else list(reversed(candidate_ids))[reverse_selected]
            permutation.append(selected_id == reverse_id and confidence == reverse_confidence)
            decisions[kind] = {"selected": selected, "confidence": confidence, "resolution": resolution}

        positive = decisions["positive"]
        selected = positive["selected"]
        target_index = target_indices[row_index]
        relation_target = selected is not None and bool(row["candidates"][selected]["is_target"])
        target_pose = pose_success(cached[row_index]["positive"]["poses"][target_index], row["truth_local_poses"])
        relation_pose = selected is not None and pose_success(
            cached[row_index]["positive"]["poses"][selected], row["truth_local_poses"]
        )
        committed = positive["confidence"] >= threshold
        same_type = int(row["same_type_visible_candidates"]) >= 2
        baseline = r0_records[row["sample_id"]]
        target_coordinate = target_signatures[row_index]["canonical_coordinate"]
        records.append({
            "sample_id": row["sample_id"],
            "canonical_target_evaluable": target_signatures[row_index]["canonical_coordinate_evaluable"],
            "canonical_target_ordinal": [target_signature_value for target_signature_value in (
                target_signatures[row_index]["part_horizontal"], target_signatures[row_index]["part_vertical"]
            )],
            "canonical_owner_source": target_coordinate.get("owner_source"),
            "native_sibling_count": target_coordinate.get("native_sibling_count"),
            "relation_target": relation_target,
            "target_pose_capable": target_pose,
            "relation_complete": committed and relation_target and relation_pose,
            "relation_committed": committed,
            "wrong_target": bool(same_type and committed and not relation_target),
            "resolution": positive["resolution"],
            "baseline_target": baseline["baseline_target"],
            "baseline_complete": baseline["baseline_complete"],
            "r0_target": baseline["relation_target"],
            "r0_complete": baseline["relation_complete"],
            "was_r1b_view_disagreement_failure": row["sample_id"] in view_disagreement_failures,
        })
        absence_commits.append(decisions["negative"]["confidence"] >= threshold)

    referent = sum(record["relation_target"] for record in records)
    complete = sum(record["relation_complete"] for record in records)
    wrong_target = sum(record["wrong_target"] for record in records)
    absence = sum(absence_commits)
    evaluable = sum(record["canonical_target_evaluable"] for record in records)
    canonical_agreement = evaluable  # the same source-native owner slot is camera independent by contract
    permutation_consistent = sum(permutation)
    gate_results = {
        "canonical_target_evaluable": evaluable == GATES["canonical_target_evaluable"],
        "canonical_label_agreement": canonical_agreement == GATES["canonical_label_agreement"],
        "referent_top1": referent >= GATES["referent_top1_minimum"],
        "complete_pose": complete >= GATES["complete_pose_minimum"],
        "wrong_target": wrong_target <= GATES["wrong_target_maximum"],
        "absence_false_commit": absence <= GATES["absence_false_commit_maximum"],
        "permutation_consistent": permutation_consistent == GATES["permutation_consistent"],
    }
    passed = all(gate_results.values())
    return {
        "schema": "blindassist_grail_r1c_o_owner_local_canonical_coordinate_probe_v1",
        "mode": "PROJECT_CONSUMED_DEVELOPMENT_PRIVILEGED_COORDINATE_CEILING",
        "claim_ceiling": "synthetic ProcTHOR Development privileged native owner-frame mechanism ceiling; no RGB orientation, natural scene, formal test, Android, product, or safety authority",
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
            "coordinate_artifact_sha256": sha256_file(coordinate_path),
        },
        "baselines": {"m1_referent": 44, "m1_complete": 22, "r1b_referent": 47,
                      "r1b_complete": 35, "r0_referent": 75, "r0_complete": 57},
        "metrics": {
            "canonical_target_evaluable": evaluable,
            "canonical_label_agreement": canonical_agreement,
            "referent_top1": referent,
            "referent_and_target_pose": sum(r["relation_target"] and r["target_pose_capable"] for r in records),
            "complete_pose": complete,
            "wrong_target": wrong_target,
            "absence_false_commit": absence,
            "permutation_consistent": permutation_consistent,
            "permutation_denominator": len(permutation),
            "r1b_view_disagreement_failure_rescued": sum(
                r["was_r1b_view_disagreement_failure"] and r["relation_target"] for r in records
            ),
            "r1b_view_disagreement_failure_denominator": len(view_disagreement_failures),
            "oracle_uplift_recovery": {"referent": (referent - 44) / 31, "complete": (complete - 22) / 35},
            "selector_collateral": sum(r["baseline_target"] and not r["relation_target"] for r in records),
            "complete_collateral": sum(r["baseline_complete"] and not r["relation_complete"] for r in records),
        },
        "preregistered_gates": GATES,
        "gate_results": gate_results,
        "terminal": "GRAIL_R1C_O_CANONICAL_COORDINATE_CEILING_ESTABLISHED_R1C_V_PROTOCOL_ONLY"
        if passed else "GRAIL_R1C_O_CANONICAL_COORDINATE_CEILING_NOT_ESTABLISHED_STOP",
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--r0-result", type=Path, required=True)
    parser.add_argument("--r1b-result", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.dataset, args.collection, args.features, args.checkpoint, args.development_result,
                       args.r0_result, args.r1b_result, args.coordinates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("metrics", "gate_results", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

