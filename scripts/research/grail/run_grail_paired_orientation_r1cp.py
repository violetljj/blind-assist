#!/usr/bin/env python3
"""Evaluate frozen OA-V2 paired orientation on the fresh GRAIL-R1C-P cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grail_paired_orientation_r1cp import (
    consensus_index,
    ordinals_from_basis,
    orientation_matrix,
    paired_mode_bases,
    projected_basis,
    reference_mode_matrices,
)
from grail_relational_r0 import load_houses, relation_signatures, select_with_projected_relations
from grail_visual_orientation_r1cv import group_members
from run_grail_m1 import GrailModel, local_match_features, negative_reference_indices, pose_success, sha256_file


ARMS = ("OA_V2_INDEPENDENT_ABSOLUTE_DIAGNOSTIC", "OA_V2_PAIRED_RELATIVE_FINAL")
FIELDS = ("semantic_type", "sibling_ordinal", "nearby_type")
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


def _target(candidates: list[dict[str, Any]]) -> int:
    return next(index for index, candidate in enumerate(candidates) if candidate["is_target"])


def _with_ordinals(house: dict[str, Any], candidates: list[dict[str, Any]],
                   ordinals: list[tuple[str, str]]) -> list[dict[str, Any]]:
    signatures = relation_signatures(house, candidates)
    for signature, ordinal in zip(signatures, ordinals):
        signature["part_horizontal"], signature["part_vertical"] = ordinal
    return signatures


def _reference_ordinals(candidates: list[dict[str, Any]], groups: list[int], target_index: int,
                        basis: dict[str, Any]) -> list[tuple[str, str]]:
    members = group_members(candidates, groups)
    target_key = (groups[target_index], candidates[target_index]["object_type"])
    bases = {key: {"evaluable": False, "right": None, "down": None} for key in members}
    bases[target_key] = basis
    return ordinals_from_basis(candidates, groups, bases)


def _predicted_modes(row: dict[str, Any], reference_row: dict[str, Any], record: dict[str, Any],
                     kind: str, arm: str) -> list[dict[str, Any]]:
    reference = record["references"][kind]
    ref_prediction = reference["reference_absolute"]
    alpha = 1 if arm == "OA_V2_INDEPENDENT_ABSOLUTE_DIAGNOSTIC" else int(ref_prediction["alpha"])
    ref_matrices = reference_mode_matrices(
        ref_prediction["azimuth"], ref_prediction["elevation"], ref_prediction["roll"], alpha
    )
    if not ref_matrices:
        return []
    query_groups = record["query_groups"]
    modes = []
    for mode_index, ref_matrix in enumerate(ref_matrices):
        query_bases = {}
        for group in record["group_predictions"]:
            key = (int(group["group"]), group["object_type"])
            if arm == "OA_V2_INDEPENDENT_ABSOLUTE_DIAGNOSTIC":
                prediction = group["independent_absolute"]
                query_bases[key] = projected_basis(orientation_matrix(
                    prediction["azimuth"], prediction["elevation"], prediction["roll"]
                ))
            else:
                relative = group["paired_relative"][kind]
                query_bases[key] = paired_mode_bases(
                    ref_prediction["azimuth"], ref_prediction["elevation"], ref_prediction["roll"], alpha,
                    relative["azimuth"], relative["elevation"], relative["roll"],
                )[mode_index]["query"]
        query_ordinals = ordinals_from_basis(row["candidates"], query_groups, query_bases)
        ref_target = int(reference["reference_target_index"])
        ref_ordinals = _reference_ordinals(
            reference_row["candidates"], reference["reference_groups"], ref_target, projected_basis(ref_matrix)
        )
        modes.append({"query_ordinals": query_ordinals, "reference_ordinals": ref_ordinals})
    return modes


@torch.inference_mode()
def evaluate(dataset: Path, collection_path: Path, query_features_path: Path, reference_features_path: Path,
             checkpoint_path: Path, development_path: Path, predictions_path: Path,
             native_coordinates_path: Path) -> dict[str, Any]:
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    query = torch.load(query_features_path, weights_only=False)
    reference = torch.load(reference_features_path, weights_only=False)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    native = json.loads(native_coordinates_path.read_text(encoding="utf-8"))
    collection_hash = sha256_file(collection_path)
    if any(value["collection_sha256"] != collection_hash for value in (query, reference, predictions, native)):
        raise ValueError("R1C-P collection identity mismatch")
    rows, reference_rows = query["rows"], reference["rows"]
    if len(rows) != 78 or collection["wrong_target_examples"] != 43:
        raise ValueError("R1C-P frozen denominator mismatch")
    if [row["sample_id"] for row in rows] != [record["sample_id"] for record in predictions["records"]]:
        raise ValueError("R1C-P prediction alignment mismatch")
    houses = load_houses(dataset, {int(row["house_index"]) for row in rows})
    native_rows = {row["sample_id"]: row for row in native["rows"]}
    negative_indices = negative_reference_indices(rows)
    threshold = float(development["thresholds"]["GRAIL"])
    if abs(threshold - 0.9353410602) > 1e-10:
        raise ValueError("R1C-P frozen threshold drift")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GrailModel(checkpoint["dim"]).to(device)
    model.load_state_dict(checkpoint["grail"])
    model.eval()
    cached = []
    for index, row in enumerate(rows):
        kinds = {}
        for kind, reference_index in (("positive", index), ("negative", negative_indices[index])):
            scores, poses = [], []
            for candidate in row["candidates"]:
                match = candidate["local_match"] if kind == "positive" else local_match_features(
                    candidate["tokens"].astype(np.float32), rows[reference_index]["reference_tokens"].astype(np.float32)
                )
                logit, predicted_pose = model(
                    torch.tensor(row["query_embedding"], device=device),
                    torch.tensor(rows[reference_index]["reference_embedding"], device=device),
                    torch.tensor(candidate["embedding"], device=device),
                    torch.tensor(candidate["geometry"], device=device),
                    torch.tensor(match, device=device),
                )
                scores.append(float(torch.sigmoid(logit)))
                poses.append(predicted_pose.cpu().tolist())
            kinds[kind] = {"reference_index": reference_index, "scores": scores, "poses": poses}
        cached.append(kinds)

    def privileged_decision(row_index: int, kind: str) -> tuple[int | None, float]:
        row, values = rows[row_index], cached[row_index][kind]
        reference_index = values["reference_index"]
        target_index = _target(reference_rows[reference_index]["candidates"])
        target_id = reference_rows[reference_index]["candidates"][target_index]["object_id"]
        target_coordinate = native_rows[rows[reference_index]["sample_id"]]["candidates"][target_id]
        target_signature = relation_signatures(
            houses[int(rows[reference_index]["house_index"])], reference_rows[reference_index]["candidates"]
        )[target_index]
        target_signature["part_horizontal"] = target_coordinate["horizontal"]
        target_signature["part_vertical"] = target_coordinate["vertical"]
        query_native = native_rows[row["sample_id"]]["candidates"]
        ordinals = [(query_native[candidate["object_id"]]["horizontal"],
                     query_native[candidate["object_id"]]["vertical"]) for candidate in row["candidates"]]
        signatures = _with_ordinals(houses[int(row["house_index"])], row["candidates"], ordinals)
        selected, confidence, _ = select_with_projected_relations(
            target_signature, signatures, values["scores"], [tuple(c["bbox"]) for c in row["candidates"]], FIELDS
        )
        return selected, confidence

    privileged_records = []
    for index, row in enumerate(rows):
        selected, confidence = privileged_decision(index, "positive")
        target_index = _target(row["candidates"])
        target = selected is not None and row["candidates"][selected]["is_target"]
        committed = confidence >= threshold
        privileged_records.append({
            "target": target,
            "complete": bool(committed and target and pose_success(cached[index]["positive"]["poses"][selected], row["truth_local_poses"])),
        })

    arms = {}
    alpha_histogram = {"0": 0, "1": 0, "2": 0, "4": 0}
    for prediction in predictions["records"]:
        alpha_histogram[str(prediction["references"]["positive"]["reference_absolute"]["alpha"])] += 1
    for arm in ARMS:
        records, absence, permutation = [], [], []
        for index, (row, prediction) in enumerate(zip(rows, predictions["records"])):
            decisions = {}
            for kind in ("positive", "negative"):
                values = cached[index][kind]
                reference_index = values["reference_index"]
                modes = _predicted_modes(row, reference_rows[reference_index], prediction, kind, arm)
                selected_modes, confidences, mode_permutation = [], [], []
                for mode in modes:
                    ref_target = _target(reference_rows[reference_index]["candidates"])
                    target_signature = _with_ordinals(
                        houses[int(rows[reference_index]["house_index"])],
                        reference_rows[reference_index]["candidates"], mode["reference_ordinals"],
                    )[ref_target]
                    query_signatures = _with_ordinals(
                        houses[int(row["house_index"])], row["candidates"], mode["query_ordinals"]
                    )
                    selected, confidence, _ = select_with_projected_relations(
                        target_signature, query_signatures, values["scores"],
                        [tuple(candidate["bbox"]) for candidate in row["candidates"]], FIELDS,
                    )
                    reverse_selected, reverse_confidence, _ = select_with_projected_relations(
                        target_signature, list(reversed(query_signatures)), list(reversed(values["scores"])),
                        list(reversed([tuple(candidate["bbox"]) for candidate in row["candidates"]])), FIELDS,
                    )
                    reverse_original = None if reverse_selected is None else len(row["candidates"]) - 1 - reverse_selected
                    mode_permutation.append(selected == reverse_original and confidence == reverse_confidence)
                    selected_modes.append(selected)
                    confidences.append(confidence)
                permutation.append(all(mode_permutation))
                selected = consensus_index(selected_modes)
                decisions[kind] = {"selected": selected, "confidence": min(confidences) if selected is not None else 0.0,
                                   "mode_count": len(modes), "mode_consensus": selected is not None}
            positive = decisions["positive"]
            selected = positive["selected"]
            target_index = _target(row["candidates"])
            relation_target = selected is not None and row["candidates"][selected]["is_target"]
            committed = positive["confidence"] >= threshold
            complete = bool(committed and relation_target and pose_success(
                cached[index]["positive"]["poses"][selected], row["truth_local_poses"]
            ))
            positive_modes = _predicted_modes(row, reference_rows[index], prediction, "positive", arm)
            slot_agree = False
            if positive_modes:
                slot_agree = all(
                    mode["query_ordinals"][target_index] == mode["reference_ordinals"][_target(reference_rows[index]["candidates"])]
                    and "NOT_EVALUABLE" not in mode["query_ordinals"][target_index]
                    for mode in positive_modes
                )
            records.append({
                "sample_id": row["sample_id"], "cross_view_slot_agree": slot_agree,
                "relation_target": relation_target, "complete": complete,
                "wrong_target": bool(row["same_type_visible_candidates"] >= 2 and committed and not relation_target),
                "mode_count": positive["mode_count"], "mode_consensus": positive["mode_consensus"],
                "privileged_target": privileged_records[index]["target"],
                "privileged_complete": privileged_records[index]["complete"],
            })
            absence.append(decisions["negative"]["confidence"] >= threshold)
        arms[arm] = {
            "cross_view_canonical_slot_agreement": sum(record["cross_view_slot_agree"] for record in records),
            "referent_top1": sum(record["relation_target"] for record in records),
            "complete_pose": sum(record["complete"] for record in records),
            "wrong_target": sum(record["wrong_target"] for record in records),
            "absence_false_commit": sum(absence),
            "mode_consensus": sum(record["mode_consensus"] for record in records),
            "mode_unknown": sum(not record["mode_consensus"] for record in records),
            "selector_collateral": sum(r["privileged_target"] and not r["relation_target"] for r in records),
            "complete_collateral": sum(r["privileged_complete"] and not r["complete"] for r in records),
            "permutation_consistent": sum(permutation), "permutation_denominator": len(permutation),
            "records": records,
        }
    final = arms["OA_V2_PAIRED_RELATIVE_FINAL"]
    gates = {
        "cross_view_canonical_slot_agreement": final["cross_view_canonical_slot_agreement"] >= GATES["cross_view_slot_agreement_minimum"],
        "referent_top1": final["referent_top1"] >= GATES["referent_top1_minimum"],
        "complete_pose": final["complete_pose"] >= GATES["complete_pose_minimum"],
        "wrong_target": final["wrong_target"] <= GATES["wrong_target_maximum"],
        "absence_false_commit": final["absence_false_commit"] <= GATES["absence_false_commit_maximum"],
        "permutation_consistent": final["permutation_consistent"] == GATES["permutation_consistent"],
        "selector_collateral": final["selector_collateral"] == 0,
        "complete_collateral": final["complete_collateral"] == 0,
    }
    terminal = ("GRAIL_R1C_P_PAIRED_VISUAL_OWNER_ORIENTATION_ESTABLISHED_FORMAL_TEST_ONLY" if all(gates.values())
                else "GRAIL_R1C_P_PAIRED_RGB_OWNER_ORIENTATION_NOT_ESTABLISHED_STOP_BEFORE_DEPTH_GEOMETRY")
    return {
        "schema": "blindassist_grail_r1c_p_paired_orientation_result_v1",
        "mode": "FRESH_HOUSE_DISJOINT_FIXED_ZERO_SHOT_OA_V2",
        "frozen_inputs": {
            "collection_sha256": collection_hash, "query_features_sha256": sha256_file(query_features_path),
            "reference_features_sha256": sha256_file(reference_features_path),
            "checkpoint_sha256": sha256_file(checkpoint_path), "predictions_sha256": sha256_file(predictions_path),
            "native_coordinates_sha256": sha256_file(native_coordinates_path), "grail_threshold": threshold,
        },
        "denominators": {"positive": 78, "wrong_target": 43, "absence": 78},
        "symmetry_alpha_histogram": alpha_histogram,
        "fresh_privileged_owner_local": {
            "referent_top1": sum(record["target"] for record in privileged_records),
            "complete_pose": sum(record["complete"] for record in privileged_records),
        },
        "arms": arms, "preregistered_gates": GATES, "gate_results": gates, "terminal": terminal,
        "claim_ceiling": "one fixed OA-V2 checkpoint on one fresh house-disjoint synthetic ProcTHOR cohort",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--native-coordinates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.collection, args.query_features, args.reference_features,
                      args.checkpoint, args.development_result, args.predictions, args.native_coordinates)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "fresh_privileged_owner_local": result["fresh_privileged_owner_local"],
        "arms": {name: {key: value for key, value in arm.items() if key != "records"}
                 for name, arm in result["arms"].items()},
        "gate_results": result["gate_results"], "terminal": result["terminal"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
