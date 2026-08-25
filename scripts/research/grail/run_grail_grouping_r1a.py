#!/usr/bin/env python3
"""Run the training-free GRAIL-R1A obtainable grouping Development probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from grail_grouping_r1a import (
    aligned_context_score,
    predict_groups,
    predicted_ordinals,
    select_by_predicted_ordinal,
    select_carrier,
    shifted_context_score,
)
from grail_relational_r0 import _candidate_record, door_record, flatten_objects, load_houses, relation_signatures
from run_grail_m1 import GrailModel, local_match_features, negative_reference_indices, pose_success, sha256_file


ARMS: tuple[tuple[str, str], ...] = (
    ("APPEARANCE_CARRIER_CONTROL", "appearance"),
    ("ALIGNED_CONTEXT_DINO", "aligned"),
    ("SHIFT2_CONTEXT_DINO", "shift2"),
)


def _truth_roots(house: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    object_index, _ = flatten_objects(house.get("objects", []))
    doors = {door["id"]: door_record(door) for door in house.get("doors", [])}
    return [_candidate_record(candidate, object_index, doors)["root_id"] for candidate in candidates]


def _grouping_metrics(rows: list[dict[str, Any]], houses: dict[int, dict[str, Any]],
                      signatures: list[list[dict[str, Any]]]) -> tuple[
                          dict[str, Any], list[list[tuple[str, str]]], list[dict[str, Any]]
                      ]:
    tp = fp = fn = tn = exact_rows = ordinal_correct = ordinal_total = target_ordinal_correct = 0
    all_ordinals: list[list[tuple[str, str]]] = []
    row_diagnostics: list[dict[str, Any]] = []
    for row, row_signatures in zip(rows, signatures):
        candidates = row["candidates"]
        truth_roots = _truth_roots(houses[int(row["house_index"])], candidates)
        groups = predict_groups(candidates)
        ordinals = predicted_ordinals(candidates, groups)
        all_ordinals.append(ordinals)
        exact = True
        for first in range(len(candidates)):
            oracle_ordinal = (row_signatures[first]["part_horizontal"], row_signatures[first]["part_vertical"])
            ordinal_correct += ordinals[first] == oracle_ordinal
            ordinal_total += 1
            for second in range(first + 1, len(candidates)):
                if candidates[first]["object_type"] != candidates[second]["object_type"]:
                    continue
                truth = truth_roots[first] == truth_roots[second]
                predicted = groups[first] == groups[second]
                tp += truth and predicted
                fp += not truth and predicted
                fn += truth and not predicted
                tn += not truth and not predicted
                exact = exact and truth == predicted
        exact_rows += exact
        target_index = next(index for index, candidate in enumerate(candidates) if candidate["is_target"])
        oracle_target = (row_signatures[target_index]["part_horizontal"], row_signatures[target_index]["part_vertical"])
        target_correct = ordinals[target_index] == oracle_target
        target_ordinal_correct += target_correct
        row_diagnostics.append({
            "query_partition_exact": exact,
            "query_target_ordinal_correct": target_correct,
        })
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    return {
        "same_type_pair_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "same_root_pair_precision": precision,
        "same_root_pair_recall": recall,
        "same_root_pair_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "different_root_pair_specificity": specificity,
        "pair_balanced_accuracy": (recall + specificity) / 2.0,
        "exact_partition_rows": exact_rows,
        "exact_partition_denominator": len(rows),
        "candidate_ordinal_correct": ordinal_correct,
        "candidate_ordinal_denominator": ordinal_total,
        "target_candidate_ordinal_correct": target_ordinal_correct,
        "target_candidate_ordinal_denominator": len(rows),
    }, all_ordinals, row_diagnostics


def _context_scores(mode: str, candidates: list[dict[str, Any]], reference_tokens: np.ndarray,
                    appearance_scores: list[float]) -> list[float]:
    if mode == "appearance":
        return appearance_scores
    scorer: Callable[[np.ndarray, np.ndarray], float]
    scorer = aligned_context_score if mode == "aligned" else shifted_context_score
    return [scorer(candidate["tokens"], reference_tokens) for candidate in candidates]


@torch.inference_mode()
def run_probe(dataset: Path, collection_path: Path, features_path: Path, checkpoint_path: Path,
              development_result_path: Path, r0_result_path: Path) -> dict[str, Any]:
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    features = torch.load(features_path, weights_only=False)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    development = json.loads(development_result_path.read_text(encoding="utf-8"))
    r0 = json.loads(r0_result_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != collection["dataset_sha256"]:
        raise ValueError("ProcTHOR val identity mismatch")
    if sha256_file(collection_path) != features["collection_sha256"]:
        raise ValueError("collection/features identity mismatch")
    if sha256_file(checkpoint_path) != development["checkpoint_sha256"]:
        raise ValueError("checkpoint/development-result identity mismatch")
    if r0["grail_r0"]["referent_top1"] != 75 or r0["grail_r0"]["complete_pose"] != 57:
        raise ValueError("R0 metrics mismatch")
    rows = features["rows"]
    if len(rows) != 78:
        raise ValueError(f"R1A requires the frozen 78-case Development cohort, got {len(rows)}")

    houses = load_houses(dataset, {int(row["house_index"]) for row in rows})
    signatures = [relation_signatures(houses[int(row["house_index"])], row["candidates"]) for row in rows]
    grouping, predicted_row_ordinals, grouping_row_diagnostics = _grouping_metrics(rows, houses, signatures)
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
    arms: dict[str, Any] = {}
    for arm_name, mode in ARMS:
        records: list[dict[str, Any]] = []
        absence_commits: list[bool] = []
        permutation: list[bool] = []
        positive_carrier_ordinal_correct = 0
        for row_index, row in enumerate(rows):
            ordinals = predicted_row_ordinals[row_index]
            spatial_keys = [tuple(candidate["bbox"]) for candidate in row["candidates"]]
            candidate_ids = [candidate["object_id"] for candidate in row["candidates"]]
            decisions: dict[str, Any] = {}
            for kind in ("positive", "negative"):
                values = cached[row_index][kind]
                reference_index = values["reference_index"]
                target_type = rows[reference_index]["target_type"]
                carrier_scores = _context_scores(
                    mode, row["candidates"], rows[reference_index]["reference_tokens"], values["scores"]
                )
                eligible = [
                    index for index, candidate in enumerate(row["candidates"])
                    if candidate["object_type"] == target_type
                ]
                carrier = select_carrier(carrier_scores, eligible, spatial_keys)
                target_ordinal = None if carrier is None else ordinals[carrier]
                selected, confidence, resolution = select_by_predicted_ordinal(
                    target_type, target_ordinal, row["candidates"], ordinals, values["scores"], spatial_keys
                )
                reversed_candidates = list(reversed(row["candidates"]))
                reversed_ordinals = list(reversed(ordinals))
                reversed_appearance = list(reversed(values["scores"]))
                reversed_carrier_scores = list(reversed(carrier_scores))
                reversed_spatial = list(reversed(spatial_keys))
                reversed_eligible = [
                    index for index, candidate in enumerate(reversed_candidates)
                    if candidate["object_type"] == target_type
                ]
                reverse_carrier = select_carrier(reversed_carrier_scores, reversed_eligible, reversed_spatial)
                reverse_target_ordinal = None if reverse_carrier is None else reversed_ordinals[reverse_carrier]
                reverse_selected, reverse_confidence, _ = select_by_predicted_ordinal(
                    target_type, reverse_target_ordinal, reversed_candidates, reversed_ordinals,
                    reversed_appearance, reversed_spatial,
                )
                selected_id = None if selected is None else candidate_ids[selected]
                reverse_id = None if reverse_selected is None else list(reversed(candidate_ids))[reverse_selected]
                permutation.append(selected_id == reverse_id and confidence == reverse_confidence)
                decisions[kind] = {
                    "selected": selected, "confidence": confidence, "resolution": resolution,
                    "carrier": carrier, "target_ordinal": target_ordinal,
                }
            positive = decisions["positive"]
            selected = positive["selected"]
            target_index = next(index for index, candidate in enumerate(row["candidates"]) if candidate["is_target"])
            oracle_target_ordinal = (
                signatures[row_index][target_index]["part_horizontal"],
                signatures[row_index][target_index]["part_vertical"],
            )
            positive_carrier_ordinal_correct += positive["target_ordinal"] == oracle_target_ordinal
            carrier_ordinal_correct = positive["target_ordinal"] == oracle_target_ordinal
            relation_target = selected is not None and bool(row["candidates"][selected]["is_target"])
            target_pose = pose_success(cached[row_index]["positive"]["poses"][target_index], row["truth_local_poses"])
            relation_pose = selected is not None and pose_success(
                cached[row_index]["positive"]["poses"][selected], row["truth_local_poses"]
            )
            committed = positive["confidence"] >= threshold
            same_type = int(row["same_type_visible_candidates"]) >= 2
            baseline = r0_records[row["sample_id"]]
            records.append({
                "sample_id": row["sample_id"],
                "relation_target": relation_target,
                "target_pose_capable": target_pose,
                "relation_complete": committed and relation_target and relation_pose,
                "relation_committed": committed,
                "relation_pose": relation_pose,
                "wrong_target": bool(same_type and committed and not relation_target),
                "baseline_target": baseline["baseline_target"],
                "baseline_complete": baseline["baseline_complete"],
                "r0_target": baseline["relation_target"],
                "r0_complete": baseline["relation_complete"],
                "carrier_ordinal_correct": carrier_ordinal_correct,
                **grouping_row_diagnostics[row_index],
                "resolution": positive["resolution"],
            })
            absence_commits.append(decisions["negative"]["confidence"] >= threshold)

        referent = sum(record["relation_target"] for record in records)
        joint = sum(record["relation_target"] and record["target_pose_capable"] for record in records)
        complete = sum(record["relation_complete"] for record in records)
        r0_selector_rescues = [record for record in records if not record["baseline_target"] and record["r0_target"]]
        r0_complete_rescues = [record for record in records if not record["baseline_complete"] and record["r0_complete"]]
        referent_failures = [record for record in records if not record["relation_target"]]
        failure_attribution = {
            "query_grouping_or_ordinal_error": sum(
                not record["query_partition_exact"] or not record["query_target_ordinal_correct"]
                for record in referent_failures
            ),
            "reference_context_ordinal_error_after_exact_query_grouping": sum(
                record["query_partition_exact"] and record["query_target_ordinal_correct"]
                and not record["carrier_ordinal_correct"]
                for record in referent_failures
            ),
            "ordinal_collision_or_appearance_tiebreak_after_correct_ordinals": sum(
                record["query_partition_exact"] and record["query_target_ordinal_correct"]
                and record["carrier_ordinal_correct"]
                for record in referent_failures
            ),
        }
        arms[arm_name] = {
            "carrier_mode": mode,
            "referent_top1": referent,
            "referent_and_target_pose": joint,
            "complete_pose": complete,
            "wrong_target": sum(record["wrong_target"] for record in records),
            "absence_false_commit": sum(absence_commits),
            "target_ordinal_correct": positive_carrier_ordinal_correct,
            "target_ordinal_denominator": len(rows),
            "r0_selector_rescue_recovered": sum(record["relation_target"] for record in r0_selector_rescues),
            "r0_selector_rescue_denominator": len(r0_selector_rescues),
            "r0_complete_rescue_recovered": sum(record["relation_complete"] for record in r0_complete_rescues),
            "r0_complete_rescue_denominator": len(r0_complete_rescues),
            "selector_collateral": sum(record["baseline_target"] and not record["relation_target"] for record in records),
            "complete_collateral": sum(record["baseline_complete"] and not record["relation_complete"] for record in records),
            "referent_failure_attribution": failure_attribution,
            "oracle_uplift_recovery": {
                "referent": (referent - 44) / (75 - 44),
                "complete": (complete - 22) / (57 - 22),
            },
            "permutation_consistent": sum(permutation),
            "permutation_denominator": len(permutation),
            "resolution_counts": {
                name: sum(record["resolution"] == name for record in records)
                for name in ("UNIQUE_PREDICTED_ORDINAL_MATCH", "ORDINAL_COLLISION_APPEARANCE_TIEBREAK", "NO_TARGET_TYPE_CARRIER", "NO_PREDICTED_ORDINAL_MATCH")
            },
            "records": records,
        }

    best_name = max(
        arms,
        key=lambda name: (
            arms[name]["complete_pose"], arms[name]["referent_top1"],
            -arms[name]["wrong_target"], -arms[name]["absence_false_commit"], name,
        ),
    )
    best = arms[best_name]
    terminal = "GRAIL_R1A_QUERY_GROUPING_HIGH_REFERENCE_ORDINAL_PARTIAL_FALSE_COMMIT_UNRESOLVED"
    return {
        "schema": "blindassist_grail_r1a_obtainable_grouping_probe_v1",
        "mode": "PROJECT_CONSUMED_DEVELOPMENT_TRAINING_FREE",
        "input_contract": "existing synthetic query/reference RGB, oracle candidate bbox proposals, simulator semantic candidate types, frozen DINO/M1 features; no pixel masks were stored and none are claimed",
        "claim_ceiling": "consumed synthetic ProcTHOR Development grouping/context probe; no natural RGB, learned grouping, formal test, Android, product, or safety authority",
        "frozen_inputs": {
            "positive_denominator": len(rows), "wrong_target_denominator": 43, "absence_denominator": len(rows),
            "grail_threshold": threshold, "dataset_sha256": sha256_file(dataset),
            "collection_sha256": sha256_file(collection_path), "features_sha256": sha256_file(features_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "development_result_sha256": sha256_file(development_result_path), "r0_result_sha256": sha256_file(r0_result_path),
        },
        "grouping": grouping,
        "baselines": {"m1_referent": 44, "m1_complete": 22, "r0_referent": 75, "r0_complete": 57},
        "arms": arms,
        "selected_arm": best_name,
        "terminal": terminal,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--r0-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(
        args.dataset, args.collection, args.features, args.checkpoint,
        args.development_result, args.r0_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        "grouping": result["grouping"],
        "arms": {name: {key: arm[key] for key in (
            "referent_top1", "complete_pose", "wrong_target", "absence_false_commit",
            "target_ordinal_correct", "r0_selector_rescue_recovered", "r0_complete_rescue_recovered",
            "oracle_uplift_recovery", "permutation_consistent",
        )} for name, arm in result["arms"].items()},
        "selected_arm": result["selected_arm"], "terminal": result["terminal"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
