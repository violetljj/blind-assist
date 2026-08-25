#!/usr/bin/env python3
"""Run the GRAIL-R1B bilateral full-scene reference grouping probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from grail_grouping_r1a import predict_groups, predicted_ordinals, select_by_predicted_ordinal
from grail_relational_r0 import _candidate_record, door_record, flatten_objects, load_houses, relation_signatures
from run_grail_grouping_r1a import _grouping_metrics
from run_grail_m1 import (
    GrailModel,
    VISUAL_WEIGHTS_SHA256,
    encode_images,
    expanded_crop,
    local_match_features,
    negative_reference_indices,
    pose_success,
    sha256_file,
)


ARMS = ("FULL_SCENE_RGB_BBOX", "FULL_SCENE_RGB_MASK_CENTROID")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _truth_roots(house: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    object_index, _ = flatten_objects(house.get("objects", []))
    doors = {door["id"]: door_record(door) for door in house.get("doors", [])}
    return [_candidate_record(candidate, object_index, doors)["root_id"] for candidate in candidates]


@torch.inference_mode()
def materialize_reference_features(
    supplement_path: Path, supplement_root: Path, visual_path: Path, cache_path: Path,
) -> dict[str, Any]:
    supplement_sha = sha256_file(supplement_path)
    visual_weights = visual_path / "model.safetensors"
    if sha256_file(visual_weights) != VISUAL_WEIGHTS_SHA256:
        raise ValueError("frozen DINOv2-S weights identity mismatch")
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        if cached.get("supplement_sha256") != supplement_sha:
            raise ValueError("reference feature cache supplement identity mismatch")
        if cached.get("visual_weights_sha256") != VISUAL_WEIGHTS_SHA256:
            raise ValueError("reference feature cache visual identity mismatch")
        return cached

    from transformers import AutoImageProcessor, AutoModel

    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(visual_path, local_files_only=True)
    visual = AutoModel.from_pretrained(visual_path, local_files_only=True).to(device).eval()
    feature_rows = []
    for number, row in enumerate(supplement["rows"], 1):
        full_image = Image.open(supplement_root / row["reference_full_image"]).convert("RGB")
        crops = [expanded_crop(full_image, candidate["bbox"]) for candidate in row["candidates"]]
        embeddings, _ = encode_images(crops, processor, visual, device)
        candidates = []
        for candidate, embedding in zip(row["candidates"], embeddings):
            mask = np.asarray(Image.open(supplement_root / candidate["mask_image"]).convert("L")) > 0
            if _array_sha256(mask) != candidate["mask_array_sha256"]:
                raise ValueError(f"reference mask identity mismatch: {row['sample_id']}")
            ys, xs = np.nonzero(mask)
            candidates.append({
                **candidate,
                "embedding": embedding.astype(np.float16),
                "mask_centroid": (float(xs.mean()), float(ys.mean())),
            })
        feature_rows.append({**row, "candidates": candidates})
        if number % 20 == 0:
            print(json.dumps({"state": "REFERENCE_FEATURES", "completed": number, "total": len(supplement["rows"])}), flush=True)
    result = {
        "schema": "blindassist_grail_r1b_reference_features_v1",
        "supplement_sha256": supplement_sha,
        "visual_weights_sha256": VISUAL_WEIGHTS_SHA256,
        "rows": feature_rows,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(result, temporary)
    temporary.replace(cache_path)
    del visual
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _reference_grouping(
    feature_rows: list[dict[str, Any]], houses: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[tuple[str, str]]], list[dict[str, Any]]]:
    tp = fp = fn = tn = exact_partitions = exact_owner_groups = 0
    arm_target_ordinals = {arm: [] for arm in ARMS}
    diagnostics = []
    bbox_correct = mask_correct = 0
    for row in feature_rows:
        candidates = row["candidates"]
        house = houses[int(row["house_index"])]
        truth_roots = _truth_roots(house, candidates)
        groups = predict_groups(candidates)
        bbox_ordinals = predicted_ordinals(candidates, groups)
        mask_ordinals = predicted_ordinals(candidates, groups, [tuple(candidate["mask_centroid"]) for candidate in candidates])
        oracle_signatures = relation_signatures(house, candidates)
        target_index = next(index for index, candidate in enumerate(candidates) if candidate["is_target"])
        oracle_target_ordinal = (
            oracle_signatures[target_index]["part_horizontal"],
            oracle_signatures[target_index]["part_vertical"],
        )
        arm_target_ordinals["FULL_SCENE_RGB_BBOX"].append(bbox_ordinals[target_index])
        arm_target_ordinals["FULL_SCENE_RGB_MASK_CENTROID"].append(mask_ordinals[target_index])
        bbox_correct += bbox_ordinals[target_index] == oracle_target_ordinal
        mask_correct += mask_ordinals[target_index] == oracle_target_ordinal

        partition_exact = True
        target_owner_exact = True
        for first in range(len(candidates)):
            for second in range(first + 1, len(candidates)):
                if candidates[first]["object_type"] != candidates[second]["object_type"]:
                    continue
                truth = truth_roots[first] == truth_roots[second]
                predicted = groups[first] == groups[second]
                tp += truth and predicted
                fp += not truth and predicted
                fn += truth and not predicted
                tn += not truth and not predicted
                partition_exact = partition_exact and truth == predicted
                if first == target_index or second == target_index:
                    target_owner_exact = target_owner_exact and truth == predicted
        exact_partitions += partition_exact
        exact_owner_groups += target_owner_exact
        diagnostics.append({
            "reference_partition_exact": partition_exact,
            "reference_target_owner_group_exact": target_owner_exact,
            "reference_oracle_target_ordinal": oracle_target_ordinal,
            "reference_bbox_target_ordinal_correct": bbox_ordinals[target_index] == oracle_target_ordinal,
            "reference_mask_target_ordinal_correct": mask_ordinals[target_index] == oracle_target_ordinal,
        })
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    metrics = {
        "same_type_pair_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "same_root_pair_precision": precision,
        "same_root_pair_recall": recall,
        "same_root_pair_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "different_root_pair_specificity": specificity,
        "pair_balanced_accuracy": (recall + specificity) / 2.0,
        "exact_partition_rows": exact_partitions,
        "target_owner_group_exact_rows": exact_owner_groups,
        "denominator": len(feature_rows),
        "bbox_target_ordinal_correct": bbox_correct,
        "mask_target_ordinal_correct": mask_correct,
    }
    return metrics, arm_target_ordinals, diagnostics


@torch.inference_mode()
def run_probe(
    dataset: Path, collection_path: Path, features_path: Path, checkpoint_path: Path,
    development_result_path: Path, r0_result_path: Path, supplement_path: Path,
    supplement_root: Path, reference_features_path: Path, visual_path: Path,
) -> dict[str, Any]:
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
    reference_features = materialize_reference_features(supplement_path, supplement_root, visual_path, reference_features_path)
    reference_rows = reference_features["rows"]
    if len(rows) != 78 or [row["sample_id"] for row in rows] != [row["sample_id"] for row in reference_rows]:
        raise ValueError("R1B reference supplement does not match frozen 78-case cohort")

    houses = load_houses(dataset, {int(row["house_index"]) for row in rows})
    query_signatures = [relation_signatures(houses[int(row["house_index"])], row["candidates"]) for row in rows]
    query_grouping, query_ordinals, query_diagnostics = _grouping_metrics(rows, houses, query_signatures)
    reference_grouping, reference_target_ordinals, reference_diagnostics = _reference_grouping(reference_rows, houses)

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
    for arm_name in ARMS:
        records = []
        absence_commits = []
        permutation = []
        for row_index, row in enumerate(rows):
            ordinals = query_ordinals[row_index]
            spatial_keys = [tuple(candidate["bbox"]) for candidate in row["candidates"]]
            candidate_ids = [candidate["object_id"] for candidate in row["candidates"]]
            decisions: dict[str, Any] = {}
            for kind in ("positive", "negative"):
                values = cached[row_index][kind]
                reference_index = values["reference_index"]
                target_type = rows[reference_index]["target_type"]
                target_ordinal = reference_target_ordinals[arm_name][reference_index]
                selected, confidence, resolution = select_by_predicted_ordinal(
                    target_type, target_ordinal, row["candidates"], ordinals, values["scores"], spatial_keys
                )
                reverse_selected, reverse_confidence, _ = select_by_predicted_ordinal(
                    target_type, target_ordinal, list(reversed(row["candidates"])), list(reversed(ordinals)),
                    list(reversed(values["scores"])), list(reversed(spatial_keys)),
                )
                selected_id = None if selected is None else candidate_ids[selected]
                reverse_id = None if reverse_selected is None else list(reversed(candidate_ids))[reverse_selected]
                permutation.append(selected_id == reverse_id and confidence == reverse_confidence)
                decisions[kind] = {"selected": selected, "confidence": confidence, "resolution": resolution}

            positive = decisions["positive"]
            selected = positive["selected"]
            target_index = next(index for index, candidate in enumerate(row["candidates"]) if candidate["is_target"])
            query_oracle_ordinal = (
                query_signatures[row_index][target_index]["part_horizontal"],
                query_signatures[row_index][target_index]["part_vertical"],
            )
            reference_predicted_ordinal = reference_target_ordinals[arm_name][row_index]
            reference_oracle_ordinal = reference_diagnostics[row_index]["reference_oracle_target_ordinal"]
            relation_target = selected is not None and bool(row["candidates"][selected]["is_target"])
            target_pose = pose_success(cached[row_index]["positive"]["poses"][target_index], row["truth_local_poses"])
            relation_pose = selected is not None and pose_success(cached[row_index]["positive"]["poses"][selected], row["truth_local_poses"])
            committed = positive["confidence"] >= threshold
            same_type = int(row["same_type_visible_candidates"]) >= 2
            baseline = r0_records[row["sample_id"]]
            records.append({
                "sample_id": row["sample_id"],
                "relation_target": relation_target,
                "target_pose_capable": target_pose,
                "relation_complete": committed and relation_target and relation_pose,
                "relation_committed": committed,
                "wrong_target": bool(same_type and committed and not relation_target),
                "baseline_target": baseline["baseline_target"],
                "baseline_complete": baseline["baseline_complete"],
                "r0_target": baseline["relation_target"],
                "r0_complete": baseline["relation_complete"],
                "query_target_ordinal_correct": ordinals[target_index] == query_oracle_ordinal,
                "reference_target_owner_group_exact": reference_diagnostics[row_index]["reference_target_owner_group_exact"],
                "reference_target_ordinal_correct": reference_predicted_ordinal == reference_oracle_ordinal,
                "query_reference_predicted_ordinal_agree": ordinals[target_index] == reference_predicted_ordinal,
                "query_reference_oracle_ordinal_agree": query_oracle_ordinal == reference_oracle_ordinal,
                "query_predicted_ordinal": ordinals[target_index],
                "reference_predicted_ordinal": reference_predicted_ordinal,
                "query_oracle_ordinal": query_oracle_ordinal,
                "reference_oracle_ordinal": reference_oracle_ordinal,
                "resolution": positive["resolution"],
            })
            absence_commits.append(decisions["negative"]["confidence"] >= threshold)

        referent = sum(record["relation_target"] for record in records)
        complete = sum(record["relation_complete"] for record in records)
        r0_selector_rescues = [record for record in records if not record["baseline_target"] and record["r0_target"]]
        r0_complete_rescues = [record for record in records if not record["baseline_complete"] and record["r0_complete"]]
        failures = [record for record in records if not record["relation_target"]]
        arms[arm_name] = {
            "referent_top1": referent,
            "referent_and_target_pose": sum(record["relation_target"] and record["target_pose_capable"] for record in records),
            "complete_pose": complete,
            "wrong_target": sum(record["wrong_target"] for record in records),
            "absence_false_commit": sum(absence_commits),
            "reference_target_ordinal_correct": sum(record["reference_target_ordinal_correct"] for record in records),
            "query_reference_predicted_ordinal_agreement": sum(record["query_reference_predicted_ordinal_agree"] for record in records),
            "query_reference_oracle_ordinal_agreement": sum(record["query_reference_oracle_ordinal_agree"] for record in records),
            "query_reference_predicted_horizontal_agreement": sum(
                record["query_predicted_ordinal"][0] == record["reference_predicted_ordinal"][0] for record in records
            ),
            "query_reference_predicted_vertical_agreement": sum(
                record["query_predicted_ordinal"][1] == record["reference_predicted_ordinal"][1] for record in records
            ),
            "query_reference_oracle_horizontal_agreement": sum(
                record["query_oracle_ordinal"][0] == record["reference_oracle_ordinal"][0] for record in records
            ),
            "query_reference_oracle_vertical_agreement": sum(
                record["query_oracle_ordinal"][1] == record["reference_oracle_ordinal"][1] for record in records
            ),
            "target_owner_group_exact": sum(record["reference_target_owner_group_exact"] for record in records),
            "denominator": len(records),
            "r0_selector_rescue_recovered": sum(record["relation_target"] for record in r0_selector_rescues),
            "r0_selector_rescue_denominator": len(r0_selector_rescues),
            "r0_complete_rescue_recovered": sum(record["relation_complete"] for record in r0_complete_rescues),
            "r0_complete_rescue_denominator": len(r0_complete_rescues),
            "selector_collateral": sum(record["baseline_target"] and not record["relation_target"] for record in records),
            "complete_collateral": sum(record["baseline_complete"] and not record["relation_complete"] for record in records),
            "oracle_uplift_recovery": {"referent": (referent - 44) / 31, "complete": (complete - 22) / 35},
            "permutation_consistent": sum(permutation),
            "permutation_denominator": len(permutation),
            "referent_failure_attribution": {
                "query_grouping_or_ordinal_error": sum(not record["query_target_ordinal_correct"] for record in failures),
                "reference_grouping_or_ordinal_error": sum(
                    record["query_target_ordinal_correct"] and not record["reference_target_ordinal_correct"] for record in failures
                ),
                "cross_view_coordinate_disagreement_after_view_local_ordinals_correct": sum(
                    record["query_target_ordinal_correct"] and record["reference_target_ordinal_correct"]
                    and not record["query_reference_oracle_ordinal_agree"] for record in failures
                ),
                "collision_or_appearance_tiebreak_after_ordinal_agreement": sum(
                    record["query_target_ordinal_correct"] and record["reference_target_ordinal_correct"]
                    and record["query_reference_oracle_ordinal_agree"] for record in failures
                ),
            },
            "records": records,
        }

    best_name = max(arms, key=lambda name: (
        arms[name]["complete_pose"], arms[name]["referent_top1"],
        -arms[name]["wrong_target"], -arms[name]["absence_false_commit"], name,
    ))
    return {
        "schema": "blindassist_grail_r1b_bilateral_grouping_probe_v1",
        "mode": "PROJECT_CONSUMED_DEVELOPMENT_TRAINING_FREE_REFERENCE_SOURCE_CHANGE_ONLY",
        "input_contract": "frozen query RGB+bbox grouping plus replayed full-scene reference RGB, actionable bbox proposals and masks; frozen DINO, grouping affinity, selector, pose head, threshold and evaluator",
        "claim_ceiling": "consumed synthetic ProcTHOR Development bilateral grouping diagnostic; no formal test, natural RGB, learned ownership, Android, product, or safety authority",
        "frozen_inputs": {
            "positive_denominator": len(rows), "wrong_target_denominator": 43, "absence_denominator": len(rows),
            "grail_threshold": threshold, "dataset_sha256": sha256_file(dataset),
            "collection_sha256": sha256_file(collection_path), "features_sha256": sha256_file(features_path),
            "checkpoint_sha256": sha256_file(checkpoint_path), "development_result_sha256": sha256_file(development_result_path),
            "r0_result_sha256": sha256_file(r0_result_path), "reference_supplement_sha256": sha256_file(supplement_path),
            "reference_features_sha256": sha256_file(reference_features_path), "visual_weights_sha256": VISUAL_WEIGHTS_SHA256,
        },
        "reproduction": json.loads(supplement_path.read_text(encoding="utf-8"))["summary"],
        "query_grouping": query_grouping,
        "reference_grouping": reference_grouping,
        "baselines": {"m1_referent": 44, "m1_complete": 22, "r1a_referent": 51, "r1a_complete": 38, "r0_referent": 75, "r0_complete": 57},
        "arms": arms,
        "selected_arm": best_name,
        "terminal": "GRAIL_R1B_REFERENCE_OWNERSHIP_HIGH_BUT_VIEW_LOCAL_ORDINAL_NOT_ALIGNABLE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--r0-result", type=Path, required=True)
    parser.add_argument("--reference-supplement", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--visual-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(
        args.dataset, args.collection, args.features, args.checkpoint, args.development_result,
        args.r0_result, args.reference_supplement, args.reference_root, args.reference_features, args.visual_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        "reproduction": result["reproduction"],
        "query_grouping": result["query_grouping"],
        "reference_grouping": result["reference_grouping"],
        "arms": {name: {key: arm[key] for key in (
            "referent_top1", "complete_pose", "wrong_target", "absence_false_commit",
            "reference_target_ordinal_correct", "query_reference_predicted_ordinal_agreement",
            "query_reference_oracle_ordinal_agreement", "target_owner_group_exact",
            "r0_selector_rescue_recovered", "r0_complete_rescue_recovered", "oracle_uplift_recovery",
        )} for name, arm in result["arms"].items()},
        "selected_arm": result["selected_arm"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
