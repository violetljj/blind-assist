#!/usr/bin/env python3
"""Ablate which privileged GRAIL-R0 relation fields carry its oracle uplift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grail_relational_r0 import (
    RELATION_FIELD_GROUPS,
    load_houses,
    relation_signatures,
    select_with_projected_relations,
)
from run_grail_m1 import GrailModel, local_match_features, negative_reference_indices, pose_success, sha256_file


STAIRCASE = tuple(
    ("PLUS_" + "_".join(RELATION_FIELD_GROUPS[:index]).upper(), RELATION_FIELD_GROUPS[:index])
    for index in range(1, len(RELATION_FIELD_GROUPS) + 1)
)
LEAVE_ONE_OUT = tuple(
    ("FULL_MINUS_" + omitted.upper(), tuple(group for group in RELATION_FIELD_GROUPS if group != omitted))
    for omitted in RELATION_FIELD_GROUPS
)
MINIMAL_CANDIDATES = (
    ("MINIMAL_SIBLING_ONLY", ("sibling_ordinal",)),
    ("MINIMAL_SEMANTIC_SIBLING", ("semantic_type", "sibling_ordinal")),
    ("MINIMAL_SIBLING_NEARBY_TYPE", ("sibling_ordinal", "nearby_type")),
    ("MINIMAL_SEMANTIC_SIBLING_NEARBY_TYPE", ("semantic_type", "sibling_ordinal", "nearby_type")),
    ("MINIMAL_SEMANTIC_ROOM_SIBLING_NEARBY_TYPE", (
        "semantic_type", "room_types", "sibling_ordinal", "nearby_type",
    )),
)


def _metrics(records: list[dict[str, Any]], absence_commits: list[bool], permutation: list[bool]) -> dict[str, Any]:
    referent = sum(row["relation_target"] for row in records)
    joint = sum(row["relation_target"] and row["target_pose_capable"] for row in records)
    complete = sum(row["relation_complete"] for row in records)
    resolutions = {
        name: sum(row["resolution"] == name for row in records)
        for name in ("UNIQUE_RELATION_MATCH", "RELATION_COLLISION_APPEARANCE_TIEBREAK", "NO_EXACT_RELATION_MATCH")
    }
    return {
        "referent_top1": referent,
        "referent_and_target_pose": joint,
        "complete_pose": complete,
        "wrong_target": sum(row["wrong_target"] for row in records),
        "absence_false_commit": sum(absence_commits),
        "resolution_counts": resolutions,
        "permutation_consistent": sum(permutation),
        "permutation_denominator": len(permutation),
        "oracle_uplift_recovery": {
            "referent": (referent - 44) / (75 - 44),
            "complete": (complete - 22) / (57 - 22),
        },
    }


@torch.inference_mode()
def run_ablation(dataset: Path, collection_path: Path, features_path: Path, checkpoint_path: Path,
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
        raise ValueError("R0 result identity/metrics mismatch")
    rows = features["rows"]
    if len(rows) != 78:
        raise ValueError(f"observability ablation requires the frozen 78-case dev cohort, got {len(rows)}")

    houses = load_houses(dataset, {int(row["house_index"]) for row in rows})
    signatures = [relation_signatures(houses[int(row["house_index"])], row["candidates"]) for row in rows]
    target_signatures = [row_signatures[next(i for i, c in enumerate(row["candidates"]) if c["is_target"])]
                         for row, row_signatures in zip(rows, signatures)]
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

    variants: dict[str, Any] = {}
    for name, groups in STAIRCASE + LEAVE_ONE_OUT + MINIMAL_CANDIDATES:
        records, absence_commits, permutation = [], [], []
        for row_index, row in enumerate(rows):
            candidate_ids = [candidate["object_id"] for candidate in row["candidates"]]
            spatial_keys = [tuple(candidate["bbox"]) for candidate in row["candidates"]]
            decisions: dict[str, Any] = {}
            for kind in ("positive", "negative"):
                values = cached[row_index][kind]
                selected, confidence, resolution = select_with_projected_relations(
                    target_signatures[values["reference_index"]], signatures[row_index], values["scores"], spatial_keys, groups
                )
                reverse_selected, reverse_confidence, _ = select_with_projected_relations(
                    target_signatures[values["reference_index"]], list(reversed(signatures[row_index])),
                    list(reversed(values["scores"])), list(reversed(spatial_keys)), groups,
                )
                selected_id = None if selected is None else candidate_ids[selected]
                reverse_id = None if reverse_selected is None else list(reversed(candidate_ids))[reverse_selected]
                permutation.append(selected_id == reverse_id and confidence == reverse_confidence)
                decisions[kind] = (selected, confidence, resolution)
            selected, confidence, resolution = decisions["positive"]
            target_index = next(i for i, candidate in enumerate(row["candidates"]) if candidate["is_target"])
            relation_target = selected is not None and bool(row["candidates"][selected]["is_target"])
            target_pose = pose_success(cached[row_index]["positive"]["poses"][target_index], row["truth_local_poses"])
            relation_pose = selected is not None and pose_success(
                cached[row_index]["positive"]["poses"][selected], row["truth_local_poses"]
            )
            committed = confidence >= threshold
            same_type = int(row["same_type_visible_candidates"]) >= 2
            records.append({
                "relation_target": relation_target,
                "target_pose_capable": target_pose,
                "relation_complete": committed and relation_target and relation_pose,
                "wrong_target": bool(same_type and committed and not relation_target),
                "resolution": resolution,
            })
            absence_commits.append(decisions["negative"][1] >= threshold)
        variants[name] = {"field_groups": list(groups), **_metrics(records, absence_commits, permutation)}

    full = variants[STAIRCASE[-1][0]]
    expected = r0["grail_r0"]
    for key in ("referent_top1", "referent_and_target_pose", "complete_pose", "wrong_target", "absence_false_commit"):
        if full[key] != expected[key]:
            raise ValueError(f"full-signature R0 replay drift for {key}: {full[key]} != {expected[key]}")
    return {
        "schema": "blindassist_grail_r1_signature_observability_ablation_v1",
        "mode": "PROJECT_CONSUMED_DEVELOPMENT_DIAGNOSTIC",
        "claim_ceiling": "same consumed synthetic ProcTHOR Development privileged-metadata diagnostic; field attribution is order-sensitive and does not establish RGB/text obtainability",
        "frozen_inputs": {
            "positive_denominator": 78,
            "wrong_target_denominator": 43,
            "absence_denominator": 78,
            "grail_threshold": threshold,
            "dataset_sha256": sha256_file(dataset),
            "collection_sha256": sha256_file(collection_path),
            "features_sha256": sha256_file(features_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "development_result_sha256": sha256_file(development_result_path),
            "r0_result_sha256": sha256_file(r0_result_path),
        },
        "baselines": {"m1_referent": 44, "m1_complete": 22, "r0_referent": 75, "r0_complete": 57},
        "staircase_order": list(RELATION_FIELD_GROUPS),
        "variants": variants,
        "terminal": "GRAIL_R1_SIGNATURE_OBSERVABILITY_ABLATION_COMPLETE",
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
    result = run_ablation(
        args.dataset, args.collection, args.features, args.checkpoint, args.development_result, args.r0_result
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"terminal": result["terminal"], "variants": result["variants"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
