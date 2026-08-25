#!/usr/bin/env python3
"""Run GRAIL-R0 with frozen M1 pose head and privileged native relations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grail_relational_r0 import load_houses, relation_signatures, select_with_relational_oracle
from run_grail_m1 import GrailModel, local_match_features, negative_reference_indices, pose_success, sha256_file


@torch.inference_mode()
def run_probe(dataset: Path, collection_path: Path, features_path: Path, checkpoint_path: Path,
              development_result_path: Path) -> dict[str, Any]:
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    features = torch.load(features_path, weights_only=False)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    development = json.loads(development_result_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != collection["dataset_sha256"]:
        raise ValueError("ProcTHOR val identity mismatch")
    if sha256_file(collection_path) != features["collection_sha256"]:
        raise ValueError("collection/features identity mismatch")
    if sha256_file(checkpoint_path) != development["checkpoint_sha256"]:
        raise ValueError("checkpoint/development-result identity mismatch")
    rows = features["rows"]
    if len(rows) != 78:
        raise ValueError(f"GRAIL-R0 requires the frozen 78-case dev cohort, got {len(rows)}")

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
    records = []
    absence_commits = []
    permutation = []

    for row_index, row in enumerate(rows):
        per_kind: dict[str, Any] = {}
        for kind, reference_index in (("positive", row_index), ("negative", negative_indices[row_index])):
            reference_embedding = rows[reference_index]["reference_embedding"]
            candidate_scores, candidate_poses = [], []
            for candidate in row["candidates"]:
                match_np = candidate["local_match"] if kind == "positive" else local_match_features(
                    candidate["tokens"].astype(np.float32),
                    rows[reference_index]["reference_tokens"].astype(np.float32),
                )
                logit, poses = model(
                    torch.tensor(row["query_embedding"], device=device),
                    torch.tensor(reference_embedding, device=device),
                    torch.tensor(candidate["embedding"], device=device),
                    torch.tensor(candidate["geometry"], device=device),
                    torch.tensor(match_np, device=device),
                )
                candidate_scores.append(float(torch.sigmoid(logit)))
                candidate_poses.append(poses.cpu().tolist())
            candidate_ids = [candidate["object_id"] for candidate in row["candidates"]]
            spatial_keys = [tuple(candidate["bbox"]) for candidate in row["candidates"]]
            selected, confidence, resolution = select_with_relational_oracle(
                target_signatures[reference_index], signatures[row_index], candidate_scores, spatial_keys,
            )
            reverse_selected, reverse_confidence, _ = select_with_relational_oracle(
                target_signatures[reference_index], list(reversed(signatures[row_index])),
                list(reversed(candidate_scores)), list(reversed(spatial_keys)),
            )
            reverse_object_id = None if reverse_selected is None else list(reversed(candidate_ids))[reverse_selected]
            selected_object_id = None if selected is None else candidate_ids[selected]
            permutation.append(selected_object_id == reverse_object_id and confidence == reverse_confidence)
            per_kind[kind] = {
                "selected": selected,
                "confidence": confidence,
                "committed": confidence >= threshold,
                "resolution": resolution,
                "poses": None if selected is None else candidate_poses[selected],
                "candidate_poses": candidate_poses,
                "appearance_scores": candidate_scores,
            }
        baseline_selected = int(np.argmax(per_kind["positive"]["appearance_scores"]))
        baseline_committed = max(per_kind["positive"]["appearance_scores"]) >= threshold
        baseline_target = bool(row["candidates"][baseline_selected]["is_target"])
        positive_selected = per_kind["positive"]["selected"]
        target_index = next(i for i, candidate in enumerate(row["candidates"]) if candidate["is_target"])
        positive_poses = per_kind["positive"]["candidate_poses"]
        baseline_pose = pose_success(positive_poses[baseline_selected], row["truth_local_poses"])
        relation_pose = positive_selected is not None and pose_success(positive_poses[positive_selected], row["truth_local_poses"])
        target_pose = pose_success(positive_poses[target_index], row["truth_local_poses"])
        relation_target = positive_selected is not None and bool(row["candidates"][positive_selected]["is_target"])
        baseline_complete = baseline_committed and baseline_target and baseline_pose
        relation_complete = per_kind["positive"]["committed"] and relation_target and relation_pose
        same_type_case = int(row["same_type_visible_candidates"]) >= 2
        records.append({
            "sample_id": row["sample_id"], "target_type": row["target_type"], "same_type_case": same_type_case,
            "baseline_target": baseline_target, "relation_target": relation_target,
            "target_pose_capable": target_pose, "baseline_complete": baseline_complete,
            "relation_complete": relation_complete, "relation_committed": per_kind["positive"]["committed"],
            "relation_selected_object_id": None if positive_selected is None else row["candidates"][positive_selected]["object_id"],
            "relation_resolution": per_kind["positive"]["resolution"],
            "relation_wrong_target": bool(same_type_case and per_kind["positive"]["committed"] and not relation_target),
        })
        absence_commits.append(bool(per_kind["negative"]["committed"]))

    baseline_top1 = sum(record["baseline_target"] for record in records)
    baseline_joint = sum(record["baseline_target"] and record["target_pose_capable"] for record in records)
    baseline_complete = sum(record["baseline_complete"] for record in records)
    relation_top1 = sum(record["relation_target"] for record in records)
    relation_joint = sum(record["relation_target"] and record["target_pose_capable"] for record in records)
    relation_complete = sum(record["relation_complete"] for record in records)
    if (baseline_top1, baseline_joint, baseline_complete) != (44, 34, 22):
        raise ValueError(f"frozen M1 diagnostic drift: {(baseline_top1, baseline_joint, baseline_complete)}")
    resolution_counts = {name: sum(record["relation_resolution"] == name for record in records) for name in (
        "UNIQUE_RELATION_MATCH", "RELATION_COLLISION_APPEARANCE_TIEBREAK", "NO_EXACT_RELATION_MATCH",
    )}
    result = {
        "schema": "blindassist_grail_r0_privileged_relational_oracle_probe_v1",
        "mode": "REVERSIBLE_EXPLORATION_PROJECT_CONSUMED_DEVELOPMENT",
        "claim_ceiling": "synthetic ProcTHOR Development mechanism probe with privileged native metadata; no visual relation extraction, formal test, natural scene, Android, product, or safety claim",
        "frozen_inputs": {
            "positive_denominator": len(rows), "wrong_target_denominator": sum(r["same_type_case"] for r in records),
            "absence_denominator": len(absence_commits), "grail_threshold": threshold,
            "dataset_sha256": sha256_file(dataset), "collection_sha256": sha256_file(collection_path),
            "features_sha256": sha256_file(features_path), "checkpoint_sha256": sha256_file(checkpoint_path),
            "development_result_sha256": sha256_file(development_result_path),
        },
        "m1_reference_only": {
            "referent_top1": baseline_top1, "referent_and_target_pose": baseline_joint,
            "complete_pose": baseline_complete, "strongest_simple_baseline_b1": development["metrics"]["B1"]["pose_success"],
        },
        "grail_r0": {
            "referent_top1": relation_top1, "referent_and_target_pose": relation_joint,
            "complete_pose": relation_complete,
            "wrong_target": sum(record["relation_wrong_target"] for record in records),
            "absence_false_commit": sum(absence_commits),
            "permutation_consistent": sum(permutation), "permutation_denominator": len(permutation),
            "resolution_counts": resolution_counts,
        },
        "causal_diagnostics": {
            "target_pose_oracle": sum(record["target_pose_capable"] for record in records),
            "selector_fail_case_rescue": sum(not r["baseline_target"] and r["relation_target"] for r in records),
            "selector_collateral": sum(r["baseline_target"] and not r["relation_target"] for r in records),
            "complete_fail_case_rescue": sum(not r["baseline_complete"] and r["relation_complete"] for r in records),
            "complete_collateral": sum(r["baseline_complete"] and not r["relation_complete"] for r in records),
        },
        "terminal": "GRAIL_R0_RELATIONAL_INFORMATION_CAN_BREAK_REFERENT_BOTTLENECK"
        if relation_top1 >= 60 and relation_complete > development["metrics"]["B1"]["pose_success"]
        else "GRAIL_R0_RELATIONAL_ORACLE_NO_CLEAR_UPLIFT_STOP",
        "records": records,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.dataset, args.collection, args.features, args.checkpoint, args.development_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("m1_reference_only", "grail_r0", "causal_diagnostics", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
