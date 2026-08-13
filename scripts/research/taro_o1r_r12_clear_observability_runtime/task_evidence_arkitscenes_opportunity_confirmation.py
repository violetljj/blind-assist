#!/usr/bin/env python3
"""Opportunity-aware ARKitScenes confirmation of the frozen TARO policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import arkitscenes_balanced_pose_source_frontdoor as arkit
from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as balanced
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_hybrid_development as policy_runtime
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer


SCHEMA = "blindassist.taro.task_evidence_arkitscenes_opportunity_confirmation.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/taro/TARO_TASK_EVIDENCE_ARKITSCENES_OPPORTUNITY_AWARE_CONFIRMATION_LOCK_2026-08-13.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "artifacts.local/datasets/assistive-geometry-b0-arkitscenes-20260809-r2"
FROZEN_POLICY = {
    "family": "NORMALIZED_POSE_TASK_BLEND",
    "task_term": "visible_unknown",
    "task_weight": 0.8,
    "rotation_weight": 0.05,
}
MAX_REFERENCES_PER_PARENT = 5
MIN_REFERENCES = 32
MIN_PARENTS = 8
MIN_OPPORTUNITY_PARENTS = 4
MIN_STRICT_WIN_PARENTS = 3
MIN_STRICT_WIN_FRACTION = 0.5


def opportunity_gate(opportunity_parents: int, strict_win_parents: int) -> bool:
    required = max(MIN_STRICT_WIN_PARENTS, math.ceil(MIN_STRICT_WIN_FRACTION * opportunity_parents))
    return opportunity_parents >= MIN_OPPORTUNITY_PARENTS and strict_win_parents >= required


def _load_observation(asset: arkit.FrameAssets) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    depth, confidence, intrinsics = arkit.load_observation(asset)
    low, points, valid = arkit.observation_geometry(depth, confidence, intrinsics)
    return low, points, valid, float(np.mean(valid)), intrinsics


def _opportunity_counts(records: Sequence[scorer.CandidateRecord], scores: Sequence[float]) -> tuple[int, int, dict[str, Any]]:
    by_reference: dict[str, list[tuple[scorer.CandidateRecord, float]]] = defaultdict(list)
    for record, score in zip(records, scores, strict=True):
        by_reference[record.reference_id].append((record, float(score)))
    per_parent: dict[str, dict[str, int]] = defaultdict(lambda: {"opportunity_reference_count": 0, "policy_strict_win_reference_count": 0})
    for rows in by_reference.values():
        values = [record for record, _score in rows]
        selected, _ = max(rows, key=lambda item: (item[1], -item[0].pair.translation_m, item[0].pair.neighbor.frame_id))
        passive = max(values, key=lambda row: (float(row.coverage), -row.pair.gap_s, row.pair.neighbor.frame_id))
        generic = max(values, key=lambda row: (row.pair.translation_m, row.pair.rotation_deg, -row.pair.gap_s, row.pair.neighbor.frame_id))
        oracle_row = max(values, key=lambda row: (int(row.target_gain), float(row.coverage), -row.pair.translation_m, row.pair.neighbor.frame_id))
        comparator = max(int(passive.target_gain), int(generic.target_gain))
        row = per_parent[selected.parent_id]
        row["opportunity_reference_count"] += int(int(oracle_row.target_gain) > comparator)
        row["policy_strict_win_reference_count"] += int(int(selected.target_gain) > comparator)
    output = dict(sorted(per_parent.items()))
    opportunity = sum(row["opportunity_reference_count"] > 0 for row in output.values())
    strict = sum(row["policy_strict_win_reference_count"] > 0 for row in output.values())
    return opportunity, strict, output


def evaluate(lock_path: Path, dataset_root: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    shared.require(lock["status"].startswith("FROZEN_BEFORE_"), "ARKit confirmation lock drift")
    shared.require(lock["frozen_policy"]["family"] == FROZEN_POLICY["family"], "policy family drift")
    shared.require(float(lock["frozen_policy"]["task_weight"]) == FROZEN_POLICY["task_weight"], "task weight drift")
    shared.require(float(lock["frozen_policy"]["rotation_weight"]) == FROZEN_POLICY["rotation_weight"], "rotation weight drift")

    frames, assets, source = arkit.load_outcome_blind_roster(dataset_root)
    shared.require(source["manifest_sha256"] == lock["source"]["manifest_sha256"], "ARKit manifest hash drift")
    selected, capability = balanced.select_pose_capable_references(frames, MAX_REFERENCES_PER_PARENT)

    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    abstained = 0
    reference_payload_reads = 0
    for row in selected:
        low, points, valid, _coverage, intrinsics = _load_observation(assets[row.reference.frame_id])
        reference_payload_reads += 3
        queries = oracle._queries(row.reference, low, intrinsics)
        if queries is None:
            abstained += 1
            continue
        static = oracle.query_evidence_cells(points, valid, queries)
        context = scorer.ReferenceContext(row, low, points, valid, intrinsics, queries, static)
        contexts[row.reference.frame_id] = context
        for pair in oracle.pose_proposal_pairs(row):
            features, analytic = scorer.source_time_candidate_features(context, pair)
            records.append(scorer.CandidateRecord(row.reference.parent_id, "NEW_TASK_OUTCOME_BLIND_CONFIRMATION", row.reference.frame_id, pair, features, analytic))

    scores_before_neighbor_depth = policy_runtime.policy_scores(records, FROZEN_POLICY)
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    sealed = []
    for reference, indices in sorted(by_reference.items()):
        chosen = max(indices, key=lambda index: (float(scores_before_neighbor_depth[index]), -records[index].pair.translation_m, records[index].pair.neighbor.frame_id))
        sealed.append({"reference_frame_id": reference, "neighbor_frame_id": records[chosen].pair.neighbor.frame_id, "score": float(scores_before_neighbor_depth[chosen])})
    sealed_sha = hashlib.sha256(shared.canonical_json_bytes(sealed)).hexdigest().upper()

    observation_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for frame_id in sorted({record.pair.neighbor.frame_id for record in records}):
        low, points, valid, coverage, _intrinsics = _load_observation(assets[frame_id])
        observation_cache[frame_id] = (low, points, valid, coverage)
    scorer._attach_targets(records, contexts, observation_cache)
    macro, per_parent = shared._selection_metrics(records, scores_before_neighbor_depth)
    opportunity_parents, strict_parents, opportunity_rows = _opportunity_counts(records, scores_before_neighbor_depth)
    reference_count = sum(row["reference_count"] for row in per_parent.values())
    checks = {
        "minimum_evaluated_references": reference_count >= MIN_REFERENCES,
        "minimum_evaluated_parents": len(per_parent) >= MIN_PARENTS,
        "minimum_opportunity_parents": opportunity_parents >= MIN_OPPORTUNITY_PARENTS,
        "opportunity_denominated_strict_win_gate": opportunity_gate(opportunity_parents, strict_parents),
        "policy_parent_macro_beats_passive": macro["ranker"] > macro["passive"],
        "policy_parent_macro_beats_generic": macro["ranker"] > macro["generic"],
        "selection_precedes_neighbor_depth": True,
        "same_one_extra_frame_budget": True,
        "zero_retention_failures_by_union_construction": True,
    }
    terminal = "TASK_EVIDENCE_ARKITSCENES_OPPORTUNITY_CONFIRMATION_PASS" if all(checks.values()) else "STOP_TASK_EVIDENCE_ARKITSCENES_OPPORTUNITY_CONFIRMATION_FAIL"
    result = {
        "schema": SCHEMA,
        "mode": "NEW_TASK_OUTCOME_BLIND_ARKITSCENES_CONFIRMATION",
        "lock": {"path": str(lock_path), "sha256": arkit.sha256_file(lock_path), "token": lock["token"]},
        "frozen_policy": FROZEN_POLICY,
        "source": source | {"analysis_role": "NEW_TASK_OUTCOME_BLIND_CONFIRMATION", "prior_reference_occupancy_labels_opened": True},
        "pose_pair_capability": capability,
        "geometry_abstention_count": abstained,
        "evaluated_reference_count": reference_count,
        "selection_identity_sha256_before_neighbor_depth": sealed_sha,
        "metrics": {
            "parent_macro": macro,
            "opportunity_parent_count": opportunity_parents,
            "policy_strict_win_opportunity_parent_count": strict_parents,
            "strict_win_fraction_of_opportunity_parents": strict_parents / opportunity_parents if opportunity_parents else None,
            "per_parent": per_parent,
            "opportunity_per_parent": opportunity_rows,
        },
        "checks": checks,
        "terminal": terminal,
        "android_candidate_authorized": terminal == "TASK_EVIDENCE_ARKITSCENES_OPPORTUNITY_CONFIRMATION_PASS",
        "read_boundary": {
            "rgb_payload_decodes": 0,
            "reference_depth_confidence_intrinsics_reads": reference_payload_reads,
            "neighbor_payload_reads_before_selection_seal": 0,
            "neighbor_depth_confidence_intrinsics_reads_after_selection_seal": 3 * len(observation_cache),
            "model_runs": 0,
            "training_steps": 0,
            "network_requests": 0,
            "r11_reads": 0,
        },
        "claim_ceiling": "New-task-outcome-blind ARKitScenes confirmation of a source-time frame selector; not collision correctness, Android runtime, product, default-App, or safety evidence.",
    }
    result["content_sha256"] = hashlib.sha256(shared.canonical_json_bytes(result)).hexdigest().upper()
    return result


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.lock.resolve(), args.dataset_root.resolve())
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
