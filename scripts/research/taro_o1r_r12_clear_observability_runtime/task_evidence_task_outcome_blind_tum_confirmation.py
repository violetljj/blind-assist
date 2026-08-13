#!/usr/bin/env python3
"""One-shot task-outcome-blind TUM confirmation for the frozen TARO policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as balanced
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_hybrid_development as policy_runtime
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_task_outcome_blind_tum_confirmation.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/taro/TARO_TASK_EVIDENCE_TASK_OUTCOME_BLIND_TUM_CONFIRMATION_LOCK_2026-08-13.json"
DEFAULT_MANIFEST = REPO_ROOT / "docs/research/taro/TARO_TASK_EVIDENCE_TASK_OUTCOME_BLIND_TUM_CONFIRMATION_COHORT_R0_2026-08-13.json"
FROZEN_POLICY = {
    "family": "NORMALIZED_POSE_TASK_BLEND",
    "task_term": "visible_unknown",
    "task_weight": 0.8,
    "rotation_weight": 0.05,
}
MAX_REFERENCES_PER_PARENT = 5
MIN_REFERENCES = 16
MIN_PARENTS = 4
MIN_STRICT_WIN_PARENTS = 3


def evaluate(lock_path: Path, manifest_path: Path, verify_archive_hashes: bool = True) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    shared.require(lock["status"].startswith("FROZEN_BEFORE_"), "confirmation lock status drift")
    shared.require(lock["frozen_policy"]["family"] == FROZEN_POLICY["family"], "policy family drift")
    shared.require(float(lock["frozen_policy"]["task_weight"]) == FROZEN_POLICY["task_weight"], "task weight drift")
    shared.require(float(lock["frozen_policy"]["rotation_weight"]) == FROZEN_POLICY["rotation_weight"], "rotation weight drift")

    frames, assets, source = tum.load_outcome_blind_roster((manifest_path,), verify_archive_hashes)
    selected, capability = balanced.select_pose_capable_references(frames, MAX_REFERENCES_PER_PARENT)
    shared.require(capability["eligible_parent_count"] >= MIN_PARENTS, "confirmation source lacks four pose-capable parents")

    # Reference depth is a runtime scorer input. Neighbor depth remains unopened.
    reference_observations = scorer._load_observations([row.reference.frame_id for row in selected], assets)
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    abstained = 0
    for row in selected:
        low, points, valid, _coverage = reference_observations[row.reference.frame_id]
        k = bonn._scaled_intrinsics(assets[row.reference.frame_id].intrinsics, tum.NATIVE_SIZE_WH, tum.LOW_SIZE_WH)
        queries = oracle._queries(row.reference, low, k)
        if queries is None:
            abstained += 1
            continue
        static = oracle.query_evidence_cells(points, valid, queries)
        context = scorer.ReferenceContext(row, low, points, valid, k, queries, static)
        contexts[row.reference.frame_id] = context
        for pair in oracle.pose_proposal_pairs(row):
            features, analytic = scorer.source_time_candidate_features(context, pair)
            records.append(scorer.CandidateRecord(row.reference.parent_id, "TASK_OUTCOME_BLIND_CONFIRMATION", row.reference.frame_id, pair, features, analytic))

    scores_before_neighbor_depth = policy_runtime.policy_scores(records, FROZEN_POLICY)
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    sealed = []
    for reference, indices in sorted(by_reference.items()):
        chosen = max(indices, key=lambda index: (float(scores_before_neighbor_depth[index]), -records[index].pair.translation_m, records[index].pair.neighbor.frame_id))
        sealed.append({
            "reference_frame_id": reference,
            "neighbor_frame_id": records[chosen].pair.neighbor.frame_id,
            "score": float(scores_before_neighbor_depth[chosen]),
        })
    sealed_sha = hashlib.sha256(shared.canonical_json_bytes(sealed)).hexdigest().upper()

    # Target-side replay begins only after the exact selections above are sealed.
    neighbor_observations = scorer._load_observations([record.pair.neighbor.frame_id for record in records], assets)
    scorer._attach_targets(records, contexts, neighbor_observations)
    macro, per_parent = shared._selection_metrics(records, scores_before_neighbor_depth)
    strict_parents = sum(row["strict_win_reference_count"] > 0 for row in per_parent.values())
    reference_count = sum(row["reference_count"] for row in per_parent.values())
    checks = {
        "minimum_evaluated_references": reference_count >= MIN_REFERENCES,
        "minimum_evaluated_parents": len(per_parent) >= MIN_PARENTS,
        "minimum_strict_win_parents": strict_parents >= MIN_STRICT_WIN_PARENTS,
        "policy_parent_macro_beats_passive": macro["ranker"] > macro["passive"],
        "policy_parent_macro_beats_generic": macro["ranker"] > macro["generic"],
        "selection_precedes_neighbor_depth": True,
        "same_one_extra_frame_budget": True,
        "zero_retention_failures_by_union_construction": True,
    }
    terminal = "TASK_EVIDENCE_TASK_OUTCOME_BLIND_TUM_CONFIRMATION_PASS" if all(checks.values()) else "STOP_TASK_EVIDENCE_TASK_OUTCOME_BLIND_TUM_CONFIRMATION_FAIL"
    result = {
        "schema": SCHEMA,
        "mode": "TASK_OUTCOME_BLIND_PARENT_DISJOINT_CONFIRMATION",
        "lock": {"path": str(lock_path), "sha256": tum.sha256_file(lock_path), "token": lock["token"]},
        "manifest": {"path": str(manifest_path), "sha256": tum.sha256_file(manifest_path)},
        "frozen_policy": FROZEN_POLICY,
        "source": source | {"analysis_role": "TASK_OUTCOME_BLIND_CONFIRMATION"},
        "pose_pair_capability": capability,
        "geometry_abstention_count": abstained,
        "evaluated_reference_count": reference_count,
        "selection_identity_sha256_before_neighbor_depth": sealed_sha,
        "metrics": {"parent_macro": macro, "strict_win_parent_count": strict_parents, "per_parent": per_parent},
        "checks": checks,
        "terminal": terminal,
        "android_candidate_authorized": terminal == "TASK_EVIDENCE_TASK_OUTCOME_BLIND_TUM_CONFIRMATION_PASS",
        "read_boundary": {
            "rgb_payload_decodes": 0,
            "reference_depth_is_scorer_input": True,
            "neighbor_depth_reads_before_selection_seal": 0,
            "neighbor_depth_reads_after_selection_seal": len({record.pair.neighbor.frame_id for record in records}),
            "model_runs": 0,
            "training_steps": 0,
            "network_requests": 0,
            "r11_reads": 0,
        },
        "claim_ceiling": "Task-outcome-blind parent-disjoint TUM confirmation of a source-time frame selector; not collision correctness, Android runtime, product, default-App, or safety evidence.",
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
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-archive-hash-verification", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.lock.resolve(), args.manifest.resolve(), not args.skip_archive_hash_verification)
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
