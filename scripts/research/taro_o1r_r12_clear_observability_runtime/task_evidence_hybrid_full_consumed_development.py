#!/usr/bin/env python3
"""Full-consumed TUM/Bonn Development for the frozen TARO hybrid family.

This keeps R17's candidate family and gates unchanged, but uses every TUM parent
whose task-evidence target was already consumed by R13/R14 instead of retaining
the obsolete six-parent FIT split from the earlier scorer experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as balanced
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_hybrid_development as r17
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_hybrid_full_consumed_development.v1"


def _build_all_consumed_tum() -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    frames, assets, source = tum.load_outcome_blind_roster(tum.DEFAULT_MANIFESTS, verify_archive_hashes=False)
    selected, capability = balanced.select_pose_capable_references(frames, oracle.MAX_REFERENCES_PER_PARENT)
    role_by_parent = {row["parent_id"]: row["cohort_role"] for row in source["parents"]}
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
            role = f"CONSUMED_{role_by_parent[row.reference.parent_id]}_DEVELOPMENT"
            records.append(scorer.CandidateRecord(row.reference.parent_id, role, row.reference.frame_id, pair, features, analytic))
    observations = scorer._load_observations([record.pair.neighbor.frame_id for record in records], assets)
    scorer._attach_targets(records, contexts, observations)
    return records, {"source": source, "capability": capability}, abstained


def evaluate(bonn_root: Path) -> dict[str, Any]:
    tum_records, tum_source, tum_abstained = _build_all_consumed_tum()
    bonn_contexts, bonn_records, bonn_capability, bonn_abstained = shared._bonn_contexts_and_records(bonn_root)
    shared._attach_bonn_targets(bonn_records, bonn_contexts)

    candidates = []
    for policy in r17.candidate_policy_specs():
        metrics = {
            "TUM_RGBD": r17._source_metrics(tum_records, policy),
            "BONN_RGBD_DYNAMIC": r17._source_metrics(bonn_records, policy),
        }
        candidates.append({"policy": policy, "metrics": metrics, "admissible": r17.policy_is_admissible(metrics)})
    admissible = [row for row in candidates if row["admissible"]]
    selected = max(admissible, key=r17._selection_value) if admissible else None
    compact = [
        {
            "policy": row["policy"],
            "admissible": row["admissible"],
            "metrics": {
                source: {
                    "parent_macro": metrics["parent_macro"],
                    "strict_win_parent_count": metrics["strict_win_parent_count"],
                    "parent_count": metrics["parent_count"],
                    "reference_count": metrics["reference_count"],
                }
                for source, metrics in row["metrics"].items()
            },
        }
        for row in candidates
    ]
    terminal = "TASK_EVIDENCE_HYBRID_FULL_CONSUMED_DEVELOPMENT_PASS" if selected is not None else "STOP_TASK_EVIDENCE_HYBRID_FULL_CONSUMED_NO_POLICY"
    result = {
        "schema": SCHEMA,
        "mode": "FULL_CONSUMED_MULTI_SOURCE_DEVELOPMENT",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside frozen body/path capsules; UNKNOWN remains unknown.",
        "candidate_family_identity_sha256": hashlib.sha256(shared.canonical_json_bytes(r17.candidate_policy_specs())).hexdigest().upper(),
        "candidate_family_unchanged_from_r17": True,
        "gate_unchanged_from_r17": True,
        "selection_policy": "require every source family to beat passive and generic with at least four strict-win parents; then maximize worst-source relative gain over generic",
        "sources": {
            "TUM_RGBD": {"disposition": "ALL_TASK_EVIDENCE_OUTCOMES_CONSUMED_BY_R13_R14", "source": tum_source, "geometry_abstention_count": tum_abstained},
            "BONN_RGBD_DYNAMIC": {"disposition": "TASK_EVIDENCE_OUTCOMES_CONSUMED_BY_R16_R17", "pose_capability": bonn_capability, "geometry_abstention_count": bonn_abstained},
        },
        "candidate_summaries": compact,
        "admissible_candidate_count": len(admissible),
        "selected": selected,
        "terminal": terminal,
        "fresh_confirmation_source_lock_authorized": selected is not None,
        "android_candidate_authorized": False,
        "read_boundary": {"rgb_payload_decodes": 0, "network_requests": 0, "r11_reads": 0},
        "claim_ceiling": "Fully consumed TUM/Bonn Development only; not fresh Confirmation, collision correctness, Android, product, default-App, or safety evidence.",
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
    parser.add_argument("--bonn-root", type=Path, default=shared.DEFAULT_BONN_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.bonn_root.resolve())
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
