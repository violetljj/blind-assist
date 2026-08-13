#!/usr/bin/env python3
"""Constraint-first TARO task-evidence scorer selection and Bonn transfer.

R15 selected the largest FIT macro mean and only then applied its hard gates,
which discarded an already-admissible fixed analytic scorer. This successor
freezes the corrected policy: discard candidates that fail any FIT gate first,
then select the remaining candidate with the largest held-parent-out macro.
No Bonn task-evidence target is decoded unless an admissible candidate exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as r15


SCHEMA = "blindassist.taro.task_evidence_constrained_selector_bonn_transfer.v1"


def _macro(candidate: Mapping[str, Any]) -> Mapping[str, float]:
    return candidate.get("fit_lopo_parent_macro", candidate.get("fit_parent_macro"))


def candidate_is_admissible(candidate: Mapping[str, Any]) -> bool:
    macro = _macro(candidate)
    return (
        int(candidate["strict_win_parent_count"]) >= r15.MIN_FIT_STRICT_WIN_PARENTS
        and float(macro["ranker"]) > float(macro["passive"])
        and float(macro["ranker"]) > float(macro["generic"])
    )


def select_admissible_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    admissible = [candidate for candidate in candidates if candidate_is_admissible(candidate)]
    if not admissible:
        return None
    return max(
        admissible,
        key=lambda candidate: (
            float(_macro(candidate)["ranker"]),
            int(candidate["strict_win_parent_count"]),
            candidate["family"] == "ANALYTIC",
            json.dumps(candidate, sort_keys=True),
        ),
    )


def _pairwise_lopo_scores(records: Sequence[Any], penalty: float) -> np.ndarray:
    features = r15.reference_relative_features(records)
    parents = sorted({record.parent_id for record in records})
    scores = np.zeros(len(records), dtype=np.float64)
    for held in parents:
        train_indices = [index for index, record in enumerate(records) if record.parent_id != held]
        test_indices = [index for index, record in enumerate(records) if record.parent_id == held]
        model = r15.fit_pairwise_ranker([records[index] for index in train_indices], features[train_indices], penalty)
        scores[test_indices] = r15.predict_pairwise(features[test_indices], model)
    return scores


def choose_constraint_first(records: Sequence[Any]) -> tuple[dict[str, Any], Any | None, np.ndarray | None]:
    r15_selection, _r15_model, _r15_scores = r15.choose_fit_only_ranker(records)
    candidates = r15_selection["candidates"]
    selected = select_admissible_candidate(candidates)
    summary = {
        "policy": "HARD_FIT_GATES_FIRST_THEN_MAXIMIZE_HELD_PARENT_OUT_MACRO",
        "candidates": candidates,
        "r15_unconstrained_selected": r15_selection["selected"],
        "admissible_candidate_count": sum(candidate_is_admissible(candidate) for candidate in candidates),
        "selected": selected,
    }
    if selected is None:
        return summary, None, None
    if selected["family"] == "ANALYTIC":
        model: Any = str(selected["name"])
        scores = np.asarray([record.analytic[model] for record in records], dtype=np.float64)
    else:
        penalty = float(selected["penalty"])
        features = r15.reference_relative_features(records)
        scores = _pairwise_lopo_scores(records, penalty)
        model = r15.fit_pairwise_ranker(records, features, penalty)
    return summary, model, scores


def evaluate(bonn_root: Path) -> dict[str, Any]:
    fit_records, fit_source, fit_abstained = r15._build_tum_fit()
    selection, model, fit_scores = choose_constraint_first(fit_records)
    if model is None or fit_scores is None:
        result = {
            "schema": SCHEMA,
            "mode": "FIT_ONLY_NO_BONN_TARGET_OPENED",
            "fit_source": fit_source,
            "fit_geometry_abstention_count": fit_abstained,
            "fit_model_selection": selection,
            "confirmation": None,
            "terminal": "STOP_NO_ADMISSIBLE_FIT_SCORER",
            "android_candidate_authorized": False,
            "fresh_confirmation_source_lock_authorized": False,
            "read_boundary": {"bonn_reference_depth_reads": 0, "bonn_neighbor_depth_reads": 0, "network_requests": 0},
            "claim_ceiling": "TUM FIT-only Development evidence; Bonn task-evidence transfer was not opened.",
        }
        result["content_sha256"] = hashlib.sha256(r15.canonical_json_bytes(result)).hexdigest().upper()
        return result

    fit_macro, fit_rows = r15._selection_metrics(fit_records, fit_scores)
    fit_strict_parents = sum(row["strict_win_reference_count"] > 0 for row in fit_rows.values())
    fit_checks = {
        "minimum_fit_parents": len(fit_rows) >= r15.MIN_FIT_PARENTS,
        "minimum_fit_strict_win_parents": fit_strict_parents >= r15.MIN_FIT_STRICT_WIN_PARENTS,
        "fit_ranker_macro_beats_passive": fit_macro["ranker"] > fit_macro["passive"],
        "fit_ranker_macro_beats_generic": fit_macro["ranker"] > fit_macro["generic"],
    }
    r15.require(all(fit_checks.values()), "constraint-first selector returned a non-admissible candidate")

    contexts, records, capability, abstained = r15._bonn_contexts_and_records(bonn_root)
    features = r15.reference_relative_features(records)
    if isinstance(model, str):
        scores_before_depth = np.asarray([record.analytic[model] for record in records], dtype=np.float64)
    else:
        scores_before_depth = r15.predict_pairwise(features, model)

    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    sealed = []
    for reference, indices in sorted(by_reference.items()):
        chosen = max(indices, key=lambda index: (float(scores_before_depth[index]), -records[index].pair.translation_m, records[index].pair.neighbor.frame_id))
        sealed.append({"reference_frame_id": reference, "neighbor_frame_id": records[chosen].pair.neighbor.frame_id, "score": float(scores_before_depth[chosen])})
    sealed_sha = hashlib.sha256(r15.canonical_json_bytes(sealed)).hexdigest().upper()

    # The new task target is opened only after the scorer choices are sealed.
    r15._attach_bonn_targets(records, contexts)
    transfer_macro, transfer_rows = r15._selection_metrics(records, scores_before_depth)
    strict_parents = sum(row["strict_win_reference_count"] > 0 for row in transfer_rows.values())
    reference_count = sum(row["reference_count"] for row in transfer_rows.values())
    checks = {
        "minimum_transfer_references": reference_count >= r15.MIN_CONFIRMATION_REFERENCES,
        "minimum_transfer_parents": len(transfer_rows) >= r15.MIN_CONFIRMATION_PARENTS,
        "minimum_transfer_strict_win_parents": strict_parents >= r15.MIN_CONFIRMATION_STRICT_WIN_PARENTS,
        "scorer_macro_beats_passive": transfer_macro["ranker"] > transfer_macro["passive"],
        "scorer_macro_beats_generic": transfer_macro["ranker"] > transfer_macro["generic"],
        "selection_precedes_neighbor_depth": True,
        "same_one_frame_budget": True,
        "zero_retention_failures_by_union_construction": True,
    }
    terminal = "TASK_EVIDENCE_CONSTRAINED_SELECTOR_BONN_TRANSFER_PASS" if all(checks.values()) else "STOP_CONSTRAINED_SELECTOR_BONN_TRANSFER_FAIL"
    result = {
        "schema": SCHEMA,
        "mode": "TUM_FIT_CONSTRAINT_FIRST_THEN_BONN_CONSUMED_SOURCE_NEW_TASK_TRANSFER",
        "fit_source": fit_source,
        "fit_geometry_abstention_count": fit_abstained,
        "fit_model_selection": selection,
        "fit_frozen_model": r15._model_summary(model),
        "fit_metrics": {"parent_macro": fit_macro, "strict_win_parent_count": fit_strict_parents, "per_parent": fit_rows},
        "fit_checks": fit_checks,
        "transfer": {
            "source_family": "BONN_RGBD_DYNAMIC",
            "source_root": str(bonn_root),
            "prior_occupancy_task_outcome_opened": True,
            "task_evidence_outcome_previously_opened": False,
            "pose_capability": capability,
            "geometry_abstention_count": abstained,
            "evaluated_reference_count": reference_count,
            "selection_identity_sha256_before_neighbor_depth": sealed_sha,
            "parent_macro": transfer_macro,
            "strict_win_parent_count": strict_parents,
            "per_parent": transfer_rows,
            "checks": checks,
        },
        "terminal": terminal,
        "android_candidate_authorized": False,
        "fresh_confirmation_source_lock_authorized": terminal == "TASK_EVIDENCE_CONSTRAINED_SELECTOR_BONN_TRANSFER_PASS",
        "read_boundary": {"rgb_payload_decodes": 0, "bonn_neighbor_depth_in_selection": False, "network_requests": 0, "r11_reads": 0},
        "claim_ceiling": "TUM FIT plus Bonn consumed-source/new-task Development transfer; not fresh-source confirmation, collision correctness, Android, product, default-App, or safety evidence.",
    }
    result["content_sha256"] = hashlib.sha256(r15.canonical_json_bytes(result)).hexdigest().upper()
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
    parser.add_argument("--bonn-root", type=Path, default=r15.DEFAULT_BONN_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.bonn_root.resolve())
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
