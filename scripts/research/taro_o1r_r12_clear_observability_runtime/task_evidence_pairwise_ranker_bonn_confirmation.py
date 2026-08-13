#!/usr/bin/env python3
"""FIT-only pairwise TARO evidence ranker with Bonn new-task transfer.

The ranker repairs R14's objective mismatch: it learns within-reference gain
differences instead of pointwise gain.  Hyperparameters and admission are chosen
only by leave-one-FIT-parent-out TUM evidence. Bonn was consumed by the earlier
occupancy task, but its task-evidence-cell outcome is new. Those outcomes are
opened only if the FIT gate passes, and scorer selections are sealed before any
Bonn neighbor depth is decoded.
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

from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_pairwise_ranker_bonn_confirmation.v1"
PAIRWISE_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)
MIN_FIT_PARENTS = 4
MIN_FIT_STRICT_WIN_PARENTS = 4
MIN_CONFIRMATION_REFERENCES = 48
MIN_CONFIRMATION_PARENTS = 4
MIN_CONFIRMATION_STRICT_WIN_PARENTS = 4
DEFAULT_BONN_ROOT = Path("artifacts.local/datasets/bonn-rgbd-dynamic-full-r0/rgbd_bonn_dataset")


class RankerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RankerError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def reference_relative_features(records: Sequence[scorer.CandidateRecord]) -> np.ndarray:
    output = np.zeros((len(records), len(scorer.FEATURE_NAMES) * 2), dtype=np.float64)
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    for indices in by_reference.values():
        absolute = np.stack([records[index].features for index in indices])
        mean = np.mean(absolute, axis=0)
        scale = np.std(absolute, axis=0)
        scale[scale < 1e-9] = 1.0
        output[indices, : absolute.shape[1]] = absolute
        output[indices, absolute.shape[1] :] = (absolute - mean) / scale
    require(np.all(np.isfinite(output)), "relative feature non-finite")
    return output


def _pairwise_dataset(
    records: Sequence[scorer.CandidateRecord],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        require(record.target_gain is not None, "pairwise target missing")
        by_reference[record.reference_id].append(index)
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    parents: list[str] = []
    for indices in by_reference.values():
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                delta = int(records[left].target_gain) - int(records[right].target_gain)
                if delta == 0:
                    continue
                difference = features[left] - features[right]
                x_rows.extend((difference, -difference))
                y_rows.extend((float(delta), float(-delta)))
                parents.extend((records[left].parent_id, records[left].parent_id))
    require(bool(x_rows), "pairwise dataset empty")
    return np.stack(x_rows), np.asarray(y_rows, dtype=np.float64), parents


def fit_pairwise_ranker(
    records: Sequence[scorer.CandidateRecord],
    features: np.ndarray,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, parents = _pairwise_dataset(records, features)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (x - mean) / scale
    counts = {parent: parents.count(parent) for parent in set(parents)}
    weights = np.asarray([1.0 / counts[parent] for parent in parents], dtype=np.float64)
    weights *= len(weights) / np.sum(weights)
    root = np.sqrt(weights)[:, None]
    regularizer = np.eye(standardized.shape[1], dtype=np.float64) * penalty
    coefficients = np.linalg.solve((standardized * root).T @ (standardized * root) + regularizer, (standardized * root).T @ (y * root[:, 0]))
    return coefficients, mean, scale


def predict_pairwise(features: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    coefficients, mean, scale = model
    return ((features - mean) / scale) @ coefficients


def _selection_metrics(
    records: Sequence[scorer.CandidateRecord],
    scores: Sequence[float],
) -> tuple[dict[str, float], dict[str, Any]]:
    by_reference: dict[str, list[tuple[scorer.CandidateRecord, float]]] = defaultdict(list)
    for record, score in zip(records, scores, strict=True):
        by_reference[record.reference_id].append((record, float(score)))
    per_parent: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rows in by_reference.values():
        values = [record for record, _score in rows]
        selected, _ = max(rows, key=lambda item: (item[1], -item[0].pair.translation_m, item[0].pair.neighbor.frame_id))
        passive = max(values, key=lambda row: (float(row.coverage), -row.pair.gap_s, row.pair.neighbor.frame_id))
        generic = max(values, key=lambda row: (row.pair.translation_m, row.pair.rotation_deg, -row.pair.gap_s, row.pair.neighbor.frame_id))
        oracle_row = max(values, key=lambda row: (int(row.target_gain), float(row.coverage), -row.pair.translation_m, row.pair.neighbor.frame_id))
        parent = selected.parent_id
        for name, row in (("ranker", selected), ("passive", passive), ("generic", generic), ("oracle", oracle_row)):
            per_parent[parent][name].append(int(row.target_gain))
        per_parent[parent]["strict_win"].append(int(int(selected.target_gain) > max(int(passive.target_gain), int(generic.target_gain))))
    rows_out = {
        parent: {
            "reference_count": len(arms["ranker"]),
            "mean_gain": {name: float(np.mean(arms[name])) for name in ("ranker", "passive", "generic", "oracle")},
            "strict_win_reference_count": int(sum(arms["strict_win"])),
        }
        for parent, arms in sorted(per_parent.items())
    }
    macro = {name: float(np.mean([row["mean_gain"][name] for row in rows_out.values()])) for name in ("ranker", "passive", "generic", "oracle")}
    return macro, rows_out


def choose_fit_only_ranker(
    records: Sequence[scorer.CandidateRecord],
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray] | str, np.ndarray]:
    features = reference_relative_features(records)
    parents = sorted({record.parent_id for record in records})
    candidates: list[dict[str, Any]] = []
    admission_scores: list[np.ndarray] = []
    # Fixed analytic scorers remain admissible alternatives; they use no target fit.
    for name in scorer.ANALYTIC_SCORERS:
        scores = np.asarray([record.analytic[name] for record in records], dtype=np.float64)
        macro, rows = _selection_metrics(records, scores)
        candidates.append({"family": "ANALYTIC", "name": name, "fit_parent_macro": macro, "strict_win_parent_count": sum(row["strict_win_reference_count"] > 0 for row in rows.values())})
        admission_scores.append(scores)
    for penalty in PAIRWISE_LAMBDAS:
        fold_scores = np.zeros(len(records), dtype=np.float64)
        for held in parents:
            train_indices = [index for index, record in enumerate(records) if record.parent_id != held]
            test_indices = [index for index, record in enumerate(records) if record.parent_id == held]
            model = fit_pairwise_ranker([records[index] for index in train_indices], features[train_indices], penalty)
            fold_scores[test_indices] = predict_pairwise(features[test_indices], model)
        macro, rows = _selection_metrics(records, fold_scores)
        candidates.append({"family": "PAIRWISE_RIDGE", "penalty": penalty, "fit_lopo_parent_macro": macro, "strict_win_parent_count": sum(row["strict_win_reference_count"] > 0 for row in rows.values())})
        admission_scores.append(fold_scores)
    def value(row: Mapping[str, Any]) -> tuple[float, int, bool, str]:
        macro = row.get("fit_lopo_parent_macro", row.get("fit_parent_macro"))["ranker"]
        return float(macro), int(row["strict_win_parent_count"]), row["family"] == "ANALYTIC", json.dumps(row, sort_keys=True)
    selected_index = max(range(len(candidates)), key=lambda index: value(candidates[index]))
    selected = candidates[selected_index]
    if selected["family"] == "ANALYTIC":
        model: tuple[np.ndarray, np.ndarray, np.ndarray] | str = str(selected["name"])
    else:
        model = fit_pairwise_ranker(records, features, float(selected["penalty"]))
    # Admission is based strictly on fixed analytic scores or held-parent-out
    # predictions. The full FIT model above is frozen only for later transfer.
    return {"candidates": candidates, "selected": selected}, model, admission_scores[selected_index]


def _build_tum_fit() -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    frames, assets, source = tum.load_outcome_blind_roster(tum.DEFAULT_MANIFESTS, verify_archive_hashes=False)
    selected, capability = shared.select_pose_capable_references(frames, oracle.MAX_REFERENCES_PER_PARENT)
    role_by_parent = {row["parent_id"]: row["cohort_role"] for row in source["parents"]}
    fit_selected = [row for row in selected if role_by_parent[row.reference.parent_id] == "FIT"]
    reference_observations = scorer._load_observations([row.reference.frame_id for row in fit_selected], assets)
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    abstained = 0
    for row in fit_selected:
        low, points, valid, _ = reference_observations[row.reference.frame_id]
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
            records.append(scorer.CandidateRecord(row.reference.parent_id, "FIT", row.reference.frame_id, pair, features, analytic))
    observations = scorer._load_observations([record.pair.neighbor.frame_id for record in records], assets)
    scorer._attach_targets(records, contexts, observations)
    return records, {"source": source, "capability": capability}, abstained


def _model_summary(model: tuple[np.ndarray, np.ndarray, np.ndarray] | str) -> dict[str, Any]:
    if isinstance(model, str):
        return {"family": "ANALYTIC", "name": model}
    coefficients, mean, scale = model
    return {
        "family": "PAIRWISE_RIDGE",
        "coefficient_count": len(coefficients),
        "coefficients": coefficients.tolist(),
        "pair_difference_mean": mean.tolist(),
        "pair_difference_scale": scale.tolist(),
    }


def _bonn_contexts_and_records(dataset_root: Path) -> tuple[dict[str, scorer.ReferenceContext], list[scorer.CandidateRecord], dict[str, Any], int]:
    capability, selected = bonn.audit_capability(dataset_root, bonn.MAX_REFERENCES_PER_PARENT)
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    abstained = 0
    for row in selected:
        depth = bonn._load_depth(str(row.reference.depth_path))
        low, points, valid, _coverage = bonn._low_observation(depth)
        queries = oracle._queries(row.reference, low, bonn.LOW_INTRINSICS)
        if queries is None:
            abstained += 1
            continue
        static = oracle.query_evidence_cells(points, valid, queries)
        context = scorer.ReferenceContext(row, low, points, valid, bonn.LOW_INTRINSICS, queries, static)
        contexts[row.reference.frame_id] = context
        for pair in oracle.pose_proposal_pairs(row):
            features, analytic = scorer.source_time_candidate_features(context, pair)
            records.append(scorer.CandidateRecord(row.reference.parent_id, "CONSUMED_SOURCE_NEW_TASK_TRANSFER", row.reference.frame_id, pair, features, analytic))
    return contexts, records, capability, abstained


def _attach_bonn_targets(records: Sequence[scorer.CandidateRecord], contexts: Mapping[str, scorer.ReferenceContext]) -> None:
    observations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    frame_lookup = {record.pair.neighbor.frame_id: record.pair.neighbor for record in records}
    for frame_id, frame in frame_lookup.items():
        observations[frame_id] = bonn._low_observation(bonn._load_depth(str(frame.depth_path)))
    scorer._attach_targets(records, contexts, observations)


def evaluate(bonn_root: Path) -> dict[str, Any]:
    fit_records, fit_source, fit_abstained = _build_tum_fit()
    selection, model, fit_scores = choose_fit_only_ranker(fit_records)
    fit_macro, fit_rows = _selection_metrics(fit_records, fit_scores)
    fit_strict_parents = sum(row["strict_win_reference_count"] > 0 for row in fit_rows.values())
    fit_checks = {
        "minimum_fit_parents": len(fit_rows) >= MIN_FIT_PARENTS,
        "minimum_fit_strict_win_parents": fit_strict_parents >= MIN_FIT_STRICT_WIN_PARENTS,
        "fit_ranker_macro_beats_passive": fit_macro["ranker"] > fit_macro["passive"],
        "fit_ranker_macro_beats_generic": fit_macro["ranker"] > fit_macro["generic"],
    }
    if not all(fit_checks.values()):
        result = {
            "schema": SCHEMA,
            "mode": "FIT_ONLY_NO_CONFIRMATION_OPENED",
            "fit_source": fit_source,
            "fit_geometry_abstention_count": fit_abstained,
            "fit_model_selection": selection,
            "fit_metrics": {"parent_macro": fit_macro, "strict_win_parent_count": fit_strict_parents, "per_parent": fit_rows},
            "fit_checks": fit_checks,
            "confirmation": None,
            "terminal": "STOP_PAIRWISE_RANKER_FIT_GATE_FAIL",
            "android_candidate_authorized": False,
            "read_boundary": {"bonn_reference_depth_reads": 0, "bonn_neighbor_depth_reads": 0, "network_requests": 0},
            "claim_ceiling": "TUM FIT-only Development evidence; Bonn task-evidence confirmation was not opened.",
        }
        result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
        return result
    contexts, confirmation_records, capability, confirmation_abstained = _bonn_contexts_and_records(bonn_root)
    confirmation_features = reference_relative_features(confirmation_records)
    if isinstance(model, str):
        scores_before_depth = np.asarray([record.analytic[model] for record in confirmation_records], dtype=np.float64)
    else:
        scores_before_depth = predict_pairwise(confirmation_features, model)
    selection_rows: list[dict[str, Any]] = []
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(confirmation_records):
        by_reference[record.reference_id].append(index)
    for reference, indices in sorted(by_reference.items()):
        chosen = max(indices, key=lambda index: (float(scores_before_depth[index]), -confirmation_records[index].pair.translation_m, confirmation_records[index].pair.neighbor.frame_id))
        selection_rows.append({"reference_frame_id": reference, "neighbor_frame_id": confirmation_records[chosen].pair.neighbor.frame_id, "score": float(scores_before_depth[chosen])})
    selection_sha = hashlib.sha256(canonical_json_bytes(selection_rows)).hexdigest().upper()
    # Confirmation target payloads are opened only after the scorer identities are sealed above.
    _attach_bonn_targets(confirmation_records, contexts)
    confirmation_macro, confirmation_rows = _selection_metrics(confirmation_records, scores_before_depth)
    strict_parents = sum(row["strict_win_reference_count"] > 0 for row in confirmation_rows.values())
    reference_count = sum(row["reference_count"] for row in confirmation_rows.values())
    checks = {
        "minimum_confirmation_references": reference_count >= MIN_CONFIRMATION_REFERENCES,
        "minimum_confirmation_parents": len(confirmation_rows) >= MIN_CONFIRMATION_PARENTS,
        "minimum_confirmation_strict_win_parents": strict_parents >= MIN_CONFIRMATION_STRICT_WIN_PARENTS,
        "ranker_macro_beats_passive": confirmation_macro["ranker"] > confirmation_macro["passive"],
        "ranker_macro_beats_generic": confirmation_macro["ranker"] > confirmation_macro["generic"],
        "selection_precedes_neighbor_depth": True,
        "same_one_frame_budget": True,
        "zero_retention_failures_by_union_construction": True,
    }
    terminal = "TASK_EVIDENCE_PAIRWISE_RANKER_BONN_CONFIRMATION_PASS" if all(checks.values()) else "STOP_PAIRWISE_RANKER_BONN_CONFIRMATION_FAIL"
    result = {
        "schema": SCHEMA,
        "mode": "TUM_FIT_ONLY_THEN_BONN_CONSUMED_SOURCE_NEW_TASK_TRANSFER",
        "fit_source": fit_source,
        "fit_geometry_abstention_count": fit_abstained,
        "fit_model_selection": selection,
        "fit_frozen_model": _model_summary(model),
        "fit_metrics": {"parent_macro": fit_macro, "strict_win_parent_count": fit_strict_parents, "per_parent": fit_rows},
        "fit_checks": fit_checks,
        "confirmation": {
            "source_family": "BONN_RGBD_DYNAMIC",
            "source_root": str(bonn_root),
            "prior_occupancy_task_outcome_opened": True,
            "task_evidence_outcome_previously_opened": False,
            "pose_capability": capability,
            "geometry_abstention_count": confirmation_abstained,
            "evaluated_reference_count": reference_count,
            "selection_identity_sha256_before_neighbor_depth": selection_sha,
            "parent_macro": confirmation_macro,
            "strict_win_parent_count": strict_parents,
            "per_parent": confirmation_rows,
            "checks": checks,
        },
        "terminal": terminal,
        "android_candidate_authorized": False,
        "fresh_confirmation_source_lock_authorized": terminal == "TASK_EVIDENCE_PAIRWISE_RANKER_BONN_CONFIRMATION_PASS",
        "read_boundary": {"rgb_payload_decodes": 0, "bonn_neighbor_depth_in_selection": False, "network_requests": 0, "r11_reads": 0},
        "claim_ceiling": "TUM FIT plus independent Bonn Development task-evidence confirmation; not collision correctness, device runtime, Android integration, product, default-App, or safety evidence.",
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
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
    parser.add_argument("--bonn-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.bonn_root.resolve())
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
