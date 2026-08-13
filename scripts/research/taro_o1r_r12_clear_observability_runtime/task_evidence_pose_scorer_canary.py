#!/usr/bin/env python3
"""R14 source-time pose/static-evidence scorer replay for TARO R13 gain.

The scorer sees only the reference evidence grid, relative pose, intrinsics, and
analytic candidate-frustum geometry.  FIT neighbor depth generates training and
model-selection targets.  EVALUATION candidate identities are sealed before any
EVALUATION neighbor depth is opened; those payloads are label-side replay only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_pose_scorer_canary.v1"
MIN_EVALUATION_REFERENCES = 16
MIN_EVALUATION_PARENTS = 4
MIN_STRICT_WIN_PARENTS = 4
RIDGE_LAMBDAS = (0.1, 1.0, 10.0, 100.0)
ANALYTIC_SCORERS = ("visible_unknown", "unknown_parallax", "occluded_parallax", "far_unknown_parallax")
FEATURE_NAMES = (
    "gap_s", "translation_m", "rotation_deg", "translation_x", "translation_y", "translation_z",
    "abs_translation_x", "abs_translation_y", "abs_translation_z", "static_fraction",
    "visible_unknown", "unknown_parallax", "occluded_parallax", "far_unknown_parallax",
) + tuple(f"static_query_{index}" for index in range(9)) + tuple(f"static_along_{index}" for index in range(6)) + tuple(f"static_height_{index}" for index in range(4))


class ScorerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScorerError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


@dataclass
class ReferenceContext:
    row: bonn.ReferenceSupport
    low_depth: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    intrinsics: np.ndarray
    queries: list[dict[str, Any]]
    static: np.ndarray


@dataclass
class CandidateRecord:
    parent_id: str
    role: str
    reference_id: str
    pair: bonn.Pair
    features: np.ndarray
    analytic: dict[str, float]
    target_gain: int | None = None
    coverage: float | None = None


def _cell_centers(query: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    origin, _, lateral, heading = adapter._query_receipt_vectors(dict(query))
    up = adapter._normalize_vector(query["virtual_query_frame"]["gravity_up_camera_xyz"], "R14_QUERY_UP_INVALID")
    side = adapter._normalize_vector(np.cross(heading, up), "R14_QUERY_SIDE_INVALID")
    path_origin = origin + float(query["path_lateral_offset_m"]) * lateral
    along = (oracle.ALONG_BIN_EDGES_M[:-1] + oracle.ALONG_BIN_EDGES_M[1:]) / 2.0
    across = (oracle.ACROSS_BIN_EDGES_M[:-1] + oracle.ACROSS_BIN_EDGES_M[1:]) / 2.0
    height = (oracle.HEIGHT_BIN_EDGES_M[:-1] + oracle.HEIGHT_BIN_EDGES_M[1:]) / 2.0
    aa, cc, hh = np.meshgrid(along, across, height, indexing="ij")
    points = path_origin + aa[..., None] * heading + cc[..., None] * side + hh[..., None] * up
    return np.ascontiguousarray(points.reshape(-1, 3)), np.ascontiguousarray(aa.reshape(-1))


def source_time_candidate_features(
    context: ReferenceContext,
    pair: bonn.Pair,
) -> tuple[np.ndarray, dict[str, float]]:
    relative = np.linalg.inv(context.row.reference.camera_to_world) @ pair.neighbor.camera_to_world
    inverse = np.linalg.inv(relative)
    unknown_masks: list[np.ndarray] = []
    ref_points: list[np.ndarray] = []
    along_values: list[np.ndarray] = []
    for query_index, query in enumerate(context.queries):
        centers, along = _cell_centers(query)
        ref_points.append(centers)
        along_values.append(along)
        unknown_masks.append(~context.static[query_index].reshape(-1))
    points_ref = np.concatenate(ref_points, axis=0)
    along = np.concatenate(along_values)
    unknown = np.concatenate(unknown_masks)
    points_neighbor = points_ref @ inverse[:3, :3].T + inverse[:3, 3]
    ref_z = points_ref[:, 2]
    neighbor_z = points_neighbor[:, 2]
    width, height = tum.LOW_SIZE_WH
    k = context.intrinsics
    ref_u = k[0, 0] * points_ref[:, 0] / np.maximum(ref_z, 1e-9) + k[0, 2]
    ref_v = k[1, 1] * points_ref[:, 1] / np.maximum(ref_z, 1e-9) + k[1, 2]
    nei_u = k[0, 0] * points_neighbor[:, 0] / np.maximum(neighbor_z, 1e-9) + k[0, 2]
    nei_v = k[1, 1] * points_neighbor[:, 1] / np.maximum(neighbor_z, 1e-9) + k[1, 2]
    visible = unknown & (neighbor_z >= adapter.DEPTH_RANGE_M[0]) & (neighbor_z <= adapter.DEPTH_RANGE_M[1]) & (nei_u >= 0.0) & (nei_u < width) & (nei_v >= 0.0) & (nei_v < height)
    parallax = np.sqrt((nei_u - ref_u) ** 2 + (nei_v - ref_v) ** 2)
    parallax_weight = np.clip(parallax / 20.0, 0.0, 1.0)
    ref_col = np.clip(np.rint(ref_u).astype(np.int64), 0, width - 1)
    ref_row = np.clip(np.rint(ref_v).astype(np.int64), 0, height - 1)
    sampled = context.low_depth[ref_row, ref_col]
    sample_valid = context.valid[ref_row, ref_col]
    occluded = visible & sample_valid & (sampled + 0.05 < ref_z)
    far_weight = np.clip(along / adapter.HORIZON_M, 0.0, 1.0)
    analytic = {
        "visible_unknown": float(np.sum(visible)),
        "unknown_parallax": float(np.sum(parallax_weight * visible)),
        "occluded_parallax": float(np.sum(parallax_weight * occluded)),
        "far_unknown_parallax": float(np.sum(parallax_weight * visible * far_weight)),
    }
    static = context.static
    translation = relative[:3, 3]
    values = [
        pair.gap_s, pair.translation_m, pair.rotation_deg,
        *translation.tolist(), *np.abs(translation).tolist(), float(np.mean(static)),
        *(analytic[name] for name in ANALYTIC_SCORERS),
        *(float(np.mean(static[index])) for index in range(9)),
        *(float(np.mean(static[:, index])) for index in range(6)),
        *(float(np.mean(static[:, :, :, index])) for index in range(4)),
    ]
    result = np.asarray(values, dtype=np.float64)
    require(result.shape == (len(FEATURE_NAMES),) and np.all(np.isfinite(result)), "feature vector drift")
    return result, analytic


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-9] = 1.0
    return (x - mean) / scale, mean, scale


def _ridge_fit(x: np.ndarray, y: np.ndarray, parent_ids: Sequence[str], penalty: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    standardized, mean, scale = _standardize_fit(x)
    design = np.concatenate((np.ones((len(x), 1)), standardized), axis=1)
    counts = {parent: parent_ids.count(parent) for parent in set(parent_ids)}
    weights = np.asarray([1.0 / counts[parent] for parent in parent_ids], dtype=np.float64)
    weights *= len(weights) / np.sum(weights)
    root = np.sqrt(weights)[:, None]
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve((design * root).T @ (design * root) + regularizer, (design * root).T @ (y * root[:, 0]))
    return coefficients, mean, scale


def _ridge_predict(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    coefficients, mean, scale = model
    design = np.concatenate((np.ones((len(x), 1)), (x - mean) / scale), axis=1)
    return design @ coefficients


def _selected_target(records: Sequence[CandidateRecord], scores: Sequence[float]) -> dict[str, float]:
    by_reference: dict[str, list[tuple[CandidateRecord, float]]] = defaultdict(list)
    for record, score in zip(records, scores, strict=True):
        by_reference[record.reference_id].append((record, float(score)))
    output: dict[str, list[int]] = defaultdict(list)
    for rows in by_reference.values():
        selected, _score = max(rows, key=lambda item: (item[1], -item[0].pair.translation_m, item[0].pair.neighbor.frame_id))
        require(selected.target_gain is not None, "target unavailable during FIT selection")
        output[selected.parent_id].append(int(selected.target_gain))
    return {parent: float(np.mean(values)) for parent, values in output.items()}


def choose_fit_model(records: Sequence[CandidateRecord]) -> tuple[dict[str, Any], Any]:
    parents = sorted({record.parent_id for record in records})
    require(len(parents) >= 4 and all(record.target_gain is not None for record in records), "FIT target support insufficient")
    candidates: list[dict[str, Any]] = []
    for name in ANALYTIC_SCORERS:
        scores = [record.analytic[name] for record in records]
        macro = float(np.mean(list(_selected_target(records, scores).values())))
        candidates.append({"family": "ANALYTIC", "name": name, "fit_parent_macro_selected_gain": macro})
    x_all = np.stack([record.features for record in records])
    y_all = np.asarray([record.target_gain for record in records], dtype=np.float64)
    for penalty in RIDGE_LAMBDAS:
        fold_values: list[float] = []
        for held in parents:
            train = [index for index, record in enumerate(records) if record.parent_id != held]
            test = [index for index, record in enumerate(records) if record.parent_id == held]
            model = _ridge_fit(x_all[train], y_all[train], [records[index].parent_id for index in train], penalty)
            scores = _ridge_predict(x_all[test], model)
            fold_values.extend(_selected_target([records[index] for index in test], scores).values())
        candidates.append({"family": "RIDGE", "penalty": penalty, "fit_parent_lopo_macro_selected_gain": float(np.mean(fold_values))})
    def score(row: Mapping[str, Any]) -> float:
        return float(row.get("fit_parent_lopo_macro_selected_gain", row.get("fit_parent_macro_selected_gain", -np.inf)))
    selected = max(candidates, key=lambda row: (score(row), row["family"] == "ANALYTIC", json.dumps(row, sort_keys=True)))
    if selected["family"] == "ANALYTIC":
        model: Any = selected["name"]
    else:
        model = _ridge_fit(x_all, y_all, [record.parent_id for record in records], float(selected["penalty"]))
    return {"candidates": candidates, "selected": selected}, model


def _score_records(records: Sequence[CandidateRecord], model_spec: Mapping[str, Any], model: Any) -> np.ndarray:
    if model_spec["family"] == "ANALYTIC":
        return np.asarray([record.analytic[str(model)] for record in records], dtype=np.float64)
    return _ridge_predict(np.stack([record.features for record in records]), model)


def _load_observations(frame_ids: Sequence[str], assets: Mapping[str, tum.DepthAsset]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    depths = tum.load_depth_frame_ids(sorted(set(frame_ids)), assets)
    output = {}
    for frame_id, depth in depths.items():
        low, points, valid = tum._low_observation(depth, assets[frame_id].intrinsics)
        output[frame_id] = (low, points, valid, float(np.mean(valid)))
    return output


def _attach_targets(
    records: Sequence[CandidateRecord],
    contexts: Mapping[str, ReferenceContext],
    observations: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
) -> None:
    for record in records:
        _low, points, valid, coverage = observations[record.pair.neighbor.frame_id]
        context = contexts[record.reference_id]
        transformed = oracle._transform_points(points, context.row.reference, record.pair.neighbor)
        observed = oracle.query_evidence_cells(transformed, valid, context.queries)
        record.target_gain = int(np.sum(observed & ~context.static))
        record.coverage = coverage


def _arm_parent_metrics(
    records: Sequence[CandidateRecord],
    scorer_scores: Sequence[float],
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    by_reference: dict[str, list[tuple[CandidateRecord, float]]] = defaultdict(list)
    for record, score in zip(records, scorer_scores, strict=True):
        by_reference[record.reference_id].append((record, float(score)))
    per_parent: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    receipts: list[dict[str, Any]] = []
    for reference, rows in sorted(by_reference.items()):
        values = [record for record, _ in rows]
        learned, learned_score = max(rows, key=lambda item: (item[1], -item[0].pair.translation_m, item[0].pair.neighbor.frame_id))
        passive = max(values, key=lambda row: (float(row.coverage), -row.pair.gap_s, row.pair.neighbor.frame_id))
        generic = max(values, key=lambda row: (row.pair.translation_m, row.pair.rotation_deg, -row.pair.gap_s, row.pair.neighbor.frame_id))
        micro = min(values, key=lambda row: (abs(row.pair.translation_m - bonn.MICRO_TARGET_TRANSLATION_M), row.pair.rotation_deg, row.pair.gap_s, row.pair.neighbor.frame_id))
        task = max(values, key=lambda row: (int(row.target_gain), float(row.coverage), -row.pair.translation_m, row.pair.neighbor.frame_id))
        selections = {"pose_scorer": learned, "passive": passive, "fixed_micro": micro, "generic_max_parallax": generic, "task_evidence_oracle": task}
        for name, row in selections.items():
            per_parent[row.parent_id][name].append(int(row.target_gain))
        comparator = max(int(passive.target_gain), int(generic.target_gain))
        per_parent[learned.parent_id]["strict_win"].append(int(int(learned.target_gain) > comparator))
        receipts.append({"reference_frame_id": reference, "pose_scorer_score": learned_score, "selected": {name: {"neighbor_frame_id": row.pair.neighbor.frame_id, "novel_evidence_cells": int(row.target_gain)} for name, row in selections.items()}})
    rows_out: dict[str, Any] = {}
    for parent, arms in sorted(per_parent.items()):
        rows_out[parent] = {"reference_count": len(arms["pose_scorer"]), "mean_novel_evidence_cells": {name: float(np.mean(arms[name])) for name in ("pose_scorer", "passive", "fixed_micro", "generic_max_parallax", "task_evidence_oracle")}, "pose_scorer_strict_win_reference_count": int(sum(arms["strict_win"]))}
    macro = {name: float(np.mean([row["mean_novel_evidence_cells"][name] for row in rows_out.values()])) for name in ("pose_scorer", "passive", "fixed_micro", "generic_max_parallax", "task_evidence_oracle")}
    return macro, rows_out, receipts


def evaluate() -> dict[str, Any]:
    frames, assets, source = tum.load_outcome_blind_roster(tum.DEFAULT_MANIFESTS, verify_archive_hashes=False)
    selected, capability = shared.select_pose_capable_references(frames, oracle.MAX_REFERENCES_PER_PARENT)
    role_by_parent = {row["parent_id"]: row["cohort_role"] for row in source["parents"]}
    reference_ids = [row.reference.frame_id for row in selected]
    reference_observations = _load_observations(reference_ids, assets)
    contexts: dict[str, ReferenceContext] = {}
    records: list[CandidateRecord] = []
    abstained = 0
    for row in selected:
        low, points, valid, _coverage = reference_observations[row.reference.frame_id]
        low_intrinsics = bonn._scaled_intrinsics(assets[row.reference.frame_id].intrinsics, tum.NATIVE_SIZE_WH, tum.LOW_SIZE_WH)
        queries = oracle._queries(row.reference, low, low_intrinsics)
        if queries is None:
            abstained += 1
            continue
        static = oracle.query_evidence_cells(points, valid, queries)
        context = ReferenceContext(row, low, points, valid, low_intrinsics, queries, static)
        contexts[row.reference.frame_id] = context
        role = role_by_parent[row.reference.parent_id]
        for pair in oracle.pose_proposal_pairs(row):
            features, analytic = source_time_candidate_features(context, pair)
            records.append(CandidateRecord(row.reference.parent_id, role, row.reference.frame_id, pair, features, analytic))
    fit_records = [record for record in records if record.role == "FIT"]
    evaluation_records = [record for record in records if record.role == "EVALUATION"]
    fit_ids = [record.pair.neighbor.frame_id for record in fit_records]
    fit_observations = _load_observations(fit_ids, assets)
    _attach_targets(fit_records, contexts, fit_observations)
    model_selection, model = choose_fit_model(fit_records)
    eval_scores_before_target = _score_records(evaluation_records, model_selection["selected"], model)
    eval_selection_rows: list[dict[str, Any]] = []
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(evaluation_records):
        by_reference[record.reference_id].append(index)
    for reference, indices in sorted(by_reference.items()):
        chosen = max(indices, key=lambda index: (float(eval_scores_before_target[index]), -evaluation_records[index].pair.translation_m, evaluation_records[index].pair.neighbor.frame_id))
        eval_selection_rows.append({"reference_frame_id": reference, "neighbor_frame_id": evaluation_records[chosen].pair.neighbor.frame_id, "score": float(eval_scores_before_target[chosen])})
    eval_selection_sha = hashlib.sha256(canonical_json_bytes(eval_selection_rows)).hexdigest().upper()
    # Evaluation neighbor payloads are opened only after scorer identities above are frozen.
    eval_ids = [record.pair.neighbor.frame_id for record in evaluation_records]
    eval_observations = _load_observations(eval_ids, assets)
    _attach_targets(evaluation_records, contexts, eval_observations)
    macro, per_parent, receipts = _arm_parent_metrics(evaluation_records, eval_scores_before_target)
    evaluated_parents = len(per_parent)
    evaluated_references = sum(row["reference_count"] for row in per_parent.values())
    strict_win_parents = sum(row["pose_scorer_strict_win_reference_count"] > 0 for row in per_parent.values())
    checks = {
        "minimum_evaluation_references": evaluated_references >= MIN_EVALUATION_REFERENCES,
        "minimum_evaluation_parents": evaluated_parents >= MIN_EVALUATION_PARENTS,
        "minimum_strict_win_parents": strict_win_parents >= MIN_STRICT_WIN_PARENTS,
        "pose_scorer_macro_beats_passive": macro["pose_scorer"] > macro["passive"],
        "pose_scorer_macro_beats_generic": macro["pose_scorer"] > macro["generic_max_parallax"],
        "same_one_frame_budget": True,
        "zero_retention_failures_by_union_construction": True,
        "evaluation_selection_precedes_evaluation_neighbor_depth_reads": True,
    }
    terminal = "TASK_EVIDENCE_POSE_SCORER_DEVELOPMENT_REPLAY_PASS" if all(checks.values()) else "STOP_TASK_EVIDENCE_POSE_SCORER_NOT_BETTER_THAN_BASELINES"
    result = {
        "schema": SCHEMA,
        "mode": "POST_R13_CONSUMED_DEVELOPMENT_ROLE_REPLAY",
        "source": source,
        "pose_pair_capability": capability,
        "feature_contract": {"feature_names": list(FEATURE_NAMES), "inputs": ["reference static evidence grid", "relative pose", "intrinsics", "candidate frustum geometry"], "neighbor_depth_in_scorer_input": False, "unknown_is_negative": False},
        "fit": {"parent_ids": sorted({record.parent_id for record in fit_records}), "candidate_row_count": len(fit_records), "model_selection": model_selection, "target": "novel observed query-evidence cell count from FIT neighbor depth"},
        "evaluation_firewall": {"parent_ids": sorted({record.parent_id for record in evaluation_records}), "candidate_row_count": len(evaluation_records), "selection_identity_sha256_before_neighbor_depth": eval_selection_sha, "evaluation_neighbor_depth_opened_after_selection": True, "r13_previously_opened_same_parents": True, "fresh_confirmation": False},
        "evaluated_reference_count": evaluated_references,
        "geometry_abstention_count": abstained,
        "metrics": {"parent_macro_novel_evidence_cells_per_reference": macro, "strict_win_parent_count": strict_win_parents, "per_parent": per_parent},
        "checks": checks,
        "terminal": terminal,
        "untouched_confirmation_authorized": terminal == "TASK_EVIDENCE_POSE_SCORER_DEVELOPMENT_REPLAY_PASS",
        "selection_receipt_sha256": hashlib.sha256(canonical_json_bytes(receipts)).hexdigest().upper(),
        "read_boundary": {"rgb_payload_decodes": 0, "evaluation_neighbor_depth_in_selection": False, "model_type": model_selection["selected"]["family"], "network_requests": 0, "r11_reads": 0},
        "claim_ceiling": "Post-R13 consumed TUM Development role replay only; not fresh confirmation, Android, product, default-App, or safety evidence.",
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate()
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
