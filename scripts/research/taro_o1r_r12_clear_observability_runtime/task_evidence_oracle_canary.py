#!/usr/bin/env python3
"""R13 TARO task-conditioned query-evidence oracle headroom canary.

This is a new falsifiable task after R12's three independent sources exposed a
structurally saturated CLEAR label.  It asks whether one pose-valid observation,
chosen from the same outcome-blind proposal set and with the same one-frame
budget, adds more observed body/path query cells than passive or generic pose
rules.  UNKNOWN cells remain unknown; they are never converted to negatives.
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

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_conditioned_query_evidence_oracle_canary.v1"
ARM_NAMES = ("static", "passive", "fixed_micro", "generic_max_parallax", "task_evidence_oracle")
MAX_REFERENCES_PER_PARENT = 5
MIN_EVALUATED_REFERENCES = 48
MIN_OPPORTUNITY_PARENTS = 4
MIN_TASK_WIN_PARENTS = 4
ALONG_BIN_EDGES_M = np.linspace(adapter.MINIMUM_FORWARD_M, adapter.HORIZON_M, 7)
ACROSS_BIN_EDGES_M = np.linspace(-adapter.CAPSULE_RADIUS_M, adapter.CAPSULE_RADIUS_M, 4)
HEIGHT_BIN_EDGES_M = np.asarray([-0.05, adapter.OBSTACLE_HEIGHT_RANGE_M[0], 0.50, 1.00, adapter.OBSTACLE_HEIGHT_RANGE_M[1]], dtype=np.float64)
MINIMUM_POINTS_PER_EVIDENCE_CELL = 2
POSE_TRANSLATION_TARGETS_M = (0.04, 0.05, 0.06, 0.07, 0.08)
POSE_GAP_TARGETS_S = (0.15, 0.50, 1.00)


class CanaryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def pose_proposal_pairs(row: bonn.ReferenceSupport) -> tuple[bonn.Pair, ...]:
    """Freeze a small pose-only proposal pool before any neighbor depth is read."""
    require(bool(row.candidates) and bool(row.micro_candidates), "reference lacks legal proposal pairs")
    chosen: dict[str, bonn.Pair] = {}

    def add(pair: bonn.Pair) -> None:
        chosen[pair.neighbor.frame_id] = pair

    add(min(row.micro_candidates, key=lambda pair: (abs(pair.translation_m - bonn.MICRO_TARGET_TRANSLATION_M), pair.rotation_deg, pair.gap_s, pair.neighbor.frame_id)))
    add(max(row.candidates, key=lambda pair: (pair.translation_m, pair.rotation_deg, -pair.gap_s, pair.neighbor.frame_id)))
    for target in POSE_TRANSLATION_TARGETS_M:
        add(min(row.candidates, key=lambda pair: (abs(pair.translation_m - target), pair.rotation_deg, pair.gap_s, pair.neighbor.frame_id)))
    for target in POSE_GAP_TARGETS_S:
        add(min(row.candidates, key=lambda pair: (abs(pair.gap_s - target), abs(pair.translation_m - bonn.MICRO_TARGET_TRANSLATION_M), pair.neighbor.frame_id)))
    return tuple(sorted(chosen.values(), key=lambda pair: pair.neighbor.frame_id))


def query_evidence_cells(
    points_hw3: np.ndarray,
    valid_hw: np.ndarray,
    queries: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    shape = (len(queries), len(ALONG_BIN_EDGES_M) - 1, len(ACROSS_BIN_EDGES_M) - 1, len(HEIGHT_BIN_EDGES_M) - 1)
    output = np.zeros(shape, dtype=bool)
    for query_index, query in enumerate(queries):
        along, across, height = r7_canary._query_coordinates(points_hw3, query)
        eligible = valid_hw & (along >= ALONG_BIN_EDGES_M[0]) & (along <= ALONG_BIN_EDGES_M[-1]) & (across >= ACROSS_BIN_EDGES_M[0]) & (across <= ACROSS_BIN_EDGES_M[-1]) & (height >= HEIGHT_BIN_EDGES_M[0]) & (height <= HEIGHT_BIN_EDGES_M[-1])
        if not np.any(eligible):
            continue
        ai = np.clip(np.digitize(along[eligible], ALONG_BIN_EDGES_M) - 1, 0, shape[1] - 1)
        ci = np.clip(np.digitize(across[eligible], ACROSS_BIN_EDGES_M) - 1, 0, shape[2] - 1)
        hi = np.clip(np.digitize(height[eligible], HEIGHT_BIN_EDGES_M) - 1, 0, shape[3] - 1)
        flat = (ai * shape[2] + ci) * shape[3] + hi
        counts = np.bincount(flat, minlength=shape[1] * shape[2] * shape[3])
        output[query_index] = counts.reshape(shape[1:]) >= MINIMUM_POINTS_PER_EVIDENCE_CELL
    return output


def _transform_points(points: np.ndarray, reference: bonn.Frame, neighbor: bonn.Frame) -> np.ndarray:
    relative = np.linalg.inv(reference.camera_to_world) @ neighbor.camera_to_world
    flat = points.reshape(-1, 3)
    transformed = flat @ relative[:3, :3].T + relative[:3, 3]
    return np.ascontiguousarray(transformed.reshape(points.shape), dtype=np.float64)


def _queries(reference: bonn.Frame, low_depth: np.ndarray, low_intrinsics: np.ndarray) -> list[dict[str, Any]] | None:
    up = adapter._normalize_vector(reference.camera_to_world[:3, :3].T @ tum.WORLD_UP, "TUM_GRAVITY_INVALID")
    plane = prospective._fit_depth_plane(low_depth, low_intrinsics, up)
    if not plane["evaluable"]:
        return None
    return prospective._build_queries(reference.frame_id, hashlib.sha256(reference.frame_id.encode("utf-8")).hexdigest().upper(), round(reference.timestamp_s * 1_000_000_000), plane)


def _empty_arm_counts() -> dict[str, int]:
    return {"reference_count": 0, "static_evidence_cells": 0, "final_evidence_cells": 0, "novel_evidence_cells": 0, "retention_failures": 0, "extra_frame_count": 0}


def _accumulate(counts: dict[str, int], static: np.ndarray, final: np.ndarray, extra: bool) -> None:
    counts["reference_count"] += 1
    counts["static_evidence_cells"] += int(np.sum(static))
    counts["final_evidence_cells"] += int(np.sum(final))
    counts["novel_evidence_cells"] += int(np.sum(final & ~static))
    counts["retention_failures"] += int(np.sum(static & ~final))
    counts["extra_frame_count"] += int(extra)


def evaluate(
    manifests: Sequence[Path] = tum.DEFAULT_MANIFESTS,
    limit: int = MAX_REFERENCES_PER_PARENT,
) -> dict[str, Any]:
    frames, assets, source = tum.load_outcome_blind_roster(manifests, verify_archive_hashes=False)
    selected, capability = shared.select_pose_capable_references(frames, limit)
    require(capability["selected_reference_count"] >= MIN_EVALUATED_REFERENCES, "NOT_EVALUABLE_PAIR_SUPPORT")
    proposals = {row.reference.frame_id: pose_proposal_pairs(row) for row in selected}
    frame_lookup = {frame.frame_id: frame for frame in frames}
    needed_ids = sorted({row.reference.frame_id for row in selected} | {pair.neighbor.frame_id for row in selected for pair in proposals[row.reference.frame_id]})
    depth_cache = tum.load_depth_frame_ids(needed_ids, assets)
    observations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for frame_id in needed_ids:
        asset = assets[frame_id]
        low, points, valid = tum._low_observation(depth_cache[frame_id], asset.intrinsics)
        observations[frame_id] = (low, points, valid, float(np.mean(valid)))
    totals = {name: _empty_arm_counts() for name in ARM_NAMES}
    per_parent: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: {name: _empty_arm_counts() for name in ARM_NAMES})
    evaluated = abstained = 0
    reference_receipts: list[dict[str, Any]] = []
    parent_task_wins: dict[str, int] = defaultdict(int)
    parent_opportunities: dict[str, int] = defaultdict(int)
    for row in selected:
        reference_id = row.reference.frame_id
        low, reference_points, reference_valid, _coverage = observations[reference_id]
        asset = assets[reference_id]
        low_intrinsics = bonn._scaled_intrinsics(asset.intrinsics, tum.NATIVE_SIZE_WH, tum.LOW_SIZE_WH)
        queries = _queries(row.reference, low, low_intrinsics)
        if queries is None:
            abstained += 1
            continue
        static = query_evidence_cells(reference_points, reference_valid, queries)
        candidates: list[dict[str, Any]] = []
        for pair in proposals[reference_id]:
            _neighbor_low, points, valid, coverage = observations[pair.neighbor.frame_id]
            transformed = _transform_points(points, row.reference, pair.neighbor)
            observed = query_evidence_cells(transformed, valid, queries)
            final = static | observed
            candidates.append({"pair": pair, "coverage": coverage, "final": final, "gain": int(np.sum(final & ~static))})
        require(bool(candidates), "empty proposal evaluation")
        passive = max(candidates, key=lambda item: (item["coverage"], -item["pair"].gap_s, item["pair"].neighbor.frame_id))
        micro_pair = min(row.micro_candidates, key=lambda pair: (abs(pair.translation_m - bonn.MICRO_TARGET_TRANSLATION_M), pair.rotation_deg, pair.gap_s, pair.neighbor.frame_id))
        by_id = {item["pair"].neighbor.frame_id: item for item in candidates}
        micro = by_id[micro_pair.neighbor.frame_id]
        generic = max(candidates, key=lambda item: (item["pair"].translation_m, item["pair"].rotation_deg, -item["pair"].gap_s, item["pair"].neighbor.frame_id))
        task = max(candidates, key=lambda item: (item["gain"], item["coverage"], -item["pair"].translation_m, item["pair"].neighbor.frame_id))
        arms = {"static": {"final": static, "pair": None}, "passive": passive, "fixed_micro": micro, "generic_max_parallax": generic, "task_evidence_oracle": task}
        for name, item in arms.items():
            _accumulate(totals[name], static, item["final"], name != "static")
            _accumulate(per_parent[row.reference.parent_id][name], static, item["final"], name != "static")
        comparator_gain = max(int(passive["gain"]), int(generic["gain"]))
        parent_opportunities[row.reference.parent_id] += int(task["gain"] > 0)
        parent_task_wins[row.reference.parent_id] += int(task["gain"] > comparator_gain)
        reference_receipts.append({
            "reference_frame_id": reference_id,
            "proposal_count": len(candidates),
            "static_evidence_cell_count": int(np.sum(static)),
            "selected": {name: {"neighbor_frame_id": item["pair"].neighbor.frame_id, "novel_evidence_cells": int(item["gain"])} for name, item in (("passive", passive), ("fixed_micro", micro), ("generic_max_parallax", generic), ("task_evidence_oracle", task))},
        })
        evaluated += 1
    parent_metrics: dict[str, dict[str, Any]] = {}
    for parent, arms in sorted(per_parent.items()):
        parent_metrics[parent] = {
            "reference_count": arms["static"]["reference_count"],
            "novel_evidence_cells_per_reference": {name: (arms[name]["novel_evidence_cells"] / arms[name]["reference_count"] if arms[name]["reference_count"] else None) for name in ARM_NAMES},
            "task_strict_win_reference_count": parent_task_wins[parent],
            "task_opportunity_reference_count": parent_opportunities[parent],
        }
    macro = {
        name: float(np.mean([row["novel_evidence_cells_per_reference"][name] for row in parent_metrics.values() if row["novel_evidence_cells_per_reference"][name] is not None]))
        for name in ARM_NAMES
    }
    opportunity_parents = sum(row["task_opportunity_reference_count"] > 0 for row in parent_metrics.values())
    task_win_parents = sum(row["task_strict_win_reference_count"] > 0 for row in parent_metrics.values())
    checks = {
        "minimum_evaluated_references": evaluated >= MIN_EVALUATED_REFERENCES,
        "minimum_opportunity_parents": opportunity_parents >= MIN_OPPORTUNITY_PARENTS,
        "minimum_task_win_parents": task_win_parents >= MIN_TASK_WIN_PARENTS,
        "task_macro_beats_passive": macro["task_evidence_oracle"] > macro["passive"],
        "task_macro_beats_generic": macro["task_evidence_oracle"] > macro["generic_max_parallax"],
        "all_arms_same_extra_frame_budget": all(totals[name]["extra_frame_count"] == evaluated for name in ARM_NAMES if name != "static"),
        "zero_retention_failures": all(totals[name]["retention_failures"] == 0 for name in ARM_NAMES),
    }
    terminal = "TASK_CONDITIONED_QUERY_EVIDENCE_ORACLE_HEADROOM_PASS" if all(checks.values()) else "STOP_TASK_CONDITIONED_QUERY_EVIDENCE_NO_HEADROOM"
    result = {
        "schema": SCHEMA,
        "mode": "REVERSIBLE_EXPLORATION_PROJECT_CONSUMED_DEVELOPMENT",
        "task_definition": "One extra pose-valid frame should maximize newly observed spatial evidence cells inside nine frozen body/path capsules; unobserved cells remain UNKNOWN and are never negatives.",
        "source": source,
        "pose_pair_capability": capability,
        "proposal_policy": {"inputs": "pose only", "translation_targets_m": list(POSE_TRANSLATION_TARGETS_M), "gap_targets_s": list(POSE_GAP_TARGETS_S), "same_pool_for_all_non_static_arms": True, "neighbor_payload_reads_after_proposal_freeze": len(needed_ids)},
        "evidence_grid": {"along_bin_edges_m": ALONG_BIN_EDGES_M.tolist(), "across_bin_edges_m": ACROSS_BIN_EDGES_M.tolist(), "height_bin_edges_m": HEIGHT_BIN_EDGES_M.tolist(), "minimum_points_per_cell": MINIMUM_POINTS_PER_EVIDENCE_CELL, "unknown_is_negative": False},
        "evaluated_reference_count": evaluated,
        "geometry_abstention_count": abstained,
        "metrics": {"totals": totals, "parent_macro_novel_evidence_cells_per_reference": macro, "opportunity_parent_count": opportunity_parents, "task_strict_win_parent_count": task_win_parents, "per_parent": parent_metrics},
        "evaluability_and_decision_checks": checks,
        "terminal": terminal,
        "learned_pose_scorer_design_authorized": terminal == "TASK_CONDITIONED_QUERY_EVIDENCE_ORACLE_HEADROOM_PASS",
        "reference_receipt_sha256": hashlib.sha256(canonical_json_bytes(reference_receipts)).hexdigest().upper(),
        "read_boundary": {"rgb_payload_decodes": 0, "depth_payload_reads_before_pose_proposal_freeze": 0, "model_runs": 0, "training_steps": 0, "network_requests": 0, "r11_reads": 0},
        "claim_ceiling": "Consumed TUM RGB-D Development oracle-headroom evidence only; not a learned policy, collision classifier, fresh Confirmation, Android, product, default-App, or safety evidence.",
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
