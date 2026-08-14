#!/usr/bin/env python3
"""R26 fixed task-cell disocclusion release scorer for TARO.

Unlike the earlier parallax summaries, this scorer explicitly back-projects
the foreground point that occludes each unknown task cell, transforms both
points into a candidate camera, and requires their projected separation to
cross an observed foreground boundary in the reference depth map.  Candidate
depth is target-side only and is never read by the scorer.
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

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer


SCHEMA = "blindassist.taro.task_evidence_disocclusion_release_scorer.v1"
DEPTH_OCCLUSION_MARGIN_M = 0.05
MAX_FOREGROUND_EDGE_SEARCH_PX = 12
BOUNDARY_CLEARANCE_MARGIN_PX = 1.0
MINIMUM_OVERRIDE_RELEASE_CELL_ADVANTAGE = 1
RELEASE_MARGIN_NORMALIZER_PX = 4.0


class DisocclusionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DisocclusionError(message)


def _foreground_edge_distances(
    low_depth: np.ndarray,
    valid: np.ndarray,
    reference_u: np.ndarray,
    reference_v: np.ndarray,
    direction_u: np.ndarray,
    direction_v: np.ndarray,
    target_depth: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Return the first observed pixel that is no longer in front of a cell.

    Invalid/out-of-frame samples are not treated as free space.  This makes the
    boundary test fail closed when the reference depth cannot support it.
    """

    require(low_depth.ndim == 2 and valid.shape == low_depth.shape, "reference depth shape drift")
    shape = reference_u.shape
    require(
        reference_v.shape == shape
        and direction_u.shape == shape
        and direction_v.shape == shape
        and target_depth.shape == shape
        and active.shape == shape,
        "edge-search input shape drift",
    )
    height, width = low_depth.shape
    output = np.full(shape, float(MAX_FOREGROUND_EDGE_SEARCH_PX + 1), dtype=np.float64)
    unresolved = active.copy()
    for step in range(1, MAX_FOREGROUND_EDGE_SEARCH_PX + 1):
        columns = np.rint(reference_u + float(step) * direction_u).astype(np.int64)
        rows = np.rint(reference_v + float(step) * direction_v).astype(np.int64)
        inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        safe_columns = np.clip(columns, 0, width - 1)
        safe_rows = np.clip(rows, 0, height - 1)
        sampled_valid = inside & valid[safe_rows, safe_columns]
        no_longer_foreground = sampled_valid & (
            low_depth[safe_rows, safe_columns] + DEPTH_OCCLUSION_MARGIN_M >= target_depth
        )
        found = unresolved & no_longer_foreground
        output[found] = float(step)
        unresolved &= ~found
    return output


def disocclusion_candidate_features(
    context: scorer.ReferenceContext,
    pair: bonn.Pair,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute fixed source-time task-cell disocclusion evidence."""

    task_points: list[np.ndarray] = []
    unknown_masks: list[np.ndarray] = []
    for query_index, query in enumerate(context.queries):
        centers, _along = scorer._cell_centers(query)
        task_points.append(centers)
        unknown_masks.append(~context.static[query_index].reshape(-1))
    points_reference = np.concatenate(task_points, axis=0)
    unknown = np.concatenate(unknown_masks)

    relative = np.linalg.inv(context.row.reference.camera_to_world) @ pair.neighbor.camera_to_world
    reference_to_candidate = np.linalg.inv(relative)
    points_candidate = (
        points_reference @ reference_to_candidate[:3, :3].T
        + reference_to_candidate[:3, 3]
    )
    reference_z = points_reference[:, 2]
    candidate_z = points_candidate[:, 2]
    intrinsics = context.intrinsics
    height, width = context.low_depth.shape

    reference_u = intrinsics[0, 0] * points_reference[:, 0] / np.maximum(reference_z, 1e-9) + intrinsics[0, 2]
    reference_v = intrinsics[1, 1] * points_reference[:, 1] / np.maximum(reference_z, 1e-9) + intrinsics[1, 2]
    candidate_u = intrinsics[0, 0] * points_candidate[:, 0] / np.maximum(candidate_z, 1e-9) + intrinsics[0, 2]
    candidate_v = intrinsics[1, 1] * points_candidate[:, 1] / np.maximum(candidate_z, 1e-9) + intrinsics[1, 2]
    reference_inside = (
        (reference_z >= adapter.DEPTH_RANGE_M[0])
        & (reference_z <= adapter.DEPTH_RANGE_M[1])
        & (reference_u >= 0.0)
        & (reference_u < width)
        & (reference_v >= 0.0)
        & (reference_v < height)
    )
    candidate_inside = (
        (candidate_z >= adapter.DEPTH_RANGE_M[0])
        & (candidate_z <= adapter.DEPTH_RANGE_M[1])
        & (candidate_u >= 0.0)
        & (candidate_u < width)
        & (candidate_v >= 0.0)
        & (candidate_v < height)
    )
    reference_columns = np.clip(np.rint(reference_u).astype(np.int64), 0, width - 1)
    reference_rows = np.clip(np.rint(reference_v).astype(np.int64), 0, height - 1)
    foreground_depth = context.low_depth[reference_rows, reference_columns]
    foreground_valid = context.valid[reference_rows, reference_columns]
    occluded = (
        unknown
        & reference_inside
        & candidate_inside
        & foreground_valid
        & (foreground_depth + DEPTH_OCCLUSION_MARGIN_M < reference_z)
    )

    foreground_reference = np.stack(
        (
            (reference_u - intrinsics[0, 2]) * foreground_depth / intrinsics[0, 0],
            (reference_v - intrinsics[1, 2]) * foreground_depth / intrinsics[1, 1],
            foreground_depth,
        ),
        axis=1,
    )
    foreground_candidate = (
        foreground_reference @ reference_to_candidate[:3, :3].T
        + reference_to_candidate[:3, 3]
    )
    foreground_candidate_z = foreground_candidate[:, 2]
    foreground_candidate_u = (
        intrinsics[0, 0]
        * foreground_candidate[:, 0]
        / np.maximum(foreground_candidate_z, 1e-9)
        + intrinsics[0, 2]
    )
    foreground_candidate_v = (
        intrinsics[1, 1]
        * foreground_candidate[:, 1]
        / np.maximum(foreground_candidate_z, 1e-9)
        + intrinsics[1, 2]
    )
    delta_u = candidate_u - foreground_candidate_u
    delta_v = candidate_v - foreground_candidate_v
    separation = np.sqrt(delta_u * delta_u + delta_v * delta_v)
    finite_foreground_projection = (
        np.isfinite(foreground_candidate_u)
        & np.isfinite(foreground_candidate_v)
        & (foreground_candidate_z > 1e-6)
    )
    direction_u = np.divide(delta_u, separation, out=np.zeros_like(delta_u), where=separation > 1e-9)
    direction_v = np.divide(delta_v, separation, out=np.zeros_like(delta_v), where=separation > 1e-9)
    edge_distance = _foreground_edge_distances(
        context.low_depth,
        context.valid,
        reference_u,
        reference_v,
        direction_u,
        direction_v,
        reference_z,
        occluded & finite_foreground_projection,
    )
    release_margin = separation - edge_distance - BOUNDARY_CLEARANCE_MARGIN_PX
    robust_release = occluded & finite_foreground_projection & (release_margin > 0.0)
    normalized_margin = np.clip(release_margin / RELEASE_MARGIN_NORMALIZER_PX, 0.0, 1.0)
    analytic = {
        "robust_release_cell_count": float(np.sum(robust_release)),
        "release_margin_sum": float(np.sum(normalized_margin * robust_release)),
        "eligible_occluded_cell_count": float(np.sum(occluded)),
        "candidate_visible_unknown_cell_count": float(np.sum(unknown & candidate_inside)),
        "foreground_separation_sum": float(np.sum(np.clip(separation / 20.0, 0.0, 1.0) * occluded)),
    }
    features = np.asarray(
        [
            analytic["robust_release_cell_count"],
            analytic["release_margin_sum"],
            analytic["eligible_occluded_cell_count"],
            analytic["candidate_visible_unknown_cell_count"],
            pair.translation_m,
            pair.rotation_deg,
        ],
        dtype=np.float64,
    )
    require(features.shape == (6,) and np.all(np.isfinite(features)), "R26 feature drift")
    return features, analytic


def _generic_index(records: Sequence[scorer.CandidateRecord], indices: Sequence[int]) -> int:
    return max(
        indices,
        key=lambda index: (
            records[index].pair.translation_m,
            records[index].pair.rotation_deg,
            -records[index].pair.gap_s,
            records[index].pair.neighbor.frame_id,
        ),
    )


def primary_selection_scores(
    records: Sequence[scorer.CandidateRecord],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Override generic only for one additional robust release-cell prediction."""

    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    scores = np.zeros(len(records), dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    overrides = 0
    for reference_id, indices in sorted(by_reference.items()):
        generic = _generic_index(records, indices)
        best = max(
            indices,
            key=lambda index: (
                records[index].analytic["robust_release_cell_count"],
                records[index].analytic["release_margin_sum"],
                records[index].pair.translation_m,
                records[index].pair.rotation_deg,
                records[index].pair.neighbor.frame_id,
            ),
        )
        advantage = int(
            records[best].analytic["robust_release_cell_count"]
            - records[generic].analytic["robust_release_cell_count"]
        )
        selected = best if advantage >= MINIMUM_OVERRIDE_RELEASE_CELL_ADVANTAGE else generic
        scores[selected] = 1.0
        overrides += int(selected != generic)
        receipts.append(
            {
                "reference_id": reference_id,
                "generic_neighbor_id": records[generic].pair.neighbor.frame_id,
                "selected_neighbor_id": records[selected].pair.neighbor.frame_id,
                "predicted_release_cell_advantage": advantage,
                "overrode_generic": selected != generic,
            }
        )
    receipt_sha256 = hashlib.sha256(r21.shared.canonical_json_bytes(receipts)).hexdigest().upper()
    return scores, {
        "reference_count": len(by_reference),
        "generic_override_count": overrides,
        "generic_fallback_count": len(by_reference) - overrides,
        "selection_receipt_sha256": receipt_sha256,
    }


def ungated_release_scores(records: Sequence[scorer.CandidateRecord]) -> np.ndarray:
    return np.asarray(
        [
            record.analytic["robust_release_cell_count"] * 1000.0
            + record.analytic["release_margin_sum"]
            for record in records
        ],
        dtype=np.float64,
    )


def evaluate(bonn_root: Path, arkit_root: Path) -> dict[str, Any]:
    datasets = {
        "TUM_RGBD": r21._build_tum_records(disocclusion_candidate_features),
        "BONN_RGBD_DYNAMIC": r21._build_bonn_records(bonn_root, disocclusion_candidate_features),
        "ARKITSCENES": r21._build_arkit_records(arkit_root, disocclusion_candidate_features),
    }
    source_results: dict[str, Any] = {}
    for source, (records, receipt, abstentions) in datasets.items():
        scores, selection = primary_selection_scores(records)
        source_results[source] = {
            "candidate_count": len(records),
            "parent_count": len({record.parent_id for record in records}),
            "geometry_abstention_count": abstentions,
            "source_receipt": receipt,
            "selection": selection,
            "primary_metrics": r21.fold_metrics(records, scores),
            "ungated_release_metrics": r21.fold_metrics(records, ungated_release_scores(records)),
        }
    all_primary_checks_pass = all(
        all(row["primary_metrics"]["checks"].values()) for row in source_results.values()
    )
    terminal = (
        "TASK_EVIDENCE_DISOCCLUSION_RELEASE_THREE_SOURCE_PASS"
        if all_primary_checks_pass
        else "STOP_TASK_EVIDENCE_DISOCCLUSION_RELEASE_TRANSFER_FAIL"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "CONSUMED_THREE_SOURCE_DEVELOPMENT_FIXED_ANALYTIC",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside nine frozen body/path capsules; UNKNOWN remains unknown.",
        "mechanism": {
            "name": "TASK_CELL_FOREGROUND_BOUNDARY_DISOCCLUSION_RELEASE",
            "description": "Transform each unknown task cell and its reference-depth foreground occluder into the candidate camera; count a release only when projected separation crosses the observed foreground boundary plus a fixed one-pixel margin.",
            "depth_occlusion_margin_m": DEPTH_OCCLUSION_MARGIN_M,
            "maximum_foreground_edge_search_px": MAX_FOREGROUND_EDGE_SEARCH_PX,
            "boundary_clearance_margin_px": BOUNDARY_CLEARANCE_MARGIN_PX,
            "minimum_override_release_cell_advantage": MINIMUM_OVERRIDE_RELEASE_CELL_ADVANTAGE,
            "generic_fallback": True,
            "neighbor_depth_in_scorer_input": False,
            "training_steps": 0,
            "parameters_fit_from_targets": 0,
        },
        "sources": source_results,
        "terminal": terminal,
        "fresh_confirmation_source_lock_authorized": all_primary_checks_pass,
        "android_candidate_authorized": False,
        "read_boundary": {
            "reference_depth_in_scorer_input": True,
            "candidate_depth_in_scorer_input": False,
            "candidate_depth_used_only_after_selection_for_consumed_development_metric": True,
            "rgb_payload_decodes": 0,
            "network_requests": 0,
        },
        "claim_ceiling": "Consumed three-source Development evidence only. A PASS would authorize a fresh task-outcome-blind source lock, not collision correctness, Android integration, product, default-App, navigation, or safety claims.",
    }
    result["content_sha256"] = hashlib.sha256(r21.shared.canonical_json_bytes(result)).hexdigest().upper()
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
    parser.add_argument("--bonn-root", type=Path, default=r21.balanced.DEFAULT_BONN_ROOT)
    parser.add_argument("--arkit-root", type=Path, default=r21.DEFAULT_ARKIT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.bonn_root.resolve(), args.arkit_root.resolve())
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
