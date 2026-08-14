#!/usr/bin/env python3
"""R27 query-aligned RGB reprojection visibility scorer for TARO.

The reference RGB-D frame is forward-warped into each historical candidate
with a z-buffer.  Unknown task cells are predicted to gain evidence only where
their candidate projection is not explained by the warped reference view, or
where the direct correspondence has a robust photometric residual.  Candidate
depth remains target-side only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_rgb_query_interaction_ranker as r25


SCHEMA = "blindassist.taro.task_evidence_reprojection_visibility_scorer.v1"
WARP_COVERAGE_DILATION_PX = 1
PHOTOMETRIC_RESIDUAL_QUANTILE = 0.90
MINIMUM_OVERRIDE_NOVEL_CELL_ADVANTAGE = 1


class ReprojectionVisibilityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReprojectionVisibilityError(message)


def _forward_z_buffer_warp(
    context: scorer.ReferenceContext,
    pair: Any,
    reference_luma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-warp reference luma and retain the nearest depth per pixel."""

    depth = context.low_depth
    valid = context.valid
    require(depth.ndim == 2 and valid.shape == depth.shape, "R27 reference depth shape drift")
    require(reference_luma.shape == depth.shape, "R27 reference luma shape drift")
    height, width = depth.shape
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    intrinsics = context.intrinsics
    points_reference = np.stack(
        (
            (columns - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    ).reshape(-1, 3)
    relative = np.linalg.inv(context.row.reference.camera_to_world) @ pair.neighbor.camera_to_world
    reference_to_candidate = np.linalg.inv(relative)
    points_candidate = (
        points_reference @ reference_to_candidate[:3, :3].T
        + reference_to_candidate[:3, 3]
    )
    candidate_z = points_candidate[:, 2]
    candidate_u = intrinsics[0, 0] * points_candidate[:, 0] / np.maximum(candidate_z, 1e-9) + intrinsics[0, 2]
    candidate_v = intrinsics[1, 1] * points_candidate[:, 1] / np.maximum(candidate_z, 1e-9) + intrinsics[1, 2]
    source_valid = valid.reshape(-1)
    admitted = (
        source_valid
        & (candidate_z >= adapter.DEPTH_RANGE_M[0])
        & (candidate_z <= adapter.DEPTH_RANGE_M[1])
        & (candidate_u >= 0.0)
        & (candidate_u < width)
        & (candidate_v >= 0.0)
        & (candidate_v < height)
    )
    admitted_indices = np.flatnonzero(admitted)
    warped_luma = np.zeros((height, width), dtype=np.float32)
    warped_depth = np.full((height, width), np.inf, dtype=np.float64)
    direct_coverage = np.zeros((height, width), dtype=bool)
    if admitted_indices.size:
        destination_columns = np.rint(candidate_u[admitted_indices]).astype(np.int64)
        destination_rows = np.rint(candidate_v[admitted_indices]).astype(np.int64)
        destination_flat = destination_rows * width + destination_columns
        order = np.lexsort((candidate_z[admitted_indices], destination_flat))
        sorted_flat = destination_flat[order]
        first = np.concatenate((np.asarray([True]), sorted_flat[1:] != sorted_flat[:-1]))
        winners = admitted_indices[order[first]]
        winning_flat = destination_flat[order[first]]
        warped_depth.reshape(-1)[winning_flat] = candidate_z[winners]
        warped_luma.reshape(-1)[winning_flat] = reference_luma.reshape(-1)[winners].astype(np.float32)
        direct_coverage.reshape(-1)[winning_flat] = True
    kernel_size = 2 * WARP_COVERAGE_DILATION_PX + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    explained_coverage = cv2.dilate(direct_coverage.astype(np.uint8), kernel, iterations=1).astype(bool)
    return warped_luma, direct_coverage, explained_coverage


def reprojection_visibility_features(
    context: scorer.ReferenceContext,
    pair: Any,
    reference_planes: np.ndarray,
    candidate_planes: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    reference_luma = reference_planes[0].astype(np.float32)
    candidate_luma = candidate_planes[0].astype(np.float32)
    warped_luma, direct_coverage, explained_coverage = _forward_z_buffer_warp(
        context, pair, reference_luma
    )
    residual_map = np.abs(candidate_luma - warped_luma)
    direct_residuals = residual_map[direct_coverage]
    residual_threshold = (
        float(np.quantile(direct_residuals, PHOTOMETRIC_RESIDUAL_QUANTILE))
        if direct_residuals.size
        else float("inf")
    )
    high_residual = direct_coverage & (residual_map > residual_threshold)

    points_rows: list[np.ndarray] = []
    unknown_rows: list[np.ndarray] = []
    for query_index, query in enumerate(context.queries):
        centers, _along = scorer._cell_centers(query)
        points_rows.append(centers)
        unknown_rows.append(~context.static[query_index].reshape(-1))
    points_reference = np.concatenate(points_rows, axis=0)
    unknown = np.concatenate(unknown_rows)
    relative = np.linalg.inv(context.row.reference.camera_to_world) @ pair.neighbor.camera_to_world
    reference_to_candidate = np.linalg.inv(relative)
    points_candidate = (
        points_reference @ reference_to_candidate[:3, :3].T
        + reference_to_candidate[:3, 3]
    )
    intrinsics = context.intrinsics
    candidate_z = points_candidate[:, 2]
    candidate_u = intrinsics[0, 0] * points_candidate[:, 0] / np.maximum(candidate_z, 1e-9) + intrinsics[0, 2]
    candidate_v = intrinsics[1, 1] * points_candidate[:, 1] / np.maximum(candidate_z, 1e-9) + intrinsics[1, 2]
    height, width = context.low_depth.shape
    candidate_inside = (
        unknown
        & (candidate_z >= adapter.DEPTH_RANGE_M[0])
        & (candidate_z <= adapter.DEPTH_RANGE_M[1])
        & (candidate_u >= 0.0)
        & (candidate_u < width)
        & (candidate_v >= 0.0)
        & (candidate_v < height)
    )
    candidate_columns = np.clip(np.rint(candidate_u).astype(np.int64), 0, width - 1)
    candidate_rows = np.clip(np.rint(candidate_v).astype(np.int64), 0, height - 1)
    unexplained = candidate_inside & ~explained_coverage[candidate_rows, candidate_columns]
    inconsistent = candidate_inside & high_residual[candidate_rows, candidate_columns]
    novel = unexplained | inconsistent
    candidate_gradient = candidate_planes[1, candidate_rows, candidate_columns].astype(np.float32)
    candidate_texture = candidate_planes[2, candidate_rows, candidate_columns].astype(np.float32)
    appearance_strength = np.maximum(candidate_gradient, candidate_texture) * candidate_inside
    analytic = {
        "reprojection_novel_cell_count": float(np.sum(novel)),
        "unexplained_warp_hole_cell_count": float(np.sum(unexplained)),
        "photometric_inconsistent_cell_count": float(np.sum(inconsistent)),
        "novel_appearance_strength_sum": float(np.sum(appearance_strength * novel)),
        "candidate_visible_unknown_cell_count": float(np.sum(candidate_inside)),
        "direct_warp_coverage_fraction": float(np.mean(direct_coverage)),
        "explained_warp_coverage_fraction": float(np.mean(explained_coverage)),
        "photometric_residual_threshold": residual_threshold,
    }
    features = np.asarray(
        [
            analytic["reprojection_novel_cell_count"],
            analytic["unexplained_warp_hole_cell_count"],
            analytic["photometric_inconsistent_cell_count"],
            analytic["novel_appearance_strength_sum"],
            analytic["candidate_visible_unknown_cell_count"],
            analytic["direct_warp_coverage_fraction"],
            pair.translation_m,
            pair.rotation_deg,
        ],
        dtype=np.float64,
    )
    require(features.shape == (8,) and np.all(np.isfinite(features)), "R27 feature drift")
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
                records[index].analytic["reprojection_novel_cell_count"],
                records[index].analytic["novel_appearance_strength_sum"],
                records[index].pair.translation_m,
                records[index].pair.rotation_deg,
                records[index].pair.neighbor.frame_id,
            ),
        )
        advantage = int(
            records[best].analytic["reprojection_novel_cell_count"]
            - records[generic].analytic["reprojection_novel_cell_count"]
        )
        selected = best if advantage >= MINIMUM_OVERRIDE_NOVEL_CELL_ADVANTAGE else generic
        scores[selected] = 1.0
        overrides += int(selected != generic)
        receipts.append(
            {
                "reference_id": reference_id,
                "generic_neighbor_id": records[generic].pair.neighbor.frame_id,
                "selected_neighbor_id": records[selected].pair.neighbor.frame_id,
                "predicted_novel_cell_advantage": advantage,
                "overrode_generic": selected != generic,
            }
        )
    return scores, {
        "reference_count": len(by_reference),
        "generic_override_count": overrides,
        "generic_fallback_count": len(by_reference) - overrides,
        "selection_receipt_sha256": hashlib.sha256(r21.shared.canonical_json_bytes(receipts)).hexdigest().upper(),
    }


def ungated_scores(records: Sequence[scorer.CandidateRecord]) -> np.ndarray:
    return np.asarray(
        [
            record.analytic["reprojection_novel_cell_count"] * 1000.0
            + record.analytic["novel_appearance_strength_sum"]
            for record in records
        ],
        dtype=np.float64,
    )


def _build_tum_records() -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    store = r25.RgbStore(r25._tum_rgb_assets(r21.TUM_MANIFESTS), "frozen TUM rgb.txt identities and source manifests")

    def feature(context: scorer.ReferenceContext, pair: Any) -> tuple[np.ndarray, dict[str, float]]:
        return reprojection_visibility_features(
            context, pair, store.planes(pair.reference), store.planes(pair.neighbor)
        )

    try:
        records, source, abstained = r21._build_tum_records(feature)
        source["rgb_signal_receipt"] = store.receipt()
        return records, source, abstained
    finally:
        store.close()


def _build_bonn_records(root: Path) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    store = r25.RgbStore({}, "Bonn rgb.txt-associated direct RGB paths")

    def feature(context: scorer.ReferenceContext, pair: Any) -> tuple[np.ndarray, dict[str, float]]:
        return reprojection_visibility_features(
            context, pair, store.planes(pair.reference), store.planes(pair.neighbor)
        )

    try:
        records, source, abstained = r21._build_bonn_records(root, feature)
        source["rgb_signal_receipt"] = store.receipt()
        return records, source, abstained
    finally:
        store.close()


def _build_arkit_records(root: Path) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    store = r25.RgbStore(r25._arkit_rgb_assets(root), "ARKitScenes manifest lowres_wide identities")

    def feature(context: scorer.ReferenceContext, pair: Any) -> tuple[np.ndarray, dict[str, float]]:
        return reprojection_visibility_features(
            context, pair, store.planes(pair.reference), store.planes(pair.neighbor)
        )

    try:
        records, source, abstained = r21._build_arkit_records(root, feature)
        source["rgb_signal_receipt"] = store.receipt()
        return records, source, abstained
    finally:
        store.close()


def evaluate(bonn_root: Path, arkit_root: Path) -> dict[str, Any]:
    datasets = {
        "TUM_RGBD": _build_tum_records(),
        "BONN_RGBD_DYNAMIC": _build_bonn_records(bonn_root),
        "ARKITSCENES": _build_arkit_records(arkit_root),
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
            "ungated_metrics": r21.fold_metrics(records, ungated_scores(records)),
        }
    passed = all(all(row["primary_metrics"]["checks"].values()) for row in source_results.values())
    terminal = (
        "TASK_EVIDENCE_REPROJECTION_VISIBILITY_THREE_SOURCE_PASS"
        if passed
        else "STOP_TASK_EVIDENCE_REPROJECTION_VISIBILITY_TRANSFER_FAIL"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "CONSUMED_THREE_SOURCE_DEVELOPMENT_FIXED_REPROJECTION",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside nine frozen body/path capsules; UNKNOWN remains unknown.",
        "mechanism": {
            "name": "QUERY_ALIGNED_RGBD_TO_RGB_Z_BUFFER_REPROJECTION_VISIBILITY",
            "warp_coverage_dilation_px": WARP_COVERAGE_DILATION_PX,
            "photometric_residual_quantile": PHOTOMETRIC_RESIDUAL_QUANTILE,
            "minimum_override_novel_cell_advantage": MINIMUM_OVERRIDE_NOVEL_CELL_ADVANTAGE,
            "generic_fallback": True,
            "candidate_depth_in_scorer_input": False,
            "training_steps": 0,
            "parameters_fit_from_targets": 0,
        },
        "sources": source_results,
        "terminal": terminal,
        "fresh_confirmation_source_lock_authorized": passed,
        "android_candidate_authorized": False,
        "read_boundary": {
            "reference_rgb_and_depth_in_scorer_input": True,
            "candidate_rgb_in_scorer_input": True,
            "candidate_depth_in_scorer_input": False,
            "candidate_depth_used_only_after_selection_for_consumed_development_metric": True,
            "network_requests": 0,
        },
        "claim_ceiling": "Consumed three-source Development evidence only. A PASS would authorize a fresh task-outcome-blind source lock, not collision correctness, Android, product, default-App, navigation, or safety claims.",
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
    parser.add_argument("--bonn-root", type=Path, default=r21.shared.DEFAULT_BONN_ROOT)
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
