#!/usr/bin/env python3
"""Task-query tensor representation for the frozen TARO learned ranker.

R21 proved source-family holdout macro transfer but not broad opportunity-parent
coverage. This successor changes only representation: candidate geometry is no
longer collapsed across the nine body/path queries. Network, loss, optimizer,
seeds, residual bound, and LOFO gates remain exactly R21.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_query_tensor_learned_ranker.v1"
QUERY_COUNT = 9
CELL_SHAPE = (
    len(oracle.ALONG_BIN_EDGES_M) - 1,
    len(oracle.ACROSS_BIN_EDGES_M) - 1,
    len(oracle.HEIGHT_BIN_EDGES_M) - 1,
)
GEOMETRY_CHANNELS = ("visible", "parallax", "occluded_parallax", "far_parallax")
STATIC_TENSOR_FEATURE_COUNT = QUERY_COUNT * CELL_SHAPE[0] * CELL_SHAPE[2]
QUERY_GEOMETRY_FEATURE_COUNT = QUERY_COUNT * len(GEOMETRY_CHANNELS)
ALONG_GEOMETRY_FEATURE_COUNT = CELL_SHAPE[0] * len(GEOMETRY_CHANNELS)
HEIGHT_GEOMETRY_FEATURE_COUNT = CELL_SHAPE[2] * len(GEOMETRY_CHANNELS)
TOTAL_FEATURE_COUNT = len(scorer.FEATURE_NAMES) + STATIC_TENSOR_FEATURE_COUNT + QUERY_GEOMETRY_FEATURE_COUNT + ALONG_GEOMETRY_FEATURE_COUNT + HEIGHT_GEOMETRY_FEATURE_COUNT


def query_tensor_candidate_features(
    context: scorer.ReferenceContext,
    pair: Any,
) -> tuple[np.ndarray, dict[str, float]]:
    base_features, analytic = scorer.source_time_candidate_features(context, pair)
    shared.require(context.static.shape == (QUERY_COUNT, *CELL_SHAPE), "query tensor static shape drift")
    relative = np.linalg.inv(context.row.reference.camera_to_world) @ pair.neighbor.camera_to_world
    inverse = np.linalg.inv(relative)
    points_rows = []
    along_rows = []
    for query in context.queries:
        centers, along = scorer._cell_centers(query)
        points_rows.append(centers)
        along_rows.append(along)
    points_ref = np.stack(points_rows, axis=0)
    along = np.stack(along_rows, axis=0)
    unknown = ~context.static.reshape(QUERY_COUNT, -1)
    points_neighbor = points_ref @ inverse[:3, :3].T + inverse[:3, 3]
    ref_z = points_ref[..., 2]
    neighbor_z = points_neighbor[..., 2]
    width, height = tum.LOW_SIZE_WH
    k = context.intrinsics
    ref_u = k[0, 0] * points_ref[..., 0] / np.maximum(ref_z, 1e-9) + k[0, 2]
    ref_v = k[1, 1] * points_ref[..., 1] / np.maximum(ref_z, 1e-9) + k[1, 2]
    nei_u = k[0, 0] * points_neighbor[..., 0] / np.maximum(neighbor_z, 1e-9) + k[0, 2]
    nei_v = k[1, 1] * points_neighbor[..., 1] / np.maximum(neighbor_z, 1e-9) + k[1, 2]
    visible = unknown & (neighbor_z >= adapter.DEPTH_RANGE_M[0]) & (neighbor_z <= adapter.DEPTH_RANGE_M[1]) & (nei_u >= 0.0) & (nei_u < width) & (nei_v >= 0.0) & (nei_v < height)
    parallax_weight = np.clip(np.sqrt((nei_u - ref_u) ** 2 + (nei_v - ref_v) ** 2) / 20.0, 0.0, 1.0)
    ref_col = np.clip(np.rint(ref_u).astype(np.int64), 0, width - 1)
    ref_row = np.clip(np.rint(ref_v).astype(np.int64), 0, height - 1)
    sampled = context.low_depth[ref_row, ref_col]
    sample_valid = context.valid[ref_row, ref_col]
    occluded = visible & sample_valid & (sampled + 0.05 < ref_z)
    far_weight = np.clip(along / adapter.HORIZON_M, 0.0, 1.0)
    channels = np.stack(
        (
            visible.astype(np.float64),
            parallax_weight * visible,
            parallax_weight * occluded,
            parallax_weight * visible * far_weight,
        ),
        axis=-1,
    ).reshape(QUERY_COUNT, *CELL_SHAPE, len(GEOMETRY_CHANNELS))
    static_tensor = np.mean(context.static, axis=2).reshape(-1)
    query_geometry = np.sum(channels, axis=(1, 2, 3)).reshape(-1)
    along_geometry = np.sum(channels, axis=(0, 2, 3)).reshape(-1)
    height_geometry = np.sum(channels, axis=(0, 1, 2)).reshape(-1)
    features = np.concatenate((base_features, static_tensor, query_geometry, along_geometry, height_geometry)).astype(np.float64)
    shared.require(features.shape == (TOTAL_FEATURE_COUNT,) and np.all(np.isfinite(features)), "query tensor feature drift")
    return features, analytic


def evaluate(bonn_root: Path, arkit_root: Path) -> dict[str, Any]:
    datasets = {
        "TUM_RGBD": r21._build_tum_records(query_tensor_candidate_features),
        "BONN_RGBD_DYNAMIC": r21._build_bonn_records(bonn_root, query_tensor_candidate_features),
        "ARKITSCENES": r21._build_arkit_records(arkit_root, query_tensor_candidate_features),
    }
    result = r21.run_lofo(
        datasets,
        schema=SCHEMA,
        feature_contract={
            "representation": "base source-time features plus reference static query-along-height tensor and candidate query/along/height geometry channels",
            "base_feature_names": list(scorer.FEATURE_NAMES),
            "static_tensor_shape": [QUERY_COUNT, CELL_SHAPE[0], CELL_SHAPE[2]],
            "candidate_geometry_channels": list(GEOMETRY_CHANNELS),
            "candidate_query_shape": [QUERY_COUNT, len(GEOMETRY_CHANNELS)],
            "candidate_along_shape": [CELL_SHAPE[0], len(GEOMETRY_CHANNELS)],
            "candidate_height_shape": [CELL_SHAPE[2], len(GEOMETRY_CHANNELS)],
            "raw_feature_count": TOTAL_FEATURE_COUNT,
            "per_reference_zscore": True,
            "per_reference_minmax": True,
            "neighbor_depth_in_input": False,
            "network_loss_optimizer_seeds_and_gates_unchanged_from_r21": True,
        },
    )
    result["representation_successor_of"] = "blindassist.taro.task_evidence_cross_source_learned_ranker.v1"
    result["content_sha256"] = hashlib.sha256(shared.canonical_json_bytes({key: value for key, value in result.items() if key != "content_sha256"})).hexdigest().upper()
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
