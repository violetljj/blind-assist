#!/usr/bin/env python3
"""R31 multi-candidate reliability and occlusion-consistency Development.

The scorer consumes reference RGB-D, candidate RGB, source-native poses, and
the complete frozen candidate set for one reference.  Candidate depth remains
target-side only.  Per-cell rigid-warp reliability is encoded before a
permutation-equivariant candidate-set transformer predicts utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_openloris_home_frontdoor as openloris
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_query_conditioned_no_regret_ranker as r23
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_reprojection_visibility_scorer as r27
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_rgb_query_interaction_ranker as r25
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_multi_candidate_reliability_consistency.v7"
CACHE_SCHEMA = "blindassist.taro.r31_feature_cache.v1"
SOURCE_NAMES = ("ARKITSCENES", "BONN_RGBD_DYNAMIC", "TUM_RGBD", "OPENLORIS_HOME")
QUERY_COUNT = 9
CELL_COUNT_PER_QUERY = (
    (len(oracle.ALONG_BIN_EDGES_M) - 1)
    * (len(oracle.ACROSS_BIN_EDGES_M) - 1)
    * (len(oracle.HEIGHT_BIN_EDGES_M) - 1)
)
CELL_CHANNEL_NAMES = (
    "candidate_visible_unknown",
    "unexplained_warp_hole",
    "robust_photometric_inconsistency",
    "candidate_appearance_strength",
    "direct_rigid_correspondence",
    "normalized_affine_residual",
)
GLOBAL_FEATURE_NAMES = (
    "robust_novel_cell_count",
    "unexplained_warp_hole_cell_count",
    "robust_photometric_inconsistent_cell_count",
    "robust_novel_appearance_strength_sum",
    "candidate_visible_unknown_cell_count",
    "direct_warp_coverage_fraction",
    "explained_warp_coverage_fraction",
    "affine_gain",
    "affine_bias",
    "affine_residual_median",
    "affine_residual_mad",
    "affine_residual_p90",
    "affine_inlier_fraction",
    "translation_m",
    "rotation_deg",
    "gap_s",
)
SET_CONSISTENCY_FEATURE_NAMES = (
    "novel_peer_support_mean",
    "peer_supported_novel_cell_count",
    "isolated_novel_cell_count",
    "novel_mask_deviation_from_set_mean",
    "novel_jaccard_with_generic",
    "visible_peer_support_mean",
    "supported_novel_appearance_sum",
    "hole_peer_support_mean",
    "photometric_peer_support_mean",
)
SEEDS = (31011, 31029, 31047)
EPOCHS = 240
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0002
REGRESSION_WEIGHT = 0.15
GENERIC_RELATIVE_WEIGHT = 0.25
OPPORTUNITY_LOSS_WEIGHT = 1.0
EMBED_DIM = 64
ATTENTION_HEADS = 4
TRANSFORMER_LAYERS = 2
LCB_Z = 0.5
RESIDUAL_SCALE = 1.0


class R31Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R31Error(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


@dataclass(frozen=True)
class DetailedCandidate:
    global_features: np.ndarray
    cell_features: np.ndarray
    analytic: dict[str, float]


@dataclass
class SourceDataset:
    source: str
    records: list[scorer.CandidateRecord]
    cells: np.ndarray
    receipt: dict[str, Any]
    abstentions: int


def _robust_affine_residual(
    warped_luma: np.ndarray,
    candidate_luma: np.ndarray,
    direct_coverage: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    require(warped_luma.shape == candidate_luma.shape == direct_coverage.shape, "R31 affine input shape drift")
    reference = warped_luma[direct_coverage].astype(np.float64)
    candidate = candidate_luma[direct_coverage].astype(np.float64)
    if reference.size < 64:
        gain = 1.0
        bias = 0.0
    else:
        reference_q10, reference_q90 = np.quantile(reference, [0.10, 0.90])
        candidate_q10, candidate_q90 = np.quantile(candidate, [0.10, 0.90])
        reference_span = max(float(reference_q90 - reference_q10), 1e-3)
        gain = float(np.clip((candidate_q90 - candidate_q10) / reference_span, 0.5, 2.0))
        bias = float(np.median(candidate - gain * reference))
    residual = np.abs(candidate_luma.astype(np.float64) - (gain * warped_luma.astype(np.float64) + bias))
    direct_residual = residual[direct_coverage]
    if direct_residual.size:
        median = float(np.median(direct_residual))
        mad = float(np.median(np.abs(direct_residual - median)))
        p90 = float(np.quantile(direct_residual, 0.90))
        threshold = max(0.05, median + 3.0 * max(1.4826 * mad, 1e-3))
        inlier_fraction = float(np.mean(direct_residual <= threshold))
    else:
        median = mad = p90 = 0.0
        threshold = float("inf")
        inlier_fraction = 0.0
    return np.ascontiguousarray(residual, dtype=np.float32), {
        "affine_gain": gain,
        "affine_bias": bias,
        "affine_residual_median": median,
        "affine_residual_mad": mad,
        "affine_residual_p90": p90,
        "affine_residual_threshold": float(threshold),
        "affine_inlier_fraction": inlier_fraction,
    }


def reliability_consistency_features(
    context: scorer.ReferenceContext,
    pair: bonn.Pair,
    reference_planes: np.ndarray,
    candidate_planes: np.ndarray,
) -> DetailedCandidate:
    reference_luma = reference_planes[0].astype(np.float32)
    candidate_luma = candidate_planes[0].astype(np.float32)
    warped_luma, direct_coverage, explained_coverage = r27._forward_z_buffer_warp(
        context, pair, reference_luma
    )
    residual, reliability = _robust_affine_residual(warped_luma, candidate_luma, direct_coverage)

    points_rows: list[np.ndarray] = []
    unknown_rows: list[np.ndarray] = []
    for query_index, query in enumerate(context.queries):
        centers, _along = scorer._cell_centers(query)
        points_rows.append(centers)
        unknown_rows.append(~context.static[query_index].reshape(-1))
    points_reference = np.concatenate(points_rows, axis=0)
    unknown = np.concatenate(unknown_rows)
    require(len(points_reference) == QUERY_COUNT * CELL_COUNT_PER_QUERY, "R31 query cell count drift")

    relative = np.linalg.inv(context.row.reference.camera_to_world) @ pair.neighbor.camera_to_world
    reference_to_candidate = np.linalg.inv(relative)
    points_candidate = points_reference @ reference_to_candidate[:3, :3].T + reference_to_candidate[:3, 3]
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
    columns = np.clip(np.rint(candidate_u).astype(np.int64), 0, width - 1)
    rows = np.clip(np.rint(candidate_v).astype(np.int64), 0, height - 1)
    direct_cells = candidate_inside & direct_coverage[rows, columns]
    holes = candidate_inside & ~explained_coverage[rows, columns]
    robust_photo = direct_cells & (residual[rows, columns] > reliability["affine_residual_threshold"])
    novel = holes | robust_photo
    appearance = np.maximum(
        candidate_planes[1, rows, columns].astype(np.float32),
        candidate_planes[2, rows, columns].astype(np.float32),
    ) * candidate_inside
    normalized_residual = np.zeros_like(candidate_z, dtype=np.float32)
    scale = max(reliability["affine_residual_median"] + 3.0 * max(1.4826 * reliability["affine_residual_mad"], 1e-3), 0.05)
    normalized_residual[candidate_inside] = np.clip(residual[rows[candidate_inside], columns[candidate_inside]] / scale, 0.0, 5.0) / 5.0

    cell_features = np.stack(
        (
            candidate_inside.astype(np.float32),
            holes.astype(np.float32),
            robust_photo.astype(np.float32),
            appearance.astype(np.float32),
            direct_cells.astype(np.float32),
            normalized_residual,
        ),
        axis=1,
    ).reshape(QUERY_COUNT, CELL_COUNT_PER_QUERY, len(CELL_CHANNEL_NAMES)).transpose(0, 2, 1)
    analytic = {
        "robust_novel_cell_count": float(np.sum(novel)),
        "unexplained_warp_hole_cell_count": float(np.sum(holes)),
        "robust_photometric_inconsistent_cell_count": float(np.sum(robust_photo)),
        "robust_novel_appearance_strength_sum": float(np.sum(appearance * novel)),
        "candidate_visible_unknown_cell_count": float(np.sum(candidate_inside)),
        "direct_warp_coverage_fraction": float(np.mean(direct_coverage)),
        "explained_warp_coverage_fraction": float(np.mean(explained_coverage)),
        **reliability,
    }
    global_features = np.asarray(
        [
            analytic["robust_novel_cell_count"],
            analytic["unexplained_warp_hole_cell_count"],
            analytic["robust_photometric_inconsistent_cell_count"],
            analytic["robust_novel_appearance_strength_sum"],
            analytic["candidate_visible_unknown_cell_count"],
            analytic["direct_warp_coverage_fraction"],
            analytic["explained_warp_coverage_fraction"],
            analytic["affine_gain"],
            analytic["affine_bias"],
            analytic["affine_residual_median"],
            analytic["affine_residual_mad"],
            analytic["affine_residual_p90"],
            analytic["affine_inlier_fraction"],
            pair.translation_m,
            pair.rotation_deg,
            pair.gap_s,
        ],
        dtype=np.float64,
    )
    require(
        global_features.shape == (len(GLOBAL_FEATURE_NAMES),)
        and cell_features.shape == (QUERY_COUNT, len(CELL_CHANNEL_NAMES), CELL_COUNT_PER_QUERY)
        and np.all(np.isfinite(global_features))
        and np.all(np.isfinite(cell_features)),
        "R31 feature drift",
    )
    return DetailedCandidate(global_features, np.ascontiguousarray(cell_features, dtype=np.float32), analytic)


def _detail_key(reference_id: str, neighbor_id: str) -> str:
    return f"{reference_id}\u0000{neighbor_id}"


def _standard_source_dataset(source: str, bonn_root: Path, arkit_root: Path) -> SourceDataset:
    details: dict[str, np.ndarray] = {}
    if source == "TUM_RGBD":
        store = r25.RgbStore(r25._tum_rgb_assets(r21.TUM_MANIFESTS), "R31 TUM RGB")
        builder = lambda feature: r21._build_tum_records(feature)
    elif source == "BONN_RGBD_DYNAMIC":
        store = r25.RgbStore({}, "R31 Bonn RGB")
        builder = lambda feature: r21._build_bonn_records(bonn_root, feature)
    elif source == "ARKITSCENES":
        store = r25.RgbStore(r25._arkit_rgb_assets(arkit_root), "R31 ARKit RGB")
        builder = lambda feature: r21._build_arkit_records(arkit_root, feature)
    else:
        raise R31Error(f"unsupported standard source: {source}")

    def feature(context: scorer.ReferenceContext, pair: bonn.Pair) -> tuple[np.ndarray, dict[str, float]]:
        detail = reliability_consistency_features(
            context, pair, store.planes(pair.reference), store.planes(pair.neighbor)
        )
        details[_detail_key(context.row.reference.frame_id, pair.neighbor.frame_id)] = detail.cell_features
        return detail.global_features, detail.analytic

    try:
        records, receipt, abstentions = builder(feature)
        receipt["rgb_signal_receipt"] = store.receipt()
    finally:
        store.close()
    cells = np.stack([details[_detail_key(record.reference_id, record.pair.neighbor.frame_id)] for record in records])
    return SourceDataset(source, records, cells, receipt, abstentions)


def _openloris_dataset(source_root: Path, groundtruth_root: Path) -> SourceDataset:
    frames, assets, source = openloris.load_outcome_blind_roster(source_root, groundtruth_root)
    selected, capability = openloris.balanced.select_pose_capable_references(
        frames, openloris.MAX_REFERENCES_PER_PARENT
    )
    proposals, candidate_identity_sha = openloris._candidate_identity(selected)
    store = openloris.PayloadStore(assets)
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    cell_rows: list[np.ndarray] = []
    for row in selected:
        low, points, valid, _coverage = store.observation(row.reference.frame_id)
        asset = assets[row.reference.frame_id]
        low_intrinsics = bonn._scaled_intrinsics(
            asset.intrinsics, openloris.CROPPED_SIZE_WH, tum.LOW_SIZE_WH
        )
        queries, _receipt = openloris._calibration_queries(row.reference, asset.camera_height_m)
        static = oracle.query_evidence_cells(points, valid, queries)
        context = scorer.ReferenceContext(row, low, points, valid, low_intrinsics, queries, static)
        contexts[row.reference.frame_id] = context
        reference_planes = store.planes(row.reference)
        for pair in proposals[row.reference.frame_id]:
            detail = reliability_consistency_features(
                context, pair, reference_planes, store.planes(pair.neighbor)
            )
            records.append(
                scorer.CandidateRecord(
                    row.reference.parent_id,
                    "CONSUMED_DEVELOPMENT",
                    row.reference.frame_id,
                    pair,
                    detail.global_features,
                    detail.analytic,
                )
            )
            cell_rows.append(detail.cell_features)
    observations = {
        frame_id: store.observation(frame_id)
        for frame_id in sorted({record.pair.neighbor.frame_id for record in records})
    }
    scorer._attach_targets(records, contexts, observations)
    receipt = {
        "source": source,
        "capability": capability,
        "candidate_identity_sha256": candidate_identity_sha,
        "payload_receipt": store.receipt(),
        "prior_fresh_result_content_sha256": "9F8F0F3991F04BB539108A86E8751A0BC3B41B19C3D1C1F068435F5C37BEFFBA",
    }
    return SourceDataset("OPENLORIS_HOME", records, np.stack(cell_rows), receipt, 0)


def _cache_path(cache_root: Path, source: str) -> Path:
    return cache_root / f"{source.lower()}-v1.npz"


def save_cache(cache_root: Path, dataset: SourceDataset) -> dict[str, Any]:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_root, dataset.source)
    require(not path.exists(), f"R31 cache already exists: {path}")
    records = dataset.records
    np.savez_compressed(
        path,
        schema=np.asarray(CACHE_SCHEMA),
        source=np.asarray(dataset.source),
        parent_ids=np.asarray([record.parent_id for record in records]),
        reference_ids=np.asarray([record.reference_id for record in records]),
        neighbor_ids=np.asarray([record.pair.neighbor.frame_id for record in records]),
        gap_s=np.asarray([record.pair.gap_s for record in records], dtype=np.float64),
        translation_m=np.asarray([record.pair.translation_m for record in records], dtype=np.float64),
        rotation_deg=np.asarray([record.pair.rotation_deg for record in records], dtype=np.float64),
        targets=np.asarray([record.target_gain for record in records], dtype=np.int32),
        coverage=np.asarray([record.coverage for record in records], dtype=np.float64),
        global_features=np.stack([record.features for record in records]).astype(np.float32),
        cell_features=dataset.cells.astype(np.float16),
        receipt_json=np.asarray(json.dumps(dataset.receipt, sort_keys=True, separators=(",", ":"))),
        abstentions=np.asarray(dataset.abstentions, dtype=np.int64),
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "candidate_count": len(records),
    }


def _frame_from_id(frame_id: str) -> bonn.Frame:
    parent_id, timestamp = frame_id.rsplit(":", 1)
    return bonn.Frame(parent_id, float(timestamp), Path(frame_id), Path(frame_id), np.eye(4, dtype=np.float64))


def load_cache(cache_root: Path, source: str) -> SourceDataset:
    path = _cache_path(cache_root, source)
    require(path.is_file(), f"R31 cache absent: {path}")
    with np.load(path, allow_pickle=False) as value:
        require(str(value["schema"]) == CACHE_SCHEMA and str(value["source"]) == source, "R31 cache identity drift")
        parent_ids = value["parent_ids"].astype(str)
        reference_ids = value["reference_ids"].astype(str)
        neighbor_ids = value["neighbor_ids"].astype(str)
        gap = value["gap_s"].astype(np.float64)
        translation = value["translation_m"].astype(np.float64)
        rotation = value["rotation_deg"].astype(np.float64)
        targets = value["targets"].astype(np.int64)
        coverage = value["coverage"].astype(np.float64)
        global_features = value["global_features"].astype(np.float64)
        cells = value["cell_features"].astype(np.float32)
        receipt = json.loads(str(value["receipt_json"]))
        abstentions = int(value["abstentions"])
    records: list[scorer.CandidateRecord] = []
    for index in range(len(parent_ids)):
        reference = _frame_from_id(reference_ids[index])
        neighbor = _frame_from_id(neighbor_ids[index])
        pair = bonn.Pair(reference, neighbor, float(gap[index]), float(translation[index]), float(rotation[index]))
        analytic = {
            name: float(global_features[index, position])
            for position, name in enumerate(GLOBAL_FEATURE_NAMES)
        }
        records.append(
            scorer.CandidateRecord(
                str(parent_ids[index]),
                "CONSUMED_DEVELOPMENT_CACHE",
                str(reference_ids[index]),
                pair,
                global_features[index],
                analytic,
                int(targets[index]),
                float(coverage[index]),
            )
        )
    require(cells.shape[0] == len(records), "R31 cached cell alignment drift")
    return SourceDataset(source, records, cells, receipt, abstentions)


def _set_consistency_rows(dataset: SourceDataset) -> np.ndarray:
    records = dataset.records
    output = np.zeros((len(records), len(SET_CONSISTENCY_FEATURE_NAMES)), dtype=np.float64)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.reference_id].append(index)
    for indices in groups.values():
        cells = dataset.cells[indices]
        visible = cells[:, :, 0, :] > 0.5
        holes = cells[:, :, 1, :] > 0.5
        photometric = cells[:, :, 2, :] > 0.5
        appearance = cells[:, :, 3, :].astype(np.float64)
        novel = holes | photometric
        count = len(indices)
        novel_frequency = np.mean(novel, axis=0)
        visible_frequency = np.mean(visible, axis=0)
        hole_frequency = np.mean(holes, axis=0)
        photometric_frequency = np.mean(photometric, axis=0)
        generic_global = r27._generic_index(records, indices)
        generic_local = indices.index(generic_global)
        generic_novel = novel[generic_local]
        for local_index, record_index in enumerate(indices):
            peer_novel = (
                (np.sum(novel, axis=0) - novel[local_index]) / float(count - 1)
                if count > 1
                else np.zeros_like(novel_frequency, dtype=np.float64)
            )
            peer_visible = (
                (np.sum(visible, axis=0) - visible[local_index]) / float(count - 1)
                if count > 1
                else np.zeros_like(visible_frequency, dtype=np.float64)
            )
            peer_holes = (
                (np.sum(holes, axis=0) - holes[local_index]) / float(count - 1)
                if count > 1
                else np.zeros_like(hole_frequency, dtype=np.float64)
            )
            peer_photometric = (
                (np.sum(photometric, axis=0) - photometric[local_index]) / float(count - 1)
                if count > 1
                else np.zeros_like(photometric_frequency, dtype=np.float64)
            )
            candidate_novel = novel[local_index]
            candidate_visible = visible[local_index]
            novel_count = max(int(np.sum(candidate_novel)), 1)
            visible_count = max(int(np.sum(candidate_visible)), 1)
            intersection = int(np.sum(candidate_novel & generic_novel))
            union = int(np.sum(candidate_novel | generic_novel))
            output[record_index] = np.asarray(
                [
                    float(np.sum(peer_novel * candidate_novel) / novel_count),
                    float(np.sum(candidate_novel & (peer_novel >= 0.5))),
                    float(np.sum(candidate_novel & (peer_novel < 0.25))),
                    float(np.mean(np.abs(candidate_novel.astype(np.float64) - novel_frequency))),
                    float(intersection / union) if union else 1.0,
                    float(np.sum(peer_visible * candidate_visible) / visible_count),
                    float(np.sum(appearance[local_index] * candidate_novel * peer_novel)),
                    float(np.sum(peer_holes * holes[local_index]) / max(int(np.sum(holes[local_index])), 1)),
                    float(
                        np.sum(peer_photometric * photometric[local_index])
                        / max(int(np.sum(photometric[local_index])), 1)
                    ),
                ],
                dtype=np.float64,
            )
    require(np.all(np.isfinite(output)), "R31 set-consistency feature non-finite")
    return output


@dataclass(frozen=True)
class SetInputs:
    global_tokens: np.ndarray
    cell_tokens: np.ndarray
    base: np.ndarray
    padding_mask: np.ndarray
    flat_indices: np.ndarray


class SetTransform:
    def __init__(self, mean: np.ndarray, scale: np.ndarray, set_mean: np.ndarray, set_scale: np.ndarray):
        self.mean = mean
        self.scale = scale
        self.set_mean = set_mean
        self.set_scale = set_scale

    @classmethod
    def fit(cls, datasets: Sequence[SourceDataset]) -> "SetTransform":
        values = np.concatenate([np.stack([record.features for record in dataset.records]) for dataset in datasets])
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale[scale < 1e-6] = 1.0
        set_values = np.concatenate([_set_consistency_rows(dataset) for dataset in datasets])
        set_mean = np.mean(set_values, axis=0)
        set_scale = np.std(set_values, axis=0)
        set_scale[set_scale < 1e-6] = 1.0
        return cls(mean, scale, set_mean, set_scale)

    def apply(self, datasets: Sequence[SourceDataset]) -> SetInputs:
        records: list[scorer.CandidateRecord] = []
        cells: list[np.ndarray] = []
        set_rows: list[np.ndarray] = []
        sources: list[str] = []
        for dataset in datasets:
            records.extend(dataset.records)
            cells.extend(dataset.cells)
            set_rows.extend(_set_consistency_rows(dataset))
            sources.extend([dataset.source] * len(dataset.records))
        raw = np.stack([record.features for record in records]).astype(np.float64)
        set_values = np.stack(set_rows).astype(np.float64)
        local_z = np.zeros_like(raw)
        local_unit = np.zeros_like(raw)
        set_local_z = np.zeros_like(set_values)
        set_local_unit = np.zeros_like(set_values)
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, (source, record) in enumerate(zip(sources, records, strict=True)):
            groups[(source, record.reference_id)].append(index)
        for indices in groups.values():
            local = raw[indices]
            local_mean = np.mean(local, axis=0)
            local_scale = np.std(local, axis=0)
            local_scale[local_scale < 1e-6] = 1.0
            local_z[indices] = (local - local_mean) / local_scale
            low = np.min(local, axis=0)
            span = np.max(local, axis=0) - low
            span[span < 1e-6] = 1.0
            local_unit[indices] = (local - low) / span
            local_set = set_values[indices]
            local_set_mean = np.mean(local_set, axis=0)
            local_set_scale = np.std(local_set, axis=0)
            local_set_scale[local_set_scale < 1e-6] = 1.0
            set_local_z[indices] = (local_set - local_set_mean) / local_set_scale
            set_low = np.min(local_set, axis=0)
            set_span = np.max(local_set, axis=0) - set_low
            set_span[set_span < 1e-6] = 1.0
            set_local_unit[indices] = (local_set - set_low) / set_span
        transformed = np.concatenate(
            (
                (raw - self.mean) / self.scale,
                local_z,
                local_unit,
                (set_values - self.set_mean) / self.set_scale,
                set_local_z,
                set_local_unit,
            ),
            axis=1,
        )
        maximum = max(len(indices) for indices in groups.values())
        group_rows = list(groups.values())
        global_tokens = np.zeros((len(group_rows), maximum, transformed.shape[1]), dtype=np.float32)
        cell_tokens = np.zeros(
            (len(group_rows), maximum, QUERY_COUNT, len(CELL_CHANNEL_NAMES) * CELL_COUNT_PER_QUERY),
            dtype=np.float32,
        )
        base = np.zeros((len(group_rows), maximum), dtype=np.float32)
        padding = np.ones((len(group_rows), maximum), dtype=bool)
        flat_indices = np.full((len(group_rows), maximum), -1, dtype=np.int64)
        translation_index = GLOBAL_FEATURE_NAMES.index("translation_m")
        rotation_index = GLOBAL_FEATURE_NAMES.index("rotation_deg")
        for row, indices in enumerate(group_rows):
            count = len(indices)
            global_tokens[row, :count] = np.clip(transformed[indices], -6.0, 6.0)
            selected_cells = np.stack([cells[index] for index in indices])
            cell_tokens[row, :count] = selected_cells.reshape(count, QUERY_COUNT, -1)
            base[row, :count] = local_unit[indices, translation_index] + 0.20 * local_unit[indices, rotation_index]
            padding[row, :count] = False
            flat_indices[row, :count] = indices
        require(np.all(np.isfinite(global_tokens)) and np.all(np.isfinite(cell_tokens)), "R31 transformed feature non-finite")
        return SetInputs(global_tokens, cell_tokens, base, padding, flat_indices)

    def receipt_sha256(self) -> str:
        return hashlib.sha256(
            self.mean.tobytes()
            + self.scale.tobytes()
            + self.set_mean.tobytes()
            + self.set_scale.tobytes()
        ).hexdigest().upper()


class CandidateSetRanker(nn.Module):
    def __init__(self, global_width: int, cell_width: int):
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.Linear(global_width, 96), nn.GELU(), nn.Linear(96, EMBED_DIM)
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_width, 160), nn.GELU(), nn.Linear(160, EMBED_DIM)
        )
        self.query_embedding = nn.Parameter(torch.empty(QUERY_COUNT, EMBED_DIM))
        nn.init.normal_(self.query_embedding, mean=0.0, std=0.02)
        self.candidate_fusion = nn.Sequential(
            nn.Linear(EMBED_DIM * 3, 128), nn.GELU(), nn.Linear(128, EMBED_DIM)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=192,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=TRANSFORMER_LAYERS)
        self.utility_output = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, 96), nn.GELU(), nn.Linear(96, 2)
        )
        self.opportunity_output = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, 96), nn.GELU(), nn.Linear(96, 1)
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        cell_tokens: torch.Tensor,
        base: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_embedding = self.global_encoder(global_tokens)
        cell_embedding = self.cell_encoder(cell_tokens) + self.query_embedding.view(1, 1, QUERY_COUNT, EMBED_DIM)
        cell_mean = cell_embedding.mean(dim=2)
        cell_max = cell_embedding.max(dim=2).values
        candidate = self.candidate_fusion(torch.cat((global_embedding, cell_mean, cell_max), dim=-1))
        encoded = self.set_encoder(candidate, src_key_padding_mask=padding_mask)
        representation = torch.cat((candidate, encoded), dim=-1)
        output = self.utility_output(representation)
        score = base + RESIDUAL_SCALE * torch.tanh(output[..., 0])
        log_variance = torch.clamp(output[..., 1], min=-5.0, max=1.0)
        opportunity_logit = self.opportunity_output(representation.detach()).squeeze(-1)
        opportunity_logit = opportunity_logit.masked_fill(padding_mask, -1e9)
        return score.masked_fill(padding_mask, -1e9), log_variance, opportunity_logit


def _torch_inputs(inputs: SetInputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(inputs.global_tokens),
        torch.from_numpy(inputs.cell_tokens),
        torch.from_numpy(inputs.base),
        torch.from_numpy(inputs.padding_mask),
        torch.from_numpy(inputs.flat_indices),
    )


def _flat_scores(padded: torch.Tensor, padding: torch.Tensor, flat_indices: torch.Tensor) -> torch.Tensor:
    values = padded[~padding]
    indices = flat_indices[~padding]
    return values[torch.argsort(indices)]


def _combine(datasets: Sequence[SourceDataset]) -> tuple[list[scorer.CandidateRecord], list[str]]:
    records: list[scorer.CandidateRecord] = []
    sources: list[str] = []
    for dataset in datasets:
        records.extend(dataset.records)
        sources.extend([dataset.source] * len(dataset.records))
    return records, sources


def _generic_relative_pairs(
    records: Sequence[scorer.CandidateRecord],
    sources: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(len(records) == len(sources), "R31 generic-relative source alignment drift")
    by_reference: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (source, record) in enumerate(zip(sources, records, strict=True)):
        by_reference[(source, record.reference_id)].append(index)
    rows: list[tuple[int, int, str, str, float]] = []
    for (source, _reference_id), indices in by_reference.items():
        generic = r27._generic_index(records, indices)
        generic_gain = int(records[generic].target_gain)
        for index in indices:
            gain = int(records[index].target_gain)
            if index == generic or gain == generic_gain:
                continue
            high, low = (index, generic) if gain > generic_gain else (generic, index)
            rows.append((high, low, source, records[index].parent_id, float(abs(gain - generic_gain))))
    require(bool(rows), "R31 generic-relative pair set empty")
    source_parents: dict[str, set[str]] = defaultdict(set)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for _high, _low, source, parent, _magnitude in rows:
        source_parents[source].add(parent)
        counts[(source, parent)] += 1
    source_count = len(source_parents)
    weights = np.asarray(
        [
            magnitude
            / (
                source_count
                * len(source_parents[source])
                * counts[(source, parent)]
            )
            for _high, _low, source, parent, magnitude in rows
        ],
        dtype=np.float64,
    )
    weights *= len(weights) / np.sum(weights)
    return (
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        weights.astype(np.float32),
    )


def _opportunity_targets(
    records: Sequence[scorer.CandidateRecord],
    sources: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label candidates that strictly beat both frozen generic and passive selectors."""
    require(len(records) == len(sources), "R31 opportunity source alignment drift")
    by_reference: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (source, record) in enumerate(zip(sources, records, strict=True)):
        by_reference[(source, record.reference_id)].append(index)
    labels = np.zeros(len(records), dtype=np.float32)
    generic_anchors = np.zeros(len(records), dtype=np.int64)
    passive_anchors = np.zeros(len(records), dtype=np.int64)
    active = np.ones(len(records), dtype=bool)
    for indices in by_reference.values():
        generic = r27._generic_index(records, indices)
        passive = max(
            indices,
            key=lambda index: (
                float(records[index].coverage),
                -records[index].pair.gap_s,
                records[index].pair.neighbor.frame_id,
            ),
        )
        baseline_gain = max(int(records[generic].target_gain), int(records[passive].target_gain))
        for index in indices:
            labels[index] = float(index != generic and int(records[index].target_gain) > baseline_gain)
            generic_anchors[index] = generic
            passive_anchors[index] = passive
            active[index] = index not in {generic, passive}
    require(
        set(np.unique(labels[active]).tolist()) == {0.0, 1.0},
        "R31 opportunity labels lost a class",
    )

    source_parents: dict[str, set[str]] = defaultdict(set)
    class_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    parent_classes: dict[tuple[str, str], set[int]] = defaultdict(set)
    for source, record, label, is_active in zip(sources, records, labels, active, strict=True):
        if not is_active:
            continue
        label_int = int(label)
        source_parents[source].add(record.parent_id)
        parent_classes[(source, record.parent_id)].add(label_int)
        class_counts[(source, record.parent_id, label_int)] += 1
    source_count = len(source_parents)
    weights = np.zeros(len(records), dtype=np.float64)
    for index, (source, record, label, is_active) in enumerate(
        zip(sources, records, labels, active, strict=True)
    ):
        if not is_active:
            continue
        weights[index] = 1.0 / (
            source_count
            * len(source_parents[source])
            * len(parent_classes[(source, record.parent_id)])
            * class_counts[(source, record.parent_id, int(label))]
        )
    weights *= int(np.sum(active)) / np.sum(weights)
    return (
        labels,
        weights.astype(np.float32),
        generic_anchors,
        passive_anchors,
        np.flatnonzero(active).astype(np.int64),
    )


def train_ranker(
    datasets: Sequence[SourceDataset],
    transform: SetTransform,
    seed: int,
) -> tuple[CandidateSetRanker, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    inputs = transform.apply(datasets)
    global_tokens, cells, base, padding, flat_indices = _torch_inputs(inputs)
    records, sources = _combine(datasets)
    high_np, low_np, pair_weights_np = r21._pair_indices(records, sources)
    generic_high_np, generic_low_np, generic_weights_np = _generic_relative_pairs(records, sources)
    targets_np, candidate_weights_np = r23._balanced_candidate_targets(records, sources)
    (
        opportunity_targets_np,
        opportunity_weights_np,
        opportunity_generic_np,
        opportunity_passive_np,
        opportunity_active_np,
    ) = _opportunity_targets(records, sources)
    high = torch.from_numpy(high_np)
    low = torch.from_numpy(low_np)
    pair_weights = torch.from_numpy(pair_weights_np)
    generic_high = torch.from_numpy(generic_high_np)
    generic_low = torch.from_numpy(generic_low_np)
    generic_weights = torch.from_numpy(generic_weights_np)
    targets = torch.from_numpy(targets_np)
    candidate_weights = torch.from_numpy(candidate_weights_np)
    opportunity_targets = torch.from_numpy(opportunity_targets_np)
    opportunity_weights = torch.from_numpy(opportunity_weights_np)
    opportunity_generic = torch.from_numpy(opportunity_generic_np)
    opportunity_passive = torch.from_numpy(opportunity_passive_np)
    opportunity_active = torch.from_numpy(opportunity_active_np)
    model = CandidateSetRanker(global_tokens.shape[-1], cells.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_rank = final_generic_relative = final_regression = final_opportunity = float("nan")
    for _epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        padded_scores, padded_log_variance, padded_opportunity_logits = model(
            global_tokens, cells, base, padding
        )
        scores = _flat_scores(padded_scores, padding, flat_indices)
        log_variance = _flat_scores(padded_log_variance, padding, flat_indices)
        opportunity_logits = _flat_scores(padded_opportunity_logits, padding, flat_indices)
        rank_loss = torch.mean(F.softplus(-(scores[high] - scores[low])) * pair_weights)
        generic_relative_loss = torch.mean(
            F.softplus(-(scores[generic_high] - scores[generic_low])) * generic_weights
        )
        regression_rows = 0.5 * (torch.exp(-log_variance) * (scores - targets) ** 2 + log_variance)
        regression_loss = torch.mean(regression_rows * candidate_weights)
        opportunity_margins = opportunity_logits - opportunity_logits[opportunity_generic]
        opportunity_loss = torch.mean(
            F.binary_cross_entropy_with_logits(
                opportunity_margins[opportunity_active],
                opportunity_targets[opportunity_active],
                reduction="none",
            )
            * opportunity_weights[opportunity_active]
        )
        loss = (
            rank_loss
            + GENERIC_RELATIVE_WEIGHT * generic_relative_loss
            + REGRESSION_WEIGHT * regression_loss
            + OPPORTUNITY_LOSS_WEIGHT * opportunity_loss
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_rank = float(rank_loss.detach())
        final_generic_relative = float(generic_relative_loss.detach())
        final_regression = float(regression_loss.detach())
        final_opportunity = float(opportunity_loss.detach())
    state = b"".join(value.detach().cpu().numpy().tobytes() for _name, value in sorted(model.state_dict().items()))
    return model, {
        "seed": seed,
        "epochs": EPOCHS,
        "pair_count": len(high_np),
        "generic_relative_pair_count": len(generic_high_np),
        "opportunity_positive_count": int(np.sum(opportunity_targets_np[opportunity_active_np])),
        "opportunity_negative_count": int(
            len(opportunity_active_np) - np.sum(opportunity_targets_np[opportunity_active_np])
        ),
        "opportunity_anchor_excluded_count": int(len(records) - len(opportunity_active_np)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_pairwise_loss": final_rank,
        "final_generic_relative_loss": final_generic_relative,
        "final_regression_loss": final_regression,
        "final_opportunity_loss": final_opportunity,
        "model_state_sha256": hashlib.sha256(state).hexdigest().upper(),
    }


def predict(
    dataset: SourceDataset,
    transform: SetTransform,
    models: Sequence[CandidateSetRanker],
) -> tuple[np.ndarray, np.ndarray]:
    inputs = transform.apply([dataset])
    global_tokens, cells, base, padding, flat_indices = _torch_inputs(inputs)
    utility_rows: list[np.ndarray] = []
    opportunity_rows: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            padded, _log_variance, padded_opportunity = model(global_tokens, cells, base, padding)
            utility_rows.append(
                _flat_scores(padded, padding, flat_indices).cpu().numpy().astype(np.float64)
            )
            opportunity_rows.append(
                _flat_scores(padded_opportunity, padding, flat_indices).cpu().numpy().astype(np.float64)
            )
    return np.stack(utility_rows), np.stack(opportunity_rows)


def gated_scores(
    records: Sequence[scorer.CandidateRecord],
    utility_ensemble: np.ndarray,
    opportunity_ensemble: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(
        utility_ensemble.ndim == 2 and utility_ensemble.shape[1] == len(records),
        "R31 utility ensemble shape drift",
    )
    require(
        opportunity_ensemble.shape == utility_ensemble.shape,
        "R31 opportunity ensemble shape drift",
    )
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    output = np.zeros(len(records), dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    accepted = 0
    beneficial = harmful = neutral = strict_beneficial = 0
    outcomes_available = all(
        record.target_gain is not None and record.coverage is not None for record in records
    )
    utility_mean = np.mean(utility_ensemble, axis=0)
    for reference_id, indices in sorted(by_reference.items()):
        generic = r27._generic_index(records, indices)
        opportunity_baseline = opportunity_ensemble[:, generic]
        opportunity_margins = {
            index: opportunity_ensemble[:, index] - opportunity_baseline for index in indices
        }
        opportunity_lcbs = {
            index: float(np.mean(margin) - LCB_Z * np.std(margin, ddof=0))
            for index, margin in opportunity_margins.items()
        }
        utility_proposal = max(
            indices,
            key=lambda index: (utility_mean[index], records[index].pair.neighbor.frame_id),
        )
        utility_proposal_margin = utility_ensemble[:, utility_proposal] - utility_ensemble[:, generic]
        utility_proposal_lcb = float(
            np.mean(utility_proposal_margin)
            - LCB_Z * np.std(utility_proposal_margin, ddof=0)
        )
        opportunity_proposal = max(
            indices,
            key=lambda index: (opportunity_lcbs[index], records[index].pair.neighbor.frame_id),
        )
        opportunity_utility_margin = (
            utility_ensemble[:, opportunity_proposal] - utility_ensemble[:, generic]
        )
        opportunity_fallback = (
            opportunity_proposal != generic
            and opportunity_lcbs[opportunity_proposal] > 0.0
            and float(np.mean(opportunity_utility_margin)) >= 0.0
        )
        if utility_proposal != generic and utility_proposal_lcb > 0.0:
            proposal = utility_proposal
            decision_lane = "UTILITY_LCB"
        elif opportunity_fallback:
            proposal = opportunity_proposal
            decision_lane = "OPPORTUNITY_FALLBACK"
        else:
            proposal = generic
            decision_lane = "GENERIC_FALLBACK"
        eligible = [index for index in indices if opportunity_lcbs[index] > 0.0]
        differences = utility_ensemble[:, proposal] - utility_ensemble[:, generic]
        lower_confidence = float(np.mean(differences) - LCB_Z * np.std(differences, ddof=0))
        accept = proposal != generic
        selected = proposal if accept else generic
        output[selected] = 1.0
        accepted += int(accept)
        if accept and outcomes_available:
            passive = max(
                indices,
                key=lambda index: (
                    float(records[index].coverage),
                    -records[index].pair.gap_s,
                    records[index].pair.neighbor.frame_id,
                ),
            )
            proposal_gain = int(records[proposal].target_gain)
            generic_gain = int(records[generic].target_gain)
            passive_gain = int(records[passive].target_gain)
            beneficial += int(proposal_gain > generic_gain)
            harmful += int(proposal_gain < generic_gain)
            neutral += int(proposal_gain == generic_gain)
            strict_beneficial += int(proposal_gain > max(generic_gain, passive_gain))
        receipts.append(
            {
                "reference_id": reference_id,
                "generic_neighbor_id": records[generic].pair.neighbor.frame_id,
                "proposal_neighbor_id": records[proposal].pair.neighbor.frame_id,
                "selected_neighbor_id": records[selected].pair.neighbor.frame_id,
                "ensemble_advantage_mean": float(np.mean(differences)),
                "ensemble_advantage_std": float(np.std(differences, ddof=0)),
                "lower_confidence_advantage": lower_confidence,
                "opportunity_margin_mean": float(np.mean(opportunity_margins[proposal])),
                "opportunity_margin_std": float(np.std(opportunity_margins[proposal], ddof=0)),
                "opportunity_margin_lcb": opportunity_lcbs[proposal],
                "eligible_candidate_count": len(eligible),
                "decision_lane": decision_lane,
                "accepted": accept,
            }
        )
    return output, {
        "reference_count": len(by_reference),
        "accepted_override_count": accepted,
        "generic_fallback_count": len(by_reference) - accepted,
            "outcome_diagnostics_available": outcomes_available,
            "beneficial_override_count": beneficial if outcomes_available else None,
            "harmful_override_count": harmful if outcomes_available else None,
            "neutral_override_count": neutral if outcomes_available else None,
            "strict_beneficial_override_count": strict_beneficial if outcomes_available else None,
        "decision_receipt_sha256": hashlib.sha256(canonical_json_bytes(receipts)).hexdigest().upper(),
    }


def _write_prediction_cache_exclusive(
    path: Path,
    records: Sequence[scorer.CandidateRecord],
    utility_ensemble: np.ndarray,
    opportunity_ensemble: np.ndarray,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    identities = [
        {
            "parent_id": record.parent_id,
            "reference_id": record.reference_id,
            "neighbor_id": record.pair.neighbor.frame_id,
        }
        for record in records
    ]
    identity_sha256 = hashlib.sha256(canonical_json_bytes(identities)).hexdigest().upper()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(SCHEMA),
            record_identity_sha256=np.asarray(identity_sha256),
            utility_ensemble=utility_ensemble,
            opportunity_ensemble=opportunity_ensemble,
        )
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "record_identity_sha256": identity_sha256,
    }


def run_lofo(
    datasets: Mapping[str, SourceDataset],
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    for held_source in SOURCE_NAMES:
        train = [datasets[source] for source in SOURCE_NAMES if source != held_source]
        transform = SetTransform.fit(train)
        models: list[CandidateSetRanker] = []
        training_receipts = []
        for seed in SEEDS:
            model, receipt = train_ranker(train, transform, seed)
            models.append(model)
            training_receipts.append(receipt)
        held = datasets[held_source]
        utility_ensemble, opportunity_ensemble = predict(held, transform, models)
        gated, selection = gated_scores(held.records, utility_ensemble, opportunity_ensemble)
        prediction_receipt = None
        if prediction_root is not None:
            prediction_receipt = _write_prediction_cache_exclusive(
                prediction_root / f"{held_source.lower()}-v1.npz",
                held.records,
                utility_ensemble,
                opportunity_ensemble,
            )
        folds[held_source] = {
            "training_sources": [dataset.source for dataset in train],
            "held_source_excluded_from_normalizer_and_fit": True,
            "training_candidate_count": sum(len(dataset.records) for dataset in train),
            "training_receipts": training_receipts,
            "normalizer_sha256": transform.receipt_sha256(),
            "development_prediction_receipt": prediction_receipt,
            "selection": selection,
            "gated_metrics": r21.fold_metrics(held.records, gated),
            "ungated_metrics": r21.fold_metrics(held.records, np.mean(utility_ensemble, axis=0)),
        }
    passed = all(all(row["gated_metrics"]["checks"].values()) for row in folds.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "CONSUMED_FOUR_SOURCE_DEVELOPMENT_LEAVE_ONE_SOURCE_FAMILY_OUT",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside nine frozen body/path capsules; UNKNOWN remains unknown.",
        "mechanism": {
            "name": "MULTI_CANDIDATE_RIGID_WARP_RELIABILITY_AND_OCCLUSION_CONSISTENCY",
            "cell_channel_names": list(CELL_CHANNEL_NAMES),
            "global_feature_names": list(GLOBAL_FEATURE_NAMES),
            "set_consistency_feature_names": list(SET_CONSISTENCY_FEATURE_NAMES),
            "query_count": QUERY_COUNT,
            "cells_per_query": CELL_COUNT_PER_QUERY,
            "candidate_set_transformer_layers": TRANSFORMER_LAYERS,
            "embedding_width": EMBED_DIM,
            "attention_heads": ATTENTION_HEADS,
            "ensemble_seeds": list(SEEDS),
            "epochs": EPOCHS,
            "lcb_z": LCB_Z,
            "generic_relative_loss_weight": GENERIC_RELATIVE_WEIGHT,
            "regression_loss_weight": REGRESSION_WEIGHT,
            "opportunity_loss_weight": OPPORTUNITY_LOSS_WEIGHT,
            "opportunity_training_coordinate": "WITHIN_REFERENCE_MARGIN_OVER_SOURCE_TIME_GENERIC",
            "opportunity_gradient_boundary": "DETACHED_FROM_UTILITY_BACKBONE",
            "selection_policy": "UTILITY_LCB_THEN_OPPORTUNITY_MARGIN_WITH_NONNEGATIVE_UTILITY_MEAN",
            "candidate_depth_in_scorer_input": False,
            "inference_anchor_uses_target_coverage": False,
        },
        "sources": {
            source: {
                "candidate_count": len(dataset.records),
                "parent_count": len({record.parent_id for record in dataset.records}),
                "geometry_abstention_count": dataset.abstentions,
                "cache_receipt": dataset.receipt.get("cache_receipt"),
            }
            for source, dataset in datasets.items()
        },
        "folds": folds,
        "terminal": "TARO_R31_FOUR_SOURCE_LOFO_PASS" if passed else "STOP_TARO_R31_FOUR_SOURCE_LOFO_FAIL",
        "consumed_development_pass": passed,
        "fresh_confirmation_authorized": passed,
        "android_candidate_authorized": False,
        "product_authorized": False,
        "safety_authorized": False,
        "read_boundary": {
            "reference_rgb_and_depth_in_scorer_input": True,
            "candidate_rgb_in_scorer_input": True,
            "candidate_depth_in_scorer_input": False,
            "candidate_depth_role": "CONSUMED_DEVELOPMENT_TARGET_ONLY",
            "held_source_targets_in_fit": False,
            "network_requests": 0,
        },
        "claim_ceiling": "Consumed four-source leave-one-family-out Development only. A PASS authorizes a new untouched confirmation lock, not Android, product, deployment, collision, navigation, or safety claims.",
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
    return result


def build_caches(
    cache_root: Path,
    bonn_root: Path,
    arkit_root: Path,
    openloris_root: Path,
    groundtruth_root: Path,
) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for source in SOURCE_NAMES:
        if source == "OPENLORIS_HOME":
            dataset = _openloris_dataset(openloris_root, groundtruth_root)
        else:
            dataset = _standard_source_dataset(source, bonn_root, arkit_root)
        receipts[source] = save_cache(cache_root, dataset)
    return receipts


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
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--bonn-root", type=Path, default=r21.shared.DEFAULT_BONN_ROOT)
    parser.add_argument("--arkit-root", type=Path, default=r21.DEFAULT_ARKIT_ROOT)
    parser.add_argument("--openloris-root", type=Path, required=True)
    parser.add_argument("--groundtruth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-root", type=Path)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    cache_receipts = None
    if args.build_cache:
        cache_receipts = build_caches(
            cache_root,
            args.bonn_root.resolve(),
            args.arkit_root.resolve(),
            args.openloris_root.resolve(),
            args.groundtruth_root.resolve(),
        )
    datasets = {source: load_cache(cache_root, source) for source in SOURCE_NAMES}
    if cache_receipts is None:
        cache_receipts = {
            source: {
                "path": str(_cache_path(cache_root, source)),
                "bytes": _cache_path(cache_root, source).stat().st_size,
                "sha256": hashlib.sha256(_cache_path(cache_root, source).read_bytes()).hexdigest().upper(),
                "candidate_count": len(dataset.records),
            }
            for source, dataset in datasets.items()
        }
    for source, dataset in datasets.items():
        dataset.receipt["cache_receipt"] = cache_receipts[source]
    result = run_lofo(
        datasets,
        args.prediction_root.resolve() if args.prediction_root is not None else None,
    )
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
