#!/usr/bin/env python3
"""R32 source-anchored monocular candidate geometry Development.

The scorer derives candidate geometry from candidate RGB with a frozen metric
Depth Anything V2 model.  A robust scale anchor is fitted only between the
same reference RGB prediction and the runtime-available reference depth.
Candidate sensor depth remains target-only and is opened by the source
builders only after every candidate score for a source reference is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy import ndimage

from scripts.research.hftf.produce_external_rgb_metric_depth_observations import (
    DepthAnythingV2MetricSource,
)
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_openloris_home_frontdoor as openloris
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_rgb_query_interaction_ranker as r25
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_source_anchored_monocular_geometry.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_NAMES = ("ARKITSCENES", "BONN_RGBD_DYNAMIC", "TUM_RGBD", "OPENLORIS_HOME")
DEFAULT_MODEL_REPO = REPO_ROOT / "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/models/depth-anything-v2-metric-hypersim-small"
    / "depth_anything_v2_metric_hypersim_vits.pth"
)
DEFAULT_IDENTITY_CACHE_ROOT = (
    REPO_ROOT / "artifacts.local/cache/taro-r31-reliability-consistency-r0"
)
EXPECTED_CHECKPOINT_SHA256 = "B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545"
EXPECTED_DPT_SHA256 = "120698FD0AD9A4D169B4B9EF26035AD0CB11EED4089AF53E07E38F5717024B1D"
MINIMUM_ANCHOR_PIXELS = 256
ANCHOR_MINIMUM_LOG_BAND = 0.10


class R32Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R32Error(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def robust_reference_scale(
    source_depth: np.ndarray,
    predicted_depth: np.ndarray,
    source_valid: np.ndarray,
) -> tuple[float, dict[str, float]]:
    require(
        source_depth.shape == predicted_depth.shape == source_valid.shape,
        "R32 reference anchor shape drift",
    )
    valid = (
        source_valid
        & np.isfinite(source_depth)
        & np.isfinite(predicted_depth)
        & (source_depth > 0.0)
        & (predicted_depth > 0.0)
    )
    require(int(np.sum(valid)) >= MINIMUM_ANCHOR_PIXELS, "R32 reference anchor support insufficient")
    log_ratio = np.log(source_depth[valid] / predicted_depth[valid])
    center = float(np.median(log_ratio))
    mad = float(np.median(np.abs(log_ratio - center)))
    band = max(ANCHOR_MINIMUM_LOG_BAND, 3.0 * 1.4826 * mad)
    inliers = np.abs(log_ratio - center) <= band
    require(int(np.sum(inliers)) >= MINIMUM_ANCHOR_PIXELS, "R32 robust anchor support insufficient")
    log_scale = float(np.median(log_ratio[inliers]))
    scale = float(math.exp(log_scale))
    anchored_error = np.abs(np.log((predicted_depth[valid] * scale) / source_depth[valid]))
    require(math.isfinite(scale) and scale > 0.0, "R32 reference scale invalid")
    return scale, {
        "reference_anchor_pixel_count": float(np.sum(valid)),
        "reference_anchor_inlier_count": float(np.sum(inliers)),
        "reference_anchor_inlier_fraction": float(np.mean(inliers)),
        "reference_anchor_scale": scale,
        "reference_anchor_median_abs_log_error": float(np.median(anchored_error)),
        "reference_anchor_p90_abs_log_error": float(np.quantile(anchored_error, 0.90)),
    }


class SourceAnchoredMonocularGeometry:
    def __init__(self, model_repo: Path, checkpoint: Path, device: str) -> None:
        dpt_path = model_repo / "metric_depth/depth_anything_v2/dpt.py"
        require(model_repo.is_dir() and dpt_path.is_file(), "R32 model source absent")
        require(checkpoint.is_file(), "R32 checkpoint absent")
        require(sha256_file(dpt_path) == EXPECTED_DPT_SHA256, "R32 model source drift")
        require(sha256_file(checkpoint) == EXPECTED_CHECKPOINT_SHA256, "R32 checkpoint drift")
        self.source = DepthAnythingV2MetricSource(
            model_repo,
            checkpoint,
            device=device,
            input_size=518,
            precision="fp32",
        )
        self._low_depth: dict[str, np.ndarray] = {}
        self._anchor: dict[str, tuple[float, dict[str, float]]] = {}
        self.model_run_count = 0

    def low_depth(self, frame: bonn.Frame, rgb: np.ndarray) -> np.ndarray:
        cached = self._low_depth.get(frame.frame_id)
        if cached is not None:
            return cached
        prediction, _metadata = self.source.infer(rgb, {})
        require(
            prediction.ndim == 2 and np.all(np.isfinite(prediction)) and np.all(prediction > 0.0),
            f"R32 monocular prediction invalid: {frame.frame_id}",
        )
        low = cv2.resize(prediction, tum.LOW_SIZE_WH, interpolation=cv2.INTER_LINEAR)
        low = np.ascontiguousarray(low, dtype=np.float64)
        self._low_depth[frame.frame_id] = low
        self.model_run_count += 1
        return low

    def candidate_features(
        self,
        context: scorer.ReferenceContext,
        pair: bonn.Pair,
        reference_rgb: np.ndarray,
        candidate_rgb: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        anchor = self._anchor.get(context.row.reference.frame_id)
        if anchor is None:
            reference_prediction = self.low_depth(context.row.reference, reference_rgb)
            anchor = robust_reference_scale(context.low_depth, reference_prediction, context.valid)
            self._anchor[context.row.reference.frame_id] = anchor
        scale, anchor_receipt = anchor
        candidate_depth = self.low_depth(pair.neighbor, candidate_rgb) * scale
        valid = (
            np.isfinite(candidate_depth)
            & (candidate_depth >= adapter.DEPTH_RANGE_M[0])
            & (candidate_depth <= adapter.DEPTH_RANGE_M[1])
        )
        maximum = ndimage.maximum_filter(
            np.where(valid, candidate_depth, -np.inf), size=3, mode="constant", cval=-np.inf
        )
        minimum = ndimage.minimum_filter(
            np.where(valid, candidate_depth, np.inf), size=3, mode="constant", cval=np.inf
        )
        stable = (
            valid
            & np.isfinite(maximum)
            & np.isfinite(minimum)
            & ((maximum - minimum) <= bonn.LOCAL_STABILITY_RANGE_M)
        )
        rows, columns = np.indices(candidate_depth.shape, dtype=np.float64)
        intrinsics = context.intrinsics
        points = np.stack(
            (
                (columns - intrinsics[0, 2]) * candidate_depth / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) * candidate_depth / intrinsics[1, 1],
                candidate_depth,
            ),
            axis=-1,
        )
        transformed = oracle._transform_points(points, context.row.reference, pair.neighbor)
        observed = oracle.query_evidence_cells(transformed, stable, context.queries)
        novel = observed & ~context.static
        analytic = {
            **anchor_receipt,
            "predicted_observed_cell_count": float(np.sum(observed)),
            "predicted_novel_cell_count": float(np.sum(novel)),
            "predicted_candidate_valid_fraction": float(np.mean(valid)),
            "predicted_candidate_stable_fraction": float(np.mean(stable)),
        }
        features = np.asarray(
            [
                analytic["predicted_novel_cell_count"],
                analytic["predicted_observed_cell_count"],
                analytic["predicted_candidate_valid_fraction"],
                analytic["predicted_candidate_stable_fraction"],
                analytic["reference_anchor_median_abs_log_error"],
                analytic["reference_anchor_p90_abs_log_error"],
                pair.translation_m,
                pair.rotation_deg,
                pair.gap_s,
            ],
            dtype=np.float64,
        )
        require(features.shape == (9,) and np.all(np.isfinite(features)), "R32 feature drift")
        return features, analytic

    def receipt(self) -> dict[str, Any]:
        return {
            "model": "DEPTH_ANYTHING_V2_METRIC_HYPERSIM_VITS",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "dpt_source_sha256": EXPECTED_DPT_SHA256,
            "model_run_count": self.model_run_count,
            "unique_prediction_count": len(self._low_depth),
            "reference_anchor_count": len(self._anchor),
        }


def _standard_source_records(
    source: str,
    predictor: SourceAnchoredMonocularGeometry,
    bonn_root: Path,
    arkit_root: Path,
    identity_cache_root: Path | None,
    tum_manifests: Sequence[Path] = r21.TUM_MANIFESTS,
) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    if source == "TUM_RGBD":
        store = r25.RgbStore(r25._tum_rgb_assets(tum_manifests), "R32 TUM RGB")
        builder = lambda feature: _tum_records(tum_manifests, feature)
    elif source == "BONN_RGBD_DYNAMIC":
        store = r25.RgbStore({}, "R32 Bonn RGB")
        builder = lambda feature: r21._build_bonn_records(bonn_root, feature)
    elif source == "ARKITSCENES":
        store = r25.RgbStore(r25._arkit_rgb_assets(arkit_root), "R32 ARKit RGB")
        builder = lambda feature: r21._build_arkit_records(arkit_root, feature)
    else:
        raise R32Error(f"unsupported R32 source: {source}")

    if identity_cache_root is not None:
        identity_path = identity_cache_root / f"{source.lower()}-v1.npz"
        require(identity_path.is_file(), f"R32 identity cache absent: {identity_path}")
        with np.load(identity_path, allow_pickle=False) as value:
            frame_ids = np.concatenate((value["reference_ids"].astype(str), value["neighbor_ids"].astype(str)))
        store.preload(frame_ids.tolist())

    def feature(context: scorer.ReferenceContext, pair: bonn.Pair) -> tuple[np.ndarray, dict[str, float]]:
        return predictor.candidate_features(
            context,
            pair,
            store.rgb(pair.reference),
            store.rgb(pair.neighbor),
        )

    try:
        records, receipt, abstentions = builder(feature)
        receipt["rgb_signal_receipt"] = store.receipt()
    finally:
        store.close()
    return records, receipt, abstentions


def _tum_records(
    manifest_paths: Sequence[Path],
    feature_fn: Any,
) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    frames, assets, source = tum.load_outcome_blind_roster(
        manifest_paths, verify_archive_hashes=False
    )
    selected, capability = openloris.balanced.select_pose_capable_references(
        frames, oracle.MAX_REFERENCES_PER_PARENT
    )
    reference_observations = scorer._load_observations(
        [row.reference.frame_id for row in selected], assets
    )
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    abstained = 0
    for row in selected:
        low, points, valid, _coverage = reference_observations[row.reference.frame_id]
        intrinsics = bonn._scaled_intrinsics(
            assets[row.reference.frame_id].intrinsics, tum.NATIVE_SIZE_WH, tum.LOW_SIZE_WH
        )
        queries = oracle._queries(row.reference, low, intrinsics)
        if queries is None:
            abstained += 1
            continue
        static = oracle.query_evidence_cells(points, valid, queries)
        context = scorer.ReferenceContext(row, low, points, valid, intrinsics, queries, static)
        contexts[row.reference.frame_id] = context
        for pair in oracle.pose_proposal_pairs(row):
            features, analytic = feature_fn(context, pair)
            records.append(
                scorer.CandidateRecord(
                    row.reference.parent_id,
                    "CONSUMED_DEVELOPMENT",
                    row.reference.frame_id,
                    pair,
                    features,
                    analytic,
                )
            )
    observations = scorer._load_observations(
        [record.pair.neighbor.frame_id for record in records], assets
    )
    scorer._attach_targets(records, contexts, observations)
    return records, {"source": source, "capability": capability}, abstained


def available_tum_manifest(output_root: Path) -> Path:
    rows, _receipts = tum._manifest_rows(r21.TUM_MANIFESTS)
    available = [
        row
        for row in rows
        if (REPO_ROOT / str(row["source_path"])).resolve().exists()
    ]
    require(len(available) >= 4, "R32 available TUM parent count insufficient")
    payload_rows = [
        {key: value for key, value in row.items() if key not in {"cohort_role", "cohort_token"}}
        for row in available
    ]
    value = {
        "schema": tum.COHORT_SCHEMA,
        "status": "FROZEN_BEFORE_R32_PARTIAL_FEATURE_MATERIALIZATION",
        "token": "TARO_R32_AVAILABLE_TUM_FEATURE_MATERIALIZATION_R0",
        "fit_parents": [
            row for row, source in zip(payload_rows, available, strict=True)
            if source["cohort_role"] == "FIT"
        ],
        "evaluation_parents": [
            row for row, source in zip(payload_rows, available, strict=True)
            if source["cohort_role"] == "EVALUATION"
        ],
    }
    path = output_root / "available-tum-manifest.json"
    write_exclusive(path, value)
    return path


def _openloris_records(
    predictor: SourceAnchoredMonocularGeometry,
    source_root: Path,
    groundtruth_root: Path,
) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    frames, assets, source = openloris.load_outcome_blind_roster(source_root, groundtruth_root)
    selected, capability = openloris.balanced.select_pose_capable_references(
        frames, openloris.MAX_REFERENCES_PER_PARENT
    )
    proposals, candidate_identity_sha = openloris._candidate_identity(selected)
    store = openloris.PayloadStore(assets)
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    for row in selected:
        low, points, valid, _coverage = store.observation(row.reference.frame_id)
        asset = assets[row.reference.frame_id]
        low_intrinsics = bonn._scaled_intrinsics(
            asset.intrinsics, openloris.CROPPED_SIZE_WH, tum.LOW_SIZE_WH
        )
        queries, _receipt = openloris._calibration_queries(
            row.reference, asset.camera_height_m
        )
        static = oracle.query_evidence_cells(points, valid, queries)
        context = scorer.ReferenceContext(row, low, points, valid, low_intrinsics, queries, static)
        contexts[row.reference.frame_id] = context
        reference_rgb = store.rgb(row.reference)
        for pair in proposals[row.reference.frame_id]:
            features, analytic = predictor.candidate_features(
                context,
                pair,
                reference_rgb,
                store.rgb(pair.neighbor),
            )
            records.append(
                scorer.CandidateRecord(
                    row.reference.parent_id,
                    "CONSUMED_DEVELOPMENT",
                    row.reference.frame_id,
                    pair,
                    features,
                    analytic,
                )
            )
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
    }
    return records, receipt, 0


def analytic_scores(records: Sequence[scorer.CandidateRecord]) -> np.ndarray:
    scores = np.asarray(
        [record.analytic["predicted_novel_cell_count"] for record in records], dtype=np.float64
    )
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    for indices in by_reference.values():
        generic = max(
            indices,
            key=lambda index: (
                records[index].pair.translation_m,
                records[index].pair.rotation_deg,
                -records[index].pair.gap_s,
                records[index].pair.neighbor.frame_id,
            ),
        )
        scores[generic] += 1e-6
    return scores


def save_feature_cache(
    root: Path,
    source: str,
    records: Sequence[scorer.CandidateRecord],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{source.lower()}-v1.npz"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(SCHEMA),
            source=np.asarray(source),
            reference_ids=np.asarray([record.reference_id for record in records]),
            neighbor_ids=np.asarray([record.pair.neighbor.frame_id for record in records]),
            features=np.stack([record.features for record in records]).astype(np.float32),
        )
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "candidate_count": len(records),
    }


def run(
    predictor: SourceAnchoredMonocularGeometry,
    bonn_root: Path,
    arkit_root: Path,
    openloris_root: Path,
    groundtruth_root: Path,
    identity_cache_root: Path | None = None,
    feature_cache_root: Path | None = None,
    sources: Sequence[str] = SOURCE_NAMES,
    tum_manifests: Sequence[Path] = r21.TUM_MANIFESTS,
) -> dict[str, Any]:
    datasets: dict[str, tuple[list[scorer.CandidateRecord], dict[str, Any], int]] = {}
    for source in sources:
        if source == "OPENLORIS_HOME":
            datasets[source] = _openloris_records(predictor, openloris_root, groundtruth_root)
        else:
            datasets[source] = _standard_source_records(
                source,
                predictor,
                bonn_root,
                arkit_root,
                identity_cache_root,
                tum_manifests,
            )
        if feature_cache_root is not None:
            datasets[source][1]["r32_feature_cache_receipt"] = save_feature_cache(
                feature_cache_root, source, datasets[source][0]
            )
    source_results: dict[str, Any] = {}
    for source, (records, receipt, abstentions) in datasets.items():
        metrics = r21.fold_metrics(records, analytic_scores(records))
        source_results[source] = {
            "candidate_count": len(records),
            "parent_count": len({record.parent_id for record in records}),
            "geometry_abstention_count": abstentions,
            "source_receipt": receipt,
            "metrics": metrics,
        }
    complete = tuple(sources) == SOURCE_NAMES
    passed = complete and all(
        all(row["metrics"]["checks"].values()) for row in source_results.values()
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "CONSUMED_FOUR_SOURCE_DEVELOPMENT_DETERMINISTIC_TRANSFER",
        "mechanism": {
            "name": "REFERENCE_SENSOR_ANCHORED_MONOCULAR_CANDIDATE_GEOMETRY",
            "frozen_model": predictor.receipt(),
            "reference_anchor": "ROBUST_MEDIAN_LOG_SCALE_WITH_MAD_INLIERS",
            "candidate_geometry": "RGB_DERIVED_DEPTH_BACKPROJECTED_INTO_FROZEN_REFERENCE_TASK_CAPSULES",
            "tie_policy": "GENERIC_ON_EQUAL_PREDICTED_NOVEL_CELL_COUNT",
            "trainable_parameter_count": 0,
        },
        "sources": source_results,
        "terminal": (
            "TARO_R32_FOUR_SOURCE_DEVELOPMENT_PASS"
            if passed
            else (
                "STOP_TARO_R32_FOUR_SOURCE_DEVELOPMENT_FAIL"
                if complete
                else "TARO_R32_PARTIAL_FEATURE_MATERIALIZATION_ONLY"
            )
        ),
        "consumed_development_pass": passed,
        "fresh_confirmation_authorized": passed,
        "android_candidate_authorized": False,
        "product_authorized": False,
        "safety_authorized": False,
        "read_boundary": {
            "reference_rgb_and_sensor_depth_in_scorer_input": True,
            "candidate_rgb_in_scorer_input": True,
            "candidate_rgb_derived_depth_in_scorer_input": True,
            "candidate_sensor_depth_in_scorer_input": False,
            "candidate_sensor_depth_role": "CONSUMED_DEVELOPMENT_TARGET_ONLY_AFTER_SCORE_SEAL",
            "network_requests": 0,
        },
        "claim_ceiling": "Consumed four-source Development only. A PASS authorizes a new untouched confirmation lock, not Android, product, collision, navigation, deployment, or safety claims.",
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
    return result


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-repo", type=Path, default=DEFAULT_MODEL_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bonn-root", type=Path, default=r21.shared.DEFAULT_BONN_ROOT)
    parser.add_argument("--arkit-root", type=Path, default=r21.DEFAULT_ARKIT_ROOT)
    parser.add_argument("--openloris-root", type=Path, required=True)
    parser.add_argument("--groundtruth-root", type=Path, required=True)
    parser.add_argument("--identity-cache-root", type=Path, default=DEFAULT_IDENTITY_CACHE_ROOT)
    parser.add_argument("--feature-cache-root", type=Path)
    parser.add_argument("--sources", nargs="+", choices=SOURCE_NAMES, default=list(SOURCE_NAMES))
    parser.add_argument("--available-tum-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictor = SourceAnchoredMonocularGeometry(
        args.model_repo.resolve(), args.checkpoint.resolve(), args.device
    )
    feature_cache_root = (
        args.feature_cache_root.resolve() if args.feature_cache_root is not None else None
    )
    tum_manifests: Sequence[Path] = r21.TUM_MANIFESTS
    if args.available_tum_only:
        require(feature_cache_root is not None, "R32 available TUM mode requires feature cache root")
        tum_manifests = (available_tum_manifest(feature_cache_root),)
    result = run(
        predictor,
        args.bonn_root.resolve(),
        args.arkit_root.resolve(),
        args.openloris_root.resolve(),
        args.groundtruth_root.resolve(),
        args.identity_cache_root.resolve() if args.identity_cache_root is not None else None,
        feature_cache_root,
        tuple(args.sources),
        tum_manifests,
    )
    write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
