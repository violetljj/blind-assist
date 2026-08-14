#!/usr/bin/env python3
"""R37 pose-constrained candidate monocular geometry Development.

Frozen candidate-RGB monocular depth is calibrated per candidate by projecting
runtime-available reference sensor depth through the known relative pose.  The
robust overlap fit never reads candidate sensor depth; candidate sensor depth
is opened only by the consumed source builders after every score is sealed.
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

import numpy as np
from scipy import ndimage

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_reprojection_visibility_scorer as r27
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_source_anchored_monocular_geometry as r32
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_pose_constrained_candidate_monocular_geometry.v1"
SOURCE_NAMES = r32.SOURCE_NAMES
MINIMUM_ALIGNMENT_PIXELS = 96
MINIMUM_ALIGNMENT_BAND_M = 0.08
MAXIMUM_ALIGNMENT_ITERATIONS = 4
AFFINE_SCALE_RANGE = (0.25, 4.0)
AFFINE_SHIFT_RANGE_M = (-2.0, 2.0)
CONFIDENCE_RESIDUAL_SCALE_M = 0.25
CONFIDENCE_FULL_SUPPORT_PIXELS = 512
OCCLUSION_MARGIN_M = 0.15


class R37Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R37Error(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def reference_depth_in_candidate(
    context: scorer.ReferenceContext,
    pair: bonn.Pair,
) -> np.ndarray:
    """Forward z-buffer reference sensor depth into the candidate view."""
    require(
        context.row.reference.frame_id == pair.reference.frame_id,
        "R37 context/pair reference identity drift",
    )
    candidate_points = oracle._transform_points(
        context.points,
        pair.neighbor,
        pair.reference,
    )
    height, width = context.low_depth.shape
    intrinsics = context.intrinsics
    flat = candidate_points.reshape(-1, 3)
    source_valid = context.valid.reshape(-1)
    z = flat[:, 2]
    projectable = source_valid & np.isfinite(flat).all(axis=1) & (z > 0.0)
    safe_z = np.where(projectable, z, 1.0)
    columns = np.rint(intrinsics[0, 0] * flat[:, 0] / safe_z + intrinsics[0, 2]).astype(np.int64)
    rows = np.rint(intrinsics[1, 1] * flat[:, 1] / safe_z + intrinsics[1, 2]).astype(np.int64)
    inside = projectable & (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    output = np.full(height * width, np.inf, dtype=np.float64)
    indices = rows[inside] * width + columns[inside]
    np.minimum.at(output, indices, z[inside])
    return output.reshape(height, width)


def robust_affine_alignment(
    candidate_depth: np.ndarray,
    projected_reference_depth: np.ndarray,
) -> tuple[float, float, np.ndarray, dict[str, Any]]:
    require(candidate_depth.shape == projected_reference_depth.shape, "R37 alignment shape drift")
    overlap = (
        np.isfinite(candidate_depth)
        & (candidate_depth > 0.0)
        & np.isfinite(projected_reference_depth)
        & (projected_reference_depth > 0.0)
    )
    overlap_count = int(np.sum(overlap))
    if overlap_count < MINIMUM_ALIGNMENT_PIXELS:
        return 1.0, 0.0, candidate_depth.copy(), {
            "alignment_evaluable": False,
            "alignment_failure": "POSE_OVERLAP_SUPPORT_INSUFFICIENT",
            "alignment_overlap_pixel_count": overlap_count,
            "alignment_inlier_pixel_count": 0,
            "alignment_inlier_fraction": 0.0,
            "alignment_scale": 1.0,
            "alignment_shift_m": 0.0,
            "alignment_median_abs_residual_m": 10.0,
            "alignment_p90_abs_residual_m": 10.0,
        }
    x = candidate_depth[overlap].astype(np.float64)
    y = projected_reference_depth[overlap].astype(np.float64)
    inliers = np.ones(x.shape, dtype=bool)
    scale = 1.0
    shift = 0.0
    for _iteration in range(MAXIMUM_ALIGNMENT_ITERATIONS):
        if int(np.sum(inliers)) < MINIMUM_ALIGNMENT_PIXELS:
            return 1.0, 0.0, candidate_depth.copy(), {
                "alignment_evaluable": False,
                "alignment_failure": "ROBUST_OVERLAP_SUPPORT_INSUFFICIENT",
                "alignment_overlap_pixel_count": overlap_count,
                "alignment_inlier_pixel_count": int(np.sum(inliers)),
                "alignment_inlier_fraction": float(np.mean(inliers)),
                "alignment_scale": 1.0,
                "alignment_shift_m": 0.0,
                "alignment_median_abs_residual_m": 10.0,
                "alignment_p90_abs_residual_m": 10.0,
            }
        design = np.stack((x[inliers], np.ones(int(np.sum(inliers)), dtype=np.float64)), axis=1)
        scale, shift = np.linalg.lstsq(design, y[inliers], rcond=None)[0]
        require(math.isfinite(float(scale)) and math.isfinite(float(shift)), "R37 affine fit invalid")
        residual = scale * x + shift - y
        center = float(np.median(residual[inliers]))
        mad = float(np.median(np.abs(residual[inliers] - center)))
        band = max(MINIMUM_ALIGNMENT_BAND_M, 3.0 * 1.4826 * mad)
        updated = np.abs(residual - center) <= band
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    if int(np.sum(inliers)) >= MINIMUM_ALIGNMENT_PIXELS:
        design = np.stack((x[inliers], np.ones(int(np.sum(inliers)), dtype=np.float64)), axis=1)
        scale, shift = np.linalg.lstsq(design, y[inliers], rcond=None)[0]
    supported = (
        int(np.sum(inliers)) >= MINIMUM_ALIGNMENT_PIXELS
        and AFFINE_SCALE_RANGE[0] <= scale <= AFFINE_SCALE_RANGE[1]
        and AFFINE_SHIFT_RANGE_M[0] <= shift <= AFFINE_SHIFT_RANGE_M[1]
    )
    if not supported:
        return 1.0, 0.0, candidate_depth.copy(), {
            "alignment_evaluable": False,
            "alignment_failure": "AFFINE_PARAMETERS_OR_SUPPORT_UNSUPPORTED",
            "alignment_overlap_pixel_count": overlap_count,
            "alignment_inlier_pixel_count": int(np.sum(inliers)),
            "alignment_inlier_fraction": float(np.mean(inliers)),
            "alignment_scale": 1.0,
            "alignment_shift_m": 0.0,
            "alignment_median_abs_residual_m": 10.0,
            "alignment_p90_abs_residual_m": 10.0,
        }
    residual = scale * x + shift - y
    inlier_abs = np.abs(residual[inliers])
    aligned = scale * candidate_depth + shift
    receipt = {
        "alignment_evaluable": True,
        "alignment_failure": None,
        "alignment_overlap_pixel_count": overlap_count,
        "alignment_inlier_pixel_count": int(np.sum(inliers)),
        "alignment_inlier_fraction": float(np.mean(inliers)),
        "alignment_scale": float(scale),
        "alignment_shift_m": float(shift),
        "alignment_median_abs_residual_m": float(np.median(inlier_abs)),
        "alignment_p90_abs_residual_m": float(np.quantile(inlier_abs, 0.90)),
    }
    return float(scale), float(shift), aligned, receipt


class PoseConstrainedCandidateMonocularGeometry(r32.SourceAnchoredMonocularGeometry):
    def __init__(self, model_repo: Path, checkpoint: Path, device: str) -> None:
        super().__init__(model_repo, checkpoint, device)
        self.alignment_count = 0

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
            anchor = r32.robust_reference_scale(context.low_depth, reference_prediction, context.valid)
            self._anchor[context.row.reference.frame_id] = anchor
        reference_scale, anchor_receipt = anchor
        source_candidate_depth = self.low_depth(pair.neighbor, candidate_rgb) * reference_scale
        projected_reference = reference_depth_in_candidate(context, pair)
        _scale, _shift, candidate_depth, alignment = robust_affine_alignment(
            source_candidate_depth,
            projected_reference,
        )
        self.alignment_count += 1
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
        overlap = np.isfinite(projected_reference) & valid
        closer = overlap & (candidate_depth < projected_reference - OCCLUSION_MARGIN_M)
        farther = overlap & (candidate_depth > projected_reference + OCCLUSION_MARGIN_M)
        support_confidence = min(
            1.0,
            alignment["alignment_inlier_pixel_count"] / CONFIDENCE_FULL_SUPPORT_PIXELS,
        )
        residual_confidence = math.exp(
            -alignment["alignment_median_abs_residual_m"] / CONFIDENCE_RESIDUAL_SCALE_M
        )
        alignment_confidence = float(
            alignment["alignment_inlier_fraction"] * support_confidence * residual_confidence
        )
        predicted_novel = float(np.sum(novel))
        analytic = {
            **anchor_receipt,
            **alignment,
            "alignment_confidence": alignment_confidence,
            "predicted_observed_cell_count": float(np.sum(observed)),
            "predicted_novel_cell_count": predicted_novel,
            "confidence_weighted_predicted_novel_cell_count": predicted_novel * alignment_confidence,
            "predicted_candidate_valid_fraction": float(np.mean(valid)),
            "predicted_candidate_stable_fraction": float(np.mean(stable)),
            "pose_overlap_fraction": float(np.mean(np.isfinite(projected_reference))),
            "overlap_closer_fraction": float(np.mean(closer[overlap])) if np.any(overlap) else 0.0,
            "overlap_farther_fraction": float(np.mean(farther[overlap])) if np.any(overlap) else 0.0,
        }
        features = np.asarray(
            [
                analytic["confidence_weighted_predicted_novel_cell_count"],
                analytic["predicted_novel_cell_count"],
                analytic["predicted_observed_cell_count"],
                analytic["alignment_confidence"],
                analytic["alignment_inlier_fraction"],
                analytic["alignment_median_abs_residual_m"],
                analytic["alignment_p90_abs_residual_m"],
                analytic["alignment_scale"],
                analytic["alignment_shift_m"],
                analytic["pose_overlap_fraction"],
                analytic["overlap_closer_fraction"],
                analytic["overlap_farther_fraction"],
                analytic["predicted_candidate_valid_fraction"],
                analytic["predicted_candidate_stable_fraction"],
                pair.translation_m,
                pair.rotation_deg,
                pair.gap_s,
            ],
            dtype=np.float64,
        )
        require(features.shape == (17,) and np.all(np.isfinite(features)), "R37 feature drift")
        return features, analytic

    def receipt(self) -> dict[str, Any]:
        return {
            **super().receipt(),
            "pose_constrained_alignment_count": self.alignment_count,
            "alignment": {
                "minimum_pixels": MINIMUM_ALIGNMENT_PIXELS,
                "minimum_band_m": MINIMUM_ALIGNMENT_BAND_M,
                "maximum_iterations": MAXIMUM_ALIGNMENT_ITERATIONS,
                "affine_scale_range": list(AFFINE_SCALE_RANGE),
                "affine_shift_range_m": list(AFFINE_SHIFT_RANGE_M),
                "confidence_residual_scale_m": CONFIDENCE_RESIDUAL_SCALE_M,
                "confidence_full_support_pixels": CONFIDENCE_FULL_SUPPORT_PIXELS,
                "occlusion_margin_m": OCCLUSION_MARGIN_M,
            },
        }


def analytic_scores(records: Sequence[scorer.CandidateRecord]) -> np.ndarray:
    scores = np.asarray(
        [record.analytic["confidence_weighted_predicted_novel_cell_count"] for record in records],
        dtype=np.float64,
    )
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    for indices in by_reference.values():
        generic = r27._generic_index(records, indices)
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
    predictor: PoseConstrainedCandidateMonocularGeometry,
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
            datasets[source] = r32._openloris_records(
                predictor,
                openloris_root,
                groundtruth_root,
            )
        else:
            datasets[source] = r32._standard_source_records(
                source,
                predictor,
                bonn_root,
                arkit_root,
                identity_cache_root,
                tum_manifests,
            )
        if feature_cache_root is not None:
            datasets[source][1]["r37_feature_cache_receipt"] = save_feature_cache(
                feature_cache_root,
                source,
                datasets[source][0],
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
            "name": "POSE_CONSTRAINED_CANDIDATE_MONOCULAR_GEOMETRY",
            "frozen_model": predictor.receipt(),
            "reference_anchor": "ROBUST_REFERENCE_LOG_SCALE",
            "candidate_alignment": "ROBUST_AFFINE_TO_POSE_PROJECTED_REFERENCE_SENSOR_DEPTH",
            "candidate_geometry": "ALIGNED_RGB_DERIVED_DEPTH_BACKPROJECTED_INTO_FROZEN_REFERENCE_TASK_CAPSULES",
            "score": "ALIGNMENT_CONFIDENCE_TIMES_PREDICTED_NOVEL_TASK_CELLS",
            "tie_policy": "GENERIC_ON_EQUAL_SCORE",
            "trainable_parameter_count": 0,
        },
        "sources": source_results,
        "terminal": (
            "TARO_R37_FOUR_SOURCE_DEVELOPMENT_PASS"
            if passed
            else (
                "STOP_TARO_R37_FOUR_SOURCE_DEVELOPMENT_FAIL"
                if complete
                else "TARO_R37_PARTIAL_DEVELOPMENT_RESULT"
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
            "pose_projected_reference_sensor_depth_in_alignment": True,
            "candidate_sensor_depth_in_scorer_or_alignment_input": False,
            "candidate_sensor_depth_role": "CONSUMED_DEVELOPMENT_TARGET_ONLY_AFTER_SCORE_SEAL",
            "network_requests": 0,
        },
        "claim_ceiling": "Consumed-source Development only. A PASS authorizes a separate target-untouched confirmation lock, not Android, product, collision, navigation, deployment, or safety claims.",
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
    parser.add_argument("--model-repo", type=Path, default=r32.DEFAULT_MODEL_REPO)
    parser.add_argument("--checkpoint", type=Path, default=r32.DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bonn-root", type=Path, default=r21.shared.DEFAULT_BONN_ROOT)
    parser.add_argument("--arkit-root", type=Path, default=r21.DEFAULT_ARKIT_ROOT)
    parser.add_argument("--openloris-root", type=Path, required=True)
    parser.add_argument("--groundtruth-root", type=Path, required=True)
    parser.add_argument("--identity-cache-root", type=Path, default=r32.DEFAULT_IDENTITY_CACHE_ROOT)
    parser.add_argument("--feature-cache-root", type=Path)
    parser.add_argument("--sources", nargs="+", choices=SOURCE_NAMES, default=list(SOURCE_NAMES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictor = PoseConstrainedCandidateMonocularGeometry(
        args.model_repo.resolve(),
        args.checkpoint.resolve(),
        args.device,
    )
    result = run(
        predictor,
        args.bonn_root.resolve(),
        args.arkit_root.resolve(),
        args.openloris_root.resolve(),
        args.groundtruth_root.resolve(),
        args.identity_cache_root.resolve() if args.identity_cache_root is not None else None,
        args.feature_cache_root.resolve() if args.feature_cache_root is not None else None,
        tuple(args.sources),
    )
    write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
