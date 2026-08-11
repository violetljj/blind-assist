#!/usr/bin/env python3
"""Materialize source-depth and camera-angular boundary factors for Bonn FIT parents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_ag_st_factor_labels import PROVENANCE_SOURCE_NATIVE, TIER_A_SOURCE
from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_student_bonn_depth import (
    BONN_INTRINSICS,
    DEFAULT_BONN_ARCHIVE,
    DEFAULT_BONN_CATALOG,
    DEFAULT_BONN_RECEIPT,
    DEFAULT_BONN_ROOT,
    fixed_frame_pairs,
    load_cohort_indices,
    load_depth_native,
    validate_source_receipts,
)
from materialize_ag_st_continuous_boundary_factors import continuous_boundary_factors
from materialize_ag_st_source_native_boundary_corpus import conservative_source_boundary
from run_ag_st_angular_boundary_resize_canary import (
    ANGULAR_SOFT_SIGMA_RAD,
    camera_angular_boundary_factors,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COHORT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_BONN_MIXED_DOMAIN_COHORT_R0_2026-08-10.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-bonn-fit-angular-factor-labels-r0"
)
SOURCE_ID = "bonn_rgbd_fit"
DEPTH_QUANTIZATION_UNCERTAINTY_M = 0.0001
BOUNDARY_UNCERTAINTY_PX = 0.5


def frame_id(parent_id: str, rgb_row_index: int) -> str:
    require(parent_id.startswith("rgbd_bonn_"), "Bonn parent identity drift")
    require(rgb_row_index >= 0, "negative Bonn RGB row index")
    return f"{SOURCE_ID}__{parent_id}__rgb_{rgb_row_index:06d}"


def build_factor_payload(
    depth_m: np.ndarray,
    source_valid: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.asarray(source_valid, dtype=np.bool_)
    k = np.asarray(intrinsics, dtype=np.float64)
    require(depth.ndim == 2 and depth.shape == valid.shape, "Bonn factor depth shape drift")
    require(k.shape == (3, 3) and k[0, 0] > 0.0 and k[1, 1] > 0.0, "Bonn K invalid")
    require(np.isfinite(depth[valid]).all() and np.all(depth[valid] > 0.0), "Bonn valid depth invalid")

    boundary_core, boundary_valid = conservative_source_boundary(depth, valid, k)
    boundary_distance, boundary_soft = continuous_boundary_factors(
        boundary_core, boundary_valid
    )
    angular_distance, angular_soft = camera_angular_boundary_factors(
        boundary_core, boundary_valid, k
    )
    shape = depth.shape
    unknown = np.zeros(shape, dtype=np.uint8)
    depth_tier = np.where(valid, TIER_A_SOURCE, 0).astype(np.uint8)
    depth_provenance = np.where(valid, PROVENANCE_SOURCE_NATIVE, 0).astype(np.uint8)
    boundary_tier = np.where(boundary_valid, TIER_A_SOURCE, 0).astype(np.uint8)
    boundary_provenance = np.where(
        boundary_valid, PROVENANCE_SOURCE_NATIVE, 0
    ).astype(np.uint8)
    depth_uncertainty = np.where(
        valid, DEPTH_QUANTIZATION_UNCERTAINTY_M, np.inf
    ).astype(np.float32)
    boundary_uncertainty = np.where(
        boundary_valid, BOUNDARY_UNCERTAINTY_PX, np.inf
    ).astype(np.float32)
    metric_depth = np.where(valid, depth, np.nan).astype(np.float32)

    return {
        "metric_depth_m_hw": metric_depth,
        "metric_depth_valid_hw": valid.astype(np.uint8),
        "source_native_valid_hw": valid.astype(np.uint8),
        "quality_tier_hw": depth_tier,
        "provenance_code_hw": depth_provenance,
        "depth_uncertainty_proxy_m_hw": depth_uncertainty,
        "intrinsics_output": k,
        "support_truth_hw": np.zeros(shape, dtype=np.float32),
        "support_truth_valid_hw": unknown.copy(),
        "support_quality_tier_hw": unknown.copy(),
        "support_provenance_hw": unknown.copy(),
        "support_unknown_hw": np.ones(shape, dtype=np.uint8),
        "obstacle_evidence_truth_hw": np.zeros(shape, dtype=np.float32),
        "evidence_truth_valid_hw": unknown.copy(),
        "evidence_quality_tier_hw": unknown.copy(),
        "evidence_provenance_hw": unknown.copy(),
        "evidence_unknown_hw": np.ones(shape, dtype=np.uint8),
        "boundary_core_probability_hw": boundary_core.astype(np.float32),
        "boundary_soft_probability_hw": boundary_soft.astype(np.float32),
        "boundary_distance_px_hw": boundary_distance.astype(np.float32),
        "boundary_uncertainty_px_hw": boundary_uncertainty,
        "boundary_angular_distance_rad_hw": angular_distance.astype(np.float32),
        "boundary_angular_soft_probability_hw": angular_soft.astype(np.float32),
        "boundary_truth_valid_hw": boundary_valid.astype(np.uint8),
        "boundary_factor_valid_hw": boundary_valid.astype(np.uint8),
        "boundary_unknown_hw": (~boundary_valid).astype(np.uint8),
        "boundary_quality_tier_hw": boundary_tier,
        "boundary_provenance_hw": boundary_provenance,
    }


def run(
    dataset_root: Path,
    archive: Path,
    catalog: Path,
    receipt: Path,
    cohort_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(not output_dir.exists(), f"Bonn factor output collision: {output_dir}")
    require(cohort_path.is_file(), "Bonn cohort missing")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    fit_indices = load_cohort_indices(cohort_path, "fit")
    evaluation_indices = load_cohort_indices(cohort_path, "evaluation")
    reserve_parents = {
        str(row["parent_id"]) for row in cohort.get("reserve_parents", [])
    }
    previous_fixed = {str(value) for value in cohort.get("previously_consumed_fixed8", [])}
    fit_parents = set(fit_indices)
    require(
        fit_parents.isdisjoint(evaluation_indices)
        and fit_parents.isdisjoint(reserve_parents)
        and fit_parents.isdisjoint(previous_fixed),
        "Bonn FIT role overlap",
    )
    _, _, source_provenance = validate_source_receipts(
        dataset_root, archive, catalog, receipt, fit_parents
    )
    pairs_by_parent = fixed_frame_pairs(dataset_root, fit_indices)
    require(len(pairs_by_parent) == 8, "Bonn FIT parent count drift")

    output_dir.mkdir(parents=True, exist_ok=False)
    frames: list[dict[str, Any]] = []
    for parent_id, pairs in sorted(pairs_by_parent.items()):
        require(len(pairs) == 3, f"Bonn FIT frame count drift: {parent_id}")
        for pair in pairs:
            depth, source_valid = load_depth_native(pair.depth.absolute_path)
            payload = build_factor_payload(depth, source_valid, BONN_INTRINSICS)
            identity = frame_id(parent_id, pair.rgb.row_index)
            output_path = output_dir / f"{identity}.npz"
            np.savez_compressed(output_path, **payload)
            boundary_valid = np.asarray(payload["boundary_truth_valid_hw"], dtype=np.bool_)
            boundary_core = boundary_valid & (
                np.asarray(payload["boundary_core_probability_hw"]) >= 0.5
            )
            angular_distance = np.asarray(
                payload["boundary_angular_distance_rad_hw"], dtype=np.float32
            )
            frames.append(
                {
                    "source": SOURCE_ID,
                    "role": "FIT",
                    "parent_id": parent_id,
                    "frame_id": identity,
                    "rgb_row_index_zero_based": pair.rgb.row_index,
                    "rgb_path": str(pair.rgb.absolute_path.resolve()),
                    "rgb_storage_kind": "file",
                    "rgb_sha256": sha256_file(pair.rgb.absolute_path),
                    "depth_path": str(pair.depth.absolute_path.resolve()),
                    "depth_sha256": sha256_file(pair.depth.absolute_path),
                    "association_delta_seconds": pair.association_delta_seconds,
                    "output": str(output_path.resolve()),
                    "output_sha256": sha256_file(output_path),
                    "output_bytes": output_path.stat().st_size,
                    "shape_hw": list(depth.shape),
                    "metric_depth_valid_pixels": int(np.sum(source_valid)),
                    "support_valid_pixels": 0,
                    "obstacle_evidence_valid_pixels": 0,
                    "boundary_valid_pixels": int(np.sum(boundary_valid)),
                    "boundary_core_positive_pixels": int(np.sum(boundary_core)),
                    "angular_soft_band_pixels_le_sigma": int(
                        np.sum(boundary_valid & (angular_distance <= ANGULAR_SOFT_SIGMA_RAD))
                    ),
                }
            )

    parent_counts = {
        parent: sum(row["parent_id"] == parent for row in frames)
        for parent in sorted(fit_parents)
    }
    gates = {
        "fit_parent_count_eq_8": len(fit_parents) == 8,
        "frame_count_eq_24": len(frames) == 24,
        "three_frames_per_parent": set(parent_counts.values()) == {3},
        "fit_disjoint_from_evaluation_reserve_and_previous_fixed8": True,
        "every_frame_has_source_depth": all(
            row["metric_depth_valid_pixels"] > 0 for row in frames
        ),
        "every_parent_has_boundary_core": all(
            sum(
                row["boundary_core_positive_pixels"]
                for row in frames
                if row["parent_id"] == parent
            )
            > 0
            for parent in fit_parents
        ),
        "support_and_obstacle_remain_unknown": all(
            row["support_valid_pixels"] == 0
            and row["obstacle_evidence_valid_pixels"] == 0
            for row in frames
        ),
        "all_outputs_have_sha_receipts": all(
            len(row["output_sha256"]) == 64 for row in frames
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_bonn_fit_angular_factor_labels_v1",
        "status": "BONN_FIT_ANGULAR_FACTOR_LABELS_PASS" if passed else "BONN_FIT_ANGULAR_FACTOR_LABELS_FAIL",
        "question": "Can eight new Bonn FIT parents add source-depth and camera-angular boundary supervision while unsupported support/obstacle factors remain UNKNOWN and frozen external roles remain sealed?",
        "complete_truth_required": False,
        "inputs": {
            "cohort": str(cohort_path.resolve()),
            "cohort_sha256": sha256_file(cohort_path),
            "source_provenance": source_provenance,
        },
        "contract": {
            "metric_depth": "Bonn source-native uint16 depth divided by 5000; invalid zero remains UNKNOWN",
            "boundary": "conservative point-to-plane discontinuity derived from source depth, plus pixel and camera-angular continuous fields",
            "support": "UNKNOWN because this source slice has no locked gravity/support evidence",
            "obstacle_evidence": "UNKNOWN because support-relative obstacle evidence is unavailable",
            "teacher_filled_pixels": "absent",
            "angular_soft_sigma_rad": ANGULAR_SOFT_SIGMA_RAD,
        },
        "source": SOURCE_ID,
        "role": "FIT",
        "parent_count": len(fit_parents),
        "frame_count": len(frames),
        "sealed_parent_counts": {
            "mixed_domain_evaluation": len(evaluation_indices),
            "reserve": len(reserve_parents),
            "previous_external_fixed8": len(previous_fixed),
        },
        "coverage": {
            "metric_depth_valid_pixels": sum(row["metric_depth_valid_pixels"] for row in frames),
            "boundary_valid_pixels": sum(row["boundary_valid_pixels"] for row in frames),
            "boundary_core_positive_pixels": sum(row["boundary_core_positive_pixels"] for row in frames),
            "angular_soft_band_pixels_le_sigma": sum(
                row["angular_soft_band_pixels_le_sigma"] for row in frames
            ),
            "support_valid_pixels": 0,
            "obstacle_evidence_valid_pixels": 0,
        },
        "gates": gates,
        "frames": frames,
        "decision": {
            "bonn_fit_factor_corpus_ready": passed,
            "complete_truth_required": False,
            "evaluation_or_reserve_consumed": False,
            "formal_f1_authority_changed": False,
        },
        "claim_boundary": "WILD_LAB source-depth and derived angular-boundary factor materialization for eight Bonn FIT parents. No complete truth, support, obstacle, license, task, safety, deployment, product, or formal F1 claim.",
    }
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    binding = {
        "schema": "blindassist_ag_st_bonn_fit_angular_rgb_binding_v1",
        "status": "BONN_FIT_ANGULAR_RGB_BINDING_PASS" if passed else "BONN_FIT_ANGULAR_RGB_BINDING_FAIL",
        "result": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "binding_count": len(frames),
        "parent_count": len(fit_parents),
        "frames": [
            {
                "source": row["source"],
                "parent_id": row["parent_id"],
                "frame_id": row["frame_id"],
                "rgb_storage_kind": row["rgb_storage_kind"],
                "rgb_path": row["rgb_path"],
                "rgb_sha256": row["rgb_sha256"],
                "label_path": row["output"],
                "label_sha256": row["output_sha256"],
                "label_shape_hw": row["shape_hw"],
            }
            for row in frames
        ],
    }
    binding_path = output_dir / "rgb_binding.json"
    with binding_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(binding, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_BONN_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_BONN_CATALOG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_BONN_RECEIPT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(
        args.dataset_root.resolve(),
        args.archive.resolve(),
        args.catalog.resolve(),
        args.receipt.resolve(),
        args.cohort.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "parent_count", "frame_count", "sealed_parent_counts", "coverage", "gates")
            },
            indent=2,
        )
    )
    return 0 if result["status"].endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
