#!/usr/bin/env python3
"""Add gravity-anchored support and boundary factors to AG-ST TUM depth labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_ag_st_factor_labels import compute_geometric_factors
from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-st-tum-third-teacher-r2/result.json"
DEFAULT_GRAVITY_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-st-tum-gravity-diagnostic-r0/result.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-st-tum-gravity-factors-r0"


def unknown_geometric_factors(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    zeros = np.zeros(shape, dtype=np.float32)
    false = np.zeros(shape, dtype=np.bool_)
    tiers = np.zeros(shape, dtype=np.uint8)
    nan = np.full(shape, np.nan, dtype=np.float32)
    return {
        "dense_normal_diagnostic_camera_xyz_hwc": np.zeros((*shape, 3), dtype=np.float32),
        "normal_valid_hw": false.copy(),
        "normal_quality_tier_hw": tiers.copy(),
        "normal_provenance_code_hw": tiers.copy(),
        "support_probability_pseudo_hw": zeros.copy(),
        "support_truth_hw": zeros.copy(),
        "support_truth_valid_hw": false.copy(),
        "support_quality_tier_hw": tiers.copy(),
        "support_provenance_code_hw": tiers.copy(),
        "support_plane_normal_camera_xyz": np.zeros(3, dtype=np.float32),
        "camera_height_m": np.asarray(np.nan, dtype=np.float32),
        "support_plane_fit_residual_diagnostic_m": np.asarray(np.nan, dtype=np.float32),
        "support_plane_valid": np.asarray(False, dtype=np.bool_),
        "support_plane_quality_tier": np.asarray(0, dtype=np.uint8),
        "support_plane_provenance_code": np.asarray(0, dtype=np.uint8),
        "support_plane_fit_source_pixel_count": np.asarray(0, dtype=np.int64),
        "support_plane_fit_teacher_pixel_count": np.asarray(0, dtype=np.int64),
        "height_above_support_m_hw": nan.copy(),
        "obstacle_evidence_truth_hw": zeros.copy(),
        "boundary_probability_pseudo_hw": zeros.copy(),
        "boundary_distance_px_hw": nan.copy(),
        "boundary_uncertainty_proxy_px_hw": nan.copy(),
        "evidence_truth_valid_hw": false.copy(),
        "evidence_quality_tier_hw": tiers.copy(),
        "evidence_provenance_code_hw": tiers.copy(),
        "physical_boundary_valid_diagnostic_hw": false.copy(),
    }


def run(source_result_path: Path, gravity_result_path: Path, output_dir: Path) -> dict[str, Any]:
    require(source_result_path.is_file(), "AG-ST TUM depth result missing")
    require(gravity_result_path.is_file(), "AG-ST TUM gravity result missing")
    require(not output_dir.exists(), f"gravity-factor output already exists: {output_dir}")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    gravity_result = json.loads(gravity_result_path.read_text(encoding="utf-8"))
    require(
        source_result.get("status")
        == "THIRD_TEACHER_NOT_PROMOTED_TWO_TEACHER_LABELS_MATERIALIZED",
        "AG-ST TUM depth source status invalid",
    )
    require(
        gravity_result.get("status") == "TUM_GRAVITY_BASIS_VALIDATED",
        "TUM gravity basis is not validated",
    )
    missing = set(gravity_result["parents_without_accelerometer"])
    receipts = source_result.get("frame_receipts")
    require(isinstance(receipts, list) and len(receipts) == 21, "AG-ST TUM frame receipt drift")
    output_dir.mkdir(parents=True)

    totals = {
        "pixels": 0,
        "eligible_pixels": 0,
        "normal_valid": 0,
        "support_valid": 0,
        "support_positive": 0,
        "evidence_valid": 0,
        "boundary_seed": 0,
        "source_support": 0,
        "teacher_support": 0,
    }
    plane_valid_frames = 0
    eligible_frames = 0
    output_receipts: list[dict[str, Any]] = []
    for receipt in receipts:
        source_path = Path(str(receipt["output_path"]))
        require(source_path.is_file(), f"AG-ST TUM NPZ missing: {source_path}")
        with np.load(source_path, allow_pickle=False) as loaded:
            label = {name: loaded[name] for name in loaded.files}
        shape = tuple(int(value) for value in label["metric_depth_m_hw"].shape)
        parent = str(receipt["parent_id"])
        gravity_eligible = parent not in missing
        if gravity_eligible:
            factors = compute_geometric_factors(
                label["metric_depth_m_hw"],
                label["metric_depth_valid_hw"],
                label["intrinsics_output"],
                label["camera_to_world_output"],
                label["quality_score_hw"],
                label["quality_tier_hw"],
                label["provenance_code_hw"],
                label["depth_uncertainty_proxy_m_hw"],
            )
            eligible_frames += 1
            totals["eligible_pixels"] += int(np.prod(shape))
        else:
            factors = unknown_geometric_factors(shape)
        label.update(factors)
        label["support_valid_hw"] = factors["support_truth_valid_hw"]
        label["boundary_evidence_valid_hw"] = factors["evidence_truth_valid_hw"]
        output_path = output_dir / source_path.name
        np.savez_compressed(output_path, **label)

        normal_valid = factors["normal_valid_hw"].astype(np.bool_)
        support_valid = factors["support_truth_valid_hw"].astype(np.bool_)
        support_positive = support_valid & (factors["support_truth_hw"] >= 0.5)
        evidence_valid = factors["evidence_truth_valid_hw"].astype(np.bool_)
        boundary_seed = evidence_valid & (factors["boundary_probability_pseudo_hw"] >= 0.5)
        support_provenance = factors["support_provenance_code_hw"]
        totals["pixels"] += int(np.prod(shape))
        totals["normal_valid"] += int(normal_valid.sum())
        totals["support_valid"] += int(support_valid.sum())
        totals["support_positive"] += int(support_positive.sum())
        totals["evidence_valid"] += int(evidence_valid.sum())
        totals["boundary_seed"] += int(boundary_seed.sum())
        totals["source_support"] += int(np.sum(support_valid & (support_provenance == 1)))
        totals["teacher_support"] += int(np.sum(support_valid & (support_provenance == 2)))
        plane_valid = bool(factors["support_plane_valid"])
        plane_valid_frames += int(plane_valid)
        output_receipts.append(
            {
                "role": receipt["role"],
                "parent_id": parent,
                "frame_id": receipt["frame_id"],
                "gravity_eligible": gravity_eligible,
                "support_plane_valid": plane_valid,
                "normal_valid_coverage": float(np.mean(normal_valid)),
                "support_truth_valid_coverage": float(np.mean(support_valid)),
                "evidence_truth_valid_coverage": float(np.mean(evidence_valid)),
                "boundary_seed_coverage": float(np.mean(boundary_seed)),
                "output_path": str(output_path.resolve()),
                "output_bytes": output_path.stat().st_size,
            }
        )

    eligible = totals["eligible_pixels"]
    return {
        "schema": "blindassist_ag_st_tum_gravity_factors_result_v1",
        "status": "TUM_GRAVITY_ANCHORED_SUPPORT_BOUNDARY_PSEUDOLABELS_MATERIALIZED",
        "source_depth_result": str(source_result_path.resolve()),
        "source_depth_result_sha256": sha256_file(source_result_path),
        "gravity_result": str(gravity_result_path.resolve()),
        "gravity_result_sha256": sha256_file(gravity_result_path),
        "frame_count": len(output_receipts),
        "gravity_eligible_frame_count": eligible_frames,
        "gravity_unknown_frame_count": len(output_receipts) - eligible_frames,
        "support_plane_valid_frames": plane_valid_frames,
        "coverage_over_gravity_eligible_pixels": {
            "normal_valid": totals["normal_valid"] / eligible if eligible else None,
            "support_truth_valid": totals["support_valid"] / eligible if eligible else None,
            "support_positive": totals["support_positive"] / eligible if eligible else None,
            "evidence_truth_valid": totals["evidence_valid"] / eligible if eligible else None,
            "boundary_seed": totals["boundary_seed"] / eligible if eligible else None,
        },
        "support_provenance_pixels": {
            "source_native": totals["source_support"],
            "teacher_derived": totals["teacher_support"],
        },
        "parents_without_accelerometer_remain_unknown": sorted(missing),
        "frame_receipts": output_receipts,
        "decision": {
            "complete_truth_required": False,
            "support_boundary_are_pseudolabels": True,
            "student_training_authorized": False,
            "next_execution": "Validate these support/boundary pseudo-labels against a gravity-native or synthetic-exact source before training a student.",
        },
        "claim_boundary": "Gravity-anchored TUM factor pseudo-labels with explicit UNKNOWN only; not source-native support/boundary truth, formal F1 authorization, task utility, product, or safety evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--gravity-result", type=Path, default=DEFAULT_GRAVITY_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.source_result, args.gravity_result, args.output_dir)
    result_path = args.output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "frame_count", "gravity_eligible_frame_count", "support_plane_valid_frames", "coverage_over_gravity_eligible_pixels")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
