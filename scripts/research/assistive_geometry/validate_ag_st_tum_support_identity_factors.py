#!/usr/bin/env python3
"""Validate identity-aware TUM factor materialization and quantify correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-tum-support-identity-factors-r0/result.json"
)
DEFAULT_PREDECESSOR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-tum-gravity-factors-r0/result.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-tum-support-identity-factors-r0/validation.json"
)


def validate(result_path: Path, predecessor_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    require(
        result.get("status") == "TUM_SEQUENCE_HEIGHT_IDENTITY_FACTORS_MATERIALIZED",
        "support-identity factor status invalid",
    )
    require(
        predecessor.get("status")
        == "TUM_GRAVITY_ANCHORED_SUPPORT_BOUNDARY_PSEUDOLABELS_MATERIALIZED",
        "predecessor factor status invalid",
    )
    old_by_frame = {
        str(receipt["frame_id"]): Path(str(receipt["output_path"]))
        for receipt in predecessor["frame_receipts"]
    }
    receipts = result.get("frame_receipts")
    require(isinstance(receipts, list) and len(receipts) == 21, "factor receipt drift")
    invariant_failures: list[str] = []
    eligible = 0
    unknown = 0
    corrected_old_positive = 0
    corrected_new_positive = 0
    corrected_pixels = 0
    output_bytes = 0
    output_hashes: list[dict[str, Any]] = []
    for receipt in receipts:
        frame_id = str(receipt["frame_id"])
        path = Path(str(receipt["output_path"]))
        require(path.is_file(), f"factor label missing: {path}")
        output_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as arrays:
            valid = arrays["metric_depth_valid_hw"].astype(np.bool_)
            identity_valid = bool(arrays["support_identity_valid"])
            plane_valid = bool(arrays["support_plane_valid"])
            support_valid = arrays["support_truth_valid_hw"].astype(np.bool_)
            support_positive = support_valid & (arrays["support_truth_hw"] >= 0.5)
            evidence_valid = arrays["evidence_truth_valid_hw"].astype(np.bool_)
            normal_valid = arrays["normal_valid_hw"].astype(np.bool_)
            if receipt["support_identity_valid"]:
                eligible += 1
                if not identity_valid or not plane_valid:
                    invariant_failures.append(f"{frame_id}: eligible identity/plane invalid")
                camera_height = float(arrays["camera_height_m"])
                expected_height = (
                    float(arrays["camera_to_world_output"][2, 3])
                    - float(arrays["support_identity_world_height_m"])
                )
                if not (0.45 <= camera_height <= 2.20 and abs(camera_height - expected_height) <= 1e-5):
                    invariant_failures.append(f"{frame_id}: camera-height identity mismatch")
                rotation = arrays["camera_to_world_output"][:3, :3].astype(np.float64)
                gravity_camera = rotation.T @ np.asarray([0.0, 0.0, 1.0])
                gravity_camera /= np.linalg.norm(gravity_camera)
                normal = arrays["support_plane_normal_camera_xyz"].astype(np.float64)
                if float(np.dot(gravity_camera, normal)) < 0.999:
                    invariant_failures.append(f"{frame_id}: support normal not gravity aligned")
                if np.any(support_valid & ~valid) or np.any(evidence_valid & ~valid):
                    invariant_failures.append(f"{frame_id}: derived validity escapes metric validity")
            else:
                unknown += 1
                if identity_valid or plane_valid:
                    invariant_failures.append(f"{frame_id}: UNKNOWN identity materialized")
                if np.any(support_valid) or np.any(evidence_valid) or np.any(normal_valid):
                    invariant_failures.append(f"{frame_id}: UNKNOWN factor pixels materialized")
            if receipt["support_identity_status"] == "ELEVATED_DOMINANT_SURFACE_REPLICATED":
                corrected_new_positive += int(np.sum(support_positive))
                corrected_pixels += int(np.prod(valid.shape))
                old_path = old_by_frame[frame_id]
                with np.load(old_path, allow_pickle=False) as old:
                    old_positive = old["support_truth_valid_hw"].astype(np.bool_) & (
                        old["support_truth_hw"] >= 0.5
                    )
                    corrected_old_positive += int(np.sum(old_positive))
        output_hashes.append(
            {
                "frame_id": frame_id,
                "output_bytes": path.stat().st_size,
                "output_sha256": sha256_file(path),
            }
        )
    require(eligible == 12 and unknown == 9, "identity eligible/UNKNOWN count drift")
    old_rate = corrected_old_positive / corrected_pixels
    new_rate = corrected_new_positive / corrected_pixels
    reduction = 1.0 - corrected_new_positive / max(1, corrected_old_positive)
    passed = not invariant_failures and reduction >= 0.90
    return {
        "schema": "blindassist_ag_st_tum_support_identity_factor_validation_v1",
        "status": (
            "TUM_SEQUENCE_HEIGHT_IDENTITY_FACTOR_INVARIANTS_PASS"
            if passed
            else "TUM_SEQUENCE_HEIGHT_IDENTITY_FACTOR_INVARIANTS_FAIL"
        ),
        "result": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "predecessor": str(predecessor_path.resolve()),
        "predecessor_sha256": sha256_file(predecessor_path),
        "frame_count": len(receipts),
        "eligible_frame_count": eligible,
        "unknown_frame_count": unknown,
        "output_total_bytes": output_bytes,
        "invariant_failure_count": len(invariant_failures),
        "invariant_failures": invariant_failures,
        "elevated_parent_correction": {
            "frame_count": 9,
            "pixel_count": corrected_pixels,
            "old_support_positive_pixels": corrected_old_positive,
            "new_support_positive_pixels": corrected_new_positive,
            "old_support_positive_coverage": old_rate,
            "new_support_positive_coverage": new_rate,
            "positive_pixel_reduction_fraction": reduction,
        },
        "output_receipts": output_hashes,
        "claim_boundary": (
            "Mechanical invariants and support-positive correction on the frozen TUM pseudo-label "
            "cohort only; not semantic walkability truth or downstream task evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--predecessor", type=Path, default=DEFAULT_PREDECESSOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"factor validation output exists: {args.output}")
    result = validate(args.result, args.predecessor)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "frame_count", "invariant_failure_count", "elevated_parent_correction")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
