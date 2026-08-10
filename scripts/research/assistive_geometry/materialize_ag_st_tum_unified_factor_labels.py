#!/usr/bin/env python3
"""Merge validated TUM depth/support factors with R9 continuous boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from materialize_ag_st_unified_factor_labels import (
    BOUNDARY_FIELDS,
    REPLACED_BASE_BOUNDARY_FIELDS,
    arrays_equal,
    merge_payload,
    require,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-tum-support-identity-factors-r0"
)
DEFAULT_BOUNDARY_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-continuous-boundary-factors-r0"
)
DEFAULT_R3_RESULT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R3_SUPPORT_IDENTITY_RESULT_2026-08-10.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-unified-factor-labels-tum7-r0"
)


def select_tum_boundary_rows(boundary_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {
        str(row["frame_id"]): row
        for row in boundary_result["frames"]
        if row["source"] == "tum_rgbd"
    }
    require(len(rows) == 21, "R9 TUM boundary frame count drift")
    return rows


def validation_sha_by_base_stem(validation: dict[str, Any]) -> dict[str, str]:
    receipts = validation["output_receipts"]
    require(len(receipts) == 21, "TUM base validation receipt count drift")
    return {str(row["frame_id"]): str(row["output_sha256"]) for row in receipts}


def base_validation_frame_id(base_stem: str) -> str:
    parts = base_stem.split("__", 1)
    require(len(parts) == 2 and parts[0] in {"fit", "evaluation"}, "TUM role/frame id malformed")
    return parts[1]


def run(
    base_dir: Path,
    boundary_dir: Path,
    r3_result_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for path in (base_dir, boundary_dir):
        require(path.is_dir(), f"unified TUM input missing: {path}")
    require(r3_result_path.is_file(), f"R3 authorization receipt missing: {r3_result_path}")
    require(not output_dir.exists(), f"unified TUM output exists: {output_dir}")

    base_result_path = base_dir / "result.json"
    base_validation_path = base_dir / "validation.json"
    boundary_result_path = boundary_dir / "result.json"
    require(
        base_result_path.is_file()
        and base_validation_path.is_file()
        and boundary_result_path.is_file(),
        "unified TUM input receipt missing",
    )
    base_result = json.loads(base_result_path.read_text(encoding="utf-8"))
    base_validation = json.loads(base_validation_path.read_text(encoding="utf-8"))
    boundary_result = json.loads(boundary_result_path.read_text(encoding="utf-8"))
    r3_result = json.loads(r3_result_path.read_text(encoding="utf-8"))

    require(
        base_result.get("status") == "TUM_SEQUENCE_HEIGHT_IDENTITY_FACTORS_MATERIALIZED",
        "TUM sequence-identity factors incomplete",
    )
    require(
        base_validation.get("status") == "TUM_SEQUENCE_HEIGHT_IDENTITY_FACTOR_INVARIANTS_PASS",
        "TUM sequence-identity factor validation failed",
    )
    require(
        bool(r3_result.get("decision", {}).get("wild_lab_masked_depth_support_training_authorized")),
        "R3 did not authorize TUM sequence-identity depth/support training",
    )
    require(
        boundary_result.get("status") == "CONTINUOUS_BOUNDARY_FACTORS_PASS",
        "R9 continuous boundary factors incomplete",
    )

    base_paths = {path.stem: path for path in base_dir.glob("*.npz")}
    boundary_rows = select_tum_boundary_rows(boundary_result)
    validation_shas = validation_sha_by_base_stem(base_validation)
    require(len(base_paths) == len(boundary_rows) == 21, "unified TUM frame count drift")
    require(set(base_paths) == set(boundary_rows), "TUM base/R9 frame identity mismatch")

    output_dir.mkdir(parents=True, exist_ok=False)
    receipts: list[dict[str, Any]] = []
    total_pixels = 0
    coverage_counts = {
        "metric_depth": 0,
        "normal": 0,
        "support": 0,
        "obstacle_evidence": 0,
        "continuous_boundary": 0,
    }
    copied_field_count: int | None = None
    base_receipt_matches = True
    unknown_is_not_negative = True

    for frame_id in sorted(base_paths):
        base_path = base_paths[frame_id]
        row = boundary_rows[frame_id]
        boundary_path = Path(row["output"])
        require(boundary_path.is_file(), f"R9 TUM boundary NPZ missing: {boundary_path}")
        require(
            sha256_file(boundary_path) == row["output_sha256"],
            "R9 TUM boundary NPZ SHA drift",
        )
        validation_frame_id = base_validation_frame_id(frame_id)
        require(validation_frame_id in validation_shas, "TUM base validation frame receipt missing")
        base_receipt_matches &= sha256_file(base_path) == validation_shas[validation_frame_id]

        with np.load(base_path) as values:
            base = {key: np.asarray(values[key]) for key in values.files}
        with np.load(boundary_path) as values:
            boundary = {key: np.asarray(values[key]) for key in values.files}
        payload = merge_payload(base, boundary)
        if copied_field_count is None:
            copied_field_count = len(set(base) - REPLACED_BASE_BOUNDARY_FIELDS)
        require(
            copied_field_count == len(set(base) - REPLACED_BASE_BOUNDARY_FIELDS),
            "TUM base field schema varies by frame",
        )

        output_path = output_dir / f"{frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path) as written:
            written_payload = {key: np.asarray(written[key]) for key in written.files}
        require(set(written_payload) == set(payload), "unified TUM output field set drift")
        require(
            all(arrays_equal(written_payload[key], value) for key, value in payload.items()),
            "unified TUM output array drift",
        )

        valid = np.asarray(boundary["boundary_truth_valid_hw"], dtype=np.bool_)
        unknown = np.asarray(boundary["boundary_unknown_hw"], dtype=np.bool_)
        distance = np.asarray(boundary["boundary_distance_px_hw"], dtype=np.float32)
        unknown_is_not_negative &= bool(np.all(unknown == ~valid))
        unknown_is_not_negative &= bool(np.all(np.isnan(distance[~valid])))
        frame_pixels = int(valid.size)
        total_pixels += frame_pixels
        coverage_counts["metric_depth"] += int(np.sum(base["metric_depth_valid_hw"]))
        coverage_counts["normal"] += int(np.sum(base["normal_valid_hw"]))
        coverage_counts["support"] += int(np.sum(base["support_truth_valid_hw"]))
        coverage_counts["obstacle_evidence"] += int(np.sum(base["evidence_truth_valid_hw"]))
        coverage_counts["continuous_boundary"] += int(np.sum(valid))
        receipts.append(
            {
                "source": "tum_rgbd",
                "parent_id": str(row["parent_id"]),
                "frame_id": frame_id,
                "role": frame_id.split("__", 1)[0],
                "base_path": str(base_path.resolve()),
                "base_sha256": sha256_file(base_path),
                "boundary_path": str(boundary_path.resolve()),
                "boundary_sha256": sha256_file(boundary_path),
                "output": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "output_bytes": output_path.stat().st_size,
                "field_count": len(payload),
                "shape_hw": [int(value) for value in valid.shape],
                "boundary_valid_pixels": int(np.sum(valid)),
                "boundary_core_positive_pixels": int(
                    np.sum(valid & (boundary["boundary_core_probability_hw"] >= 0.5))
                ),
                "boundary_soft_band_pixels_le_3px": int(np.sum(valid & (distance <= 3.0))),
            }
        )

    parent_count = len({row["parent_id"] for row in receipts})
    field_count = receipts[0]["field_count"]
    gates = {
        "r3_depth_support_training_authorized": True,
        "tum_base_invariants_pass": True,
        "parent_count_eq_7": parent_count == 7,
        "frame_count_eq_21": len(receipts) == 21,
        "base_r9_identity_match_21_of_21": set(base_paths) == set(boundary_rows),
        "base_sha_matches_validation_21_of_21": base_receipt_matches,
        "base_nonboundary_arrays_exact": True,
        "r9_boundary_arrays_exact": True,
        "unknown_is_never_negative": unknown_is_not_negative,
        "every_output_has_sha_receipt": all(len(row["output_sha256"]) == 64 for row in receipts),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_tum_unified_factor_labels_v1",
        "status": "TUM_UNIFIED_FACTOR_LABELS_PASS" if passed else "TUM_UNIFIED_FACTOR_LABELS_FAIL",
        "question": "Can validated TUM metric/support evidence and R9 source-native continuous boundaries be delivered as a second-source unified factor package?",
        "complete_truth_required": False,
        "inputs": {
            "base_result": str(base_result_path.resolve()),
            "base_result_sha256": sha256_file(base_result_path),
            "base_validation": str(base_validation_path.resolve()),
            "base_validation_sha256": sha256_file(base_validation_path),
            "r3_authorization": str(r3_result_path.resolve()),
            "r3_authorization_sha256": sha256_file(r3_result_path),
            "boundary_r9_result": str(boundary_result_path.resolve()),
            "boundary_r9_result_sha256": sha256_file(boundary_result_path),
        },
        "merge_contract": {
            "preserved_exactly": "all validated TUM arrays except the three previous boundary fields",
            "replaced": sorted(REPLACED_BASE_BOUNDARY_FIELDS),
            "added_from_r9": sorted(BOUNDARY_FIELDS),
            "factor_specific_masks": {
                "depth": "metric_depth_valid_hw",
                "normal": "normal_valid_hw",
                "support": "support_truth_valid_hw",
                "obstacle_evidence": "evidence_truth_valid_hw",
                "boundary": "boundary_factor_valid_hw",
            },
            "unknown_policy": "each factor loss uses only its own validity mask; UNKNOWN is never negative",
        },
        "parent_count": parent_count,
        "frame_count": len(receipts),
        "field_count_per_frame": field_count,
        "total_pixels": total_pixels,
        "coverage": {
            key: {"valid_pixels": count, "fraction_of_all_pixels": count / total_pixels}
            for key, count in coverage_counts.items()
        },
        "continuous_boundary_core_positive_pixels": sum(
            row["boundary_core_positive_pixels"] for row in receipts
        ),
        "continuous_boundary_soft_band_pixels_le_3px": sum(
            row["boundary_soft_band_pixels_le_3px"] for row in receipts
        ),
        "gates": gates,
        "frames": receipts,
        "decision": {
            "tum_second_source_unified_training_ready": passed,
            "multi_source_student_input_ready_with_arkit_r10": passed,
            "teacher_filled_boundary_training_authorized": False,
            "formal_f1_authority_changed": False,
        },
        "claim_boundary": "TUM RGB-D WILD_LAB unified selective factor labels with validated sequence support identity and source-native boundary; not complete truth, fresh evaluation, task utility, formal F1, safety, deployment, or product evidence.",
    }
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    parser.add_argument("--r3-result", type=Path, default=DEFAULT_R3_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.base_dir, args.boundary_dir, args.r3_result, args.output_dir)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "parent_count",
                    "frame_count",
                    "field_count_per_frame",
                    "coverage",
                    "gates",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
