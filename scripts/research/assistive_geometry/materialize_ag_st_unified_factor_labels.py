#!/usr/bin/env python3
"""Merge R5 depth/support factors with R9 continuous source-boundary factors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-superteacher-factor-labels-train16-r5"
)
DEFAULT_BOUNDARY_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-continuous-boundary-factors-r0"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-unified-factor-labels-train16-r0"
)
REPLACED_BASE_BOUNDARY_FIELDS = {
    "boundary_distance_px_hw",
    "boundary_probability_pseudo_hw",
    "boundary_uncertainty_proxy_px_hw",
}
BOUNDARY_FIELDS = {
    "boundary_core_probability_hw",
    "boundary_soft_probability_hw",
    "boundary_distance_px_hw",
    "boundary_uncertainty_px_hw",
    "boundary_truth_valid_hw",
    "boundary_unknown_hw",
    "boundary_quality_tier_hw",
    "boundary_provenance_hw",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in {"f", "c"}:
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def merge_payload(
    base: dict[str, np.ndarray],
    boundary: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    require(BOUNDARY_FIELDS <= set(boundary), "continuous boundary fields incomplete")
    output = {
        key: np.asarray(value)
        for key, value in base.items()
        if key not in REPLACED_BASE_BOUNDARY_FIELDS
    }
    for key in BOUNDARY_FIELDS:
        output[key] = np.asarray(boundary[key])
    output["boundary_factor_valid_hw"] = np.asarray(
        boundary["boundary_truth_valid_hw"], dtype=np.uint8
    )
    output["boundary_factor_unknown_hw"] = np.asarray(
        boundary["boundary_unknown_hw"], dtype=np.uint8
    )
    return output


def run(base_dir: Path, boundary_dir: Path, output_dir: Path) -> dict[str, Any]:
    for path in (base_dir, boundary_dir):
        require(path.is_dir(), f"unified factor input missing: {path}")
    require(not output_dir.exists(), f"unified factor output exists: {output_dir}")
    base_result_path = base_dir / "result.json"
    boundary_result_path = boundary_dir / "result.json"
    require(base_result_path.is_file() and boundary_result_path.is_file(), "input result receipt missing")
    base_result = json.loads(base_result_path.read_text(encoding="utf-8"))
    boundary_result = json.loads(boundary_result_path.read_text(encoding="utf-8"))
    require(base_result.get("status") == "COMPLETED", "R5 base factors incomplete")
    require(
        boundary_result.get("status") == "CONTINUOUS_BOUNDARY_FACTORS_PASS",
        "R9 boundary factors incomplete",
    )
    base_paths = {path.stem: path for path in base_dir.glob("*.npz")}
    boundary_rows = {
        str(row["frame_id"]): row
        for row in boundary_result["frames"]
        if row["source"] == "arkitscenes"
    }
    require(len(base_paths) == len(boundary_rows) == 48, "unified factor frame count drift")
    require(set(base_paths) == set(boundary_rows), "R5/R9 frame identity mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    receipts: list[dict[str, Any]] = []
    copied_field_count: int | None = None
    for frame_id in sorted(base_paths):
        base_path = base_paths[frame_id]
        boundary_path = Path(boundary_rows[frame_id]["output"])
        require(boundary_path.is_file(), f"R9 boundary NPZ missing: {boundary_path}")
        require(
            sha256_file(boundary_path) == boundary_rows[frame_id]["output_sha256"],
            "R9 boundary NPZ SHA drift",
        )
        with np.load(base_path) as values:
            base = {key: np.asarray(values[key]) for key in values.files}
        with np.load(boundary_path) as values:
            boundary = {key: np.asarray(values[key]) for key in values.files}
        payload = merge_payload(base, boundary)
        if copied_field_count is None:
            copied_field_count = len(set(base) - REPLACED_BASE_BOUNDARY_FIELDS)
        require(
            copied_field_count == len(set(base) - REPLACED_BASE_BOUNDARY_FIELDS),
            "R5 field schema varies by frame",
        )
        output_path = output_dir / f"{frame_id}.npz"
        np.savez_compressed(output_path, **payload)
        with np.load(output_path) as written:
            written_payload = {key: np.asarray(written[key]) for key in written.files}
        require(set(written_payload) == set(payload), "unified output field set drift")
        require(
            all(arrays_equal(written_payload[key], value) for key, value in payload.items()),
            "unified output array drift",
        )
        valid = np.asarray(boundary["boundary_truth_valid_hw"], dtype=np.bool_)
        distance = np.asarray(boundary["boundary_distance_px_hw"], dtype=np.float32)
        receipts.append(
            {
                "parent_id": frame_id.split("_", 1)[0],
                "frame_id": frame_id,
                "base_path": str(base_path.resolve()),
                "base_sha256": sha256_file(base_path),
                "boundary_path": str(boundary_path.resolve()),
                "boundary_sha256": sha256_file(boundary_path),
                "output": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "output_bytes": output_path.stat().st_size,
                "field_count": len(payload),
                "boundary_valid_pixels": int(np.sum(valid)),
                "boundary_core_positive_pixels": int(
                    np.sum(valid & (boundary["boundary_core_probability_hw"] >= 0.5))
                ),
                "boundary_soft_band_pixels_le_3px": int(np.sum(valid & (distance <= 3.0))),
            }
        )
    total_pixels = sum(int(np.prod(row["shape_hw"])) for row in boundary_rows.values())
    boundary_valid_pixels = sum(row["boundary_valid_pixels"] for row in receipts)
    gates = {
        "parent_count_eq_16": len({row["parent_id"] for row in receipts}) == 16,
        "frame_count_eq_48": len(receipts) == 48,
        "r5_r9_identity_match_48_of_48": set(base_paths) == set(boundary_rows),
        "base_nonboundary_arrays_exact": True,
        "r9_boundary_arrays_exact": True,
        "every_output_has_sha_receipt": all(len(row["output_sha256"]) == 64 for row in receipts),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_unified_factor_labels_v1",
        "status": "UNIFIED_FACTOR_LABELS_PASS" if passed else "UNIFIED_FACTOR_LABELS_FAIL",
        "question": "Can metric depth, normal, support, obstacle evidence and continuous source boundary supervision be delivered in one frame-aligned package with factor-specific masks?",
        "complete_truth_required": False,
        "inputs": {
            "base_r5_result": str(base_result_path.resolve()),
            "base_r5_result_sha256": sha256_file(base_result_path),
            "boundary_r9_result": str(boundary_result_path.resolve()),
            "boundary_r9_result_sha256": sha256_file(boundary_result_path),
        },
        "merge_contract": {
            "preserved_exactly": "all R5 arrays except its three previous boundary fields",
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
        "parent_count": len({row["parent_id"] for row in receipts}),
        "frame_count": len(receipts),
        "field_count_per_frame": receipts[0]["field_count"],
        "coverage": {
            "metric_depth": base_result["coverage"]["metric_label_coverage"],
            "normal": base_result["coverage"]["normal_coverage"],
            "support": base_result["coverage"]["support_coverage"],
            "obstacle_evidence": base_result["coverage"]["boundary_evidence_coverage"],
            "continuous_boundary": boundary_valid_pixels / total_pixels,
            "continuous_boundary_valid_pixels": boundary_valid_pixels,
            "continuous_boundary_core_positive_pixels": sum(row["boundary_core_positive_pixels"] for row in receipts),
            "continuous_boundary_soft_band_pixels_le_3px": sum(row["boundary_soft_band_pixels_le_3px"] for row in receipts),
        },
        "gates": gates,
        "frames": receipts,
        "decision": {
            "unified_masked_factor_training_ready": passed,
            "teacher_filled_boundary_training_authorized": False,
            "formal_f1_authority_changed": False,
        },
        "claim_boundary": "TRAIN-only WILD_LAB unified factor pseudo-label package with source-native boundary replacement; no complete truth, cross-source task utility, formal F1, safety, deployment, or product claim.",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.base_dir, args.boundary_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "parent_count", "frame_count", "field_count_per_frame", "coverage", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
