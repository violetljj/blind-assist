#!/usr/bin/env python3
"""Add camera-angular distance fields to all R9 continuous boundary factors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from download_b0_arkitscenes_assets import require, sha256_file
from materialize_ag_st_unified_factor_labels import arrays_equal
from run_ag_st_angular_boundary_resize_canary import (
    ANGULAR_SOFT_SIGMA_RAD,
    camera_angular_boundary_factors,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOUNDARY_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-continuous-boundary-factors-r0"
)
DEFAULT_ARKIT_UNIFIED_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-unified-factor-labels-train16-r0"
)
DEFAULT_TUM_UNIFIED_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-unified-factor-labels-tum7-r0"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-continuous-boundary-factors-angular-r0"
)
ICL_INTRINSICS_OUTPUT = np.asarray(
    [[120.3, 0.0, 79.375], [0.0, 120.0, 59.375], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
ANGULAR_FIELDS = {
    "boundary_angular_distance_rad_hw",
    "boundary_angular_soft_probability_hw",
}


def source_intrinsics(
    row: dict[str, Any],
    arkit_rows: dict[str, dict[str, Any]],
    tum_rows: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    source = str(row["source"])
    frame_id = str(row["frame_id"])
    if source == "icl_exact":
        return ICL_INTRINSICS_OUTPUT.copy(), {
            "kind": "ICL_FIXED_FLIP_DOWNSAMPLE_CALIBRATION",
            "source_path": None,
            "source_sha256": None,
        }
    lookup = arkit_rows if source == "arkitscenes" else tum_rows
    require(source in {"arkitscenes", "tum_rgbd"}, "angular boundary source invalid")
    require(frame_id in lookup, "angular boundary intrinsics frame missing")
    receipt = lookup[frame_id]
    source_path = Path(receipt["output"])
    require(source_path.is_file(), "angular boundary intrinsics payload missing")
    require(
        sha256_file(source_path) == receipt["output_sha256"],
        "angular boundary intrinsics payload SHA drift",
    )
    with np.load(source_path, allow_pickle=False) as values:
        intrinsics = np.asarray(values["intrinsics_output"], dtype=np.float64)
    return intrinsics, {
        "kind": "UNIFIED_FACTOR_INTRINSICS_OUTPUT",
        "source_path": str(source_path.resolve()),
        "source_sha256": str(receipt["output_sha256"]),
    }


def add_angular_fields(
    base: dict[str, np.ndarray],
    intrinsics: np.ndarray,
) -> dict[str, np.ndarray]:
    require(not (ANGULAR_FIELDS & set(base)), "angular boundary field collision")
    probability = np.asarray(base["boundary_core_probability_hw"], dtype=np.float32)
    valid = np.asarray(base["boundary_truth_valid_hw"], dtype=np.bool_)
    angle, soft = camera_angular_boundary_factors(probability, valid, intrinsics)
    return {
        **{key: np.asarray(value) for key, value in base.items()},
        "boundary_angular_distance_rad_hw": angle,
        "boundary_angular_soft_probability_hw": soft,
    }


def run(
    boundary_dir: Path,
    arkit_unified_dir: Path,
    tum_unified_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for path in (boundary_dir, arkit_unified_dir, tum_unified_dir):
        require(path.is_dir(), f"angular boundary input directory missing: {path}")
    require(not output_dir.exists(), f"angular boundary output exists: {output_dir}")
    boundary_result_path = boundary_dir / "result.json"
    arkit_result_path = arkit_unified_dir / "result.json"
    tum_result_path = tum_unified_dir / "result.json"
    boundary_result = json.loads(boundary_result_path.read_text(encoding="utf-8"))
    arkit_result = json.loads(arkit_result_path.read_text(encoding="utf-8"))
    tum_result = json.loads(tum_result_path.read_text(encoding="utf-8"))
    require(
        boundary_result.get("status") == "CONTINUOUS_BOUNDARY_FACTORS_PASS",
        "R9 continuous boundary input incomplete",
    )
    require(
        arkit_result.get("status") == "UNIFIED_FACTOR_LABELS_PASS",
        "R10 ARKit intrinsics input incomplete",
    )
    require(
        tum_result.get("status") == "TUM_UNIFIED_FACTOR_LABELS_PASS",
        "R13 TUM intrinsics input incomplete",
    )
    boundary_rows = list(boundary_result["frames"])
    arkit_rows = {str(row["frame_id"]): row for row in arkit_result["frames"]}
    tum_rows = {str(row["frame_id"]): row for row in tum_result["frames"]}
    require(
        len(boundary_rows) == 81 and len(arkit_rows) == 48 and len(tum_rows) == 21,
        "angular boundary input frame count drift",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    receipts: list[dict[str, Any]] = []
    base_arrays_exact = True
    unknown_preserved = True
    for row in boundary_rows:
        source_path = Path(row["output"])
        require(source_path.is_file(), "R9 boundary payload missing")
        require(
            sha256_file(source_path) == row["output_sha256"],
            "R9 boundary payload SHA drift",
        )
        with np.load(source_path, allow_pickle=False) as values:
            base = {key: np.asarray(values[key]) for key in values.files}
        intrinsics, intrinsics_receipt = source_intrinsics(row, arkit_rows, tum_rows)
        require(
            intrinsics.shape == (3, 3)
            and np.isfinite(intrinsics).all()
            and intrinsics[0, 0] > 0.0
            and intrinsics[1, 1] > 0.0,
            "angular boundary intrinsics invalid",
        )
        payload = add_angular_fields(base, intrinsics)
        output_path = output_dir / source_path.name
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as written:
            written_payload = {key: np.asarray(written[key]) for key in written.files}
        require(set(written_payload) == set(payload), "angular boundary output schema drift")
        require(
            all(arrays_equal(written_payload[key], value) for key, value in payload.items()),
            "angular boundary output array drift",
        )
        base_arrays_exact &= all(
            arrays_equal(written_payload[key], value) for key, value in base.items()
        )
        valid = np.asarray(base["boundary_truth_valid_hw"], dtype=np.bool_)
        core = valid & (np.asarray(base["boundary_core_probability_hw"]) >= 0.5)
        angle = np.asarray(payload["boundary_angular_distance_rad_hw"], dtype=np.float32)
        soft = np.asarray(
            payload["boundary_angular_soft_probability_hw"], dtype=np.float32
        )
        unknown_preserved &= bool(
            np.isnan(angle[~valid]).all()
            and np.all(soft[~valid] == 0.0)
            and np.all(angle[core] == 0.0)
            and np.all(soft[core] >= 0.5)
        )
        receipts.append(
            {
                "source": str(row["source"]),
                "parent_id": str(row["parent_id"]),
                "frame_id": str(row["frame_id"]),
                "shape_hw": list(angle.shape),
                "r9_path": str(source_path.resolve()),
                "r9_sha256": str(row["output_sha256"]),
                "intrinsics": intrinsics.tolist(),
                "intrinsics_receipt": intrinsics_receipt,
                "output": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "output_bytes": output_path.stat().st_size,
                "valid_pixels": int(np.sum(valid)),
                "core_positive_pixels": int(np.sum(core)),
                "angular_soft_band_pixels_le_sigma": int(
                    np.sum(valid & (angle <= ANGULAR_SOFT_SIGMA_RAD))
                ),
                "angular_soft_probability_mass": float(
                    np.sum(soft[valid], dtype=np.float64)
                ),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipts:
        grouped[row["source"]].append(row)
    by_source = {
        source: {
            "parent_count": len({row["parent_id"] for row in rows}),
            "frame_count": len(rows),
            "valid_pixels": sum(row["valid_pixels"] for row in rows),
            "core_positive_pixels": sum(row["core_positive_pixels"] for row in rows),
            "angular_soft_band_pixels_le_sigma": sum(
                row["angular_soft_band_pixels_le_sigma"] for row in rows
            ),
            "angular_soft_probability_mass": sum(
                row["angular_soft_probability_mass"] for row in rows
            ),
        }
        for source, rows in sorted(grouped.items())
    }
    gates = {
        "source_count_eq_3": len(by_source) == 3,
        "parent_count_eq_24": len(
            {(row["source"], row["parent_id"]) for row in receipts}
        )
        == 24,
        "frame_count_eq_81": len(receipts) == 81,
        "r9_base_arrays_exact": base_arrays_exact,
        "factor_validity_unknown_preserved": unknown_preserved,
        "every_frame_has_positive_focal_lengths": all(
            row["intrinsics"][0][0] > 0.0 and row["intrinsics"][1][1] > 0.0
            for row in receipts
        ),
        "every_source_has_core_positive_pixels": all(
            row["core_positive_pixels"] > 0 for row in by_source.values()
        ),
        "every_output_has_sha_receipt": all(
            len(row["output_sha256"]) == 64 for row in receipts
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_angular_boundary_factors_v1",
        "status": (
            "ANGULAR_BOUNDARY_FACTORS_PASS"
            if passed
            else "ANGULAR_BOUNDARY_FACTORS_FAIL"
        ),
        "question": "Can the R9 source-native/exact boundary corpus gain resolution-aware angular supervision without changing any existing factor array or UNKNOWN mask?",
        "complete_truth_required": False,
        "inputs": {
            "r9_boundary_result": str(boundary_result_path.resolve()),
            "r9_boundary_result_sha256": sha256_file(boundary_result_path),
            "r10_arkit_unified_result": str(arkit_result_path.resolve()),
            "r10_arkit_unified_result_sha256": sha256_file(arkit_result_path),
            "r13_tum_unified_result": str(tum_result_path.resolve()),
            "r13_tum_unified_result_sha256": sha256_file(tum_result_path),
        },
        "contract": {
            "preserved_exactly": "all R9 arrays",
            "boundary_angular_distance_rad_hw": "camera-ray angle to nearest valid source boundary core; clipped at 0.25 rad and NaN outside validity",
            "boundary_angular_soft_probability_hw": f"max(source core probability, exp(-angle^2/(2*{ANGULAR_SOFT_SIGMA_RAD}^2))); zero outside validity",
            "intrinsics": "per-frame frozen intrinsics_output for ARKit/TUM; fixed flip/downsample calibration for ICL exact",
            "unknown": "unchanged factor validity; unsupported pixels remain UNKNOWN and never negative",
            "teacher_filled_pixels": "absent",
        },
        "source_count": len(by_source),
        "parent_count": len({(row["source"], row["parent_id"]) for row in receipts}),
        "frame_count": len(receipts),
        "angular_soft_sigma_rad": ANGULAR_SOFT_SIGMA_RAD,
        "by_source": by_source,
        "gates": gates,
        "frames": receipts,
        "decision": {
            "angular_boundary_factor_training_ready": passed,
            "existing_pixel_boundary_fields_retired": False,
            "teacher_filled_boundary_training_authorized": False,
            "formal_f1_authority_changed": False,
        },
        "claim_boundary": "Resolution-aware angular boundary pseudo-label materialization from existing source-native/exact evidence; no learned improvement, task utility, formal F1, safety, deployment, or product claim.",
    }
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    parser.add_argument("--arkit-unified-dir", type=Path, default=DEFAULT_ARKIT_UNIFIED_DIR)
    parser.add_argument("--tum-unified-dir", type=Path, default=DEFAULT_TUM_UNIFIED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(
        args.boundary_dir.resolve(),
        args.arkit_unified_dir.resolve(),
        args.tum_unified_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "source_count",
                    "parent_count",
                    "frame_count",
                    "angular_soft_sigma_rad",
                    "by_source",
                    "gates",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
