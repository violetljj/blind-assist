#!/usr/bin/env python3
"""Materialize continuous boundary factors from the R6 source-native corpus."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINDING = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-source-native-boundary-corpus-r0/rgb_binding.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-continuous-boundary-factors-r0"
)
SOFT_SIGMA_PX = 3.0
MAX_DISTANCE_PX = 32.0


def continuous_boundary_factors(
    probability: np.ndarray,
    valid: np.ndarray,
    *,
    sigma_px: float = SOFT_SIGMA_PX,
    max_distance_px: float = MAX_DISTANCE_PX,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(probability, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.bool_)
    require(
        score.shape == mask.shape and sigma_px > 0.0 and max_distance_px > 0.0,
        "continuous boundary input invalid",
    )
    core = mask & (score >= 0.5)
    if np.any(core):
        distance = distance_transform_edt(~core).astype(np.float32)
    else:
        distance = np.full(score.shape, max_distance_px, dtype=np.float32)
    distance = np.minimum(distance, max_distance_px).astype(np.float32)
    heat = np.exp(-0.5 * np.square(distance / sigma_px)).astype(np.float32)
    soft = np.maximum(score, heat).astype(np.float32)
    soft[~mask] = 0.0
    distance[~mask] = np.nan
    return distance, np.clip(soft, 0.0, 1.0)


def materialize_frame(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    source_path = Path(row["label_path"])
    require(source_path.is_file(), f"source boundary label missing: {source_path}")
    require(sha256_file(source_path) == row["label_sha256"], "source boundary label SHA drift")
    with np.load(source_path) as values:
        probability = np.asarray(values["boundary_probability_hw"], dtype=np.float32)
        valid = np.asarray(values["boundary_truth_valid_hw"], dtype=np.bool_)
        uncertainty = np.asarray(values["boundary_uncertainty_px_hw"], dtype=np.float32)
        tier = np.asarray(values["boundary_quality_tier_hw"], dtype=np.uint8)
        provenance = np.asarray(values["boundary_provenance_hw"], dtype=np.uint8)
    expected_shape = tuple(int(value) for value in row["label_shape_hw"])
    require(
        probability.shape == valid.shape == uncertainty.shape == tier.shape == provenance.shape == expected_shape,
        "source boundary factor shape drift",
    )
    distance, soft = continuous_boundary_factors(probability, valid)
    output_path = output_dir / f"{row['source']}__{row['frame_id']}.npz"
    np.savez_compressed(
        output_path,
        boundary_core_probability_hw=probability,
        boundary_soft_probability_hw=soft,
        boundary_distance_px_hw=distance,
        boundary_uncertainty_px_hw=uncertainty,
        boundary_truth_valid_hw=valid.astype(np.uint8),
        evidence_truth_valid_hw=valid.astype(np.uint8),
        boundary_unknown_hw=(~valid).astype(np.uint8),
        boundary_quality_tier_hw=tier,
        boundary_provenance_hw=provenance,
    )
    core = valid & (probability >= 0.5)
    band = valid & (distance <= SOFT_SIGMA_PX)
    return {
        "source": str(row["source"]),
        "parent_id": str(row["parent_id"]),
        "frame_id": str(row["frame_id"]),
        "rgb_path": row.get("rgb_path"),
        "rgb_source_archive": row.get("rgb_source_archive"),
        "rgb_member": row.get("rgb_member"),
        "rgb_sha256": str(row["rgb_sha256"]),
        "source_label_path": str(source_path.resolve()),
        "source_label_sha256": str(row["label_sha256"]),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "shape_hw": list(expected_shape),
        "valid_pixels": int(np.sum(valid)),
        "core_positive_pixels": int(np.sum(core)),
        "soft_band_pixels_le_3px": int(np.sum(band)),
        "soft_probability_mass": float(np.sum(soft[valid], dtype=np.float64)),
    }


def run(binding_path: Path, output_dir: Path) -> dict[str, Any]:
    require(binding_path.is_file(), f"boundary RGB binding missing: {binding_path}")
    require(not output_dir.exists(), f"continuous boundary output exists: {output_dir}")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    require(
        binding.get("status") == "SOURCE_NATIVE_BOUNDARY_RGB_BINDING_PASS",
        "boundary RGB binding invalid",
    )
    rows = list(binding.get("frames", []))
    require(len(rows) == 81, "boundary binding frame count drift")
    output_dir.mkdir(parents=True, exist_ok=False)
    frames = [materialize_frame(row, output_dir) for row in rows]
    by_source: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frames:
        grouped[row["source"]].append(row)
    for source, selected in sorted(grouped.items()):
        by_source[source] = {
            "parent_count": len({row["parent_id"] for row in selected}),
            "frame_count": len(selected),
            "valid_pixels": sum(row["valid_pixels"] for row in selected),
            "core_positive_pixels": sum(row["core_positive_pixels"] for row in selected),
            "soft_band_pixels_le_3px": sum(row["soft_band_pixels_le_3px"] for row in selected),
            "soft_probability_mass": sum(row["soft_probability_mass"] for row in selected),
        }
    gates = {
        "source_count_eq_3": len(by_source) == 3,
        "parent_count_eq_24": len({(row["source"], row["parent_id"]) for row in frames}) == 24,
        "frame_count_eq_81": len(frames) == 81,
        "every_frame_has_valid_pixels": all(row["valid_pixels"] > 0 for row in frames),
        "every_source_has_core_positive_pixels": all(value["core_positive_pixels"] > 0 for value in by_source.values()),
        "every_output_has_sha_receipt": all(len(row["output_sha256"]) == 64 for row in frames),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_continuous_boundary_factors_v1",
        "status": "CONTINUOUS_BOUNDARY_FACTORS_PASS" if passed else "CONTINUOUS_BOUNDARY_FACTORS_FAIL",
        "question": "Can R6 source-native/exact boundary evidence be materialized as continuous factor supervision without complete truth or Teacher-filled negatives?",
        "complete_truth_required": False,
        "input": {
            "binding": str(binding_path.resolve()),
            "binding_sha256": sha256_file(binding_path),
            "frame_count": len(rows),
        },
        "contract": {
            "boundary_distance_px_hw": "Euclidean distance to nearest valid source boundary core, clipped to 32 px; NaN outside validity",
            "boundary_soft_probability_hw": "max(source probability, exp(-distance^2/(2*3px^2))); zero outside validity",
            "validity": "copied exactly from source-native/source-exact boundary operator",
            "unknown": "invalid pixels remain UNKNOWN and are never negative",
            "uncertainty_tier_provenance": "copied without promotion from the R6 source label",
            "teacher_filled_pixels": "absent",
        },
        "source_count": len(by_source),
        "parent_count": len({(row["source"], row["parent_id"]) for row in frames}),
        "frame_count": len(frames),
        "by_source": by_source,
        "gates": gates,
        "frames": frames,
        "decision": {
            "continuous_boundary_factor_training_ready": passed,
            "teacher_filled_boundary_training_authorized": False,
            "formal_f1_authority_changed": False,
        },
        "claim_boundary": "WILD_LAB continuous boundary factor labels derived from R6 source-native/exact evidence; no complete truth, task utility, formal F1, safety, deployment, or product claim.",
    }
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.binding, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "source_count", "parent_count", "frame_count", "by_source", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
