#!/usr/bin/env python3
"""Rematerialize TUM factors with sequence-level support-height identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_ag_st_factor_labels import compute_geometric_factors
from download_b0_arkitscenes_assets import require, sha256_file
from materialize_ag_st_tum_gravity_factors import unknown_geometric_factors


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-tum-third-teacher-r2/result.json"
)
DEFAULT_IDENTITY_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-tum-support-identity-r0/result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-tum-support-identity-factors-r0"
)
IDENTITY_ACCEPTED = {
    "ELEVATED_DOMINANT_SURFACE_REPLICATED",
    "LOWEST_PERSISTENT_SURFACE_REPLICATED",
}


def _identity_by_parent(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(
        result.get("status") == "TUM_DOMINANT_SUPPORT_IDENTITY_FAILURE_DETECTED",
        "support-identity diagnostic status invalid",
    )
    rows = result.get("parents")
    require(isinstance(rows, list) and len(rows) == 7, "support-identity parent drift")
    output = {str(row["parent_id"]): row for row in rows}
    require(len(output) == len(rows), "duplicate support-identity parent")
    return output


def run(source_result_path: Path, identity_result_path: Path, output_dir: Path) -> dict[str, Any]:
    require(source_result_path.is_file(), "AG-ST TUM depth result missing")
    require(identity_result_path.is_file(), "AG-ST TUM support-identity result missing")
    require(not output_dir.exists(), f"support-identity output already exists: {output_dir}")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    identity_result = json.loads(identity_result_path.read_text(encoding="utf-8"))
    require(
        source_result.get("status")
        == "THIRD_TEACHER_NOT_PROMOTED_TWO_TEACHER_LABELS_MATERIALIZED",
        "AG-ST TUM depth source status invalid",
    )
    identities = _identity_by_parent(identity_result)
    receipts = source_result.get("frame_receipts")
    require(isinstance(receipts, list) and len(receipts) == 21, "AG-ST TUM frame drift")
    output_dir.mkdir(parents=True)

    totals = {
        "identity_eligible_pixels": 0,
        "normal_valid": 0,
        "support_valid": 0,
        "support_positive": 0,
        "evidence_valid": 0,
        "boundary_seed": 0,
        "source_support": 0,
        "teacher_support": 0,
    }
    output_receipts: list[dict[str, Any]] = []
    identity_eligible_frames = 0
    floor_corrected_frames = 0
    passthrough_frames = 0
    for receipt in receipts:
        source_path = Path(str(receipt["output_path"]))
        require(source_path.is_file(), f"AG-ST TUM NPZ missing: {source_path}")
        with np.load(source_path, allow_pickle=False) as loaded:
            label = {name: loaded[name] for name in loaded.files}
        shape = tuple(int(value) for value in label["metric_depth_m_hw"].shape)
        parent = str(receipt["parent_id"])
        identity = identities[parent]
        identity_status = str(identity["status"])
        identity_eligible = identity_status in IDENTITY_ACCEPTED
        lowest_world_height = (
            float(identity["lowest_persistent_world_height_m"])
            if identity_eligible
            else None
        )
        if identity_eligible:
            camera_world_height = float(label["camera_to_world_output"][2, 3])
            camera_height = camera_world_height - float(lowest_world_height)
            factors = compute_geometric_factors(
                label["metric_depth_m_hw"],
                label["metric_depth_valid_hw"],
                label["intrinsics_output"],
                label["camera_to_world_output"],
                label["quality_score_hw"],
                label["quality_tier_hw"],
                label["provenance_code_hw"],
                label["depth_uncertainty_proxy_m_hw"],
                support_camera_height_override_m=camera_height,
                support_plane_residual_override_m=0.02,
            )
            identity_eligible_frames += 1
            floor_corrected_frames += int(
                identity_status == "ELEVATED_DOMINANT_SURFACE_REPLICATED"
            )
            passthrough_frames += int(
                identity_status == "LOWEST_PERSISTENT_SURFACE_REPLICATED"
            )
            totals["identity_eligible_pixels"] += int(np.prod(shape))
        else:
            camera_height = None
            factors = unknown_geometric_factors(shape)
        label.update(factors)
        label["support_valid_hw"] = factors["support_truth_valid_hw"]
        label["boundary_evidence_valid_hw"] = factors["evidence_truth_valid_hw"]
        label["support_identity_world_height_m"] = np.asarray(
            lowest_world_height if lowest_world_height is not None else np.nan,
            dtype=np.float32,
        )
        label["support_identity_valid"] = np.asarray(identity_eligible, dtype=np.bool_)
        output_path = output_dir / source_path.name
        np.savez_compressed(output_path, **label)

        normal_valid = factors["normal_valid_hw"].astype(np.bool_)
        support_valid = factors["support_truth_valid_hw"].astype(np.bool_)
        support_positive = support_valid & (factors["support_truth_hw"] >= 0.5)
        evidence_valid = factors["evidence_truth_valid_hw"].astype(np.bool_)
        boundary_seed = evidence_valid & (factors["boundary_probability_pseudo_hw"] >= 0.5)
        support_provenance = factors["support_provenance_code_hw"]
        totals["normal_valid"] += int(normal_valid.sum())
        totals["support_valid"] += int(support_valid.sum())
        totals["support_positive"] += int(support_positive.sum())
        totals["evidence_valid"] += int(evidence_valid.sum())
        totals["boundary_seed"] += int(boundary_seed.sum())
        totals["source_support"] += int(np.sum(support_valid & (support_provenance == 1)))
        totals["teacher_support"] += int(np.sum(support_valid & (support_provenance == 2)))
        output_receipts.append(
            {
                "role": receipt["role"],
                "parent_id": parent,
                "frame_id": receipt["frame_id"],
                "support_identity_status": identity_status,
                "support_identity_valid": identity_eligible,
                "support_identity_world_height_m": lowest_world_height,
                "camera_height_m": camera_height,
                "support_plane_valid": bool(factors["support_plane_valid"]),
                "normal_valid_coverage": float(np.mean(normal_valid)),
                "support_truth_valid_coverage": float(np.mean(support_valid)),
                "support_positive_coverage": float(np.mean(support_positive)),
                "evidence_truth_valid_coverage": float(np.mean(evidence_valid)),
                "boundary_seed_coverage": float(np.mean(boundary_seed)),
                "output_path": str(output_path.resolve()),
                "output_bytes": output_path.stat().st_size,
            }
        )

    eligible_pixels = totals["identity_eligible_pixels"]
    require(identity_eligible_frames == 12, "support-identity eligible frame drift")
    require(floor_corrected_frames == 9, "floor-corrected frame drift")
    require(passthrough_frames == 3, "lowest-plane passthrough frame drift")
    return {
        "schema": "blindassist_ag_st_tum_support_identity_factors_result_v1",
        "status": "TUM_SEQUENCE_HEIGHT_IDENTITY_FACTORS_MATERIALIZED",
        "source_depth_result": str(source_result_path.resolve()),
        "source_depth_result_sha256": sha256_file(source_result_path),
        "support_identity_result": str(identity_result_path.resolve()),
        "support_identity_result_sha256": sha256_file(identity_result_path),
        "frame_count": len(output_receipts),
        "support_identity_eligible_frame_count": identity_eligible_frames,
        "floor_corrected_frame_count": floor_corrected_frames,
        "lowest_plane_passthrough_frame_count": passthrough_frames,
        "unknown_frame_count": len(output_receipts) - identity_eligible_frames,
        "coverage_over_identity_eligible_pixels": {
            "normal_valid": totals["normal_valid"] / eligible_pixels,
            "support_truth_valid": totals["support_valid"] / eligible_pixels,
            "support_positive": totals["support_positive"] / eligible_pixels,
            "evidence_truth_valid": totals["evidence_valid"] / eligible_pixels,
            "boundary_seed": totals["boundary_seed"] / eligible_pixels,
        },
        "support_provenance_pixels": {
            "source_native": totals["source_support"],
            "teacher_derived": totals["teacher_support"],
        },
        "frame_receipts": output_receipts,
        "decision": {
            "complete_truth_required": False,
            "per_frame_dominant_plane_retired": True,
            "ambiguous_or_gravity_missing_is_unknown": True,
            "student_training_authorized": False,
            "next_execution": (
                "Validate support identity and level-change boundary direction on a "
                "synthetic-exact source, then admit only the passing factor fields to student training."
            ),
        },
        "claim_boundary": (
            "TUM sequence-height-anchored support/boundary pseudo-label materialization. "
            "The lowest persistent horizontal surface is not source-native walkability truth; "
            "no task utility, product, or safety claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--identity-result", type=Path, default=DEFAULT_IDENTITY_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.source_result, args.identity_result, args.output_dir)
    result_path = args.output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "frame_count",
                    "support_identity_eligible_frame_count",
                    "floor_corrected_frame_count",
                    "unknown_frame_count",
                    "coverage_over_identity_eligible_pixels",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
