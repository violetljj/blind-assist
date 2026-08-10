#!/usr/bin/env python3
"""Upgrade multi-Teacher labels with sequence-level support-height identity."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from build_ag_st_factor_labels import (
    PROVENANCE_SOURCE_NATIVE,
    TIER_B_ANCHORED,
    backproject_depth_grid,
    compute_dense_normals,
    compute_geometric_factors,
)
from diagnose_ag_st_tum_support_identity import SupportIdentityPolicy, _mode_candidates
from download_b0_arkitscenes_assets import require, sha256_file
from materialize_ag_st_tum_gravity_factors import unknown_geometric_factors


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-superteacher-factor-labels-multiteacher-train16-r2/result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-superteacher-factor-labels-multiteacher-train16-identity-r1"
)


def horizontal_source_world_heights(label: dict[str, np.ndarray]) -> np.ndarray:
    depth = label["metric_depth_m_hw"].astype(np.float32)
    source_valid = (
        label["metric_depth_valid_hw"].astype(np.bool_)
        & (label["provenance_code_hw"] == PROVENANCE_SOURCE_NATIVE)
        & (label["quality_tier_hw"] >= TIER_B_ANCHORED)
    )
    normals, normal_valid = compute_dense_normals(
        depth,
        source_valid,
        label["intrinsics_output"],
    )
    pose = label["camera_to_world_output"].astype(np.float64)
    rotation = pose[:3, :3]
    normals_world = np.einsum("...j,ij->...i", normals, rotation)
    points_camera = backproject_depth_grid(depth, label["intrinsics_output"])
    world_height = np.einsum("...j,j->...", points_camera, rotation[2]) + pose[2, 3]
    horizontal = (
        source_valid
        & normal_valid
        & np.isfinite(world_height)
        & (depth <= 5.0)
        & (np.abs(normals_world[..., 2]) >= math.cos(math.radians(20.0)))
    )
    return world_height[horizontal].astype(np.float64)[::2]


def _load_label(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"factor label missing: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def run(source_result_path: Path, output_dir: Path) -> dict[str, Any]:
    require(source_result_path.is_file(), "multi-Teacher factor result missing")
    require(not output_dir.exists(), f"sequence-identity output already exists: {output_dir}")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    require(
        source_result.get("schema")
        == "blindassist_ag_st_multiteacher_factor_label_factory_wild_lab_result_v1"
        and source_result.get("status") == "COMPLETED",
        "multi-Teacher factor result invalid",
    )
    receipts = source_result.get("frame_receipts")
    require(isinstance(receipts, list) and len(receipts) == 48, "multi-Teacher frame drift")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, dict[str, np.ndarray]] = {}
    for receipt in receipts:
        frame = str(receipt["frame_stem"])
        labels[frame] = _load_label(Path(str(receipt["output_path"])))
        grouped[str(receipt["parent_id"])].append(receipt)
    require(len(grouped) == 16 and all(len(rows) == 3 for rows in grouped.values()), "parent/frame drift")

    identity_policy = SupportIdentityPolicy(
        sample_stride=1,
        minimum_persistent_frames=2,
        minimum_frame_points=48,
        minimum_frame_fraction=0.001,
        minimum_total_points=192,
        minimum_total_fraction=0.001,
    )
    parent_identity: dict[str, dict[str, Any]] = {}
    for parent, rows in sorted(grouped.items()):
        frame_values: list[np.ndarray] = []
        evaluable_frames: list[str] = []
        for receipt in rows:
            frame = str(receipt["frame_stem"])
            values = horizontal_source_world_heights(labels[frame])
            if len(values) >= identity_policy.minimum_frame_points:
                frame_values.append(values)
                evaluable_frames.append(frame)
        modes = _mode_candidates(frame_values, identity_policy) if len(frame_values) >= 2 else []
        lowest = float(modes[0]["world_height_m"]) if modes else None
        camera_heights = [
            float(labels[str(receipt["frame_stem"])]["camera_to_world_output"][2, 3])
            - float(lowest)
            for receipt in rows
        ] if lowest is not None else []
        plausible_count = sum(0.45 <= height <= 2.20 for height in camera_heights)
        parent_valid = lowest is not None and plausible_count >= 2
        parent_identity[parent] = {
            "status": "SEQUENCE_HEIGHT_IDENTITY_AVAILABLE" if parent_valid else "UNKNOWN_SEQUENCE_HEIGHT_IDENTITY",
            "evaluable_frames": evaluable_frames,
            "lowest_persistent_world_height_m": lowest,
            "horizontal_modes": modes,
            "camera_heights_m": camera_heights,
            "plausible_camera_height_frame_count": plausible_count,
        }

    output_dir.mkdir(parents=True)
    output_receipts: list[dict[str, Any]] = []
    totals = {
        "pixels": 0,
        "eligible_pixels": 0,
        "support_valid": 0,
        "support_positive": 0,
        "evidence_valid": 0,
        "boundary_seed": 0,
        "old_support_positive": 0,
    }
    eligible_frames = 0
    invariant_failures: list[str] = []
    for receipt in receipts:
        parent = str(receipt["parent_id"])
        frame = str(receipt["frame_stem"])
        label = labels[frame]
        shape = label["metric_depth_m_hw"].shape
        identity = parent_identity[parent]
        lowest = identity["lowest_persistent_world_height_m"]
        camera_height = (
            float(label["camera_to_world_output"][2, 3]) - float(lowest)
            if lowest is not None
            else None
        )
        identity_valid = bool(
            identity["status"] == "SEQUENCE_HEIGHT_IDENTITY_AVAILABLE"
            and camera_height is not None
            and 0.45 <= camera_height <= 2.20
        )
        old_positive = label["support_truth_valid_hw"].astype(np.bool_) & (
            label["support_truth_hw"] >= 0.5
        )
        if identity_valid:
            factors = compute_geometric_factors(
                label["metric_depth_m_hw"],
                label["metric_depth_valid_hw"],
                label["intrinsics_output"],
                label["camera_to_world_output"],
                label["quality_score_hw"],
                label["quality_tier_hw"],
                label["provenance_code_hw"],
                label["depth_uncertainty_proxy_m_hw"],
                support_camera_height_override_m=float(camera_height),
                support_plane_residual_override_m=0.02,
            )
            eligible_frames += 1
            totals["eligible_pixels"] += int(np.prod(shape))
        else:
            factors = unknown_geometric_factors(shape)
        label.update(factors)
        label["support_valid_hw"] = factors["support_truth_valid_hw"]
        label["boundary_evidence_valid_hw"] = factors["evidence_truth_valid_hw"]
        label["support_identity_world_height_m"] = np.asarray(
            float(lowest) if lowest is not None else np.nan,
            dtype=np.float32,
        )
        label["support_identity_valid"] = np.asarray(identity_valid, dtype=np.bool_)
        output_path = output_dir / f"{frame}.npz"
        np.savez_compressed(output_path, **label)

        support_valid = factors["support_truth_valid_hw"].astype(np.bool_)
        support_positive = support_valid & (factors["support_truth_hw"] >= 0.5)
        evidence_valid = factors["evidence_truth_valid_hw"].astype(np.bool_)
        boundary_seed = evidence_valid & (factors["boundary_probability_pseudo_hw"] >= 0.5)
        totals["pixels"] += int(np.prod(shape))
        totals["support_valid"] += int(np.sum(support_valid))
        totals["support_positive"] += int(np.sum(support_positive))
        totals["evidence_valid"] += int(np.sum(evidence_valid))
        totals["boundary_seed"] += int(np.sum(boundary_seed))
        totals["old_support_positive"] += int(np.sum(old_positive))
        if identity_valid and not bool(factors["support_plane_valid"]):
            invariant_failures.append(f"{frame}: identity-valid plane invalid")
        if not identity_valid and (np.any(support_valid) or np.any(evidence_valid)):
            invariant_failures.append(f"{frame}: UNKNOWN factor materialized")
        output_receipts.append(
            {
                "frame_index": int(receipt["frame_index"]),
                "frame_stem": frame,
                "parent_id": parent,
                "support_identity_valid": identity_valid,
                "support_identity_world_height_m": lowest,
                "camera_height_m": camera_height,
                "support_truth_valid_coverage": float(np.mean(support_valid)),
                "support_positive_coverage": float(np.mean(support_positive)),
                "evidence_truth_valid_coverage": float(np.mean(evidence_valid)),
                "boundary_seed_coverage": float(np.mean(boundary_seed)),
                "output_path": str(output_path.resolve()),
                "output_bytes": output_path.stat().st_size,
            }
        )

    eligible_pixels = totals["eligible_pixels"]
    identity_parent_count = sum(
        row["status"] == "SEQUENCE_HEIGHT_IDENTITY_AVAILABLE"
        for row in parent_identity.values()
    )
    require(not invariant_failures, "sequence-identity factor invariant failure")
    return {
        "schema": "blindassist_ag_st_sequence_identity_factor_labels_result_v1",
        "status": "SEQUENCE_IDENTITY_MULTITEACHER_LABELS_MATERIALIZED",
        "source_result": str(source_result_path.resolve()),
        "source_result_sha256": sha256_file(source_result_path),
        "parent_count": len(grouped),
        "frame_count": len(receipts),
        "identity_parent_count": identity_parent_count,
        "identity_eligible_frame_count": eligible_frames,
        "unknown_frame_count": len(receipts) - eligible_frames,
        "coverage_over_identity_eligible_pixels": {
            "support_truth_valid": totals["support_valid"] / eligible_pixels if eligible_pixels else None,
            "support_positive": totals["support_positive"] / eligible_pixels if eligible_pixels else None,
            "evidence_truth_valid": totals["evidence_valid"] / eligible_pixels if eligible_pixels else None,
            "boundary_seed": totals["boundary_seed"] / eligible_pixels if eligible_pixels else None,
        },
        "old_vs_identity_support_positive_pixels_all_frames": {
            "old": totals["old_support_positive"],
            "identity": totals["support_positive"],
        },
        "invariant_failure_count": len(invariant_failures),
        "parents": [
            {"parent_id": parent, **identity}
            for parent, identity in sorted(parent_identity.items())
        ],
        "frame_receipts": output_receipts,
        "decision": {
            "complete_truth_required": False,
            "sequence_identity_applied_before_support_derivation": True,
            "unknown_frames_have_zero_factor_denominator": True,
            "student_training_authorized": eligible_frames > 0,
            "student_scope": "WILD_LAB masked depth/support only; boundary remains diagnostic",
        },
        "claim_boundary": (
            "Sequence-height-anchored multi-Teacher pseudo-labels on TRAIN parents. "
            "External ICL exact-mesh evidence supports the height-identity mechanism, but these "
            "labels are not source-native walkability truth, task utility, product, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.source_result, args.output_dir)
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
                    "parent_count",
                    "identity_parent_count",
                    "frame_count",
                    "identity_eligible_frame_count",
                    "unknown_frame_count",
                    "coverage_over_identity_eligible_pixels",
                    "old_vs_identity_support_positive_pixels_all_frames",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
