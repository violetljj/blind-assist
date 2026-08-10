#!/usr/bin/env python3
"""Test whether per-frame dominant support planes are elevated surfaces.

This is a WILD_LAB geometry canary, not a ground-truth generator.  It uses only
registered TUM sensor depth and camera-to-world pose to recover persistent
gravity-aligned height modes within each parent.  A dominant per-frame plane is
rejected as elevated when a substantially lower horizontal mode persists across
the selected source-native frames.  Ambiguous cases remain UNKNOWN.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ag_st_tum_rgbd import TumSelectedPayload, load_tum_role_payloads
from build_ag_st_factor_labels import backproject_depth_grid, compute_dense_normals
from download_b0_arkitscenes_assets import require, sha256_file
from plan_ag_st_tum_third_teacher_cohort import DEFAULT_OUTPUT as DEFAULT_COHORT


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAVITY_FACTORS = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-tum-gravity-factors-r0/result.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-tum-support-identity-r0/result.json"
)


@dataclass(frozen=True)
class SupportIdentityPolicy:
    height_bin_m: float = 0.04
    mode_radius_bins: int = 2
    minimum_persistent_frames: int = 2
    minimum_frame_points: int = 96
    minimum_frame_fraction: float = 0.002
    minimum_total_points: int = 384
    minimum_total_fraction: float = 0.002
    horizontal_tilt_degrees: float = 20.0
    maximum_depth_m: float = 5.0
    sample_stride: int = 2
    same_surface_tolerance_m: float = 0.12
    elevated_separation_m: float = 0.25


def _horizontal_world_heights(
    payload: TumSelectedPayload,
    policy: SupportIdentityPolicy,
) -> np.ndarray:
    depth, valid = payload.load_depth()
    normals_camera, normal_valid = compute_dense_normals(
        depth,
        valid,
        payload.intrinsics,
    )
    points_camera = backproject_depth_grid(depth, payload.intrinsics)
    rotation = np.asarray(payload.camera_to_world[:3, :3], dtype=np.float64)
    translation = np.asarray(payload.camera_to_world[:3, 3], dtype=np.float64)
    normals_world = np.einsum("...j,ij->...i", normals_camera, rotation)
    points_world_z = (
        np.einsum("...j,j->...", points_camera, rotation[2]) + translation[2]
    )
    horizontal = (
        valid
        & normal_valid
        & np.isfinite(points_world_z)
        & (depth <= policy.maximum_depth_m)
        & (
            np.abs(normals_world[..., 2])
            >= math.cos(math.radians(policy.horizontal_tilt_degrees))
        )
    )
    values = points_world_z[horizontal].astype(np.float64)
    require(len(values) > 0, f"no horizontal source points: {payload.parent_id}")
    return values[:: policy.sample_stride]


def _mode_candidates(
    frame_heights: list[np.ndarray],
    policy: SupportIdentityPolicy,
) -> list[dict[str, Any]]:
    require(len(frame_heights) >= policy.minimum_persistent_frames, "too few source frames")
    all_values = np.concatenate(frame_heights)
    require(len(all_values) > 0 and np.all(np.isfinite(all_values)), "invalid height samples")
    low, high = np.quantile(all_values, (0.002, 0.998))
    first_bin = int(math.floor(float(low) / policy.height_bin_m)) - 1
    last_bin = int(math.ceil(float(high) / policy.height_bin_m)) + 1
    bin_ids = np.arange(first_bin, last_bin + 1, dtype=np.int64)
    frame_counts: list[np.ndarray] = []
    for values in frame_heights:
        ids = np.floor(values / policy.height_bin_m).astype(np.int64)
        counts = np.zeros(len(bin_ids), dtype=np.int64)
        inside = (ids >= first_bin) & (ids <= last_bin)
        np.add.at(counts, ids[inside] - first_bin, 1)
        frame_counts.append(counts)
    stacked = np.stack(frame_counts, axis=0)
    total = np.sum(stacked, axis=0)
    kernel = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)
    smooth = np.convolve(total.astype(np.float64), kernel / kernel.sum(), mode="same")
    peaks = [
        index
        for index in range(1, len(bin_ids) - 1)
        if smooth[index] >= smooth[index - 1] and smooth[index] > smooth[index + 1]
    ]
    peaks.sort(key=lambda index: (-smooth[index], int(bin_ids[index])))
    selected: list[int] = []
    for index in peaks:
        if any(abs(index - prior) <= policy.mode_radius_bins for prior in selected):
            continue
        selected.append(index)

    candidates: list[dict[str, Any]] = []
    total_minimum = max(
        policy.minimum_total_points,
        int(math.ceil(policy.minimum_total_fraction * len(all_values))),
    )
    for index in selected:
        left = max(0, index - policy.mode_radius_bins)
        right = min(len(bin_ids), index + policy.mode_radius_bins + 1)
        per_frame = np.sum(stacked[:, left:right], axis=1)
        frame_minima = np.asarray(
            [
                max(
                    policy.minimum_frame_points,
                    int(math.ceil(policy.minimum_frame_fraction * len(values))),
                )
                for values in frame_heights
            ],
            dtype=np.int64,
        )
        persistent = per_frame >= frame_minima
        support_count = int(np.sum(per_frame))
        if int(np.sum(persistent)) < policy.minimum_persistent_frames:
            continue
        if support_count < total_minimum:
            continue
        center = (float(bin_ids[index]) + 0.5) * policy.height_bin_m
        near = np.abs(all_values - center) <= (
            (policy.mode_radius_bins + 0.5) * policy.height_bin_m
        )
        refined = float(np.median(all_values[near]))
        candidates.append(
            {
                "world_height_m": refined,
                "persistent_frame_count": int(np.sum(persistent)),
                "frame_count": len(frame_heights),
                "support_sample_count": support_count,
                "support_fraction": support_count / len(all_values),
                "per_frame_support_sample_count": per_frame.tolist(),
            }
        )
    return sorted(candidates, key=lambda row: float(row["world_height_m"]))


def classify_dominant_plane(
    dominant_world_height_m: float,
    lowest_persistent_world_height_m: float | None,
    policy: SupportIdentityPolicy,
) -> str:
    if lowest_persistent_world_height_m is None:
        return "UNKNOWN_NO_PERSISTENT_HORIZONTAL_MODE"
    delta = dominant_world_height_m - lowest_persistent_world_height_m
    if abs(delta) <= policy.same_surface_tolerance_m:
        return "LOWEST_PERSISTENT_SURFACE_SUPPORTED"
    if delta >= policy.elevated_separation_m:
        return "ELEVATED_DOMINANT_SURFACE_REJECTED"
    return "UNKNOWN_HEIGHT_IDENTITY_AMBIGUOUS"


def _existing_planes(result_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(
        result.get("status")
        == "TUM_GRAVITY_ANCHORED_SUPPORT_BOUNDARY_PSEUDOLABELS_MATERIALIZED",
        "gravity-factor result status invalid",
    )
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for receipt in result["frame_receipts"]:
        if not receipt["support_plane_valid"]:
            continue
        label_path = Path(str(receipt["output_path"]))
        require(label_path.is_file(), f"gravity-factor label missing: {label_path}")
        with np.load(label_path, allow_pickle=False) as arrays:
            camera_world_height = float(arrays["camera_to_world_output"][2, 3])
            camera_height = float(arrays["camera_height_m"])
        output[(str(receipt["parent_id"]), str(receipt["frame_id"]))] = {
            "camera_world_height_m": camera_world_height,
            "camera_height_above_dominant_m": camera_height,
            "dominant_plane_world_height_m": camera_world_height - camera_height,
        }
    return output


def diagnose(
    cohort_path: Path,
    gravity_factor_result_path: Path,
    policy: SupportIdentityPolicy = SupportIdentityPolicy(),
) -> dict[str, Any]:
    require(cohort_path.is_file(), "TUM cohort missing")
    require(gravity_factor_result_path.is_file(), "TUM gravity factors missing")
    payloads: list[TumSelectedPayload] = []
    for role in ("fit", "evaluation"):
        rows, _ = load_tum_role_payloads(cohort_path, role)
        payloads.extend(rows)
    existing = _existing_planes(gravity_factor_result_path)
    parent_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for parent in sorted({payload.parent_id for payload in payloads}):
        parent_payloads = [payload for payload in payloads if payload.parent_id == parent]
        eligible_payloads = [
            payload
            for payload in parent_payloads
            if (parent, f"{parent}__rgb{payload.rgb.row_index:06d}") in existing
        ]
        if not eligible_payloads:
            parent_rows.append(
                {
                    "parent_id": parent,
                    "status": "UNKNOWN_GRAVITY_UNAVAILABLE",
                    "selected_frame_count": len(parent_payloads),
                    "horizontal_modes": [],
                }
            )
            continue
        heights = [_horizontal_world_heights(payload, policy) for payload in eligible_payloads]
        modes = _mode_candidates(heights, policy)
        lowest = float(modes[0]["world_height_m"]) if modes else None
        parent_statuses: list[str] = []
        for payload, values in zip(eligible_payloads, heights, strict=True):
            frame_id = f"{parent}__rgb{payload.rgb.row_index:06d}"
            plane = existing[(parent, frame_id)]
            status = classify_dominant_plane(
                float(plane["dominant_plane_world_height_m"]),
                lowest,
                policy,
            )
            parent_statuses.append(status)
            frame_rows.append(
                {
                    "role": payload.role,
                    "parent_id": parent,
                    "frame_id": frame_id,
                    "source_horizontal_sample_count": len(values),
                    **plane,
                    "lowest_persistent_world_height_m": lowest,
                    "dominant_minus_lowest_m": (
                        float(plane["dominant_plane_world_height_m"]) - lowest
                        if lowest is not None
                        else None
                    ),
                    "status": status,
                }
            )
        status_counts = {
            status: parent_statuses.count(status) for status in sorted(set(parent_statuses))
        }
        if status_counts.get("ELEVATED_DOMINANT_SURFACE_REJECTED", 0) >= 2:
            parent_status = "ELEVATED_DOMINANT_SURFACE_REPLICATED"
        elif status_counts.get("LOWEST_PERSISTENT_SURFACE_SUPPORTED", 0) >= 2:
            parent_status = "LOWEST_PERSISTENT_SURFACE_REPLICATED"
        else:
            parent_status = "UNKNOWN_PARENT_IDENTITY_MIXED"
        parent_rows.append(
            {
                "parent_id": parent,
                "status": parent_status,
                "selected_frame_count": len(parent_payloads),
                "horizontal_modes": modes,
                "lowest_persistent_world_height_m": lowest,
                "frame_status_counts": status_counts,
            }
        )

    evaluable_frames = [row for row in frame_rows if not row["status"].startswith("UNKNOWN")]
    rejected_frames = [
        row for row in frame_rows if row["status"] == "ELEVATED_DOMINANT_SURFACE_REJECTED"
    ]
    supported_frames = [
        row for row in frame_rows if row["status"] == "LOWEST_PERSISTENT_SURFACE_SUPPORTED"
    ]
    elevated_parents = [
        row for row in parent_rows if row["status"] == "ELEVATED_DOMINANT_SURFACE_REPLICATED"
    ]
    supported_parents = [
        row for row in parent_rows if row["status"] == "LOWEST_PERSISTENT_SURFACE_REPLICATED"
    ]
    passes = bool(len(elevated_parents) >= 1 and len(supported_parents) >= 1)
    return {
        "schema": "blindassist_ag_st_tum_support_identity_diagnostic_v1",
        "status": (
            "TUM_DOMINANT_SUPPORT_IDENTITY_FAILURE_DETECTED"
            if passes
            else "TUM_SUPPORT_IDENTITY_CANARY_INCONCLUSIVE"
        ),
        "cohort": str(cohort_path.resolve()),
        "cohort_sha256": sha256_file(cohort_path),
        "gravity_factor_result": str(gravity_factor_result_path.resolve()),
        "gravity_factor_result_sha256": sha256_file(gravity_factor_result_path),
        "policy": {
            key: getattr(policy, key)
            for key in policy.__dataclass_fields__
        },
        "parent_count": len(parent_rows),
        "gravity_evaluable_parent_count": len(parent_rows) - sum(
            row["status"] == "UNKNOWN_GRAVITY_UNAVAILABLE" for row in parent_rows
        ),
        "gravity_evaluable_frame_count": len(frame_rows),
        "evaluable_identity_frame_count": len(evaluable_frames),
        "elevated_dominant_frame_count": len(rejected_frames),
        "lowest_persistent_supported_frame_count": len(supported_frames),
        "elevated_dominant_parent_count": len(elevated_parents),
        "lowest_persistent_supported_parent_count": len(supported_parents),
        "parents": parent_rows,
        "frames": frame_rows,
        "decision": {
            "complete_truth_required": False,
            "current_dominant_plane_can_be_used_as_walkable_support": False,
            "sequence_level_height_identity_required": passes,
            "student_training_authorized": False,
            "next_execution": (
                "Materialize source-depth/pose anchored lowest-persistent-plane factors, "
                "keeping ambiguous and gravity-missing parents UNKNOWN."
                if passes
                else "Add a synthetic-exact support-identity source before factor materialization."
            ),
        },
        "claim_boundary": (
            "Selected-frame TUM source-depth/pose evidence that tests whether the existing "
            "per-frame dominant horizontal plane is elevated. Lowest persistent surface is "
            "a geometry-anchored pseudo identity, not walkability ground truth, task utility, "
            "product, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--gravity-factors", type=Path, default=DEFAULT_GRAVITY_FACTORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"support-identity result already exists: {args.output}")
    result = diagnose(args.cohort, args.gravity_factors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "gravity_evaluable_parent_count",
                    "gravity_evaluable_frame_count",
                    "elevated_dominant_parent_count",
                    "lowest_persistent_supported_parent_count",
                    "elevated_dominant_frame_count",
                    "lowest_persistent_supported_frame_count",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
