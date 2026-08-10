#!/usr/bin/env python3
"""Analytic exact canary for support identity and level-change boundaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

from build_ag_st_factor_labels import (
    PROVENANCE_SOURCE_NATIVE,
    TIER_A_SOURCE,
    compute_geometric_factors,
)
from diagnose_ag_st_tum_support_identity import SupportIdentityPolicy, _mode_candidates
from download_b0_arkitscenes_assets import require


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-analytic-support-boundary-r1/result.json"
)
HEIGHT = 120
WIDTH = 160
INTRINSICS = np.asarray(
    [[140.0, 0.0, 79.5], [0.0, 140.0, 59.5], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
WORLD_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
FLOOR_ID = 1
TABLE_ID = 2


def camera_to_world_look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, WORLD_UP)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack((right, down, forward), axis=1)
    pose[:3, 3] = position
    require(np.linalg.det(pose[:3, :3]) > 0.999, "analytic camera rotation invalid")
    return pose


def render_floor_and_table(camera_to_world: np.ndarray) -> dict[str, np.ndarray]:
    rows, columns = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    rays_camera = np.stack(
        (
            (columns - INTRINSICS[0, 2]) / INTRINSICS[0, 0],
            (rows - INTRINSICS[1, 2]) / INTRINSICS[1, 1],
            np.ones((HEIGHT, WIDTH), dtype=np.float64),
        ),
        axis=-1,
    )
    rotation = camera_to_world[:3, :3]
    origin = camera_to_world[:3, 3]
    rays_world = np.einsum("...j,ij->...i", rays_camera, rotation)
    downward = rays_world[..., 2] < -1e-6
    floor_t = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)
    floor_t[downward] = (0.0 - origin[2]) / rays_world[..., 2][downward]
    table_t = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)
    table_t[downward] = (0.75 - origin[2]) / rays_world[..., 2][downward]
    table_points = origin + rays_world * table_t[..., None]
    table_inside = (
        downward
        & (table_t > 0.0)
        & (np.abs(table_points[..., 0]) <= 1.20)
        & (np.abs(table_points[..., 1]) <= 0.90)
    )
    table_t[~table_inside] = np.inf
    use_table = table_t < floor_t
    depth = np.minimum(table_t, floor_t)
    valid = np.isfinite(depth) & (depth > 0.0)
    surface = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    surface[valid] = FLOOR_ID
    surface[use_table & valid] = TABLE_ID
    depth[~valid] = 0.0
    boundary = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    for first, second in (
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
    ):
        changed = (
            valid[first]
            & valid[second]
            & (surface[first] != surface[second])
        )
        boundary[first] |= changed
        boundary[second] |= changed
    return {
        "depth_m": depth.astype(np.float32),
        "valid": valid,
        "surface_id": surface,
        "boundary": boundary,
    }


def _factors(depth: np.ndarray, valid: np.ndarray, pose: np.ndarray, override: float | None) -> dict[str, Any]:
    return compute_geometric_factors(
        depth,
        valid,
        INTRINSICS,
        pose,
        np.where(valid, 0.99, 0.0).astype(np.float32),
        np.where(valid, TIER_A_SOURCE, 0).astype(np.uint8),
        np.where(valid, PROVENANCE_SOURCE_NATIVE, 0).astype(np.uint8),
        np.where(valid, 0.005, np.inf).astype(np.float32),
        support_camera_height_override_m=override,
        support_plane_residual_override_m=0.01 if override is not None else None,
    )


def _support_metrics(factors: dict[str, Any], surface: np.ndarray) -> dict[str, float]:
    predicted = factors["support_truth_valid_hw"] & (factors["support_truth_hw"] >= 0.5)
    floor = surface == FLOOR_ID
    table = surface == TABLE_ID
    true_positive = int(np.sum(predicted & floor))
    return {
        "predicted_positive_pixels": int(np.sum(predicted)),
        "precision": true_positive / max(1, int(np.sum(predicted))),
        "floor_recall": true_positive / max(1, int(np.sum(floor))),
        "table_false_positive_rate": int(np.sum(predicted & table)) / max(1, int(np.sum(table))),
    }


def _boundary_metrics(factors: dict[str, Any], exact: np.ndarray) -> dict[str, float]:
    predicted = factors["evidence_truth_valid_hw"] & (
        factors["boundary_probability_pseudo_hw"] >= 0.5
    )
    exact_near = maximum_filter(exact.astype(np.uint8), size=5) > 0
    predicted_near = maximum_filter(predicted.astype(np.uint8), size=5) > 0
    return {
        "predicted_seed_pixels": int(np.sum(predicted)),
        "exact_boundary_pixels": int(np.sum(exact)),
        "precision_within_2px": int(np.sum(predicted & exact_near)) / max(1, int(np.sum(predicted))),
        "recall_within_2px": int(np.sum(exact & predicted_near)) / max(1, int(np.sum(exact))),
    }


def run() -> dict[str, Any]:
    positions = (
        np.asarray([-0.35, -1.55, 1.50]),
        np.asarray([0.00, -1.65, 1.55]),
        np.asarray([0.35, -1.55, 1.45]),
    )
    target = np.asarray([0.0, 0.0, 0.68])
    frames: list[dict[str, Any]] = []
    height_samples: list[np.ndarray] = []
    for index, position in enumerate(positions):
        pose = camera_to_world_look_at(position, target)
        rendered = render_floor_and_table(pose)
        default = _factors(rendered["depth_m"], rendered["valid"], pose, None)
        points = np.asarray(rendered["surface_id"], dtype=np.uint8)
        heights = np.concatenate(
            (
                np.zeros(int(np.sum(points == FLOOR_ID)), dtype=np.float64),
                np.full(int(np.sum(points == TABLE_ID)), 0.75, dtype=np.float64),
            )
        )
        height_samples.append(heights)
        frames.append(
            {
                "index": index,
                "pose": pose,
                "rendered": rendered,
                "default": default,
                "default_dominant_world_height_m": (
                    float(position[2]) - float(default["camera_height_m"])
                ),
            }
        )

    identity_policy = SupportIdentityPolicy(
        sample_stride=1,
        minimum_frame_points=64,
        minimum_total_points=192,
    )
    modes = _mode_candidates(height_samples, identity_policy)
    require(modes, "analytic support-height modes missing")
    lowest_world_height = float(modes[0]["world_height_m"])

    frame_receipts: list[dict[str, Any]] = []
    for frame in frames:
        pose = frame["pose"]
        rendered = frame["rendered"]
        camera_height = float(pose[2, 3]) - lowest_world_height
        corrected = _factors(
            rendered["depth_m"],
            rendered["valid"],
            pose,
            camera_height,
        )
        frame_receipts.append(
            {
                "frame_index": frame["index"],
                "camera_world_height_m": float(pose[2, 3]),
                "default_dominant_world_height_m": frame["default_dominant_world_height_m"],
                "lowest_persistent_world_height_m": lowest_world_height,
                "default_support": _support_metrics(frame["default"], rendered["surface_id"]),
                "corrected_support": _support_metrics(corrected, rendered["surface_id"]),
                "corrected_boundary": _boundary_metrics(corrected, rendered["boundary"]),
            }
        )

    default_elevated = sum(
        row["default_dominant_world_height_m"] - lowest_world_height >= 0.25
        for row in frame_receipts
    )
    corrected_precision = min(row["corrected_support"]["precision"] for row in frame_receipts)
    corrected_recall = min(row["corrected_support"]["floor_recall"] for row in frame_receipts)
    default_table_false_positive = min(
        row["default_support"]["table_false_positive_rate"] for row in frame_receipts
    )
    corrected_table_false_positive = max(
        row["corrected_support"]["table_false_positive_rate"] for row in frame_receipts
    )
    boundary_precision = min(
        row["corrected_boundary"]["precision_within_2px"] for row in frame_receipts
    )
    boundary_recall = min(
        row["corrected_boundary"]["recall_within_2px"] for row in frame_receipts
    )
    gates = {
        "lowest_height_error_le_0p04m": abs(lowest_world_height) <= 0.04,
        "default_elevated_all_frames": default_elevated == len(frame_receipts),
        "corrected_support_precision_ge_0p98": corrected_precision >= 0.98,
        "corrected_floor_recall_ge_0p50": corrected_recall >= 0.50,
        "default_table_false_positive_ge_0p50": default_table_false_positive >= 0.50,
        "corrected_table_false_positive_le_0p02": corrected_table_false_positive <= 0.02,
        "boundary_precision_within_2px_ge_0p50": boundary_precision >= 0.50,
        "boundary_recall_within_2px_ge_0p50": boundary_recall >= 0.50,
    }
    passed = all(gates.values())
    return {
        "schema": "blindassist_ag_st_analytic_support_boundary_canary_v1",
        "status": (
            "ANALYTIC_SUPPORT_IDENTITY_AND_BOUNDARY_PASS"
            if passed
            else "ANALYTIC_SUPPORT_IDENTITY_OR_BOUNDARY_FAIL"
        ),
        "scene": {
            "frame_count": len(frame_receipts),
            "resolution_hw": [HEIGHT, WIDTH],
            "floor_world_height_m": 0.0,
            "table_world_height_m": 0.75,
            "table_extent_xy_m": [-1.20, 1.20, -0.90, 0.90],
            "camera_positions_world_xyz_m": [position.tolist() for position in positions],
        },
        "height_modes": modes,
        "lowest_persistent_world_height_m": lowest_world_height,
        "aggregate": {
            "default_elevated_frame_count": default_elevated,
            "corrected_support_min_precision": corrected_precision,
            "corrected_support_min_floor_recall": corrected_recall,
            "default_support_min_table_false_positive_rate": default_table_false_positive,
            "corrected_support_max_table_false_positive_rate": corrected_table_false_positive,
            "corrected_boundary_min_precision_within_2px": boundary_precision,
            "corrected_boundary_min_recall_within_2px": boundary_recall,
        },
        "gates": gates,
        "frames": frame_receipts,
        "decision": {
            "support_identity_mechanics_validated": passed,
            "boundary_direction_mechanics_validated": gates[
                "boundary_precision_within_2px_ge_0p50"
            ]
            and gates["boundary_recall_within_2px_ge_0p50"],
            "student_training_authorized": False,
            "next_execution": (
                "Run the same support/boundary metrics on an external synthetic-exact RGB-D source."
            ),
        },
        "claim_boundary": (
            "Analytic exact mechanics canary for a floor plus elevated tabletop. "
            "It is not external-scene generalization, learned-model evidence, task utility, product, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"analytic canary output exists: {args.output}")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "aggregate", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
