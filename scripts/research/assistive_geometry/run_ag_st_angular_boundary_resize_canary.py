#!/usr/bin/env python3
"""No-training resize invariant for camera-angular boundary distance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from download_b0_arkitscenes_assets import require
from materialize_ag_st_continuous_boundary_factors import continuous_boundary_factors


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-angular-boundary-resize-canary-r0/result.json"
)
ANGULAR_SOFT_SIGMA_RAD = 0.012
MAX_ANGULAR_DISTANCE_RAD = 0.25


def camera_angular_boundary_factors(
    probability: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    *,
    sigma_rad: float = ANGULAR_SOFT_SIGMA_RAD,
    max_angle_rad: float = MAX_ANGULAR_DISTANCE_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(probability, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.bool_)
    k = np.asarray(intrinsics, dtype=np.float64)
    require(
        score.shape == mask.shape
        and k.shape == (3, 3)
        and float(k[0, 0]) > 0.0
        and float(k[1, 1]) > 0.0
        and sigma_rad > 0.0
        and max_angle_rad > 0.0,
        "angular boundary input invalid",
    )
    core = mask & (score >= 0.5)
    if not np.any(core):
        angle = np.full(score.shape, max_angle_rad, dtype=np.float32)
    else:
        _, nearest = distance_transform_edt(~core, return_indices=True)
        y, x = np.indices(score.shape, dtype=np.float64)
        nearest_y = nearest[0].astype(np.float64)
        nearest_x = nearest[1].astype(np.float64)

        def rays(pixel_x: np.ndarray, pixel_y: np.ndarray) -> np.ndarray:
            value = np.stack(
                (
                    (pixel_x - k[0, 2]) / k[0, 0],
                    (pixel_y - k[1, 2]) / k[1, 1],
                    np.ones(pixel_x.shape, dtype=np.float64),
                ),
                axis=-1,
            )
            return value / np.linalg.norm(value, axis=-1, keepdims=True)

        first = rays(x, y)
        second = rays(nearest_x, nearest_y)
        cosine = np.clip(np.sum(first * second, axis=-1), -1.0, 1.0)
        angle = np.minimum(np.arccos(cosine), max_angle_rad).astype(np.float32)
        angle[core] = 0.0
    heat = np.exp(-0.5 * np.square(angle / sigma_rad)).astype(np.float32)
    soft = np.maximum(score, heat).astype(np.float32)
    angle[~mask] = np.nan
    soft[~mask] = 0.0
    return angle, np.clip(soft, 0.0, 1.0)


def scaled_intrinsics(
    intrinsics: np.ndarray,
    scale_x: int,
    scale_y: int,
) -> np.ndarray:
    output = np.asarray(intrinsics, dtype=np.float64).copy()
    output[0, 0] *= scale_x
    output[0, 2] *= scale_x
    output[1, 1] *= scale_y
    output[1, 2] *= scale_y
    return output


def line_case(
    orientation: str,
    *,
    scale_x: int,
    scale_y: int,
) -> dict[str, Any]:
    require(orientation in {"vertical", "horizontal"}, "line orientation invalid")
    height, width = 120, 160
    high_height, high_width = height * scale_y, width * scale_x
    intrinsics = np.asarray(
        [[140.0, 0.0, 79.5], [0.0, 130.0, 59.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    probability = np.zeros((height, width), dtype=np.float32)
    high_probability = np.zeros((high_height, high_width), dtype=np.float32)
    if orientation == "vertical":
        probability[:, 80] = 1.0
        high_probability[:, 80 * scale_x] = 1.0
        expected_pixel_scale = float(scale_x)
    else:
        probability[60, :] = 1.0
        high_probability[60 * scale_y, :] = 1.0
        expected_pixel_scale = float(scale_y)
    valid = np.ones(probability.shape, dtype=np.bool_)
    high_valid = np.ones(high_probability.shape, dtype=np.bool_)
    angle, soft = camera_angular_boundary_factors(probability, valid, intrinsics)
    high_angle, high_soft = camera_angular_boundary_factors(
        high_probability,
        high_valid,
        scaled_intrinsics(intrinsics, scale_x, scale_y),
    )
    pixel_distance, _ = continuous_boundary_factors(probability, valid)
    high_pixel_distance, _ = continuous_boundary_factors(
        high_probability,
        high_valid,
        max_distance_px=96.0,
    )
    sampled_angle = high_angle[::scale_y, ::scale_x]
    sampled_soft = high_soft[::scale_y, ::scale_x]
    sampled_pixel = high_pixel_distance[::scale_y, ::scale_x]
    finite_pixel = pixel_distance < 31.9
    return {
        "orientation": orientation,
        "base_shape_hw": [height, width],
        "scaled_shape_hw": [high_height, high_width],
        "scale_xy": [scale_x, scale_y],
        "max_angular_distance_abs_error_rad": float(
            np.max(np.abs(sampled_angle - angle))
        ),
        "max_angular_soft_abs_error": float(np.max(np.abs(sampled_soft - soft))),
        "pixel_distance_scale_ratio_median": float(
            np.median(sampled_pixel[finite_pixel] / np.maximum(pixel_distance[finite_pixel], 1.0))
        ),
        "expected_pixel_distance_scale": expected_pixel_scale,
        "raw_pixel_distance_invariant": bool(
            np.allclose(sampled_pixel[finite_pixel], pixel_distance[finite_pixel], atol=1e-6)
        ),
    }


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"angular boundary canary output exists: {output}")
    vertical = line_case("vertical", scale_x=2, scale_y=3)
    horizontal = line_case("horizontal", scale_x=2, scale_y=3)
    probability = np.zeros((9, 11), dtype=np.float32)
    probability[4, 5] = 1.0
    valid = np.ones(probability.shape, dtype=np.bool_)
    valid[:, 0] = False
    intrinsics = np.asarray(
        [[10.0, 0.0, 5.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    angle, soft = camera_angular_boundary_factors(probability, valid, intrinsics)
    gates = {
        "vertical_angular_distance_resize_invariant": vertical[
            "max_angular_distance_abs_error_rad"
        ]
        <= 1e-6,
        "horizontal_angular_distance_resize_invariant": horizontal[
            "max_angular_distance_abs_error_rad"
        ]
        <= 1e-6,
        "vertical_angular_soft_resize_invariant": vertical["max_angular_soft_abs_error"]
        <= 1e-6,
        "horizontal_angular_soft_resize_invariant": horizontal[
            "max_angular_soft_abs_error"
        ]
        <= 1e-6,
        "vertical_raw_pixel_distance_not_invariant": not vertical[
            "raw_pixel_distance_invariant"
        ],
        "horizontal_raw_pixel_distance_not_invariant": not horizontal[
            "raw_pixel_distance_invariant"
        ],
        "core_angle_is_zero": float(angle[4, 5]) == 0.0 and float(soft[4, 5]) == 1.0,
        "unknown_remains_nan_and_zero": bool(
            np.isnan(angle[:, 0]).all() and np.all(soft[:, 0] == 0.0)
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_angular_boundary_resize_canary_v1",
        "status": (
            "ANGULAR_BOUNDARY_RESIZE_INVARIANT_PASS"
            if passed
            else "ANGULAR_BOUNDARY_RESIZE_INVARIANT_FAIL"
        ),
        "question": "Can camera-ray angular distance preserve boundary supervision across raster scaling where raw pixel distance cannot?",
        "complete_truth_required": False,
        "training_performed": False,
        "contract": {
            "distance": "acos of normalized camera-ray dot product between a pixel and its nearest valid source boundary core",
            "soft_probability": f"max(source probability, exp(-angle^2/(2*{ANGULAR_SOFT_SIGMA_RAD}^2)))",
            "max_angle_rad": MAX_ANGULAR_DISTANCE_RAD,
            "unknown": "NaN angular distance plus zero soft probability outside factor validity",
        },
        "cases": [vertical, horizontal],
        "gates": gates,
        "decision": {
            "materialize_angular_boundary_candidate": passed,
            "retire_existing_pixel_distance": False,
            "retrain_student": False,
            "next_execution": "Add angular distance/soft probability as parallel R9-derived fields over all 81 existing source-native/exact frames, with exact validity/tier/provenance preservation.",
        },
        "claim_boundary": "Analytic pinhole resize mechanics only; no learned model, empirical generalization, task utility, formal F1, safety, deployment, or product claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": result["cases"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
