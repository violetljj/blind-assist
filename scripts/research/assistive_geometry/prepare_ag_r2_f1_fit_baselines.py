#!/usr/bin/env python3
"""Freeze FIT-only nonlearned baselines and loss normalizers for AG R2 F1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-f1-source-native-labels-tum13-r0/result.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-f1-fit-baselines-tum13-r0/result.json"
)
EXPECTED_LABEL_RESULT_SHA256 = "521662011D72973BF604E9A190E65504DBAB455559A458AD412A6B8B1FC35422"
BOUNDARY_DISTANCE_SCALE_PX = 3.0
CHARBONNIER_EPSILON = 0.01
HUBER_DELTA = 0.10
MIN_SIGMA = 1.0e-3
NORMALIZATION_FLOOR = 1.0e-4
GAUSSIAN_CONSTANT = 0.5 * math.log(2.0 * math.pi)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def huber(values: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute <= delta, 0.5 * np.square(values) / delta, absolute - 0.5 * delta)


def gaussian_nll(residual: np.ndarray, sigma: float) -> np.ndarray:
    bounded = max(float(sigma), MIN_SIGMA)
    return 0.5 * np.square(residual / bounded) + math.log(bounded) + GAUSSIAN_CONSTANT


def safe_scale(value: float) -> float:
    require(math.isfinite(value), "non-finite baseline loss")
    return max(float(value), NORMALIZATION_FLOOR)


def run(label_result_path: Path, output_path: Path) -> dict[str, Any]:
    require(label_result_path.is_file(), f"label result missing: {label_result_path}")
    require(sha256_file(label_result_path) == EXPECTED_LABEL_RESULT_SHA256, "label result SHA drift")
    result = json.loads(label_result_path.read_text(encoding="utf-8"))
    require(result["passed"] is True and result["frame_count"] == 39, "frontdoor result invalid")
    fit_rows = [row for row in result["frames"] if row["role"] == "FIT"]
    require(len(fit_rows) == 27, "FIT frame count drift")

    frames: list[dict[str, Any]] = []
    depth_logs: list[np.ndarray] = []
    depth_shapes: list[np.ndarray] = []
    frame_log_scales: list[float] = []
    support_values: list[np.ndarray] = []
    support_residuals: list[np.ndarray] = []
    support_normals: list[np.ndarray] = []
    support_log_heights: list[float] = []
    obstacle_values: list[np.ndarray] = []
    boundary_distances: list[np.ndarray] = []
    total_pixels = 0
    depth_valid_pixels = 0
    evidence_valid_pixels = 0
    support_plane_valid_frames = 0

    for row in fit_rows:
        path = Path(row["output"]).resolve()
        require(path.is_file(), f"FIT payload missing: {path}")
        require(sha256_file(path) == row["output_sha256"], f"FIT payload SHA drift: {path.name}")
        with np.load(path, allow_pickle=False) as payload:
            depth = np.asarray(payload["metric_depth_m_hw"], dtype=np.float64)
            depth_valid = np.asarray(payload["metric_depth_valid_hw"], dtype=np.bool_)
            support_valid = np.asarray(payload["support_truth_valid_hw"], dtype=np.bool_)
            evidence_valid = np.asarray(payload["evidence_truth_valid_hw"], dtype=np.bool_)
            total_pixels += int(depth.size)
            depth_valid_pixels += int(depth_valid.sum())
            evidence_valid_pixels += int(evidence_valid.sum())
            valid_log_depth = np.log(depth[depth_valid])
            require(valid_log_depth.size > 0, "FIT frame has no metric depth")
            frame_scale = float(valid_log_depth.mean())
            depth_logs.append(valid_log_depth)
            depth_shapes.append(valid_log_depth - frame_scale)
            frame_log_scales.append(frame_scale)
            plane_valid = bool(np.asarray(payload["support_plane_valid"]).item())
            if plane_valid:
                support_plane_valid_frames += 1
                require(bool(support_valid.any()) and bool(evidence_valid.any()), "joint FIT frame denominator missing")
                support_values.append(np.asarray(payload["support_truth_hw"], dtype=np.float64)[support_valid])
                support_residuals.append(
                    np.asarray(payload["support_signed_plane_residual_m_hw"], dtype=np.float64)[support_valid]
                )
                normal = np.asarray(payload["support_plane_normal_camera_xyz"], dtype=np.float64)
                require(normal.shape == (3,) and np.isfinite(normal).all(), "support normal invalid")
                support_normals.append(normal / np.linalg.norm(normal))
                height = float(np.asarray(payload["camera_height_m"]).item())
                require(math.isfinite(height) and height > 0.0, "camera height invalid")
                support_log_heights.append(math.log(height))
                obstacle_values.append(
                    np.asarray(payload["obstacle_evidence_truth_hw"], dtype=np.float64)[evidence_valid]
                )
                boundary_distances.append(
                    np.asarray(payload["boundary_distance_px_hw"], dtype=np.float64)[evidence_valid]
                )
        frames.append(
            {
                "sample_id": row["sample_id"],
                "parent_id": row["parent_id"],
                "output_sha256": row["output_sha256"],
            }
        )

    expected_support_valid_frames = sum(bool(row["support_plane_valid"]) for row in fit_rows)
    require(
        support_plane_valid_frames == expected_support_valid_frames
        and support_plane_valid_frames > 0,
        "FIT support-valid frame count drift",
    )
    all_depth_logs = np.concatenate(depth_logs)
    all_support = np.concatenate(support_values)
    all_support_residuals = np.concatenate(support_residuals)
    all_obstacle = np.concatenate(obstacle_values)
    all_boundary_distance = np.concatenate(boundary_distances)
    boundary_target = np.exp(-all_boundary_distance / BOUNDARY_DISTANCE_SCALE_PX)

    baseline_log_scale = float(np.mean(frame_log_scales))
    depth_residual = all_depth_logs - baseline_log_scale
    depth_sigma = float(np.sqrt(np.mean(np.square(depth_residual))))
    support_prior = float(np.mean(all_support))
    support_residual_sigma = float(np.sqrt(np.mean(np.square(all_support_residuals))))
    normal_sum = np.sum(np.stack(support_normals), axis=0)
    support_normal = normal_sum / np.linalg.norm(normal_sum)
    camera_log_height = float(np.mean(support_log_heights))
    obstacle_prior = float(np.mean(all_obstacle))
    boundary_prior = float(np.mean(boundary_target))
    boundary_prediction_distance = min(
        -BOUNDARY_DISTANCE_SCALE_PX * math.log(max(boundary_prior, 1.0e-8)),
        32.0,
    )
    boundary_residual = all_boundary_distance - boundary_prediction_distance
    boundary_sigma = float(np.sqrt(np.mean(np.square(boundary_residual))))
    depth_valid_prior = depth_valid_pixels / total_pixels
    support_valid_prior = support_plane_valid_frames / len(fit_rows)
    evidence_valid_prior = evidence_valid_pixels / total_pixels

    normal_angles = np.arccos(
        np.clip(np.stack(support_normals) @ support_normal, -1.0, 1.0)
    )
    losses = {
        "depth_shape_log_charbonnier": float(
            np.mean(
                np.concatenate(
                    [np.sqrt(np.square(values) + CHARBONNIER_EPSILON**2) - CHARBONNIER_EPSILON for values in depth_shapes]
                )
            )
        ),
        "metric_scale_log_huber": float(huber(np.asarray(frame_log_scales) - baseline_log_scale).mean()),
        "depth_heteroscedastic_nll": float(gaussian_nll(depth_residual, depth_sigma).mean()),
        "depth_validity_brier": float(
            (depth_valid_pixels * (1.0 - depth_valid_prior) ** 2 + (total_pixels - depth_valid_pixels) * depth_valid_prior**2)
            / total_pixels
        ),
        "support_probability_brier": float(np.mean(np.square(all_support - support_prior))),
        "support_plane_angular": float(np.mean(normal_angles)),
        "camera_height_log_huber": float(
            huber(np.asarray(support_log_heights) - camera_log_height).mean()
        ),
        "support_residual_heteroscedastic_nll": float(
            gaussian_nll(all_support_residuals, support_residual_sigma).mean()
        ),
        "support_validity_brier": float(
            (support_plane_valid_frames * (1.0 - support_valid_prior) ** 2 + (len(fit_rows) - support_plane_valid_frames) * support_valid_prior**2)
            / len(fit_rows)
        ),
        "obstacle_evidence_brier": float(np.mean(np.square(all_obstacle - obstacle_prior))),
        "boundary_probability_brier": float(np.mean(np.square(boundary_target - boundary_prior))),
        "boundary_localization_heteroscedastic_nll": float(
            gaussian_nll(boundary_residual, boundary_sigma).mean()
        ),
        "evidence_validity_brier": float(
            (evidence_valid_pixels * (1.0 - evidence_valid_prior) ** 2 + (total_pixels - evidence_valid_pixels) * evidence_valid_prior**2)
            / total_pixels
        ),
    }
    result_out = {
        "schema": "blindassist_assistive_geometry_r2_f1_fit_only_nonlearned_baselines_v1",
        "status": "F1_FIT_ONLY_BASELINES_AND_NORMALIZATION_FROZEN_BEFORE_MODEL_INITIALIZATION",
        "label_result": str(label_result_path.resolve()),
        "label_result_sha256": EXPECTED_LABEL_RESULT_SHA256,
        "fit_frame_count": len(fit_rows),
        "fit_parent_count": len({row["parent_id"] for row in fit_rows}),
        "fit_support_valid_frame_count": support_plane_valid_frames,
        "denominators": {
            "total_pixels": total_pixels,
            "depth_valid_pixels": depth_valid_pixels,
            "support_valid_pixels": int(all_support.size),
            "evidence_valid_pixels": int(all_obstacle.size),
        },
        "transforms": {
            "boundary_probability_target": "exp(-boundary_distance_px / 3.0)",
            "boundary_distance_from_probability": "min(-3.0 * log(max(probability, 1e-8)), 32.0)",
            "charbonnier_epsilon": CHARBONNIER_EPSILON,
            "huber_delta": HUBER_DELTA,
            "gaussian_nll": "0.5*(residual/sigma)^2 + log(sigma) + 0.5*log(2*pi)",
            "minimum_sigma": MIN_SIGMA,
        },
        "baseline_parameters": {
            "depth_log_scale": baseline_log_scale,
            "depth_shape_constant": 1.0,
            "depth_log_sigma": math.log(max(depth_sigma, MIN_SIGMA)),
            "depth_valid_probability": depth_valid_prior,
            "support_probability": support_prior,
            "support_plane_normal_camera_xyz": support_normal.tolist(),
            "camera_height_m": math.exp(camera_log_height),
            "support_residual_sigma_m": max(support_residual_sigma, MIN_SIGMA),
            "support_valid_probability": support_valid_prior,
            "obstacle_evidence_probability": obstacle_prior,
            "boundary_probability": boundary_prior,
            "boundary_localization_sigma_px": max(boundary_sigma, MIN_SIGMA),
            "evidence_valid_probability": evidence_valid_prior,
        },
        "baseline_losses": losses,
        "optimizer_normalization": {key: safe_scale(value) for key, value in losses.items()},
        "frames": sorted(frames, key=lambda row: row["sample_id"]),
        "claim_boundary": "FIT-only deterministic statistics; no model, optimizer, selection, canary, reducer or task outcome was read.",
    }
    require(not output_path.exists(), f"baseline result exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result_out, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-result", type=Path, default=DEFAULT_LABEL_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.label_result.resolve(), args.output.resolve())
    print(json.dumps({key: result[key] for key in ("status", "fit_frame_count", "fit_parent_count", "baseline_parameters", "optimizer_normalization")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
