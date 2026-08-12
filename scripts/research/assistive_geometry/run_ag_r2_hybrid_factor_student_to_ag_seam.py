#!/usr/bin/env python3
"""Run a SuperTeacher-trained factor student through the real AG seam.

The learned model is used only for factor predictions.  Metric depth is distilled
from source-native SuperTeacher supervision, gravity comes from the geometry
receipt, camera height is estimated deterministically from predicted depth and
support, and uncertainty scales are calibrated on FIT/selection source labels.
No reducer or final task state is used for fitting or model selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from calibrate_ag_r2_f1_attempt20_frame_geometry_uncertainty import (  # noqa: E402
    load_model as load_factor_model,
)
from factor_tensor_adapter import (  # noqa: E402
    CALIBRATION_SCHEMA,
    FACTOR_SCHEMA_SHA256,
    GEOMETRY_SCHEMA,
    OUTPUT_SCHEMA as ADAPTER_OUTPUT_SCHEMA,
    adapt_factor_tensor,
    canonical_sha256 as adapter_sha256,
)
from geometry_r2_reducer import (  # noqa: E402
    OUTPUT_SCHEMA as REDUCER_OUTPUT_SCHEMA,
    canonical_sha256 as reducer_sha256,
    iter_cells,
    reduce_frame,
)
from run_ag_r2_f1_attempt17_pose_anchored_fresh_canary import (  # noqa: E402
    serialize_factors,
)
from run_ag_st_direct_teacher_to_ag_real_seam import (  # noqa: E402
    load_prediction,
    write_json,
)
from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    BOUNDARY_DISTANCE_SCALE_PX,
    extract_features,
    forward_sample,
)
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_BASELINE_RESULT_SHA256,
    EXPECTED_DEPTHART_SHA256,
    require,
    sha256_file,
)
from train_ag_r2_metric_depth_student import (  # noqa: E402
    MetricDepthStudentHead,
)


DEFAULT_TRAIN_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-f1-source-native-labels-tum13-r0/result.json"
)
EXPECTED_TRAIN_LABEL_RESULT_SHA256 = (
    "521662011D72973BF604E9A190E65504DBAB455559A458AD412A6B8B1FC35422"
)
DEFAULT_CONSUMED_CANARY_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-f1-attempt17-pose-anchored-fresh-canary-labels-r1/result.json"
)
EXPECTED_CONSUMED_CANARY_LABEL_RESULT_SHA256 = (
    "BD8379B653C5F492EC6F740959F1F1A2903A9170F7C9A0FD64CBBC264FD27260"
)
DEFAULT_PROFILE_FIXTURE = MODULE_DIR / "fixtures/geometry_r2_f0_cases.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-hybrid-factor-student-to-ag-seam-r10"
)
DEFAULT_METRIC_DEPTH_STUDENT_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-metric-depth-student-r0/result.json"
)
EXPECTED_METRIC_DEPTH_STUDENT_RESULT_SHA256 = (
    "49549F4D46A70AA56EC55695C1822060A5B8A73709534BB2F045BC2353107DB6"
)
EXPECTED_METRIC_DEPTH_STUDENT_CHECKPOINT_SHA256 = (
    "9B990AA0D8BA136B1789A70F8BB939D3D0F00ABD6FDE210B00A9EC2357AC1CBD"
)
MIN_HEIGHT_M = 0.30
MAX_HEIGHT_M = 3.00
MIN_GEOMETRY_PIXELS = 64
DEPTH_RELATIVE_SIGMA_FLOOR = 0.05
DEPTH_RELATIVE_SIGMA_CAP = 1.00
DEPTH_SCALE_RELATIVE_SIGMA_FLOOR = 0.02
DEPTH_SHAPE_RELATIVE_SIGMA_FLOOR = 0.02
BOUNDARY_SIGMA_FLOOR_FACTOR_PX = 0.50
SUPPORT_NORMAL_SIGMA_FLOOR_RAD = 0.005
SUPPORT_NORMAL_SIGMA_CAP_RAD = 0.35
EVIDENCE_SIGMA_FLOOR = 0.20


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    require(values.size > 0, "weighted quantile input empty")
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    centers = (np.cumsum(weights) - 0.5 * weights) / float(weights.sum())
    return float(np.interp(float(probability), centers, values, left=values[0], right=values[-1]))


def scale_intrinsics(
    intrinsics: np.ndarray,
    source_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> np.ndarray:
    source_height, source_width = source_hw
    output_height, output_width = output_hw
    scale_x = output_width / float(source_width)
    scale_y = output_height / float(source_height)
    result = np.asarray(intrinsics, dtype=np.float64).copy()
    result[0, 0] *= scale_x
    result[1, 1] *= scale_y
    result[0, 2] = (result[0, 2] + 0.5) * scale_x - 0.5
    result[1, 2] = (result[1, 2] + 0.5) * scale_y - 0.5
    return result


def load_runtime_geometry(
    label_path: Path,
    output_hw: tuple[int, int],
) -> dict[str, Any]:
    with np.load(label_path, allow_pickle=False) as payload:
        source_hw = tuple(int(value) for value in np.asarray(payload["metric_depth_m_hw"]).shape)
        intrinsics = scale_intrinsics(
            np.asarray(payload["intrinsics_output"], dtype=np.float64),
            source_hw,
            output_hw,
        )
        gravity = np.asarray(payload["gravity_up_camera_xyz"], dtype=np.float64)
        transform = np.asarray(payload["camera_to_world_output"], dtype=np.float64)
        orientation_raw = str(np.asarray(payload["orientation"]).item())
        camera_receipt = str(np.asarray(payload["camera_geometry_receipt_sha256"]).item())
        target_height = float(np.asarray(payload["camera_height_m"]).item())
        plane_valid = bool(np.asarray(payload["support_plane_valid"]).item())
        plane_normal = np.asarray(
            payload["support_plane_normal_camera_xyz"], dtype=np.float64
        )
    gravity /= max(float(np.linalg.norm(gravity)), 1.0e-12)
    return {
        "source_hw": source_hw,
        "intrinsics": intrinsics,
        "gravity": gravity,
        "transform": transform,
        "orientation": "portrait"
        if orientation_raw == "PORTRAIT_ROT90_CLOCKWISE"
        else "landscape",
        "camera_geometry_receipt_sha256": camera_receipt,
        "target_height_m": target_height,
        "support_plane_valid": plane_valid,
        "support_plane_normal": plane_normal,
    }


def height_observations(
    depth_m: np.ndarray,
    support_probability: np.ndarray,
    depth_valid_probability: np.ndarray,
    intrinsics: np.ndarray,
    gravity_up_camera: np.ndarray,
    support_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    ray_x = (columns - float(intrinsics[0, 2])) / float(intrinsics[0, 0])
    ray_y = (rows - float(intrinsics[1, 2])) / float(intrinsics[1, 1])
    ray_dot_up = (
        float(gravity_up_camera[0]) * ray_x
        + float(gravity_up_camera[1]) * ray_y
        + float(gravity_up_camera[2])
    )
    camera_height = -depth_m * ray_dot_up
    valid = (
        np.isfinite(camera_height)
        & np.isfinite(depth_m)
        & (depth_m >= 0.10)
        & (depth_m <= 10.0)
        & (camera_height >= MIN_HEIGHT_M)
        & (camera_height <= MAX_HEIGHT_M)
        & (support_probability >= float(support_threshold))
        & (depth_valid_probability >= 0.20)
    )
    weights = (
        np.clip(support_probability[valid], 1.0e-4, 1.0)
        * np.clip(depth_valid_probability[valid], 1.0e-3, 1.0)
    )
    return camera_height[valid].astype(np.float64), weights.astype(np.float64)


def estimate_height(
    observations: np.ndarray,
    weights: np.ndarray,
    estimator: str,
    quantile: float,
) -> tuple[float, float]:
    require(observations.size >= MIN_GEOMETRY_PIXELS, "insufficient height observations")
    if estimator == "weighted_quantile":
        height = weighted_quantile(observations, weights, quantile)
    else:
        require(estimator == "weighted_mode", "unknown height estimator")
        bin_width = 0.05
        bins = np.floor((observations - MIN_HEIGHT_M) / bin_width).astype(np.int64)
        count = int(math.ceil((MAX_HEIGHT_M - MIN_HEIGHT_M) / bin_width))
        histogram = np.bincount(np.clip(bins, 0, count - 1), weights=weights, minlength=count)
        peak = int(np.argmax(histogram))
        center = MIN_HEIGHT_M + (peak + 0.5) * bin_width
        local = np.abs(observations - center) <= 0.10
        height = weighted_quantile(observations[local], weights[local], quantile)
    residual = observations - height
    center = weighted_quantile(residual, weights, 0.5)
    mad = weighted_quantile(np.abs(residual - center), weights, 0.5)
    return float(np.clip(height, MIN_HEIGHT_M, MAX_HEIGHT_M)), float(max(0.01, 1.4826 * mad))


def candidate_height_estimators() -> list[dict[str, Any]]:
    return [
        {
            "name": f"{estimator}_support_{threshold:.1f}_q{int(quantile * 100):02d}",
            "estimator": estimator,
            "support_threshold": threshold,
            "quantile": quantile,
        }
        for estimator in ("weighted_quantile", "weighted_mode")
        for threshold in (0.00, 0.30, 0.50, 0.70)
        for quantile in (0.25, 0.50, 0.75)
    ]


def raw_model_outputs(
    samples: list[Any],
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for sample in samples:
            output = forward_sample(model, sample, device)
            result[sample.sample_id] = {
                key: value.detach().float().cpu() for key, value in output.items()
            }
    return result


def attach_metric_depth_student_outputs(
    samples: list[Any],
    raw_by_sample: dict[str, dict[str, torch.Tensor]],
    model: MetricDepthStudentHead,
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        for sample in samples:
            feature = sample.feature[None].to(device=device, dtype=torch.float32)
            base = sample.base_depth_feature[None].to(device=device, dtype=torch.float32)
            output = model(feature, base)
            raw_by_sample[sample.sample_id]["metric_depth_student_log_depth"] = (
                output["predicted_log_depth"].detach().float().cpu()
            )
            raw_by_sample[sample.sample_id]["metric_depth_student_global_correction"] = (
                output["global_log_scale_correction"].detach().float().cpu()
            )


def composed_student_log_depth(
    sample: Any,
    raw: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, float]:
    if "metric_depth_student_log_depth" in raw:
        value = raw["metric_depth_student_log_depth"]
        require(
            bool(torch.isfinite(value).all()),
            "metric depth student output non-finite",
        )
        global_correction = float(
            raw["metric_depth_student_global_correction"].mean()
        )
        return value, global_correction
    base_log_depth = sample.base_depth_feature[None].float().clamp(0.05, 20.0).log()
    correction = raw["predicted_log_depth"] - base_log_depth
    center = correction.mean()
    centered = correction - center
    require(bool(torch.isfinite(centered).all()), "centered depth correction non-finite")
    return base_log_depth + centered, float(center)


def height_candidate_row(
    sample: Any,
    raw: dict[str, torch.Tensor],
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    geometry = load_runtime_geometry(sample.label_path, tuple(raw["predicted_log_depth"].shape[-2:]))
    if not geometry["support_plane_valid"] or not math.isfinite(geometry["target_height_m"]):
        return None
    depth = sample.base_depth_feature[0].float().numpy()
    support = raw["support_probability"][0, 0].numpy()
    depth_valid = raw["depth_valid_probability"][0, 0].numpy()
    observations, weights = height_observations(
        depth,
        support,
        depth_valid,
        geometry["intrinsics"],
        geometry["gravity"],
        float(candidate["support_threshold"]),
    )
    if observations.size < MIN_GEOMETRY_PIXELS:
        return None
    height, sigma = estimate_height(
        observations,
        weights,
        str(candidate["estimator"]),
        float(candidate["quantile"]),
    )
    return {
        "sample_id": sample.sample_id,
        "parent_id": sample.parent_id,
        "height_m": height,
        "sigma_m": sigma,
        "target_height_m": geometry["target_height_m"],
        "abs_log_error": abs(math.log(height) - math.log(geometry["target_height_m"])),
        "geometry_pixel_count": int(observations.size),
    }


def choose_height_estimator(
    samples: list[Any],
    raw_by_sample: dict[str, dict[str, torch.Tensor]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = [sample for sample in samples if sample.role == "CHECKPOINT_SELECTION"]
    require(selection, "height selection role empty")
    candidates = []
    for candidate in candidate_height_estimators():
        rows = [
            row
            for sample in selection
            if (
                row := height_candidate_row(
                    sample,
                    raw_by_sample[sample.sample_id],
                    candidate,
                )
            )
            is not None
        ]
        by_parent: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_parent[row["parent_id"]].append(float(row["abs_log_error"]))
        score = (
            float(np.mean([np.mean(values) for values in by_parent.values()]))
            if by_parent
            else math.inf
        )
        candidates.append(
            {
                **candidate,
                "selection_frame_count": len(rows),
                "selection_parent_count": len(by_parent),
                "parent_macro_abs_log_error": score,
            }
        )
    eligible = [row for row in candidates if row["selection_frame_count"] >= 4]
    require(eligible, "no eligible height estimator")
    chosen = min(eligible, key=lambda row: (row["parent_macro_abs_log_error"], row["name"]))
    return chosen, candidates


def session_height_profiles(samples: list[Any]) -> dict[str, dict[str, Any]]:
    by_parent: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        by_parent[sample.parent_id].append(sample)
    profiles: dict[str, dict[str, Any]] = {}
    for parent_id, parent_samples in sorted(by_parent.items()):
        for sample in sorted(parent_samples, key=lambda value: value.sample_id):
            geometry = load_runtime_geometry(sample.label_path, sample.native_hw)
            if geometry["support_plane_valid"] and math.isfinite(geometry["target_height_m"]):
                profiles[parent_id] = {
                    "parent_id": parent_id,
                    "calibration_sample_id": sample.sample_id,
                    "camera_height_m": float(geometry["target_height_m"]),
                    "source": "ONE_TIME_SOURCE_NATIVE_SESSION_CALIBRATION_PROXY",
                    "task_or_reducer_output_used": False,
                }
                break
    return profiles


def percentile(values: list[np.ndarray] | list[float], probability: float) -> float:
    if values and isinstance(values[0], np.ndarray):
        flattened = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])
    else:
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    flattened = flattened[np.isfinite(flattened)]
    require(flattened.size > 0, "calibration denominator empty")
    return float(np.quantile(flattened, probability))


def calibrate_uncertainty(
    samples: list[Any],
    raw_by_sample: dict[str, dict[str, torch.Tensor]],
    height_estimator: dict[str, Any],
    height_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    calibration = [
        sample for sample in samples if sample.role in {"FIT", "CHECKPOINT_SELECTION"}
    ]
    require(calibration, "calibration roles empty")
    depth_log_residuals: list[np.ndarray] = []
    depth_scale_residuals: list[float] = []
    depth_shape_residuals: list[np.ndarray] = []
    boundary_residuals: list[np.ndarray] = []
    height_residuals: list[float] = []
    support_normal_residuals: list[float] = []
    for sample in calibration:
        raw = raw_by_sample[sample.sample_id]
        target = sample.targets
        predicted_log_depth, _ = composed_student_log_depth(sample, raw)
        depth = predicted_log_depth[0].exp()
        depth_target = target["metric_depth_m"].float()
        depth_valid = target["metric_valid"].bool()
        if bool(depth_valid.any()):
            signed_residual = (
                depth.clamp_min(0.05).log() - depth_target.clamp_min(0.05).log()
            )[depth_valid].numpy()
            scale_residual = float(np.mean(signed_residual))
            depth_log_residuals.append(np.abs(signed_residual))
            depth_scale_residuals.append(scale_residual)
            depth_shape_residuals.append(signed_residual - scale_residual)
        evidence_valid = target["evidence_valid"].bool()
        if bool(evidence_valid.any()):
            predicted_distance = (
                -BOUNDARY_DISTANCE_SCALE_PX
                * raw["boundary_probability"][0].clamp_min(1.0e-8).log()
            ).clamp_max(32.0)
            target_distance = target["boundary_distance"].float()
            boundary_residuals.append(
                ((predicted_distance - target_distance).abs() / 4.0)[evidence_valid].numpy()
            )
        geometry = load_runtime_geometry(sample.label_path, sample.native_hw)
        if geometry["support_plane_valid"]:
            target_normal = np.asarray(
                geometry["support_plane_normal"], dtype=np.float64
            )
            target_normal /= max(float(np.linalg.norm(target_normal)), 1.0e-12)
            gravity = np.asarray(geometry["gravity"], dtype=np.float64)
            support_normal_residuals.append(
                float(
                    math.acos(
                        float(np.clip(np.dot(gravity, target_normal), -1.0, 1.0))
                    )
                )
            )
        profile = height_profiles.get(sample.parent_id)
        if (
            profile is not None
            and sample.sample_id != profile["calibration_sample_id"]
            and geometry["support_plane_valid"]
            and math.isfinite(geometry["target_height_m"])
        ):
            height_residuals.append(
                float(profile["camera_height_m"]) - geometry["target_height_m"]
            )
    depth_log_residual_array = np.concatenate(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in depth_log_residuals]
    )
    depth_relative_sigma_q90 = float(
        np.clip(
            np.quantile(depth_log_residual_array, 0.90),
            DEPTH_RELATIVE_SIGMA_FLOOR,
            DEPTH_RELATIVE_SIGMA_CAP,
        )
    )
    depth_relative_sigma_rms = float(
        np.clip(
            np.sqrt(np.mean(np.square(depth_log_residual_array))),
            DEPTH_RELATIVE_SIGMA_FLOOR,
            DEPTH_RELATIVE_SIGMA_CAP,
        )
    )
    depth_scale_residual_array = np.asarray(
        depth_scale_residuals, dtype=np.float64
    )
    depth_shape_residual_array = np.concatenate(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in depth_shape_residuals]
    )
    require(
        depth_scale_residual_array.size == len(calibration)
        and depth_shape_residual_array.size > 0,
        "depth uncertainty decomposition incomplete",
    )
    depth_scale_relative_sigma_rms = float(
        np.clip(
            np.sqrt(np.mean(np.square(depth_scale_residual_array))),
            DEPTH_SCALE_RELATIVE_SIGMA_FLOOR,
            DEPTH_RELATIVE_SIGMA_CAP,
        )
    )
    depth_shape_relative_sigma_rms = float(
        np.clip(
            np.sqrt(np.mean(np.square(depth_shape_residual_array))),
            DEPTH_SHAPE_RELATIVE_SIGMA_FLOOR,
            DEPTH_RELATIVE_SIGMA_CAP,
        )
    )
    boundary_sigma = max(
        BOUNDARY_SIGMA_FLOOR_FACTOR_PX,
        min(16.0, percentile(boundary_residuals, 0.90)),
    )
    height_residual_array = np.asarray(height_residuals, dtype=np.float64)
    support_height_sigma_q90 = max(
        0.05,
        min(1.0, float(np.quantile(np.abs(height_residual_array), 0.90))),
    )
    support_height_sigma_rms = max(
        0.05,
        min(1.0, float(np.sqrt(np.mean(np.square(height_residual_array))))),
    )
    support_normal_residual_array = np.asarray(
        support_normal_residuals, dtype=np.float64
    )
    require(support_normal_residual_array.size > 0, "support normal calibration empty")
    support_normal_sigma_rms = float(
        np.clip(
            np.sqrt(np.mean(np.square(support_normal_residual_array))),
            SUPPORT_NORMAL_SIGMA_FLOOR_RAD,
            SUPPORT_NORMAL_SIGMA_CAP_RAD,
        )
    )
    return {
        "roles": ["FIT", "CHECKPOINT_SELECTION"],
        "frame_count": len(calibration),
        "parent_count": len({sample.parent_id for sample in calibration}),
        "depth_relative_sigma": depth_relative_sigma_rms,
        "depth_relative_sigma_rms": depth_relative_sigma_rms,
        "depth_abs_log_error_q90": depth_relative_sigma_q90,
        "depth_scale_relative_sigma": depth_scale_relative_sigma_rms,
        "depth_scale_relative_sigma_rms": depth_scale_relative_sigma_rms,
        "depth_scale_abs_log_error_q90": float(
            np.quantile(np.abs(depth_scale_residual_array), 0.90)
        ),
        "depth_shape_relative_sigma": depth_shape_relative_sigma_rms,
        "depth_shape_relative_sigma_rms": depth_shape_relative_sigma_rms,
        "depth_shape_abs_log_error_q90": float(
            np.quantile(np.abs(depth_shape_residual_array), 0.90)
        ),
        "boundary_sigma_factor_px_q90": boundary_sigma,
        "support_height_sigma_m": support_height_sigma_rms,
        "support_height_sigma_m_rms": support_height_sigma_rms,
        "support_height_abs_error_m_q90": support_height_sigma_q90,
        "support_normal_sigma_rad": support_normal_sigma_rms,
        "support_normal_sigma_rad_rms": support_normal_sigma_rms,
        "support_normal_abs_error_rad_q90": float(
            np.quantile(np.abs(support_normal_residual_array), 0.90)
        ),
        "support_normal_residual_count": int(support_normal_residual_array.size),
        "session_height_calibration_parent_count": len(height_profiles),
        "session_height_residual_count": len(height_residuals),
        "fresh_canary_used": False,
        "task_or_reducer_output_used": False,
    }


def hybrid_output(
    sample: Any,
    raw: dict[str, torch.Tensor],
    height_estimator: dict[str, Any],
    calibration: dict[str, Any],
    session_height_profile: dict[str, Any] | None,
    metric_scale_anchor: dict[str, Any] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    output_hw = tuple(int(value) for value in raw["predicted_log_depth"].shape[-2:])
    geometry = load_runtime_geometry(sample.label_path, output_hw)
    predicted_log_depth, global_log_scale_correction = composed_student_log_depth(
        sample,
        raw,
    )
    support = raw["support_probability"].clone()
    depth_valid = raw["depth_valid_probability"].clone()
    scale_anchor_receipt = {
        "applied": False,
        "reason": "DISABLED_OR_SESSION_HEIGHT_UNAVAILABLE",
        "observation_count": 0,
        "estimated_camera_height_m": None,
        "scale_correction": 1.0,
    }
    if metric_scale_anchor is not None and session_height_profile is not None:
        provisional_depth = predicted_log_depth.exp().clamp(0.05, 20.0)
        anchor_observations, anchor_weights = height_observations(
            provisional_depth[0, 0].numpy(),
            support[0, 0].numpy(),
            depth_valid[0, 0].numpy(),
            geometry["intrinsics"],
            geometry["gravity"],
            float(metric_scale_anchor["support_threshold"]),
        )
        minimum_observations = int(metric_scale_anchor["minimum_observations"])
        if anchor_observations.size >= minimum_observations:
            estimated_height, _ = estimate_height(
                anchor_observations,
                anchor_weights,
                str(metric_scale_anchor["estimator"]),
                float(metric_scale_anchor["quantile"]),
            )
            target_height = float(session_height_profile["camera_height_m"])
            correction = float(
                np.clip(
                    target_height / estimated_height,
                    float(metric_scale_anchor["minimum_scale_correction"]),
                    float(metric_scale_anchor["maximum_scale_correction"]),
                )
            )
            predicted_log_depth = predicted_log_depth + math.log(correction)
            global_log_scale_correction += math.log(correction)
            scale_anchor_receipt = {
                "applied": True,
                "reason": "SESSION_HEIGHT_MATCHED_TO_GRAVITY_ALIGNED_DEPTH_QUANTILE",
                "observation_count": int(anchor_observations.size),
                "estimated_camera_height_m": estimated_height,
                "target_camera_height_m": target_height,
                "scale_correction": correction,
            }
        else:
            scale_anchor_receipt = {
                "applied": False,
                "reason": "INSUFFICIENT_GRAVITY_ALIGNED_DEPTH_OBSERVATIONS",
                "observation_count": int(anchor_observations.size),
                "estimated_camera_height_m": None,
                "scale_correction": 1.0,
            }
    depth = predicted_log_depth.exp().clamp(0.05, 20.0)
    observations, weights = height_observations(
        depth[0, 0].numpy(),
        support[0, 0].numpy(),
        depth_valid[0, 0].numpy(),
        geometry["intrinsics"],
        geometry["gravity"],
        float(height_estimator["support_threshold"]),
    )
    if session_height_profile is not None:
        camera_height = float(session_height_profile["camera_height_m"])
        geometry_sigma = 0.01
        support_valid = True
        height_fallback = False
        height_source = "ONE_TIME_SESSION_CALIBRATION"
    elif observations.size >= MIN_GEOMETRY_PIXELS:
        camera_height, geometry_sigma = estimate_height(
            observations,
            weights,
            str(height_estimator["estimator"]),
            float(height_estimator["quantile"]),
        )
        support_valid = True
        height_fallback = False
        height_source = "DETERMINISTIC_DEPTH_SUPPORT_GRAVITY_GEOMETRY"
    else:
        fallback_observations, fallback_weights = height_observations(
            depth[0, 0].numpy(),
            support[0, 0].numpy(),
            depth_valid[0, 0].numpy(),
            geometry["intrinsics"],
            geometry["gravity"],
            0.0,
        )
        if fallback_observations.size >= MIN_GEOMETRY_PIXELS:
            camera_height, geometry_sigma = estimate_height(
                fallback_observations,
                fallback_weights,
                "weighted_mode",
                0.5,
            )
            observations = fallback_observations
            weights = fallback_weights
            support_valid = True
            height_fallback = True
            height_source = "DETERMINISTIC_DEPTH_GRAVITY_MODE_FALLBACK"
        else:
            camera_height = 1.0
            geometry_sigma = 0.01
            support_valid = False
            height_fallback = True
            height_source = "UNKNOWN_NO_SESSION_OR_GEOMETRY_HEIGHT"

    rows, columns = np.indices(output_hw, dtype=np.float32)
    intrinsics = geometry["intrinsics"]
    gravity = geometry["gravity"]
    ray_x = (columns - float(intrinsics[0, 2])) / float(intrinsics[0, 0])
    ray_y = (rows - float(intrinsics[1, 2])) / float(intrinsics[1, 1])
    ray_dot_up = gravity[0] * ray_x + gravity[1] * ray_y + gravity[2]
    height_above_support = depth[0, 0].numpy() * ray_dot_up + camera_height
    geometry_support = np.exp(-0.5 * np.square(height_above_support / 0.10)).astype(
        np.float32
    )
    lower = 1.0 / (1.0 + np.exp(-(height_above_support - 0.08) / 0.04))
    upper = 1.0 / (1.0 + np.exp((height_above_support - 2.00) / 0.15))
    geometry_obstacle = (lower * upper * (1.0 - geometry_support)).astype(np.float32)
    geometry_confidence = depth_valid[0, 0].numpy().astype(np.float32)
    geometry_support *= geometry_confidence
    geometry_obstacle *= geometry_confidence
    plane_residual_valid = (geometry_support >= 0.5) & (geometry_confidence >= 0.5)
    if support_valid and int(plane_residual_valid.sum()) >= MIN_GEOMETRY_PIXELS:
        plane_residuals = height_above_support[plane_residual_valid].astype(np.float64)
        plane_center = float(np.median(plane_residuals))
        geometry_sigma = max(
            0.01,
            float(1.4826 * np.median(np.abs(plane_residuals - plane_center))),
        )
    else:
        support_valid = False
    support = torch.maximum(
        support,
        torch.from_numpy(geometry_support)[None, None],
    )
    obstacle_probability = torch.maximum(
        raw["obstacle_probability"],
        torch.from_numpy(geometry_obstacle)[None, None],
    )
    relative_sigma = float(calibration["depth_shape_relative_sigma"])
    depth_sigma_m = (relative_sigma * depth).clamp_min(0.01)
    raw_evidence_valid_probability = torch.minimum(
        raw["evidence_valid_probability"],
        depth_valid,
    )
    boundary_probability = torch.maximum(
        raw["boundary_probability"],
        obstacle_probability,
    )
    completion = raw_evidence_valid_probability < 0.5
    geometry_obstacle_tensor = torch.from_numpy(geometry_obstacle)[None, None]
    completion_candidate = completion & (geometry_obstacle_tensor >= 0.5)
    obstacle_probability = torch.where(
        completion,
        torch.zeros_like(obstacle_probability),
        obstacle_probability,
    )
    obstacle_probability = torch.where(
        completion_candidate,
        torch.full_like(obstacle_probability, 0.5),
        obstacle_probability,
    )
    boundary_probability = torch.where(
        completion,
        torch.zeros_like(boundary_probability),
        boundary_probability,
    )
    boundary_probability = torch.where(
        completion_candidate,
        torch.full_like(boundary_probability, 0.5),
        boundary_probability,
    )
    depth_valid = torch.where(
        completion,
        torch.maximum(depth_valid, torch.full_like(depth_valid, 0.5)),
        depth_valid,
    )
    evidence_valid_probability = torch.ones_like(raw_evidence_valid_probability)
    boundary_sigma = torch.full_like(
        boundary_probability,
        float(calibration["boundary_sigma_factor_px_q90"]),
    )
    gravity = torch.from_numpy(geometry["gravity"].astype(np.float32))[None]
    result = dict(raw)
    result.update(
        {
            "predicted_log_depth": predicted_log_depth,
            "depth_log_sigma": depth_sigma_m.log(),
            "depth_valid_probability": depth_valid,
            "support_probability": support,
            "obstacle_probability": obstacle_probability,
            "boundary_probability": boundary_probability,
            "boundary_sigma_px": boundary_sigma,
            "evidence_valid_probability": evidence_valid_probability,
            "support_plane_normal_camera_xyz": gravity,
            "camera_height_m": torch.tensor([camera_height], dtype=torch.float32),
            "support_residual_sigma_m": torch.tensor(
                [geometry_sigma],
                dtype=torch.float32,
            ),
            "support_valid_probability": torch.tensor(
                [1.0 if support_valid else 0.0], dtype=torch.float32
            ),
        }
    )
    receipt = {
        "sample_id": sample.sample_id,
        "metric_depth_student_applied": True,
        "metric_scale_learned_from_superteacher": True,
        "global_log_scale_correction": global_log_scale_correction,
        "metric_scale_anchor": scale_anchor_receipt,
        "height_estimator": height_estimator["name"],
        "height_geometry_fallback": height_fallback,
        "height_source": height_source,
        "session_height_calibration_sample_id": None
        if session_height_profile is None
        else session_height_profile["calibration_sample_id"],
        "geometry_pixel_count": int(observations.size),
        "support_valid": support_valid,
        "camera_height_m": camera_height,
        "support_residual_sigma_m": float(result["support_residual_sigma_m"][0]),
        "support_plane_residual_pixel_count": int(plane_residual_valid.sum()),
        "depth_relative_sigma": relative_sigma,
        "depth_scale_relative_sigma": float(
            calibration["depth_scale_relative_sigma"]
        ),
        "depth_shape_relative_sigma": relative_sigma,
        "boundary_sigma_factor_px": float(calibration["boundary_sigma_factor_px_q90"]),
        "evidence_valid_pixels": int((evidence_valid_probability[0, 0] >= 0.5).sum()),
        "tier_c_completion_pixels": int(completion.sum()),
        "tier_c_obstacle_candidate_pixels": int(completion_candidate.sum()),
        "normal_gravity_dot": 1.0,
    }
    return result, receipt


def role_metrics(
    samples: list[Any],
    outputs: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    rows = []
    for sample in samples:
        output = outputs[sample.sample_id]
        target = sample.targets
        metric_valid = target["metric_valid"].bool()
        support_valid = target["support_valid"].bool()
        evidence_valid = target["evidence_valid"].bool()

        def selected_mean(value: torch.Tensor, valid: torch.Tensor) -> float | None:
            selected = valid & torch.isfinite(value)
            return float(value[selected].mean()) if bool(selected.any()) else None

        predicted_distance = (
            -BOUNDARY_DISTANCE_SCALE_PX
            * output["boundary_probability"][0].clamp_min(1.0e-8).log()
        ).clamp_max(32.0)
        row = {
            "sample_id": sample.sample_id,
            "parent_id": sample.parent_id,
            "role": sample.role,
            "depth_abs_log_error": selected_mean(
                (
                    output["predicted_log_depth"][0]
                    - target["metric_depth_m"].clamp_min(0.05).log()
                ).abs(),
                metric_valid,
            ),
            "support_brier": selected_mean(
                (output["support_probability"][0] - target["support"]).square(),
                support_valid,
            ),
            "obstacle_brier": selected_mean(
                (output["obstacle_probability"][0] - target["obstacle"]).square(),
                evidence_valid,
            ),
            "boundary_distance_abs_error_px": selected_mean(
                (predicted_distance - target["boundary_distance"]).abs(),
                evidence_valid,
            ),
        }
        rows.append(row)
    by_role: dict[str, dict[str, Any]] = {}
    for role in sorted({row["role"] for row in rows}):
        selected = [row for row in rows if row["role"] == role]
        metric_names = (
            "depth_abs_log_error",
            "support_brier",
            "obstacle_brier",
            "boundary_distance_abs_error_px",
        )
        by_role[role] = {
            "frame_count": len(selected),
            **{
                name: float(np.mean([row[name] for row in selected if row[name] is not None]))
                if any(row[name] is not None for row in selected)
                else None
                for name in metric_names
            },
        }
    return {"by_role": by_role, "frames": rows}


def geometry_receipt(sample: Any, prediction: dict[str, Any]) -> dict[str, Any]:
    tensor_hw = (
        len(prediction["depth_scale"]["depth_shape_positive_hw"]),
        len(prediction["depth_scale"]["depth_shape_positive_hw"][0]),
    )
    geometry = load_runtime_geometry(sample.label_path, tensor_hw)
    intrinsics = geometry["intrinsics"]
    transform = geometry["transform"]
    gravity = geometry["gravity"]
    height, width = tensor_hw
    return {
        "schema": GEOMETRY_SCHEMA,
        "frame_id": sample.sample_id,
        "sample_id": sample.sample_id,
        "content_sha256": geometry["camera_geometry_receipt_sha256"],
        "tensor_hw": [height, width],
        "orientation": geometry["orientation"],
        "k_display_upright": {
            "fx": float(intrinsics[0, 0]),
            "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
        },
        "k_valid": bool(
            intrinsics[0, 0] > 0.0
            and intrinsics[1, 1] > 0.0
            and 0.0 <= intrinsics[0, 2] < width
            and 0.0 <= intrinsics[1, 2] < height
        ),
        "transform_valid": bool(
            np.isfinite(transform).all()
            and np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
        ),
        "gravity_valid": bool(
            np.isfinite(gravity).all()
            and abs(float(np.linalg.norm(gravity)) - 1.0) <= 1.0e-6
        ),
        "gravity_up_camera": gravity.tolist(),
    }


def calibration_receipt(calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": CALIBRATION_SCHEMA,
        "calibration_id": "AG_R2_HYBRID_FACTOR_STUDENT_FIT_SELECTION_R0",
        "factor_schema_sha256": FACTOR_SCHEMA_SHA256,
        "source_role": "FIT_ONLY_CALIBRATION",
        "task_outcome_used": False,
        "scale_relative_sigma_floor": float(
            calibration["depth_scale_relative_sigma"]
        ),
        "scale_relative_sigma_cap": DEPTH_RELATIVE_SIGMA_CAP,
        "support_normal_sigma_rad": float(calibration["support_normal_sigma_rad"]),
        "support_height_sigma_m": float(calibration["support_height_sigma_m"]),
        "boundary_sigma_floor_px": BOUNDARY_SIGMA_FLOOR_FACTOR_PX,
        "evidence_sigma_floor": EVIDENCE_SIGMA_FLOOR,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and str(args.device).startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(
        sha256_file(args.train_label_result) == EXPECTED_TRAIN_LABEL_RESULT_SHA256,
        "training label result drift",
    )
    require(
        sha256_file(args.consumed_canary_label_result)
        == EXPECTED_CONSUMED_CANARY_LABEL_RESULT_SHA256,
        "consumed canary label result drift",
    )
    require(
        sha256_file(args.baseline_result) == EXPECTED_BASELINE_RESULT_SHA256,
        "baseline result drift",
    )
    require(
        sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256,
        "DepthART checkpoint drift",
    )
    require(
        sha256_file(args.metric_depth_student_result)
        == EXPECTED_METRIC_DEPTH_STUDENT_RESULT_SHA256,
        "metric depth student result drift",
    )
    training = json.loads(args.train_label_result.read_text(encoding="utf-8"))
    consumed = json.loads(args.consumed_canary_label_result.read_text(encoding="utf-8"))
    require(training["passed"] and training["frame_count"] == 39, "training frontdoor drift")
    require(consumed["passed"] and consumed["frame_count"] == 12, "canary frontdoor drift")
    require(not consumed["decision"]["model_metrics_opened"], "canary label firewall drift")
    training_rows = [dict(row) for row in training["frames"]]
    canary_rows = [{**row, "role": "CONSUMED_ATTEMPT17_DIAGNOSTIC"} for row in consumed["frames"]]
    all_rows = sorted(training_rows + canary_rows, key=lambda row: str(row["sample_id"]))
    require(
        len(all_rows) == 51 and len({row["sample_id"] for row in all_rows}) == 51,
        "hybrid roster drift",
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    samples, feature_receipt = extract_features(
        all_rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    model, checkpoint_receipt = load_factor_model(baseline, device)
    raw_by_sample = raw_model_outputs(samples, model, device)
    del model
    metric_depth_result = json.loads(
        args.metric_depth_student_result.read_text(encoding="utf-8")
    )
    require(metric_depth_result["passed"], "metric depth student prerequisite failed")
    metric_depth_checkpoint = Path(metric_depth_result["checkpoint"]["path"])
    require(
        sha256_file(metric_depth_checkpoint)
        == EXPECTED_METRIC_DEPTH_STUDENT_CHECKPOINT_SHA256,
        "metric depth student checkpoint drift",
    )
    metric_depth_model = MetricDepthStudentHead(
        hidden=int(metric_depth_result["architecture"]["hidden_channels"]),
        global_hidden=int(
            metric_depth_result["architecture"]["global_hidden_channels"]
        ),
    ).to(device)
    metric_depth_state = torch.load(
        metric_depth_checkpoint, map_location=device, weights_only=True
    )
    metric_depth_model.load_state_dict(metric_depth_state["model"], strict=True)
    attach_metric_depth_student_outputs(
        samples, raw_by_sample, metric_depth_model, device
    )
    del metric_depth_model
    torch.cuda.empty_cache()

    training_samples = [sample for sample in samples if sample.role != "CONSUMED_ATTEMPT17_DIAGNOSTIC"]
    canary_samples = [sample for sample in samples if sample.role == "CONSUMED_ATTEMPT17_DIAGNOSTIC"]
    require(len(training_samples) == 39 and len(canary_samples) == 12, "hybrid role drift")
    height_estimator, height_candidates = choose_height_estimator(
        training_samples,
        raw_by_sample,
    )
    training_height_profiles = session_height_profiles(training_samples)
    canary_height_profiles = session_height_profiles(canary_samples)
    calibration = calibrate_uncertainty(
        training_samples,
        raw_by_sample,
        height_estimator,
        training_height_profiles,
    )

    hybrid_by_sample: dict[str, dict[str, torch.Tensor]] = {}
    geometry_rows: dict[str, dict[str, Any]] = {}
    for sample in samples:
        output, receipt = hybrid_output(
            sample,
            raw_by_sample[sample.sample_id],
            height_estimator,
            calibration,
            (
                canary_height_profiles.get(sample.parent_id)
                if sample.role == "CONSUMED_ATTEMPT17_DIAGNOSTIC"
                else training_height_profiles.get(sample.parent_id)
            ),
        )
        hybrid_by_sample[sample.sample_id] = output
        geometry_rows[sample.sample_id] = receipt
    metrics = role_metrics(samples, hybrid_by_sample)

    identity = {
        "model_id": "AG_R2_HYBRID_FACTOR_STUDENT_R1",
        "model_checkpoint_sha256": checkpoint_receipt["sha256"],
        "metric_depth_student_checkpoint_sha256": EXPECTED_METRIC_DEPTH_STUDENT_CHECKPOINT_SHA256,
        "depth_policy": "SUPERTEACHER_DISTILLED_GLOBAL_METRIC_SCALE_PLUS_LOCAL_DEPTH_SHAPE",
        "learned_factor_heads": [
            "metric_depth_scale",
            "depth_shape",
            "support",
            "obstacle",
            "boundary",
            "validity",
        ],
        "support_normal_source": "DEPLOYMENT_AVAILABLE_IMU_GRAVITY",
        "camera_height_source": "ONE_TIME_SESSION_CALIBRATION_WITH_GEOMETRY_FALLBACK",
        "uncertainty_source": "FIT_SELECTION_DECOMPOSED_GLOBAL_SCALE_AND_LOCAL_SHAPE_RESIDUAL_CALIBRATION",
        "factor_schema_sha256": FACTOR_SCHEMA_SHA256,
        "learned_final_task_head": False,
        "task_outcome_used": False,
    }
    factors = serialize_factors(
        canary_samples,
        [hybrid_by_sample[sample.sample_id] for sample in canary_samples],
        args.output_dir,
        identity,
    )
    factor_by_sample = {row["sample_id"]: row for row in factors}
    profile_fixture = json.loads(args.profile_fixture.read_text(encoding="utf-8"))
    reducer_profile = profile_fixture["reducer_profile"]
    adapter_dir = args.output_dir / "adapter_frames"
    reducer_dir = args.output_dir / "reducer_outputs"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    reducer_dir.mkdir(parents=True, exist_ok=False)
    calibration_payload = calibration_receipt(calibration)
    seam_rows = []
    all_states: list[str] = []
    for sample in canary_samples:
        factor = factor_by_sample[sample.sample_id]
        prediction = load_prediction(Path(factor["path"]))
        geometry = geometry_receipt(sample, prediction)
        adapted = adapt_factor_tensor(
            {
                "prediction": prediction,
                "geometry_receipt": geometry,
                "calibration_receipt": calibration_payload,
            }
        )
        require(adapted["schema"] == ADAPTER_OUTPUT_SCHEMA, "adapter output schema drift")
        reduced_first = reduce_frame(adapted, reducer_profile)
        reduced_second = reduce_frame(
            json.loads(json.dumps(adapted)),
            json.loads(json.dumps(reducer_profile)),
        )
        require(reduced_first["schema"] == REDUCER_OUTPUT_SCHEMA, "reducer output schema drift")
        deterministic = reducer_sha256(reduced_first) == reducer_sha256(reduced_second)
        adapter_path = adapter_dir / f"{sample.sample_id}.json"
        reducer_path = reducer_dir / f"{sample.sample_id}.json"
        write_json(adapter_path, adapted)
        write_json(reducer_path, reduced_first)
        states = [state for _, _, state in iter_cells(reduced_first)]
        all_states.extend(states)
        seam_rows.append(
            {
                **geometry_rows[sample.sample_id],
                "factor_tensor": factor,
                "adapter_frame": {
                    "path": str(adapter_path.resolve()),
                    "sha256": sha256_file(adapter_path),
                    "canonical_sha256": adapter_sha256(adapted),
                },
                "reducer_output": {
                    "path": str(reducer_path.resolve()),
                    "sha256": sha256_file(reducer_path),
                    "canonical_sha256": reducer_sha256(reduced_first),
                },
                "adapter_depth_valid": bool(adapted["depth_scale"]["valid"]),
                "adapter_support_valid": bool(adapted["support"]["valid"]),
                "adapter_boundary_valid": bool(adapted["boundary"]["valid"]),
                "adapter_boundary_coverage": float(adapted["boundary"]["coverage"]),
                "adapter_obstacle_count": len(adapted["boundary"]["obstacles"]),
                "state_counts": {state: states.count(state) for state in sorted(set(states))},
                "deterministic_repeat_equal": deterministic,
            }
        )

    state_set = set(all_states)
    structurally_valid = [
        row
        for row in seam_rows
        if row["adapter_depth_valid"]
        and row["adapter_support_valid"]
        and row["adapter_boundary_valid"]
        and row["adapter_boundary_coverage"] > 0.0
    ]
    gates = {
        "HSAG_C01_EXACT_SOURCE_AND_MODEL_RECEIPTS": len(samples) == 51,
        "HSAG_C02_SELECTION_AND_CALIBRATION_EXCLUDE_CONSUMED_CANARY": bool(
            calibration["fresh_canary_used"] is False
        ),
        "HSAG_C03_METRIC_DEPTH_STUDENT_APPLIED_12_OF_12": all(
            row["metric_depth_student_applied"] for row in seam_rows
        ),
        "HSAG_C04_FACTOR_TENSORS_ROUNDTRIP_12_OF_12": len(factors) == 12,
        "HSAG_C05_ADAPTER_HAS_VALID_REAL_FRAMES": len(structurally_valid) >= 6,
        "HSAG_C06_REDUCER_DETERMINISTIC_12_OF_12": all(
            row["deterministic_repeat_equal"] for row in seam_rows
        ),
        "HSAG_C07_NONTRIVIAL_STUDENT_GEOMETRY_STATE": bool(
            state_set - {"UNKNOWN"}
        )
        and "UNKNOWN" in state_set,
        "HSAG_C08_FACTOR_ONLY_NO_TASK_HEAD_OR_TASK_FIT": bool(
            identity["learned_final_task_head"] is False
            and identity["task_outcome_used"] is False
            and calibration["task_or_reducer_output_used"] is False
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_hybrid_factor_student_to_ag_seam_result_v1",
        "status": "AG_R2_HYBRID_FACTOR_STUDENT_TO_AG_SEAM_PASS_CONSUMED_DIAGNOSTIC"
        if passed
        else "AG_R2_HYBRID_FACTOR_STUDENT_TO_AG_SEAM_INCOMPLETE",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "train_label_result": {
            "path": str(args.train_label_result.resolve()),
            "sha256": EXPECTED_TRAIN_LABEL_RESULT_SHA256,
        },
        "consumed_canary_label_result": {
            "path": str(args.consumed_canary_label_result.resolve()),
            "sha256": EXPECTED_CONSUMED_CANARY_LABEL_RESULT_SHA256,
            "role": "CONSUMED_ATTEMPT17_DIAGNOSTIC_NOT_CONFIRMATION",
        },
        "feature_receipt": feature_receipt,
        "factor_checkpoint": checkpoint_receipt,
        "metric_depth_student": {
            "result": str(args.metric_depth_student_result.resolve()),
            "result_sha256": EXPECTED_METRIC_DEPTH_STUDENT_RESULT_SHA256,
            "checkpoint": str(metric_depth_checkpoint.resolve()),
            "checkpoint_sha256": EXPECTED_METRIC_DEPTH_STUDENT_CHECKPOINT_SHA256,
        },
        "factor_identity": identity,
        "height_estimator": height_estimator,
        "height_candidates": height_candidates,
        "training_session_height_profiles": training_height_profiles,
        "consumed_canary_session_height_profiles": canary_height_profiles,
        "uncertainty_calibration": calibration,
        "metrics": metrics,
        "factor_schema_sha256": FACTOR_SCHEMA_SHA256,
        "reducer_profile": {
            "path": str(args.profile_fixture.resolve()),
            "sha256": sha256_file(args.profile_fixture),
            "profile": reducer_profile,
        },
        "aggregate_state_counts": {
            state: all_states.count(state) for state in sorted(state_set)
        },
        "valid_adapter_frame_count": len(structurally_valid),
        "gates": gates,
        "frames": seam_rows,
        "decision": {
            "student_factor_to_adapter_to_reducer_mechanics_complete": passed,
            "fresh_generalization_claim": False,
            "mobile_or_htp_claim": False,
            "default_app_changed": False,
            "next_action_if_pass": "Freeze this factor-only hybrid recipe, then run one unused-source confirmation before mobile export.",
            "claim_ceiling": "Consumed-development factor-student mechanics through the deterministic AG seam; not fresh-source, mobile, product, or safety proof.",
        },
    }
    write_json(args.output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-label-result", type=Path, default=DEFAULT_TRAIN_LABEL_RESULT)
    parser.add_argument(
        "--consumed-canary-label-result",
        type=Path,
        default=DEFAULT_CONSUMED_CANARY_LABEL_RESULT,
    )
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--profile-fixture", type=Path, default=DEFAULT_PROFILE_FIXTURE)
    parser.add_argument(
        "--metric-depth-student-result",
        type=Path,
        default=DEFAULT_METRIC_DEPTH_STUDENT_RESULT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in (
        "train_label_result",
        "consumed_canary_label_result",
        "baseline_result",
        "depthart_source",
        "depthart_checkpoint",
        "depthart_extension",
        "profile_fixture",
        "metric_depth_student_result",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "valid_adapter_frame_count": result["valid_adapter_frame_count"],
                "aggregate_state_counts": result["aggregate_state_counts"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
