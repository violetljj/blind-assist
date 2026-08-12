#!/usr/bin/env python3
"""Attempt 03: preserve learned factors and close support geometry with K + IMU gravity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_factor_learnability import (  # noqa: E402
    PRIMARY_METRICS,
    aggregate_parent_metrics,
    baseline_native_outputs,
    bootstrap_lower,
    extract_features,
    frame_metrics,
    gaussian_nll_tensor,
    load_native_targets,
    masked_mean,
    native_outputs,
    require,
    sha256_file,
)
from train_ag_r2_f1_factor_learnability_attempt02 import (  # noqa: E402
    FactorSplitHead,
    forward_sample,
)

DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT03_GRAVITY_AND_SUPPORT_GEOMETRY_EXECUTION_LOCK_R1_2026-08-11.json"
DEFAULT_FRESH_LABEL_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt03-fresh-held-labels-r1/result.json"
DEFAULT_ATTEMPT02_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-factor-learnability-attempt02-r0/result.json"
DEFAULT_BASELINE_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-fit-baselines-tum13-r0/result.json"
DEFAULT_DEPTHART_SOURCE = Path("F:/ba-data/blindassist-artifacts-20260805/models/depthart/source")
DEFAULT_DEPTHART_CHECKPOINT = DEFAULT_DEPTHART_SOURCE / "checkpoints/metric/depthart_metric_indoor_s_448.pth"
DEFAULT_DEPTHART_EXTENSION = Path("E:/codex-tools/tools/venvs/blindassist-venv-export312/Lib/site-packages/depthart_selective_scan_cuda.cp311-win_amd64.pyd")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-factor-learnability-attempt03-r1"

EXPECTED_ATTEMPT02_RESULT_SHA256 = "5D6194D8FC1BC947994A0191A7421D3286F49966C889171AF87B2C0C505E9B8A"
EXPECTED_BASELINE_RESULT_SHA256 = "EECD5C9244C6A8A467B7890AF79D7871374AC8B325A87B48AE9E052089908F44"
EXPECTED_DEPTHART_SHA256 = "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65"
EXPECTED_COMPOSITES = {
    17: "861D141FBB37B02C25FB22B1721F2D034430812DB1D7743B36024D39F30C8EA2",
    29: "E2D875BE57A120E9CF2189D19DE4D1E940AB4826039D28E1FC102648D114E9A4",
    43: "BB248B41396C06F63743B739ED56FB0A40C2C8789049B01667CA042008A9A435",
}
MIN_HEIGHT_M = 0.30
MAX_HEIGHT_M = 3.00
MIN_GEOMETRY_PIXELS = 128


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, quantile: float) -> torch.Tensor:
    require(values.ndim == weights.ndim == 1 and values.numel() == weights.numel(), "weighted quantile shape drift")
    require(values.numel() > 0 and 0.0 <= quantile <= 1.0, "weighted quantile denominator drift")
    order = torch.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order].clamp_min(0.0)
    total = ordered_weights.sum()
    require(bool(torch.isfinite(total)) and float(total) > 0.0, "weighted quantile weights invalid")
    cumulative = torch.cumsum(ordered_weights, dim=0)
    target = total * float(quantile)
    index = torch.searchsorted(cumulative, target).clamp_max(values.numel() - 1)
    return ordered_values[index]


def geometry_inputs(sample: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, bool]:
    with np.load(sample.label_path, allow_pickle=False) as payload:
        intrinsics = torch.from_numpy(np.asarray(payload["intrinsics_output"], dtype=np.float32)).to(device)
        gravity = torch.from_numpy(np.asarray(payload["gravity_up_camera_xyz"], dtype=np.float32)).to(device)
    norm = torch.linalg.vector_norm(gravity)
    valid = bool(torch.isfinite(norm)) and float(norm) > 0.5
    if valid:
        gravity = gravity / norm
    return intrinsics, gravity, valid


def height_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for depth_source in ("predicted", "base_depthart"):
        for support_threshold in (0.05, 0.15, 0.30, 0.50):
            for support_power in (1.0, 2.0):
                for scope, estimators in (
                    ("frame_camera_plane", (("weighted_quantile", (0.35, 0.50, 0.65)), ("weighted_mode", (0.50,)))),
                    ("parent_vio_world_plane", (("weighted_quantile", (0.05, 0.10, 0.20, 0.35, 0.50)), ("weighted_mode", (0.50,)))),
                ):
                    for estimator, quantiles in estimators:
                        for quantile in quantiles:
                            row = {
                                "scope": scope,
                                "depth_source": depth_source,
                                "support_threshold": support_threshold,
                                "support_power": support_power,
                                "estimator": estimator,
                                "quantile": quantile,
                                "mode_bin_m": 0.04,
                                "mode_radius_m": 0.12,
                                "metric_scale_calibration": "none",
                            }
                            rows.append(row)
                            if scope == "parent_vio_world_plane" and depth_source == "predicted":
                                rows.append({**row, "metric_scale_calibration": "vio_frame_quantile_regression"})
                            if scope == "parent_vio_world_plane" and depth_source == "base_depthart":
                                rows.append({**row, "metric_scale_calibration": "align_model_to_vio_height"})
    return rows


def sigma_candidates() -> list[dict[str, Any]]:
    rows = [{"source": "attempt02_learned", "multiplier": 1.0}]
    for source, multipliers in (
        ("geometry_mad", (0.5, 1.0, 2.0, 4.0)),
        ("geometry_rms", (0.5, 1.0, 2.0)),
        ("propagated_depth_vertical", (0.5, 1.0, 2.0, 4.0)),
        ("geometry_plus_depth", (0.5, 1.0, 2.0, 4.0)),
        ("coverage_inverse_sqrt", (0.25, 0.5, 0.75, 1.0)),
        ("coverage_complement", (0.5, 1.0, 1.5, 2.0)),
    ):
        rows.extend({"source": source, "multiplier": value} for value in multipliers)
    return rows


def geometry_height_and_sigma(
    outputs: dict[str, torch.Tensor],
    sample: Any,
    height_config: dict[str, Any],
    sigma_config: dict[str, Any],
    baseline: dict[str, Any],
    device: torch.device,
    height_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    intrinsics, gravity, gravity_valid = geometry_inputs(sample, device)
    if not gravity_valid:
        return (
            outputs["support_plane_normal_camera_xyz"],
            outputs["camera_height_m"],
            outputs["support_residual_sigma_m"],
            {"gravity_valid": False, "geometry_pixel_count": 0, "fallback": "attempt02"},
        )
    height, width = sample.native_hw
    if height_config["depth_source"] == "predicted":
        depth = outputs["predicted_log_depth"].exp()[0, 0]
    else:
        depth = F.interpolate(
            sample.base_depth_feature[None].to(device=device, dtype=torch.float32),
            (height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    support = outputs["support_probability"][0, 0]
    depth_valid = outputs["depth_valid_probability"][0, 0]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ray_x = (xx - intrinsics[0, 2]) / intrinsics[0, 0]
    ray_y = (yy - intrinsics[1, 2]) / intrinsics[1, 1]
    ray_dot_up = gravity[0] * ray_x + gravity[1] * ray_y + gravity[2]
    per_pixel_height = -depth * ray_dot_up
    valid = (
        torch.isfinite(per_pixel_height)
        & torch.isfinite(depth)
        & (depth >= 0.10)
        & (depth <= 10.0)
        & (per_pixel_height >= MIN_HEIGHT_M)
        & (per_pixel_height <= MAX_HEIGHT_M)
        & (support >= float(height_config["support_threshold"]))
        & (depth_valid >= 0.20)
    )
    count = int(valid.sum())
    if count < MIN_GEOMETRY_PIXELS:
        return (
            gravity[None],
            outputs["camera_height_m"],
            outputs["support_residual_sigma_m"],
            {"gravity_valid": True, "geometry_pixel_count": count, "fallback": "attempt02_height"},
        )
    values = per_pixel_height[valid]
    weights = support[valid].clamp_min(1.0e-4).pow(float(height_config["support_power"])) * depth_valid[valid].clamp_min(1.0e-3)
    if height_override is not None:
        camera_height = height_override.reshape(()).to(device=device, dtype=torch.float32)
    elif height_config["estimator"] == "weighted_quantile":
        camera_height = weighted_quantile(values, weights, float(height_config["quantile"]))
    else:
        bin_width = float(height_config["mode_bin_m"])
        bins = torch.floor((values - MIN_HEIGHT_M) / bin_width).long().clamp_min(0)
        bin_count = int(math.ceil((MAX_HEIGHT_M - MIN_HEIGHT_M) / bin_width))
        histogram = torch.bincount(bins.clamp_max(bin_count - 1), weights=weights, minlength=bin_count)
        peak = int(torch.argmax(histogram))
        center = MIN_HEIGHT_M + (peak + 0.5) * bin_width
        local = (values - center).abs() <= float(height_config["mode_radius_m"])
        camera_height = weighted_quantile(values[local], weights[local], float(height_config["quantile"]))
    camera_height = camera_height.clamp(MIN_HEIGHT_M, MAX_HEIGHT_M)
    signed_residual = camera_height - per_pixel_height[valid]
    if sigma_config["source"] == "attempt02_learned":
        sigma = outputs["support_residual_sigma_m"][0]
    elif sigma_config["source"] == "geometry_mad":
        center = weighted_quantile(signed_residual, weights, 0.5)
        mad = weighted_quantile((signed_residual - center).abs(), weights, 0.5)
        sigma = 1.4826 * mad * float(sigma_config["multiplier"])
    else:
        geometry_rms = torch.sqrt((weights * signed_residual.square()).sum() / weights.sum().clamp_min(1.0e-8))
        depth_sigma_log = outputs["depth_log_sigma"].exp()[0, 0][valid]
        vertical_depth_sigma = depth[valid] * depth_sigma_log * ray_dot_up[valid].abs()
        propagated = torch.sqrt((weights * vertical_depth_sigma.square()).sum() / weights.sum().clamp_min(1.0e-8))
        coverage = torch.tensor(count / float(height * width), device=device)
        if sigma_config["source"] == "geometry_rms":
            sigma = geometry_rms
        elif sigma_config["source"] == "propagated_depth_vertical":
            sigma = propagated
        elif sigma_config["source"] == "geometry_plus_depth":
            sigma = torch.sqrt(geometry_rms.square() + propagated.square())
        elif sigma_config["source"] == "coverage_inverse_sqrt":
            sigma = float(sigma_config["multiplier"]) / torch.sqrt(coverage.clamp_min(1.0e-3))
        else:
            sigma = 0.30 + float(sigma_config["multiplier"]) * (1.0 - coverage)
        if sigma_config["source"] not in {"coverage_inverse_sqrt", "coverage_complement"}:
            sigma = sigma * float(sigma_config["multiplier"])
    sigma = sigma.clamp(0.03, 2.0)
    return gravity[None], camera_height[None], sigma[None], {
        "gravity_valid": True,
        "geometry_pixel_count": count,
        "fallback": None,
        "raw_height_m": float(camera_height),
        "support_sigma_m": float(sigma),
        "height_override": height_override is not None,
    }


def parent_vio_height_context(
    prepared: list[dict[str, Any]],
    cached_outputs: list[dict[str, torch.Tensor]],
    height_config: dict[str, Any],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    if height_config.get("scope") != "parent_vio_world_plane":
        return {}
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, torch.Tensor]]]] = defaultdict(list)
    for row, outputs in zip(prepared, cached_outputs):
        grouped[row["sample"].parent_id].append((row, outputs))
    context: dict[str, dict[str, Any]] = {}
    for parent, members in grouped.items():
        world_up_rows = []
        poses = []
        for row, _ in members:
            with np.load(row["sample"].label_path, allow_pickle=False) as payload:
                pose = torch.from_numpy(np.asarray(payload["camera_to_world_output"], dtype=np.float32)).to(device)
                gravity = torch.from_numpy(np.asarray(payload["gravity_up_camera_xyz"], dtype=np.float32)).to(device)
            gravity = F.normalize(gravity, dim=0, eps=1.0e-6)
            world_up_rows.append(pose[:3, :3] @ gravity)
            poses.append(pose)
        world_up = F.normalize(torch.stack(world_up_rows).mean(dim=0), dim=0, eps=1.0e-6)
        floor_values = []
        floor_weights = []
        camera_world_heights = []
        frame_height_estimates = []
        model_height_estimates = []
        frame_counts = []
        parent_invalid = False
        for (row, outputs), pose in zip(members, poses):
            sample = row["sample"]
            height, width = sample.native_hw
            with np.load(sample.label_path, allow_pickle=False) as payload:
                intrinsics = torch.from_numpy(np.asarray(payload["intrinsics_output"], dtype=np.float32)).to(device)
            if height_config["depth_source"] == "predicted":
                depth = outputs["predicted_log_depth"].exp()[0, 0]
            else:
                depth = F.interpolate(sample.base_depth_feature[None].to(device=device, dtype=torch.float32), (height, width), mode="bilinear", align_corners=False)[0, 0]
            support = outputs["support_probability"][0, 0]
            depth_valid = outputs["depth_valid_probability"][0, 0]
            yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32), torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
            ray_x = (xx - intrinsics[0, 2]) / intrinsics[0, 0]
            ray_y = (yy - intrinsics[1, 2]) / intrinsics[1, 1]
            up_camera = pose[:3, :3].transpose(0, 1) @ world_up
            per_pixel_height = -depth * (up_camera[0] * ray_x + up_camera[1] * ray_y + up_camera[2])
            camera_world_height = torch.dot(pose[:3, 3], world_up)
            valid = (
                torch.isfinite(per_pixel_height)
                & torch.isfinite(depth)
                & (depth >= 0.10)
                & (depth <= 10.0)
                & (per_pixel_height >= MIN_HEIGHT_M)
                & (per_pixel_height <= MAX_HEIGHT_M)
                & (support >= float(height_config["support_threshold"]))
                & (depth_valid >= 0.20)
            )
            valid_count = int(valid.sum())
            if valid_count < MIN_GEOMETRY_PIXELS:
                parent_invalid = True
                break
            weights = support[valid].clamp_min(1.0e-4).pow(float(height_config["support_power"])) * depth_valid[valid].clamp_min(1.0e-3)
            frame_values = per_pixel_height[valid]
            floor_values.append(camera_world_height - frame_values)
            floor_weights.append(weights)
            camera_world_heights.append(camera_world_height)
            frame_counts.append(valid_count)
            if height_config["estimator"] == "weighted_quantile":
                frame_height = weighted_quantile(frame_values, weights, float(height_config["quantile"]))
            else:
                bin_width = float(height_config["mode_bin_m"])
                bins = torch.floor((frame_values - MIN_HEIGHT_M) / bin_width).long().clamp_min(0)
                bin_count = int(math.ceil((MAX_HEIGHT_M - MIN_HEIGHT_M) / bin_width))
                histogram = torch.bincount(bins.clamp_max(bin_count - 1), weights=weights, minlength=bin_count)
                peak = int(torch.argmax(histogram))
                center = MIN_HEIGHT_M + (peak + 0.5) * bin_width
                local = (frame_values - center).abs() <= float(height_config["mode_radius_m"])
                frame_height = weighted_quantile(frame_values[local], weights[local], 0.5)
            frame_height_estimates.append(frame_height)
            if height_config.get("metric_scale_calibration") == "align_model_to_vio_height":
                model_depth = outputs["predicted_log_depth"].exp()[0, 0]
                model_height_map = -model_depth * (up_camera[0] * ray_x + up_camera[1] * ray_y + up_camera[2])
                model_valid = (
                    torch.isfinite(model_height_map)
                    & torch.isfinite(model_depth)
                    & (model_depth >= 0.10)
                    & (model_depth <= 10.0)
                    & (model_height_map >= MIN_HEIGHT_M)
                    & (model_height_map <= MAX_HEIGHT_M)
                    & (support >= float(height_config["support_threshold"]))
                    & (depth_valid >= 0.20)
                )
                if int(model_valid.sum()) < MIN_GEOMETRY_PIXELS:
                    parent_invalid = True
                    break
                model_weights = support[model_valid].clamp_min(1.0e-4).pow(float(height_config["support_power"])) * depth_valid[model_valid].clamp_min(1.0e-3)
                model_values = model_height_map[model_valid]
                if height_config["estimator"] == "weighted_quantile":
                    model_height = weighted_quantile(model_values, model_weights, float(height_config["quantile"]))
                else:
                    bin_width = float(height_config["mode_bin_m"])
                    bins = torch.floor((model_values - MIN_HEIGHT_M) / bin_width).long().clamp_min(0)
                    bin_count = int(math.ceil((MAX_HEIGHT_M - MIN_HEIGHT_M) / bin_width))
                    histogram = torch.bincount(bins.clamp_max(bin_count - 1), weights=model_weights, minlength=bin_count)
                    peak = int(torch.argmax(histogram))
                    center = MIN_HEIGHT_M + (peak + 0.5) * bin_width
                    local = (model_values - center).abs() <= float(height_config["mode_radius_m"])
                    model_height = weighted_quantile(model_values[local], model_weights[local], 0.5)
                model_height_estimates.append(model_height)
        if parent_invalid:
            continue
        values = torch.cat(floor_values)
        weights = torch.cat(floor_weights)
        if values.numel() < MIN_GEOMETRY_PIXELS * len(members):
            continue
        depth_scale = torch.tensor(1.0, device=device)
        if height_config.get("metric_scale_calibration") == "vio_frame_quantile_regression":
            frame_heights = torch.stack(frame_height_estimates)
            camera_heights = torch.stack(camera_world_heights)
            slopes = []
            for left in range(len(members)):
                for right in range(left + 1, len(members)):
                    denominator = frame_heights[right] - frame_heights[left]
                    if abs(float(denominator)) > 0.03:
                        slope = (camera_heights[right] - camera_heights[left]) / denominator
                        if math.isfinite(float(slope)) and 0.25 <= float(slope) <= 4.0:
                            slopes.append(slope)
            if slopes:
                depth_scale = torch.median(torch.stack(slopes)).clamp(0.5, 2.0)
            floor_world_height = torch.median(camera_heights - depth_scale * frame_heights)
        elif height_config["estimator"] == "weighted_quantile":
            floor_world_height = weighted_quantile(values, weights, float(height_config["quantile"]))
        else:
            bin_width = float(height_config["mode_bin_m"])
            origin = torch.floor(values.min() / bin_width) * bin_width
            bins = torch.floor((values - origin) / bin_width).long().clamp_min(0)
            histogram = torch.bincount(bins, weights=weights)
            peak = int(torch.argmax(histogram))
            center = origin + (peak + 0.5) * bin_width
            local = (values - center).abs() <= float(height_config["mode_radius_m"])
            floor_world_height = weighted_quantile(values[local], weights[local], 0.5)
        if height_config.get("metric_scale_calibration") == "align_model_to_vio_height":
            vio_heights = torch.stack(camera_world_heights) - floor_world_height
            model_heights = torch.stack(model_height_estimates)
            ratios = vio_heights / model_heights.clamp_min(0.05)
            finite_ratios = ratios[torch.isfinite(ratios) & (ratios >= 0.25) & (ratios <= 4.0)]
            if finite_ratios.numel():
                depth_scale = torch.median(finite_ratios).clamp(0.5, 2.0)
        for (row, _), camera_world_height, count in zip(members, camera_world_heights, frame_counts):
            estimated = (camera_world_height - floor_world_height).clamp(MIN_HEIGHT_M, MAX_HEIGHT_M)
            context[row["sample"].sample_id] = {
                "height": estimated,
                "parent_id": parent,
                "floor_world_height": float(floor_world_height),
                "world_up_xyz": [float(value) for value in world_up],
                "frame_geometry_pixel_count": count,
                "depth_scale": float(depth_scale),
            }
    return context


def apply_geometry(
    outputs: dict[str, torch.Tensor],
    sample: Any,
    config: dict[str, Any],
    baseline: dict[str, Any],
    device: torch.device,
    parent_context: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    context_row = (parent_context or {}).get(sample.sample_id)
    working_outputs = dict(outputs)
    if context_row is not None and float(context_row.get("depth_scale", 1.0)) != 1.0:
        working_outputs["predicted_log_depth"] = outputs["predicted_log_depth"] + math.log(float(context_row["depth_scale"]))
    learned_height_override = (
        outputs["camera_height_m"][0]
        if config["height"].get("source") == "learned_global"
        else None
    )
    normal, height, sigma, receipt = geometry_height_and_sigma(
        working_outputs,
        sample,
        config["height"],
        config["support_sigma"],
        baseline,
        device,
        learned_height_override
        if learned_height_override is not None
        else (None if context_row is None else context_row["height"]),
    )
    receipt["height_source"] = (
        "learned_global" if learned_height_override is not None else "geometry"
    )
    if context_row is not None:
        receipt["parent_vio_context"] = {key: value for key, value in context_row.items() if key != "height"}
    result = working_outputs
    result["support_plane_normal_camera_xyz"] = normal
    result["camera_height_m"] = height
    result["support_residual_sigma_m"] = sigma
    return result, receipt


def prepare(samples: list[Any], device: torch.device) -> list[dict[str, Any]]:
    return [{"sample": sample, "target": load_native_targets(sample, device)} for sample in samples]


def cache_model_outputs(model: FactorSplitHead, prepared: list[dict[str, Any]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    rows = []
    model.eval()
    with torch.no_grad():
        for row in prepared:
            sample = row["sample"]
            rows.append(native_outputs(forward_sample(model, sample, device), sample.native_hw))
    return rows


def quantile_residual_summary(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    sigma = np.concatenate([row[0].reshape(-1) for row in pairs]).astype(np.float64, copy=False)
    residual = np.concatenate([row[1].reshape(-1) for row in pairs]).astype(np.float64, copy=False)
    finite = np.isfinite(sigma) & np.isfinite(residual)
    sigma, residual = sigma[finite], residual[finite]
    require(sigma.size >= 4, "pixel uncertainty denominator missing")
    order = np.argsort(sigma, kind="stable")
    groups = np.array_split(order, 4)
    means = [float(residual[group].mean()) for group in groups]
    return {
        "observation_count": int(sigma.size),
        "sigma_quantile_means": [float(sigma[group].mean()) for group in groups],
        "quantile_residual_means": means,
        "nondecreasing": all(a <= b + 1.0e-6 for a, b in zip(means, means[1:])),
    }


def evaluate_cached(
    prepared: list[dict[str, Any]],
    cached_outputs: list[dict[str, torch.Tensor]] | None,
    baseline: dict[str, Any],
    config: dict[str, Any] | None,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    depth_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    boundary_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    support_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    receipts = []
    parent_context = (
        parent_vio_height_context(prepared, cached_outputs, config["height"], device)
        if cached_outputs is not None and config is not None
        else {}
    )
    with torch.no_grad():
        for index, row in enumerate(prepared):
            sample, target = row["sample"], row["target"]
            if cached_outputs is None:
                outputs = baseline_native_outputs(baseline, sample.native_hw, device)
                receipt = {"baseline": True}
            else:
                require(config is not None, "model geometry config missing")
                outputs, receipt = apply_geometry(cached_outputs[index], sample, config, baseline, device, parent_context)
            metrics = frame_metrics(outputs, target)
            rows.append({"sample_id": sample.sample_id, "parent_id": sample.parent_id, "orientation": sample.orientation, "metrics": metrics})
            receipts.append({"sample_id": sample.sample_id, **receipt})
            depth_valid = target["depth_valid"]
            depth_sigma = outputs["depth_log_sigma"].exp()
            depth_residual = (outputs["predicted_log_depth"] - target["depth"].clamp_min(0.01).log()).abs()
            depth_pairs.append((depth_sigma[depth_valid].cpu().numpy(), depth_residual[depth_valid].cpu().numpy()))
            evidence_valid = target["evidence_valid"]
            boundary_distance = (-3.0 * outputs["boundary_probability"].clamp_min(1.0e-8).log()).clamp_max(32.0)
            boundary_pairs.append((outputs["boundary_sigma_px"][evidence_valid].cpu().numpy(), (boundary_distance - target["boundary_distance"])[evidence_valid].abs().cpu().numpy()))
            if bool(target["plane_valid"]):
                support_residual = target["support_residual"][target["support_valid"]].abs()
                support_pairs.append((np.asarray([float(outputs["support_residual_sigma_m"][0])]), np.asarray([float(support_residual.mean())])))
    aggregate = aggregate_parent_metrics(rows)
    uncertainty_observations = {
        "depth": quantile_residual_summary(depth_pairs),
        "boundary": quantile_residual_summary(boundary_pairs),
        "support": quantile_residual_summary(support_pairs),
    }
    return {"frames": rows, **aggregate, "uncertainty_observations": uncertainty_observations, "geometry_receipts": receipts}


def gate(evaluation: dict[str, Any], baseline_eval: dict[str, Any], seed: int) -> dict[str, Any]:
    improvements: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        values = [
            float(base[metric]) - float(evaluation["parent_metrics"][parent][metric])
            for parent, base in baseline_eval["parent_metrics"].items()
            if metric in base and metric in evaluation["parent_metrics"].get(parent, {})
        ]
        lower = bootstrap_lower(values, seed + sum(ord(char) for char in metric))
        favorable = float(np.mean(np.asarray(values) > 0.0))
        improvements[metric] = {
            "parent_improvements": values,
            "bootstrap_95_lower": lower,
            "favorable_parent_fraction": favorable,
            "passed": lower > 0.0 and favorable >= 0.75,
        }
    uncertainty = {}
    for family, nll_metric in (("depth", "depth_nll"), ("support", "support_nll"), ("boundary", "boundary_nll")):
        proper = float(baseline_eval["overall_metrics"][nll_metric] - evaluation["overall_metrics"][nll_metric])
        uncertainty[family] = {"proper_score_gain": proper, **evaluation["uncertainty_observations"][family]}
        uncertainty[family]["passed"] = proper > 0.0 and uncertainty[family]["nondecreasing"]
    return {
        "metric_improvements": improvements,
        "uncertainty": uncertainty,
        "all_primary_metrics_passed": all(value["passed"] for value in improvements.values()),
        "all_uncertainty_families_passed": all(value["passed"] for value in uncertainty.values()),
    }


def choose_geometry_config(
    prepared: list[dict[str, Any]],
    cached_outputs: list[dict[str, torch.Tensor]],
    baseline: dict[str, Any],
    baseline_eval: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    height_rows = []
    learned_sigma = {"source": "attempt02_learned", "multiplier": 1.0}
    for height_config in height_candidates():
        errors: dict[str, list[float]] = defaultdict(list)
        depth_scale_errors: dict[str, list[float]] = defaultdict(list)
        fallbacks = 0
        parent_context = parent_vio_height_context(prepared, cached_outputs, height_config, device)
        with torch.no_grad():
            for index, row in enumerate(prepared):
                outputs, receipt = apply_geometry(
                    cached_outputs[index],
                    row["sample"],
                    {"height": height_config, "support_sigma": learned_sigma},
                    baseline,
                    device,
                    parent_context,
                )
                height = outputs["camera_height_m"]
                if receipt.get("fallback"):
                    fallbacks += 1
                if bool(row["target"]["plane_valid"]):
                    error = abs(float(torch.log(height[0])) - float(torch.log(row["target"]["height"])))
                    errors[row["sample"].parent_id].append(error)
                valid_depth = row["target"]["depth_valid"]
                target_scale = masked_mean(row["target"]["depth"].clamp_min(0.01).log(), valid_depth)
                predicted_scale = masked_mean(outputs["predicted_log_depth"], valid_depth)
                depth_scale_errors[row["sample"].parent_id].append(abs(float(predicted_scale - target_scale)))
        parent_errors = {parent: float(np.mean(values)) for parent, values in errors.items()}
        parent_depth_scale_errors = {
            parent: float(np.mean(values)) for parent, values in depth_scale_errors.items()
        }
        parent_improvements = {
            parent: float(baseline_eval["parent_metrics"][parent]["camera_height_abs_log_error"] - value)
            for parent, value in parent_errors.items()
        }
        parent_depth_scale_improvements = {
            parent: float(
                baseline_eval["parent_metrics"][parent]["depth_scale_abs_log_error"] - value
            )
            for parent, value in parent_depth_scale_errors.items()
        }
        normalized_errors = [
            parent_errors[parent]
            / max(float(baseline_eval["parent_metrics"][parent]["camera_height_abs_log_error"]), 1.0e-8)
            for parent in parent_errors
        ] + [
            parent_depth_scale_errors[parent]
            / max(float(baseline_eval["parent_metrics"][parent]["depth_scale_abs_log_error"]), 1.0e-8)
            for parent in parent_depth_scale_errors
        ]
        height_rows.append(
            {
                "config": height_config,
                "config_sha256": canonical_sha(height_config),
                "parent_errors": parent_errors,
                "parent_improvements": parent_improvements,
                "parent_depth_scale_errors": parent_depth_scale_errors,
                "parent_depth_scale_improvements": parent_depth_scale_improvements,
                "maximum_normalized_height_or_depth_scale_error": max(normalized_errors),
                "all_parents_improve": (
                    len(parent_improvements) == 2
                    and len(parent_depth_scale_improvements) == 2
                    and all(value > 0.0 for value in parent_improvements.values())
                    and all(value > 0.0 for value in parent_depth_scale_improvements.values())
                ),
                "fallback_frames": fallbacks,
            }
        )
    eligible_height = [row for row in height_rows if row["all_parents_improve"]]
    selected_height = min(
        eligible_height or height_rows,
        key=lambda row: (
            row["maximum_normalized_height_or_depth_scale_error"],
            max(row["parent_errors"].values()),
            max(row["parent_depth_scale_errors"].values()),
            row["fallback_frames"],
            row["config_sha256"],
        ),
    )

    sigma_rows = []
    selected_parent_context = parent_vio_height_context(
        prepared, cached_outputs, selected_height["config"], device
    )
    for sigma_config in sigma_candidates():
        config = {"height": selected_height["config"], "support_sigma": sigma_config}
        parent_nll: dict[str, list[float]] = defaultdict(list)
        support_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        with torch.no_grad():
            for index, row in enumerate(prepared):
                outputs, _ = apply_geometry(
                    cached_outputs[index],
                    row["sample"],
                    config,
                    baseline,
                    device,
                    selected_parent_context,
                )
                target = row["target"]
                if not bool(target["plane_valid"]):
                    continue
                sigma = outputs["support_residual_sigma_m"][0]
                nll = masked_mean(
                    gaussian_nll_tensor(target["support_residual"], sigma), target["support_valid"]
                )
                parent_nll[row["sample"].parent_id].append(float(nll))
                absolute = target["support_residual"][target["support_valid"]].abs()
                support_pairs.append((np.asarray([float(sigma)]), np.asarray([float(absolute.mean())])))
        parent_macro = float(np.mean([np.mean(values) for values in parent_nll.values()]))
        support_ordering = quantile_residual_summary(support_pairs)
        proper_gain = float(baseline_eval["overall_metrics"]["support_nll"] - parent_macro)
        monotonic = bool(support_ordering["nondecreasing"])
        sigma_rows.append(
            {
                "config": sigma_config,
                "config_sha256": canonical_sha(sigma_config),
                "support_nll": parent_macro,
                "proper_score_gain": proper_gain,
                "nondecreasing": monotonic,
                "eligible": proper_gain > 0.0 and monotonic,
                "quantile_residual_means": support_ordering["quantile_residual_means"],
            }
        )
    eligible_sigma = [row for row in sigma_rows if row["eligible"]]
    selected_sigma = min(eligible_sigma or sigma_rows, key=lambda row: (row["support_nll"], row["config_sha256"]))
    selected = {"height": selected_height["config"], "support_sigma": selected_sigma["config"]}
    return selected, {
        "height_candidates": height_rows,
        "selected_height": selected_height,
        "height_selection_safe": bool(eligible_height),
        "support_sigma_candidates": sigma_rows,
        "selected_support_sigma": selected_sigma,
        "support_sigma_selection_safe": bool(eligible_sigma),
        "selected_config_sha256": canonical_sha(selected),
    }


def serialize_predictions(
    model: FactorSplitHead,
    prepared: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    prediction_dir = output_dir / "canary_predictions_seed17"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    cached = cache_model_outputs(model, prepared, device)
    parent_context = parent_vio_height_context(prepared, cached, config["height"], device)
    receipts = []
    with torch.no_grad():
        for row, raw in zip(prepared, cached):
            sample = row["sample"]
            outputs, geometry_receipt = apply_geometry(raw, sample, config, {}, device, parent_context)
            log_depth = outputs["predicted_log_depth"][0, 0]
            log_scale = log_depth.mean()
            with np.load(sample.label_path, allow_pickle=False) as source:
                camera_receipt = str(np.asarray(source["camera_geometry_receipt_sha256"]).item())
            payload = {
                "schema": np.asarray("blindassist_assistive_geometry_r2_f1_prediction_v1"),
                "sample_id": np.asarray(sample.sample_id),
                "factor_identity": np.asarray("AG_R2_F1_FACTORS_RGB_K_IMU_GEOMETRY"),
                "camera_geometry_receipt_sha256": np.asarray(camera_receipt),
                "depth_shape_positive_hw": torch.exp(log_depth - log_scale).cpu().numpy().astype(np.float32),
                "log_metric_scale_m_scalar": np.asarray(float(log_scale), dtype=np.float32),
                "depth_log_sigma_hw": outputs["depth_log_sigma"][0, 0].cpu().numpy().astype(np.float32),
                "depth_valid_probability_hw": outputs["depth_valid_probability"][0, 0].cpu().numpy().astype(np.float32),
                "metric_scale_valid": np.asarray(bool(outputs["depth_valid_probability"].mean() >= 0.5)),
                "support_probability_hw": outputs["support_probability"][0, 0].cpu().numpy().astype(np.float32),
                "support_plane_normal_camera_xyz": outputs["support_plane_normal_camera_xyz"][0].cpu().numpy().astype(np.float32),
                "camera_height_m": np.asarray(float(outputs["camera_height_m"][0]), dtype=np.float32),
                "support_residual_sigma_m": np.asarray(float(outputs["support_residual_sigma_m"][0]), dtype=np.float32),
                "support_valid": np.asarray(bool(outputs["support_valid_probability"][0] >= 0.5)),
                "obstacle_evidence_probability_hw": outputs["obstacle_probability"][0, 0].cpu().numpy().astype(np.float32),
                "boundary_probability_hw": outputs["boundary_probability"][0, 0].cpu().numpy().astype(np.float32),
                "boundary_localization_sigma_px_hw": outputs["boundary_sigma_px"][0, 0].cpu().numpy().astype(np.float32),
                "evidence_valid_hw": (outputs["evidence_valid_probability"][0, 0] >= 0.5).cpu().numpy().astype(np.bool_),
            }
            path = prediction_dir / f"{sample.sample_id}.npz"
            np.savez_compressed(path, **payload)
            receipts.append({"sample_id": sample.sample_id, "path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size, "camera_geometry_receipt_sha256": camera_receipt, "geometry": geometry_receipt})
    return receipts


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "Attempt-03 formal execution requires CUDA")
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    require(lock["status"] == "ATTEMPT03_GRAVITY_AND_SUPPORT_GEOMETRY_EXECUTION_AUTHORIZED", "Attempt-03 lock invalid")
    require(sha256_file(args.fresh_label_result) == lock["bindings"]["fresh_label_result_sha256"], "Attempt-03 fresh labels drift")
    require(sha256_file(args.attempt02_result) == EXPECTED_ATTEMPT02_RESULT_SHA256, "Attempt-02 result drift")
    require(sha256_file(args.baseline_result) == EXPECTED_BASELINE_RESULT_SHA256, "FIT baseline drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART checkpoint drift")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    fresh = json.loads(args.fresh_label_result.read_text(encoding="utf-8"))
    attempt02 = json.loads(args.attempt02_result.read_text(encoding="utf-8"))
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline = baseline_result["baseline_parameters"]
    rows_by_role = {
        role: [{**row, "role": role} for row in fresh["frames"] if row["role"] == role]
        for role in ("CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    require({key: len(value) for key, value in rows_by_role.items()} == {"CHECKPOINT_SELECTION": 6, "TRAIN_CANARY": 6}, "Attempt-03 held roster drift")
    require(set(row["parent_id"] for row in rows_by_role["CHECKPOINT_SELECTION"]).isdisjoint(row["parent_id"] for row in rows_by_role["TRAIN_CANARY"]), "Attempt-03 role parent overlap")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    selection_samples, selection_feature_receipt = extract_features(
        sorted(rows_by_role["CHECKPOINT_SELECTION"], key=lambda row: row["sample_id"]),
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    selection_prepared = prepare(selection_samples, device)
    selection_baseline = evaluate_cached(selection_prepared, None, baseline, None, device)
    seed_rows = []
    for seed_row in attempt02["seed_results"]:
        seed = int(seed_row["seed"])
        checkpoint = Path(seed_row["composite_checkpoint"]["path"])
        require(sha256_file(checkpoint) == EXPECTED_COMPOSITES[seed], f"Attempt-02 composite drift: {seed}")
        model = FactorSplitHead(baseline).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        cached = cache_model_outputs(model, selection_prepared, device)
        config, selection_receipt = choose_geometry_config(selection_prepared, cached, baseline, selection_baseline, device)
        evaluation = evaluate_cached(selection_prepared, cached, baseline, config, device)
        selection_gate = gate(evaluation, selection_baseline, seed)
        eligible = selection_gate["all_primary_metrics_passed"] and selection_gate["all_uncertainty_families_passed"]
        seed_rows.append({"seed": seed, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": EXPECTED_COMPOSITES[seed], "config": config, "config_sha256": canonical_sha(config), "selection_receipt": selection_receipt, "selection_evaluation": evaluation, "selection_gate": selection_gate, "selection_eligible": eligible})
        del cached, model
        torch.cuda.empty_cache()

    all_selection_eligible = all(row["selection_eligible"] for row in seed_rows)
    canary_feature_receipt = None
    canary_prepared: list[dict[str, Any]] = []
    canary_baseline = None
    if all_selection_eligible:
        canary_samples, canary_feature_receipt = extract_features(
            sorted(rows_by_role["TRAIN_CANARY"], key=lambda row: row["sample_id"]),
            args.depthart_source,
            args.depthart_checkpoint,
            args.depthart_extension,
            device,
        )
        canary_prepared = prepare(canary_samples, device)
        canary_baseline = evaluate_cached(canary_prepared, None, baseline, None, device)
        for row in seed_rows:
            model = FactorSplitHead(baseline).to(device)
            state = torch.load(row["checkpoint"], map_location=device, weights_only=True)
            model.load_state_dict(state["model"], strict=True)
            cached = cache_model_outputs(model, canary_prepared, device)
            evaluation = evaluate_cached(canary_prepared, cached, baseline, row["config"], device)
            row["canary_evaluation"] = evaluation
            row["canary_gate"] = gate(evaluation, canary_baseline, row["seed"])
            del cached, model
            torch.cuda.empty_cache()
    else:
        for row in seed_rows:
            row["canary_evaluation"] = None
            row["canary_gate"] = None

    passed = all_selection_eligible and all(
        row["canary_gate"]["all_primary_metrics_passed"] and row["canary_gate"]["all_uncertainty_families_passed"]
        for row in seed_rows
    )
    predictions = []
    if passed:
        seed17 = next(row for row in seed_rows if row["seed"] == 17)
        model = FactorSplitHead(baseline).to(device)
        state = torch.load(seed17["checkpoint"], map_location=device, weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        predictions = serialize_predictions(model, canary_prepared, seed17["config"], args.output_dir, device)
        del model

    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_factor_learnability_attempt03_result_v1",
        "status": "R2_F1_FACTOR_LEARNABILITY_ATTEMPT03_PASS" if passed else "R2_F1_FACTOR_LEARNABILITY_ATTEMPT03_FAIL_STOP",
        "passed": passed,
        "execution_lock": str(args.lock.resolve()),
        "execution_lock_sha256": sha256_file(args.lock),
        "feature_receipt": {"selection": selection_feature_receipt, "canary_after_all_selection_eligible": canary_feature_receipt},
        "role_frame_counts": {"CHECKPOINT_SELECTION": len(selection_samples), "TRAIN_CANARY": len(canary_prepared)},
        "selection_baseline": selection_baseline,
        "canary_baseline": canary_baseline,
        "seed_results": seed_rows,
        "prediction_receipts_seed17": predictions,
        "decision": {
            "all_seeds_passed": passed,
            "all_seeds_selection_eligible_before_canary_open": all_selection_eligible,
            "attempt02_learned_components_retrained": False,
            "support_normal_source": "deployment-available IMU gravity",
            "camera_height_source": "deterministic predicted-depth/K/gravity/support geometry",
            "reducer_or_task_outcome_read": False,
            "next_action_if_pass": "Run FactorTensorAdapter on the serialized real factor tensors, then execute the deterministic reducer seam canary.",
        },
    }
    result_path = args.output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--fresh-label-result", type=Path, default=DEFAULT_FRESH_LABEL_RESULT)
    parser.add_argument("--attempt02-result", type=Path, default=DEFAULT_ATTEMPT02_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in ("lock", "fresh_label_result", "attempt02_result", "baseline_result", "depthart_source", "depthart_checkpoint", "depthart_extension", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "selection_eligible": {str(row["seed"]): row["selection_eligible"] for row in result["seed_results"]}}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
