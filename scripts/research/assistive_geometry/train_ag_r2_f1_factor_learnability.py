#!/usr/bin/env python3
"""Execute the AG R2 F1 factor-only learnability protocol on frozen TUM labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_st_masked_student import (  # noqa: E402
    DEPTHART_PYRAMID_CHANNELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_depthart_backbone,
)
from validate_ag_r2_f1_factor_execution_lock import validate as validate_execution_lock  # noqa: E402

DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_LEARNABILITY_EXECUTION_LOCK_2026-08-11.json"
DEFAULT_LABEL_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-source-native-labels-tum13-r0/result.json"
DEFAULT_BASELINE_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-fit-baselines-tum13-r0/result.json"
DEFAULT_DEPTHART_SOURCE = Path("F:/ba-data/blindassist-artifacts-20260805/models/depthart/source")
DEFAULT_DEPTHART_CHECKPOINT = DEFAULT_DEPTHART_SOURCE / "checkpoints/metric/depthart_metric_indoor_s_448.pth"
DEFAULT_DEPTHART_EXTENSION = Path(
    "E:/codex-tools/tools/venvs/blindassist-venv-export312/Lib/site-packages/"
    "depthart_selective_scan_cuda.cp311-win_amd64.pyd"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-factor-learnability-tum13-r0"
EXPECTED_LABEL_RESULT_SHA256 = "521662011D72973BF604E9A190E65504DBAB455559A458AD412A6B8B1FC35422"
EXPECTED_BASELINE_RESULT_SHA256 = "EECD5C9244C6A8A467B7890AF79D7871374AC8B325A87B48AE9E052089908F44"
EXPECTED_DEPTHART_SHA256 = "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65"
EXPECTED_DEPTHART_EXTENSION_SHA256 = "F8AF67B697136343A45F9DD9840362F504FE76B8FC1AC569FCB02916539BC600"
BOUNDARY_DISTANCE_SCALE_PX = 3.0
CHARBONNIER_EPSILON = 0.01
HUBER_DELTA = 0.10
MIN_SIGMA = 1.0e-3
GAUSSIAN_CONSTANT = 0.5 * math.log(2.0 * math.pi)
PRIMARY_METRICS = (
    "depth_shape_abs_log_error",
    "depth_scale_abs_log_error",
    "depth_nll",
    "support_brier",
    "support_plane_angular_error_rad",
    "camera_height_abs_log_error",
    "support_nll",
    "obstacle_brier",
    "boundary_distance_abs_error_px",
    "boundary_nll",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_depthart_extension(path: Path) -> None:
    require(path.is_file(), f"DepthART CUDA extension missing: {path}")
    require(
        sha256_file(path) == EXPECTED_DEPTHART_EXTENSION_SHA256,
        "DepthART CUDA extension SHA drift",
    )
    if "depthart_selective_scan_cuda" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("depthart_selective_scan_cuda", path)
    require(spec is not None and spec.loader is not None, "DepthART CUDA extension spec invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules["depthart_selective_scan_cuda"] = module
    spec.loader.exec_module(module)


def logit(probability: float) -> float:
    bounded = min(max(float(probability), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(bounded / (1.0 - bounded))


def inverse_softplus(value: float) -> float:
    bounded = max(float(value), 1.0e-5)
    return math.log(math.expm1(bounded)) if bounded < 20.0 else bounded


def huber_tensor(value: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    absolute = value.abs()
    return torch.where(absolute <= delta, 0.5 * value.square() / delta, absolute - 0.5 * delta)


def gaussian_nll_tensor(residual: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    bounded = sigma.clamp_min(MIN_SIGMA)
    return 0.5 * (residual / bounded).square() + bounded.log() + GAUSSIAN_CONSTANT


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = valid.bool() & torch.isfinite(value)
    if not bool(selected.any()):
        return value.sum() * 0.0
    return value[selected].mean()


@dataclass
class CachedSample:
    sample_id: str
    parent_id: str
    role: str
    orientation: str
    label_path: Path
    native_hw: tuple[int, int]
    feature: torch.Tensor
    base_depth_feature: torch.Tensor
    targets: dict[str, torch.Tensor]


class FactorOnlyHead(nn.Module):
    """No-regret factor head; it has no final task or reducer outputs."""

    def __init__(self, baselines: dict[str, Any], hidden: int = 96) -> None:
        super().__init__()
        require(hidden % 8 == 0, "hidden channels must be divisible by eight")
        self.register_buffer("baseline_log_scale", torch.tensor(float(baselines["depth_log_scale"])))
        self.register_buffer(
            "baseline_normal",
            torch.tensor(baselines["support_plane_normal_camera_xyz"], dtype=torch.float32),
        )
        self.register_buffer("baseline_log_height", torch.tensor(math.log(float(baselines["camera_height_m"]))))
        input_channels = DEPTHART_PYRAMID_CHANNELS + 1
        self.project = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.context = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(8, hidden),
                    nn.GELU(),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.spatial = nn.Conv2d(hidden, 8, 1)
        self.global_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 7),
        )
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        spatial_biases = (
            0.0,
            float(baselines["depth_log_sigma"]),
            logit(float(baselines["depth_valid_probability"])),
            logit(float(baselines["support_probability"])),
            logit(float(baselines["obstacle_evidence_probability"])),
            logit(float(baselines["boundary_probability"])),
            inverse_softplus(float(baselines["boundary_localization_sigma_px"])),
            logit(float(baselines["evidence_valid_probability"])),
        )
        with torch.no_grad():
            self.spatial.bias.copy_(torch.tensor(spatial_biases))
            self.global_head[-1].bias.copy_(
                torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        inverse_softplus(float(baselines["support_residual_sigma_m"])),
                        logit(float(baselines["support_valid_probability"])),
                        -12.0,
                    ]
                )
            )

    def forward(self, feature: torch.Tensor, base_depth_m: torch.Tensor) -> dict[str, torch.Tensor]:
        guidance = base_depth_m.clamp(0.05, 20.0).log()
        latent = self.project(torch.cat([feature, guidance], dim=1))
        latent = latent + self.fuse(torch.cat([branch(latent) for branch in self.context], dim=1))
        spatial = self.spatial(latent)
        global_values = self.global_head(F.adaptive_avg_pool2d(latent, 1).flatten(1))
        depth_gate = torch.sigmoid(global_values[:, 6:7, None, None])
        candidate_log_depth = guidance + 1.5 * torch.tanh(spatial[:, 0:1])
        predicted_log_depth = self.baseline_log_scale + depth_gate * (
            candidate_log_depth - self.baseline_log_scale
        )
        normal = self.baseline_normal[None] + global_values[:, :3]
        normal = F.normalize(normal, dim=1, eps=1.0e-6)
        return {
            "predicted_log_depth": predicted_log_depth,
            "depth_log_sigma": spatial[:, 1:2].clamp(math.log(0.005), math.log(5.0)),
            "depth_valid_probability": torch.sigmoid(spatial[:, 2:3]),
            "support_probability": torch.sigmoid(spatial[:, 3:4]),
            "obstacle_probability": torch.sigmoid(spatial[:, 4:5]),
            "boundary_probability": torch.sigmoid(spatial[:, 5:6]),
            "boundary_sigma_px": F.softplus(spatial[:, 6:7]).clamp(0.05, 64.0),
            "evidence_valid_probability": torch.sigmoid(spatial[:, 7:8]),
            "support_plane_normal_camera_xyz": normal,
            "camera_height_m": torch.exp(
                (self.baseline_log_height + global_values[:, 3]).clamp(math.log(0.3), math.log(3.0))
            ),
            "support_residual_sigma_m": F.softplus(global_values[:, 4]).clamp(0.005, 5.0),
            "support_valid_probability": torch.sigmoid(global_values[:, 5]),
            "depth_gate": depth_gate[:, 0, 0, 0],
        }


def resize_nearest(value: np.ndarray, output_hw: tuple[int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(value)).float()[None, None]
    return F.interpolate(tensor, output_hw, mode="nearest")[0]


def load_downsampled_targets(path: Path, output_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        support_plane_valid = bool(np.asarray(payload["support_plane_valid"]).item())
        result = {
            "metric_depth_m": resize_nearest(np.asarray(payload["metric_depth_m_hw"], dtype=np.float32), output_hw),
            "metric_valid": resize_nearest(np.asarray(payload["metric_depth_valid_hw"], dtype=np.float32), output_hw).bool(),
            "support": resize_nearest(np.asarray(payload["support_truth_hw"], dtype=np.float32), output_hw),
            "support_valid": resize_nearest(np.asarray(payload["support_truth_valid_hw"], dtype=np.float32), output_hw).bool(),
            "support_residual": resize_nearest(
                np.nan_to_num(np.asarray(payload["support_signed_plane_residual_m_hw"], dtype=np.float32)), output_hw
            ),
            "obstacle": resize_nearest(np.asarray(payload["obstacle_evidence_truth_hw"], dtype=np.float32), output_hw),
            "boundary_distance": resize_nearest(
                np.nan_to_num(np.asarray(payload["boundary_distance_px_hw"], dtype=np.float32), nan=32.0), output_hw
            ),
            "evidence_valid": resize_nearest(np.asarray(payload["evidence_truth_valid_hw"], dtype=np.float32), output_hw).bool(),
            "support_plane_valid": torch.tensor(support_plane_valid),
            "support_normal": torch.from_numpy(
                np.nan_to_num(np.asarray(payload["support_plane_normal_camera_xyz"], dtype=np.float32))
            ),
            "camera_height_m": torch.tensor(
                float(np.nan_to_num(np.asarray(payload["camera_height_m"]).item(), nan=1.0)), dtype=torch.float32
            ),
        }
    return result


def extract_features(
    rows: list[dict[str, Any]],
    depthart_source: Path,
    depthart_checkpoint: Path,
    depthart_extension: Path,
    device: torch.device,
    *,
    load_targets: bool = True,
) -> tuple[list[CachedSample], dict[str, Any]]:
    require(device.type == "cuda", "formal F1 feature extraction requires CUDA")
    load_depthart_extension(depthart_extension)
    started = time.perf_counter()
    model, scan = load_depthart_backbone(depthart_source, depthart_checkpoint, device, 17)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cached: list[CachedSample] = []
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        label_path = Path(row["output"]).resolve()
        require(label_path.is_file() and sha256_file(label_path) == row["output_sha256"], "label payload drift")
        with np.load(label_path, allow_pickle=False) as payload:
            rgb = np.asarray(payload["rgb_u8_hwc"], dtype=np.uint8)
            intrinsics = np.asarray(payload["intrinsics_output"], dtype=np.float32)
        height, width = rgb.shape[:2]
        normalized = ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
        image = torch.from_numpy(normalized)[None].to(device)
        k = torch.from_numpy(intrinsics)[None].to(device)
        padded_height = int(math.ceil(height / 32.0) * 32)
        padded_width = int(math.ceil(width / 32.0) * 32)
        image = F.pad(image, (0, padded_width - width, 0, padded_height - height), value=0.0)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
            cameras = model.metric_depthart.cam_embedder(k, padded_height, padded_width, device)
            features = model.metric_depthart.pretrained.forward_with_adapters(
                image,
                adapters=[model.metric_depthart.daa1, model.metric_depthart.daa2, model.metric_depthart.daa3, model.metric_depthart.daa4],
                cams=list(cameras),
            )
            relative, _, pyramid = model.decode(list(features), (padded_height, padded_width))
            scale = model.metric_depthart.sfh(features[3], cameras[3])
            base_depth = relative * scale.view(-1, 1, 1, 1) * model.metric_depthart.max_depth
        feature_height = int(pyramid.shape[-2] * height / padded_height)
        feature_width = int(pyramid.shape[-1] * width / padded_width)
        require(feature_height * padded_height == pyramid.shape[-2] * height, "feature height ratio drift")
        require(feature_width * padded_width == pyramid.shape[-1] * width, "feature width ratio drift")
        pyramid = pyramid[..., :feature_height, :feature_width]
        base_feature = F.interpolate(base_depth[..., :height, :width], (feature_height, feature_width), mode="bilinear", align_corners=False)
        cached.append(
            CachedSample(
                sample_id=str(row["sample_id"]),
                parent_id=str(row["parent_id"]),
                role=str(row["role"]),
                orientation=str(row["orientation"]),
                label_path=label_path,
                native_hw=(height, width),
                feature=pyramid[0].to(dtype=torch.float16, device="cpu"),
                base_depth_feature=base_feature[0].to(dtype=torch.float16, device="cpu"),
                targets=(
                    load_downsampled_targets(
                        label_path,
                        (feature_height, feature_width),
                    )
                    if load_targets
                    else {}
                ),
            )
        )
    peak = int(torch.cuda.max_memory_allocated())
    del model
    torch.cuda.empty_cache()
    return cached, {
        "elapsed_seconds": time.perf_counter() - started,
        "frame_count": len(cached),
        "feature_shapes_chw": [
            list(shape) for shape in sorted({tuple(sample.feature.shape) for sample in cached})
        ],
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "peak_cuda_allocated_bytes": peak,
        "scan_backend": scan,
        "targets_loaded": load_targets,
    }


def compute_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    normalization: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = targets["metric_valid"]
    target_log = targets["metric_depth_m"].clamp_min(0.01).log()
    predicted_log = outputs["predicted_log_depth"]
    target_scale = masked_mean(target_log, valid)
    predicted_scale = masked_mean(predicted_log, valid)
    target_shape = target_log - target_scale
    predicted_shape = predicted_log - predicted_scale
    depth_residual = predicted_log - target_log
    boundary_target = torch.exp(-targets["boundary_distance"] / BOUNDARY_DISTANCE_SCALE_PX)
    predicted_boundary_distance = (-BOUNDARY_DISTANCE_SCALE_PX * outputs["boundary_probability"].clamp_min(1.0e-8).log()).clamp_max(32.0)
    plane_valid = targets["support_plane_valid"].bool()
    if bool(plane_valid):
        angle = torch.acos(
            torch.clamp(
                torch.sum(outputs["support_plane_normal_camera_xyz"][0] * targets["support_normal"]),
                -1.0 + 1.0e-6,
                1.0 - 1.0e-6,
            )
        )
        height_error = huber_tensor(outputs["camera_height_m"][0].log() - targets["camera_height_m"].log())
        support_nll = masked_mean(
            gaussian_nll_tensor(targets["support_residual"], outputs["support_residual_sigma_m"][0]),
            targets["support_valid"],
        )
    else:
        zero = predicted_log.sum() * 0.0
        angle = zero
        height_error = zero
        support_nll = zero
    losses = {
        "depth_shape_log_charbonnier": masked_mean(
            torch.sqrt((predicted_shape - target_shape).square() + CHARBONNIER_EPSILON**2) - CHARBONNIER_EPSILON,
            valid,
        ),
        "metric_scale_log_huber": huber_tensor(predicted_scale - target_scale),
        "depth_heteroscedastic_nll": masked_mean(
            gaussian_nll_tensor(depth_residual, outputs["depth_log_sigma"].exp()), valid
        ),
        "depth_validity_brier": (outputs["depth_valid_probability"] - valid.float()).square().mean(),
        "support_probability_brier": masked_mean(
            (outputs["support_probability"] - targets["support"]).square(), targets["support_valid"]
        ),
        "support_plane_angular": angle,
        "camera_height_log_huber": height_error,
        "support_residual_heteroscedastic_nll": support_nll,
        "support_validity_brier": (outputs["support_valid_probability"][0] - plane_valid.float()).square(),
        "obstacle_evidence_brier": masked_mean(
            (outputs["obstacle_probability"] - targets["obstacle"]).square(), targets["evidence_valid"]
        ),
        "boundary_probability_brier": masked_mean(
            (outputs["boundary_probability"] - boundary_target).square(), targets["evidence_valid"]
        ),
        "boundary_localization_heteroscedastic_nll": masked_mean(
            gaussian_nll_tensor(
                predicted_boundary_distance - targets["boundary_distance"], outputs["boundary_sigma_px"]
            ),
            targets["evidence_valid"],
        ),
        "evidence_validity_brier": (
            outputs["evidence_valid_probability"] - targets["evidence_valid"].float()
        ).square().mean(),
    }
    require(set(losses) == set(normalization), "loss/normalization field drift")
    objective = torch.stack([losses[key] / float(normalization[key]) for key in sorted(losses)]).mean()
    return objective, losses


def move_targets(targets: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in targets.items()}


def forward_sample(model: FactorOnlyHead, sample: CachedSample, device: torch.device) -> dict[str, torch.Tensor]:
    feature = sample.feature[None].to(device=device, dtype=torch.float32)
    base = sample.base_depth_feature[None].to(device=device, dtype=torch.float32)
    return model(feature, base)


def load_native_targets(sample: CachedSample, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(sample.label_path, allow_pickle=False) as payload:
        def map_value(name: str, *, nan: float = 0.0) -> torch.Tensor:
            value = np.nan_to_num(np.asarray(payload[name], dtype=np.float32), nan=nan)
            return torch.from_numpy(value)[None, None].to(device)

        return {
            "depth": map_value("metric_depth_m_hw"),
            "depth_valid": map_value("metric_depth_valid_hw").bool(),
            "support": map_value("support_truth_hw"),
            "support_valid": map_value("support_truth_valid_hw").bool(),
            "support_residual": map_value("support_signed_plane_residual_m_hw"),
            "normal": torch.from_numpy(
                np.nan_to_num(np.asarray(payload["support_plane_normal_camera_xyz"], dtype=np.float32))
            ).to(device),
            "height": torch.tensor(
                float(np.nan_to_num(np.asarray(payload["camera_height_m"]).item(), nan=1.0)), device=device
            ),
            "plane_valid": torch.tensor(bool(np.asarray(payload["support_plane_valid"]).item()), device=device),
            "obstacle": map_value("obstacle_evidence_truth_hw"),
            "boundary_distance": map_value("boundary_distance_px_hw", nan=32.0),
            "evidence_valid": map_value("evidence_truth_valid_hw").bool(),
        }


def native_outputs(outputs: dict[str, torch.Tensor], native_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
    resized: dict[str, torch.Tensor] = {}
    for key in (
        "predicted_log_depth",
        "depth_log_sigma",
        "depth_valid_probability",
        "support_probability",
        "obstacle_probability",
        "boundary_probability",
        "boundary_sigma_px",
        "evidence_valid_probability",
    ):
        resized[key] = F.interpolate(outputs[key], native_hw, mode="bilinear", align_corners=False)
    for key in (
        "support_plane_normal_camera_xyz",
        "camera_height_m",
        "support_residual_sigma_m",
        "support_valid_probability",
        "depth_gate",
    ):
        resized[key] = outputs[key]
    return resized


def frame_metrics(outputs: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, float]:
    valid = target["depth_valid"]
    log_target = target["depth"].clamp_min(0.01).log()
    log_predicted = outputs["predicted_log_depth"]
    target_scale = masked_mean(log_target, valid)
    predicted_scale = masked_mean(log_predicted, valid)
    depth_residual = log_predicted - log_target
    boundary_predicted_distance = (-BOUNDARY_DISTANCE_SCALE_PX * outputs["boundary_probability"].clamp_min(1.0e-8).log()).clamp_max(32.0)
    evidence_valid = target["evidence_valid"]
    if bool(target["plane_valid"]):
        angular = torch.acos(
            torch.clamp(
                torch.sum(outputs["support_plane_normal_camera_xyz"][0] * target["normal"]),
                -1.0 + 1.0e-6,
                1.0 - 1.0e-6,
            )
        )
        height_error = (outputs["camera_height_m"][0].log() - target["height"].log()).abs()
        support_brier = masked_mean((outputs["support_probability"] - target["support"]).square(), target["support_valid"])
        support_nll = masked_mean(
            gaussian_nll_tensor(target["support_residual"], outputs["support_residual_sigma_m"][0]),
            target["support_valid"],
        )
    else:
        angular = height_error = support_brier = support_nll = torch.tensor(float("nan"), device=log_predicted.device)
    return {
        "depth_shape_abs_log_error": float(masked_mean(((log_predicted - predicted_scale) - (log_target - target_scale)).abs(), valid)),
        "depth_scale_abs_log_error": float((predicted_scale - target_scale).abs()),
        "depth_nll": float(masked_mean(gaussian_nll_tensor(depth_residual, outputs["depth_log_sigma"].exp()), valid)),
        "depth_validity_brier": float((outputs["depth_valid_probability"] - valid.float()).square().mean()),
        "support_brier": float(support_brier),
        "support_plane_angular_error_rad": float(angular),
        "camera_height_abs_log_error": float(height_error),
        "support_nll": float(support_nll),
        "support_validity_brier": float((outputs["support_valid_probability"][0] - target["plane_valid"].float()).square()),
        "obstacle_brier": float(masked_mean((outputs["obstacle_probability"] - target["obstacle"]).square(), evidence_valid)),
        "boundary_distance_abs_error_px": float(masked_mean((boundary_predicted_distance - target["boundary_distance"]).abs(), evidence_valid)),
        "boundary_nll": float(masked_mean(gaussian_nll_tensor(boundary_predicted_distance - target["boundary_distance"], outputs["boundary_sigma_px"]), evidence_valid)),
        "evidence_validity_brier": float((outputs["evidence_valid_probability"] - evidence_valid.float()).square().mean()),
        "depth_sigma_mean": float(masked_mean(outputs["depth_log_sigma"].exp(), valid)),
        "depth_abs_residual_mean": float(masked_mean(depth_residual.abs(), valid)),
        "support_sigma_mean": float(outputs["support_residual_sigma_m"][0]),
        "support_abs_residual_mean": float(masked_mean(target["support_residual"].abs(), target["support_valid"])) if bool(target["plane_valid"]) else float("nan"),
        "boundary_sigma_mean": float(masked_mean(outputs["boundary_sigma_px"], evidence_valid)),
        "boundary_abs_residual_mean": float(masked_mean((boundary_predicted_distance - target["boundary_distance"]).abs(), evidence_valid)),
        "depth_gate": float(outputs["depth_gate"][0]),
    }


def aggregate_parent_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["parent_id"])].append(row["metrics"])
    parent_metrics: dict[str, dict[str, float]] = {}
    for parent, values in grouped.items():
        keys = sorted({key for metrics in values for key in metrics})
        parent_metrics[parent] = {}
        for key in keys:
            finite = [metrics[key] for metrics in values if key in metrics and math.isfinite(metrics[key])]
            if finite:
                parent_metrics[parent][key] = float(np.mean(finite))
    overall: dict[str, float] = {}
    for key in sorted({key for metrics in parent_metrics.values() for key in metrics}):
        values = [metrics[key] for metrics in parent_metrics.values() if key in metrics]
        overall[key] = float(np.mean(values))
    return {"parent_metrics": parent_metrics, "overall_metrics": overall}


def baseline_native_outputs(
    baseline: dict[str, Any], native_hw: tuple[int, int], device: torch.device
) -> dict[str, torch.Tensor]:
    height, width = native_hw
    def constant(value: float) -> torch.Tensor:
        return torch.full((1, 1, height, width), float(value), device=device)
    return {
        "predicted_log_depth": constant(float(baseline["depth_log_scale"])),
        "depth_log_sigma": constant(float(baseline["depth_log_sigma"])),
        "depth_valid_probability": constant(float(baseline["depth_valid_probability"])),
        "support_probability": constant(float(baseline["support_probability"])),
        "obstacle_probability": constant(float(baseline["obstacle_evidence_probability"])),
        "boundary_probability": constant(float(baseline["boundary_probability"])),
        "boundary_sigma_px": constant(float(baseline["boundary_localization_sigma_px"])),
        "evidence_valid_probability": constant(float(baseline["evidence_valid_probability"])),
        "support_plane_normal_camera_xyz": torch.tensor(baseline["support_plane_normal_camera_xyz"], device=device)[None],
        "camera_height_m": torch.tensor([baseline["camera_height_m"]], device=device),
        "support_residual_sigma_m": torch.tensor([baseline["support_residual_sigma_m"]], device=device),
        "support_valid_probability": torch.tensor([baseline["support_valid_probability"]], device=device),
        "depth_gate": torch.zeros(1, device=device),
    }


def evaluate(
    model: FactorOnlyHead | None,
    samples: list[CachedSample],
    baseline: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for sample in samples:
            target = load_native_targets(sample, device)
            outputs = (
                native_outputs(forward_sample(model, sample, device), sample.native_hw)
                if model is not None
                else baseline_native_outputs(baseline, sample.native_hw, device)
            )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "parent_id": sample.parent_id,
                    "orientation": sample.orientation,
                    "metrics": frame_metrics(outputs, target),
                }
            )
    return {"frames": rows, **aggregate_parent_metrics(rows)}


def eligibility(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current = candidate["overall_metrics"]
    reference = baseline["overall_metrics"]
    regrets = {
        key: (float(current[key]) - float(reference[key])) / max(abs(float(reference[key])), 1.0e-8)
        for key in PRIMARY_METRICS
    }
    no_worse = all(value <= 1.0e-4 for value in regrets.values())
    improves = any(value < -1.0e-4 for value in regrets.values())
    return {
        "eligible": no_worse and improves,
        "no_worse_all_primary_metrics": no_worse,
        "at_least_one_primary_improves": improves,
        "normalized_regret": regrets,
        "maximum_normalized_regret": max(regrets.values()),
    }


def save_checkpoint(path: Path, model: FactorOnlyHead, seed: int, step: int) -> dict[str, Any]:
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"checkpoint collision: {path}")
    with temporary.open("xb") as stream:
        torch.save({"seed": seed, "step": step, "model": model.state_dict()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def train_seed(
    seed: int,
    model: FactorOnlyHead,
    fit: list[CachedSample],
    selection: list[CachedSample],
    baseline: dict[str, Any],
    normalization: dict[str, float],
    schedule: list[int],
    total_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = random.Random(seed)
    selection_baseline = evaluate(None, selection, baseline, device)
    candidates: list[dict[str, Any]] = []
    loss_trace: list[dict[str, Any]] = []

    def evaluate_candidate(step: int) -> None:
        evaluation = evaluate(model, selection, baseline, device)
        gate = eligibility(evaluation, selection_baseline)
        checkpoint_path = output_dir / f"seed-{seed}-step-{step}.pt"
        receipt = save_checkpoint(checkpoint_path, model, seed, step)
        candidates.append({"step": step, "evaluation": evaluation, "eligibility": gate, "checkpoint": receipt})

    if 0 in schedule:
        evaluate_candidate(0)
    for step in range(1, total_steps + 1):
        sample = fit[generator.randrange(len(fit))]
        targets = move_targets(sample.targets, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_sample(model, sample, device)
        objective, losses = compute_losses(outputs, targets, normalization)
        require(bool(torch.isfinite(objective)), "non-finite optimizer objective")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if step <= warmup_steps:
            factor = step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            factor = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = learning_rate * factor
        optimizer.step()
        if step == 1 or step % 100 == 0:
            loss_trace.append(
                {
                    "step": step,
                    "objective": float(objective.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "components": {key: float(value.detach()) for key, value in losses.items()},
                }
            )
        if step in schedule:
            evaluate_candidate(step)
            model.train()
    eligible = [row for row in candidates if row["eligibility"]["eligible"]]
    selected = None
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                row["eligibility"]["maximum_normalized_regret"],
                row["step"],
                row["checkpoint"]["sha256"],
            ),
        )
    return {
        "seed": seed,
        "selection_baseline": selection_baseline,
        "candidates": candidates,
        "selected_checkpoint": selected["checkpoint"] if selected is not None else None,
        "selected_step": selected["step"] if selected is not None else None,
        "loss_trace": loss_trace,
    }


def monotonic_by_sigma(rows: list[dict[str, Any]], sigma_key: str, residual_key: str) -> dict[str, Any]:
    pairs = [
        (row["metrics"][sigma_key], row["metrics"][residual_key])
        for row in rows
        if math.isfinite(row["metrics"].get(sigma_key, float("nan")))
        and math.isfinite(row["metrics"].get(residual_key, float("nan")))
    ]
    require(len(pairs) >= 4, f"uncertainty denominator missing: {sigma_key}")
    ordered = sorted(pairs)
    groups = np.array_split(np.asarray(ordered, dtype=np.float64), 4)
    residual_means = [float(group[:, 1].mean()) for group in groups if len(group)]
    return {
        "quantile_residual_means": residual_means,
        "nondecreasing": all(a <= b + 1.0e-6 for a, b in zip(residual_means, residual_means[1:])),
    }


def bootstrap_lower(values: list[float], seed: int, draws: int = 10000) -> float:
    require(values, "bootstrap values empty")
    rng = np.random.default_rng(seed)
    source = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(source), size=(draws, len(source)))
    means = source[indices].mean(axis=1)
    return float(np.quantile(means, 0.025))


def canary_gate(model_eval: dict[str, Any], baseline_eval: dict[str, Any], seed: int) -> dict[str, Any]:
    improvements: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        values = []
        for parent, baseline_metrics in baseline_eval["parent_metrics"].items():
            if metric in baseline_metrics and metric in model_eval["parent_metrics"].get(parent, {}):
                values.append(float(baseline_metrics[metric]) - float(model_eval["parent_metrics"][parent][metric]))
        improvements[metric] = {
            "parent_improvements": values,
            "bootstrap_95_lower": bootstrap_lower(values, seed + sum(ord(c) for c in metric)),
            "favorable_parent_fraction": float(np.mean(np.asarray(values) > 0.0)),
            "passed": bootstrap_lower(values, seed + sum(ord(c) for c in metric)) > 0.0
            and float(np.mean(np.asarray(values) > 0.0)) >= 0.75,
        }
    uncertainty = {
        "depth": {
            "proper_score_gain": float(baseline_eval["overall_metrics"]["depth_nll"] - model_eval["overall_metrics"]["depth_nll"]),
            **monotonic_by_sigma(model_eval["frames"], "depth_sigma_mean", "depth_abs_residual_mean"),
        },
        "support": {
            "proper_score_gain": float(baseline_eval["overall_metrics"]["support_nll"] - model_eval["overall_metrics"]["support_nll"]),
            **monotonic_by_sigma(model_eval["frames"], "support_sigma_mean", "support_abs_residual_mean"),
        },
        "boundary": {
            "proper_score_gain": float(baseline_eval["overall_metrics"]["boundary_nll"] - model_eval["overall_metrics"]["boundary_nll"]),
            **monotonic_by_sigma(model_eval["frames"], "boundary_sigma_mean", "boundary_abs_residual_mean"),
        },
    }
    for values in uncertainty.values():
        values["passed"] = values["proper_score_gain"] > 0.0 and values["nondecreasing"]
    return {
        "metric_improvements": improvements,
        "uncertainty": uncertainty,
        "all_primary_metrics_passed": all(row["passed"] for row in improvements.values()),
        "all_uncertainty_families_passed": all(row["passed"] for row in uncertainty.values()),
    }


def serialize_predictions(
    model: FactorOnlyHead,
    samples: list[CachedSample],
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prediction_dir = output_dir / "canary_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    model.eval()
    with torch.no_grad():
        for sample in samples:
            outputs = native_outputs(forward_sample(model, sample, device), sample.native_hw)
            log_depth = outputs["predicted_log_depth"][0, 0]
            log_scale = log_depth.mean()
            payload = {
                "schema": np.asarray("blindassist_assistive_geometry_r2_f1_prediction_v1"),
                "sample_id": np.asarray(sample.sample_id),
                "factor_identity": np.asarray("AG_R2_F1_FACTORS_ONLY"),
                "camera_geometry_receipt_sha256": np.asarray("BOUND_IN_SOURCE_LABEL_RECEIPT"),
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
            rows.append({"sample_id": sample.sample_id, "path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size, "field_count": len(payload)})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is required")
    device = torch.device(args.device)
    validation = validate_execution_lock(args.lock)
    require(validation["passed"], "execution lock static validation failed")
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    require(lock["status"] == "F1_FACTOR_LEARNABILITY_EXECUTION_AUTHORIZED", "execution lock status invalid")
    require(sha256_file(args.label_result) == EXPECTED_LABEL_RESULT_SHA256, "label result binding drift")
    require(sha256_file(args.baseline_result) == EXPECTED_BASELINE_RESULT_SHA256, "baseline result binding drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART checkpoint binding drift")
    label_result = json.loads(args.label_result.read_text(encoding="utf-8"))
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline = baseline_result["baseline_parameters"]
    normalization = baseline_result["optimizer_normalization"]
    rows = sorted(label_result["frames"], key=lambda row: row["sample_id"])
    require(not args.output_dir.exists(), f"output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    cached, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    by_role = {role: [sample for sample in cached if sample.role == role] for role in ("FIT", "CHECKPOINT_SELECTION", "TRAIN_CANARY")}
    require({key: len(value) for key, value in by_role.items()} == {"FIT": 27, "CHECKPOINT_SELECTION": 6, "TRAIN_CANARY": 6}, "role frame count drift")
    schedule = [int(value) for value in lock["training"]["checkpoint_steps"]]
    seeds = [int(value) for value in lock["training"]["seeds"]]
    seed_results: list[dict[str, Any]] = []
    canary_baseline = evaluate(None, by_role["TRAIN_CANARY"], baseline, device)
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = FactorOnlyHead(baseline, int(lock["model"]["hidden_channels"])).to(device)
        training = train_seed(
            seed,
            model,
            by_role["FIT"],
            by_role["CHECKPOINT_SELECTION"],
            baseline,
            normalization,
            schedule,
            int(lock["training"]["optimizer_steps"]),
            float(lock["training"]["learning_rate"]),
            float(lock["training"]["weight_decay"]),
            int(lock["training"]["warmup_steps"]),
            args.output_dir,
            device,
        )
        selected = training["selected_checkpoint"]
        if selected is not None:
            state = torch.load(selected["path"], map_location=device, weights_only=True)
            model.load_state_dict(state["model"], strict=True)
            canary = evaluate(model, by_role["TRAIN_CANARY"], baseline, device)
            gate = canary_gate(canary, canary_baseline, seed)
        else:
            canary = None
            gate = None
        seed_results.append({**training, "canary_evaluation": canary, "canary_gate": gate})
        del model
        torch.cuda.empty_cache()
    passed = all(row["selected_checkpoint"] is not None and row["canary_gate"]["all_primary_metrics_passed"] and row["canary_gate"]["all_uncertainty_families_passed"] for row in seed_results)
    prediction_receipts: list[dict[str, Any]] = []
    if seed_results[0]["selected_checkpoint"] is not None:
        model = FactorOnlyHead(baseline, int(lock["model"]["hidden_channels"])).to(device)
        state = torch.load(seed_results[0]["selected_checkpoint"]["path"], map_location=device, weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        prediction_receipts = serialize_predictions(model, by_role["TRAIN_CANARY"], args.output_dir, device)
        del model
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_factor_learnability_result_v1",
        "status": "R2_F1_FACTOR_LEARNABILITY_PASS" if passed else "R2_F1_FACTOR_LEARNABILITY_FAIL_STOP",
        "passed": passed,
        "execution_lock": str(args.lock.resolve()),
        "execution_lock_sha256": sha256_file(args.lock),
        "label_result_sha256": EXPECTED_LABEL_RESULT_SHA256,
        "baseline_result_sha256": EXPECTED_BASELINE_RESULT_SHA256,
        "depthart_checkpoint_sha256": EXPECTED_DEPTHART_SHA256,
        "feature_receipt": feature_receipt,
        "role_frame_counts": {key: len(value) for key, value in by_role.items()},
        "canary_baseline": canary_baseline,
        "seed_results": seed_results,
        "prediction_receipts_seed17": prediction_receipts,
        "decision": {
            "all_predeclared_seeds_passed": passed,
            "reducer_or_task_outcome_read": False,
            "f2_authorized_if_pass": passed,
            "next_action_if_pass": "Bind the selected factor tensor receipt to the already-passed FactorTensorAdapter and deterministic reducer in a real-factor seam canary.",
        },
        "claim_boundary": "TRAIN-only factor learnability only; not task utility, deployment, product or safety evidence.",
    }
    result_path = args.output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--label-result", type=Path, default=DEFAULT_LABEL_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in (
        "lock",
        "label_result",
        "baseline_result",
        "depthart_source",
        "depthart_checkpoint",
        "depthart_extension",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "selected_steps": {str(row["seed"]): row["selected_step"] for row in result["seed_results"]}}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
