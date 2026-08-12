"""RGB+K-only AG model inference and source-context conditioning.

The public entrypoint in this module cannot accept a label path, depth array,
pose, IMU sample, reducer state, or task state.  It therefore replaces the old
label-backed feature loader for this independent confirmation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contract import ContractError, require, sha256_file

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
BOUNDARY_DISTANCE_SCALE_PX = 3.0
MIN_GEOMETRY_PIXELS = 64


def scale_intrinsics(
    intrinsics: np.ndarray,
    source_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> np.ndarray:
    value = np.asarray(intrinsics)
    require(value.dtype == np.dtype("float64") and value.shape == (3, 3), "F2_MODEL_K_SCHEMA")
    require(bool(np.all(np.isfinite(value))), "F2_MODEL_K_NONFINITE")
    require(value[0, 0] > 0.0 and value[1, 1] > 0.0 and value[2, 2] == 1.0, "F2_MODEL_K_INVALID")
    source_height, source_width = source_hw
    output_height, output_width = output_hw
    require(min(source_height, source_width, output_height, output_width) > 0, "F2_MODEL_HW_INVALID")
    scale_x = output_width / float(source_width)
    scale_y = output_height / float(source_height)
    result = value.copy()
    result[0, 0] *= scale_x
    result[1, 1] *= scale_y
    result[0, 2] = (result[0, 2] + 0.5) * scale_x - 0.5
    result[1, 2] = (result[1, 2] + 0.5) * scale_y - 0.5
    return result


def _json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(code, str(error)) from error
    require(isinstance(value, dict), code)
    return value


def _binding_path(binding: Mapping[str, Any], code: str) -> Path:
    require(set(binding) == {"role", "path", "bytes", "sha256"}, f"{code}_SCHEMA")
    path = Path(str(binding["path"])).resolve()
    require(path.is_file(), f"{code}_MISSING")
    require(type(binding["bytes"]) is int and path.stat().st_size == binding["bytes"], f"{code}_BYTES")
    require(sha256_file(path) == str(binding["sha256"]).upper(), f"{code}_SHA")
    return path


def verify_depthart_source_manifest(path: Path) -> Path:
    manifest = _json(path, "F2_DEPTHART_SOURCE_MANIFEST_READ")
    require(
        set(manifest) == {"schema", "source_root", "files"}
        and manifest["schema"] == "blindassist.depthart.source_manifest.v1"
        and isinstance(manifest["source_root"], str)
        and isinstance(manifest["files"], list)
        and bool(manifest["files"]),
        "F2_DEPTHART_SOURCE_MANIFEST_SCHEMA",
    )
    root = Path(manifest["source_root"]).resolve()
    require(root.is_dir(), "F2_DEPTHART_SOURCE_ROOT_MISSING")
    seen: set[str] = set()
    for row in manifest["files"]:
        require(isinstance(row, Mapping) and set(row) == {"path", "bytes", "sha256"}, "F2_DEPTHART_SOURCE_ROW_SCHEMA")
        relative = str(row["path"])
        require(relative and "\\" not in relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts, "F2_DEPTHART_SOURCE_ROW_PATH")
        require(relative not in seen, "F2_DEPTHART_SOURCE_ROW_DUPLICATE")
        seen.add(relative)
        member = (root / relative).resolve()
        require(root in member.parents and member.is_file(), "F2_DEPTHART_SOURCE_ROW_MISSING")
        require(type(row["bytes"]) is int and member.stat().st_size == row["bytes"], "F2_DEPTHART_SOURCE_ROW_BYTES")
        require(sha256_file(member) == str(row["sha256"]).upper(), "F2_DEPTHART_SOURCE_ROW_SHA")
    return root


@dataclass(frozen=True)
class ModelPaths:
    depthart_source: Path
    depthart_extension: Path
    depthart_checkpoint: Path
    factor_baseline_result: Path
    factor_student_result: Path
    factor_student_checkpoint: Path
    metric_student_result: Path
    metric_student_checkpoint: Path
    metric_scale_bank_result: Path
    metric_scale_bank: Path
    frozen_hybrid_recipe_result: Path

    @classmethod
    def from_bindings(cls, bindings: Mapping[str, Mapping[str, Any]]) -> ModelPaths:
        expected = {
            "DEPTHART_SOURCE_MANIFEST", "DEPTHART_EXTENSION", "DEPTHART_CHECKPOINT",
            "FACTOR_BASELINE_RESULT", "FACTOR_STUDENT_RESULT", "FACTOR_STUDENT_CHECKPOINT",
            "METRIC_STUDENT_RESULT", "METRIC_STUDENT_CHECKPOINT", "METRIC_SCALE_BANK_RESULT",
            "METRIC_SCALE_BANK", "FROZEN_HYBRID_RECIPE_RESULT",
        }
        require(set(bindings) == expected, "F2_MODEL_BINDING_ROLE_SET")
        paths = {role: _binding_path(bindings[role], f"F2_MODEL_BINDING_{role}") for role in expected}
        return cls(
            depthart_source=verify_depthart_source_manifest(paths["DEPTHART_SOURCE_MANIFEST"]),
            depthart_extension=paths["DEPTHART_EXTENSION"],
            depthart_checkpoint=paths["DEPTHART_CHECKPOINT"],
            factor_baseline_result=paths["FACTOR_BASELINE_RESULT"],
            factor_student_result=paths["FACTOR_STUDENT_RESULT"],
            factor_student_checkpoint=paths["FACTOR_STUDENT_CHECKPOINT"],
            metric_student_result=paths["METRIC_STUDENT_RESULT"],
            metric_student_checkpoint=paths["METRIC_STUDENT_CHECKPOINT"],
            metric_scale_bank_result=paths["METRIC_SCALE_BANK_RESULT"],
            metric_scale_bank=paths["METRIC_SCALE_BANK"],
            frozen_hybrid_recipe_result=paths["FROZEN_HYBRID_RECIPE_RESULT"],
        )


class RGBKFactorPredictor:
    """Load the frozen recipe once and infer factor arrays from RGB+K only."""

    def __init__(self, paths: ModelPaths, device_name: str) -> None:
        import torch

        require(torch.cuda.is_available() and device_name.startswith("cuda"), "F2_MODEL_CUDA_REQUIRED")
        self.torch = torch
        self.device = torch.device(device_name)
        from scripts.research.assistive_geometry.calibrate_ag_r2_metric_scale_residual_bank import (
            load_residual_bank,
        )
        from scripts.research.assistive_geometry.train_ag_r2_f1_attempt18_consumed_cross_domain_adaptation import (
            CrossDomainFactorHead,
        )
        from scripts.research.assistive_geometry.train_ag_r2_f1_factor_learnability import (
            load_depthart_extension,
        )
        from scripts.research.assistive_geometry.train_ag_r2_metric_depth_student import (
            MetricDepthStudentHead,
        )
        from scripts.research.assistive_geometry.train_ag_st_masked_student import (
            load_depthart_backbone,
        )

        recipe = _json(paths.frozen_hybrid_recipe_result, "F2_RECIPE_RESULT_READ")
        require(recipe.get("passed") is True and recipe.get("training_steps", 0) == 0, "F2_RECIPE_RESULT_INVALID")
        baseline_result = _json(paths.factor_baseline_result, "F2_FACTOR_BASELINE_READ")
        baseline = baseline_result.get("baseline_parameters")
        require(isinstance(baseline, dict), "F2_FACTOR_BASELINE_SCHEMA")
        factor_result = _json(paths.factor_student_result, "F2_FACTOR_RESULT_READ")
        metric_result = _json(paths.metric_student_result, "F2_METRIC_RESULT_READ")
        bank_result = _json(paths.metric_scale_bank_result, "F2_METRIC_BANK_RESULT_READ")
        require(factor_result.get("passed") is True, "F2_FACTOR_RESULT_NOT_PASS")
        require(metric_result.get("passed") is True, "F2_METRIC_RESULT_NOT_PASS")
        require(bank_result.get("passed") is True, "F2_METRIC_BANK_RESULT_NOT_PASS")
        load_depthart_extension(paths.depthart_extension)
        self.backbone, self.backbone_scan = load_depthart_backbone(
            paths.depthart_source, paths.depthart_checkpoint, self.device, 17
        )
        self.factor = CrossDomainFactorHead(baseline).to(self.device)
        factor_state = torch.load(paths.factor_student_checkpoint, map_location=self.device, weights_only=True)
        self.factor.load_state_dict(factor_state.get("model", factor_state), strict=True)
        architecture = metric_result.get("architecture")
        require(isinstance(architecture, Mapping), "F2_METRIC_ARCHITECTURE_MISSING")
        self.metric = MetricDepthStudentHead(
            hidden=int(architecture["hidden_channels"]),
            global_hidden=int(architecture["global_hidden_channels"]),
        ).to(self.device)
        metric_state = torch.load(paths.metric_student_checkpoint, map_location=self.device, weights_only=True)
        self.metric.load_state_dict(metric_state.get("model", metric_state), strict=True)
        self.factor.eval()
        self.metric.eval()
        self.backbone.eval()
        self.bank = load_residual_bank(paths.metric_scale_bank)

    def predict(self, rgb_u8_hwc: np.ndarray, intrinsics: np.ndarray) -> dict[str, Any]:
        import torch.nn.functional as F

        from scripts.research.assistive_geometry.calibrate_ag_r2_metric_scale_residual_bank import (
            predict_scale_sigma,
        )
        from scripts.research.assistive_geometry.train_ag_r2_metric_scale_uncertainty import (
            pooled_uncertainty_feature,
        )

        rgb = np.asarray(rgb_u8_hwc)
        k64 = np.asarray(intrinsics)
        require(rgb.dtype == np.dtype("uint8") and rgb.ndim == 3 and rgb.shape[2] == 3, "F2_MODEL_RGB_SCHEMA")
        require(k64.dtype == np.dtype("float64") and k64.shape == (3, 3), "F2_MODEL_K_SCHEMA")
        height, width = rgb.shape[:2]
        padded_height = int(math.ceil(height / 32.0) * 32)
        padded_width = int(math.ceil(width / 32.0) * 32)
        normalized = ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
        image = self.torch.from_numpy(normalized)[None].to(self.device)
        k = self.torch.from_numpy(k64.astype(np.float32))[None].to(self.device)
        image = F.pad(image, (0, padded_width - width, 0, padded_height - height), value=0.0)
        amp_dtype = self.torch.bfloat16 if self.torch.cuda.is_bf16_supported() else self.torch.float16
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=amp_dtype):
            cameras = self.backbone.metric_depthart.cam_embedder(k, padded_height, padded_width, self.device)
            features = self.backbone.metric_depthart.pretrained.forward_with_adapters(
                image,
                adapters=[
                    self.backbone.metric_depthart.daa1, self.backbone.metric_depthart.daa2,
                    self.backbone.metric_depthart.daa3, self.backbone.metric_depthart.daa4,
                ],
                cams=list(cameras),
            )
            relative, _, pyramid = self.backbone.decode(list(features), (padded_height, padded_width))
            scale = self.backbone.metric_depthart.sfh(features[3], cameras[3])
            base_depth = relative * scale.view(-1, 1, 1, 1) * self.backbone.metric_depthart.max_depth
        feature_height = int(pyramid.shape[-2] * height / padded_height)
        feature_width = int(pyramid.shape[-1] * width / padded_width)
        require(
            feature_height * padded_height == pyramid.shape[-2] * height
            and feature_width * padded_width == pyramid.shape[-1] * width,
            "F2_MODEL_FEATURE_RATIO",
        )
        feature = pyramid[..., :feature_height, :feature_width].float()
        base = F.interpolate(base_depth[..., :height, :width].float(), (feature_height, feature_width), mode="bilinear", align_corners=False)
        with self.torch.no_grad():
            factor = self.factor(feature, base)
            metric = self.metric(feature, base)
            pooled = pooled_uncertainty_feature(feature, base, metric["predicted_log_depth"])
            scale_sigma, _ = predict_scale_sigma(pooled[0], self.bank)
        depth = metric["predicted_log_depth"][0, 0].exp().detach().cpu().numpy().astype(np.float64)
        depth_sigma = factor["depth_log_sigma"][0, 0].exp().detach().cpu().numpy().astype(np.float64)
        depth_sigma = np.sqrt(depth_sigma**2 + float(scale_sigma) ** 2)
        depth_valid = factor["depth_valid_probability"][0, 0].detach().cpu().numpy().astype(np.float64)
        support = factor["support_probability"][0, 0].detach().cpu().numpy().astype(np.float64)
        support_scalar_known = bool(float(factor["support_valid_probability"][0]) >= 0.5)
        support_sigma = float(factor["support_residual_sigma_m"][0].detach().cpu())
        obstacle = factor["obstacle_probability"][0, 0].detach().cpu().numpy().astype(np.float64)
        boundary_probability = factor["boundary_probability"][0, 0].detach().cpu().numpy().astype(np.float64)
        boundary_distance = np.minimum(-BOUNDARY_DISTANCE_SCALE_PX * np.log(np.maximum(boundary_probability, 1.0e-8)), 32.0)
        boundary_sigma = factor["boundary_sigma_px"][0, 0].detach().cpu().numpy().astype(np.float64)
        evidence_valid = factor["evidence_valid_probability"][0, 0].detach().cpu().numpy().astype(np.float64)
        depth_known = np.isfinite(depth) & np.isfinite(depth_sigma) & (depth > 0.0) & (depth_valid >= 0.5)
        support_known = depth_known & support_scalar_known & np.isfinite(support)
        evidence_known = depth_known & np.isfinite(obstacle) & np.isfinite(boundary_distance) & np.isfinite(boundary_sigma) & (evidence_valid >= 0.5)

        def unknown(value: np.ndarray, known: np.ndarray) -> np.ndarray:
            result = np.asarray(value, dtype=np.float64).copy()
            result[~known] = np.nan
            return result

        output_hw = (feature_height, feature_width)
        return {
            "source_hw": [height, width],
            "output_hw": [feature_height, feature_width],
            "intrinsics": scale_intrinsics(k64, (height, width), output_hw),
            "depth_m": unknown(depth, depth_known),
            "depth_log_sigma": unknown(depth_sigma, depth_known),
            "depth_known": depth_known.astype(bool),
            "support_probability": unknown(support, support_known),
            "support_residual_sigma_m": unknown(np.full(output_hw, support_sigma, dtype=np.float64), support_known),
            "support_known": support_known.astype(bool),
            "obstacle_probability": unknown(obstacle, evidence_known),
            "boundary_distance_px": unknown(boundary_distance, evidence_known),
            "boundary_sigma_px": unknown(boundary_sigma, evidence_known),
            "evidence_known": evidence_known.astype(bool),
        }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    centers = (np.cumsum(weights) - 0.5 * weights) / float(np.sum(weights))
    return float(np.interp(probability, centers, values, left=values[0], right=values[-1]))


def predicted_camera_height(raw: Mapping[str, Any], gravity_up_camera: np.ndarray) -> float:
    depth = np.asarray(raw["depth_m"], dtype=np.float64)
    support = np.asarray(raw["support_probability"], dtype=np.float64)
    depth_known = np.asarray(raw["depth_known"], dtype=bool)
    support_known = np.asarray(raw["support_known"], dtype=bool)
    intrinsics = np.asarray(raw["intrinsics"], dtype=np.float64)
    gravity = np.asarray(gravity_up_camera, dtype=np.float64)
    require(gravity.shape == (3,) and bool(np.all(np.isfinite(gravity))), "F2_CONTEXT_GRAVITY_SCHEMA")
    norm = float(np.linalg.norm(gravity))
    require(norm > 0.0, "F2_CONTEXT_GRAVITY_ZERO")
    gravity = gravity / norm
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    ray_x = (columns - intrinsics[0, 2]) / intrinsics[0, 0]
    ray_y = (rows - intrinsics[1, 2]) / intrinsics[1, 1]
    ray_dot_up = gravity[0] * ray_x + gravity[1] * ray_y + gravity[2]
    heights = -depth * ray_dot_up
    selected = (
        depth_known & support_known & np.isfinite(heights) & (heights >= 0.30) & (heights <= 3.00)
        & (support >= 0.0)
    )
    require(int(np.sum(selected)) >= MIN_GEOMETRY_PIXELS, "F2_CONTEXT_HEIGHT_DENOMINATOR")
    weights = np.clip(support[selected], 1.0e-4, 1.0)
    return _weighted_quantile(heights[selected], weights, 0.75)


def condition_parent_predictions(
    raw_records: Sequence[Mapping[str, Any]],
    session_context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply exactly one calibration-derived session scale to sealed records."""

    require(
        set(session_context) == {"parent_id", "camera_height_m", "camera_height_mad_m", "gravity_up_camera_xyz"},
        "F2_CONTEXT_KEY_SET",
    )
    require(len(raw_records) == 12, "F2_CONTEXT_SCORE_COUNT")
    source_height = session_context["camera_height_m"]
    require(type(source_height) is float and math.isfinite(source_height) and 0.45 <= source_height <= 2.20, "F2_CONTEXT_SOURCE_HEIGHT")
    heights = [predicted_camera_height(row, np.asarray(session_context["gravity_up_camera_xyz"], dtype=np.float64)) for row in raw_records]
    model_height = float(np.median(np.asarray(heights, dtype=np.float64)))
    require(math.isfinite(model_height) and model_height > 0.0, "F2_CONTEXT_MODEL_HEIGHT")
    factor = source_height / model_height
    require(math.isfinite(factor) and factor > 0.0, "F2_CONTEXT_SCALE_FACTOR")
    conditioned: list[dict[str, Any]] = []
    for raw in raw_records:
        row = dict(raw)
        row["depth_m"] = np.asarray(raw["depth_m"], dtype=np.float64) * factor
        conditioned.append(row)
    receipt = {
        "parent_id": session_context["parent_id"],
        "method": "source_camera_height_divided_by_median_score_rgbk_predicted_q75_support_height",
        "source_camera_height_m": source_height,
        "raw_predicted_camera_heights_m": heights,
        "raw_predicted_camera_height_median_m": model_height,
        "session_scale_factor": factor,
        "score_truth_used": False,
        "training_or_tuning": False,
    }
    return conditioned, receipt
