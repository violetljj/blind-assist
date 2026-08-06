"""Clearance-Student Mobile R0: single-frame mobile metric/clearance student.

The implementation deliberately uses only static, QNN-friendly convolutions in
R0.  It is a development model, not an Android or safety authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ARTIFACT_ROOT = Path("artifacts.local/evidence/hftf/clearance-student-mobile-r0")
BANDS = ("left", "center", "right")


@dataclass(frozen=True)
class ClearanceStudentConfig:
    input_size: int = 384
    min_depth_m: float = 0.10
    max_depth_m: float = 8.0
    temporal_head: bool = False
    teacher: str = "canonical_da_v2_518_offline"
    encoder: str = "mobilenet_v3_small_r0_qnn_friendly_fallback"


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )


class ClearanceStudentMobileR0(nn.Module):
    """MobileNetV3-small + compact decoder and clearance-aware heads."""

    input_size = 384

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights).features
        # MobileNetV3-small feature taps are 24, 48 and 576 channels at 1/8,
        # 1/16 and 1/32 resolution respectively.
        self.lateral3 = nn.Conv2d(24, 96, 1)
        self.lateral4 = nn.Conv2d(48, 96, 1)
        self.lateral5 = nn.Conv2d(576, 96, 1)
        self.refine5 = ConvNormAct(96, 96)
        self.refine4 = ConvNormAct(96, 96)
        self.refine3 = ConvNormAct(96, 96)
        self.shared = nn.Sequential(ConvNormAct(96, 32), ConvNormAct(32, 16))
        self.depth_head = nn.Conv2d(16, 1, 1)
        self.confidence_head = nn.Conv2d(16, 1, 1)
        self.geometry_pool = nn.AdaptiveAvgPool2d(1)
        self.ground_head = nn.Linear(16, 4)  # normal xyz proxy + residual
        self.height_head = nn.Linear(16, 1)
        self.clearance_head = nn.Linear(16, 3)
        self._init_heads()

    def _init_heads(self) -> None:
        for module in (self.lateral3, self.lateral4, self.lateral5, self.refine5, self.refine4, self.refine3, self.shared):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.constant_(self.depth_head.bias, math.log(math.expm1(2.0)))
        nn.init.constant_(self.confidence_head.bias, 1.0)
        nn.init.constant_(self.height_head.bias, 1.2)
        nn.init.constant_(self.clearance_head.bias, 2.0)

    def forward(self, image: torch.Tensor, output_size: tuple[int, int] | None = None) -> dict[str, torch.Tensor]:
        value = image
        taps: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in (2, 7, 12):
                taps[index] = value
        if set(taps) != {2, 7, 12}:
            raise RuntimeError("MobileNet feature taps were not produced")
        pyramid = self.refine5(self.lateral5(taps[12]))
        pyramid = self.refine4(F.interpolate(pyramid, size=taps[7].shape[-2:], mode="bilinear", align_corners=False) + self.lateral4(taps[7]))
        pyramid = self.refine3(F.interpolate(pyramid, size=taps[2].shape[-2:], mode="bilinear", align_corners=False) + self.lateral3(taps[2]))
        shared = self.shared(F.interpolate(pyramid, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False))
        size = output_size or (self.input_size, self.input_size)
        depth = F.softplus(self.depth_head(shared)) + 0.10
        confidence = self.confidence_head(shared)
        if depth.shape[-2:] != size:
            depth = F.interpolate(depth, size=size, mode="bilinear", align_corners=True)
            confidence = F.interpolate(confidence, size=size, mode="bilinear", align_corners=True)
        pooled = self.geometry_pool(shared).flatten(1)
        ground = self.ground_head(pooled)
        height = F.softplus(self.height_head(pooled)) + 0.25
        clearance = F.softplus(self.clearance_head(pooled)) + 0.05
        return {"metric_depth": depth[:, 0], "confidence": confidence[:, 0], "ground_plane": ground, "camera_height": height[:, 0], "clearance": clearance}


def normalize_bgr_batch(images: list[torch.Tensor]) -> torch.Tensor:
    batch = torch.stack(images).float() / 255.0
    batch = batch[:, [2, 1, 0]]
    batch = F.interpolate(batch, size=(ClearanceStudentMobileR0.input_size,) * 2, mode="bilinear", align_corners=False)
    mean = batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (batch - mean) / std


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def clearance_targets(depth: torch.Tensor) -> torch.Tensor:
    """Derive fixed left/centre/right near-clearance targets from metric depth."""
    _, height, width = depth.shape
    yy = torch.arange(height, device=depth.device)[:, None].expand(height, width)
    xx = torch.arange(width, device=depth.device)[None, :].expand(height, width)
    lower = yy >= int(height * 0.55)
    bands = ((0.0, 0.33), (0.33, 0.67), (0.67, 1.0))
    outputs = []
    for lo, hi in bands:
        mask = lower & (xx >= int(width * lo)) & (xx < int(width * hi))
        values = depth[:, mask]
        values = torch.where(torch.isfinite(values) & (values >= 0.20), values, torch.full_like(values, 8.0))
        outputs.append(torch.quantile(values, 0.05, dim=1).clamp(0.20, 8.0))
    return torch.stack(outputs, dim=1)


def clearance_student_loss(pred: dict[str, torch.Tensor], truth_depth: torch.Tensor, teacher_depth: torch.Tensor, config: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = config or {}
    lo, hi = float(cfg.get("depth_clamp_m", [0.25, 6.0])[0]), float(cfg.get("depth_clamp_m", [0.25, 6.0])[1])
    p = pred["metric_depth"].float().clamp(lo, hi)
    t = truth_depth.float().clamp(lo, hi)
    teacher = teacher_depth.float().clamp(lo, hi)
    lp, lt, lteacher = torch.log(p), torch.log(t), torch.log(teacher)
    valid = torch.isfinite(truth_depth) & (truth_depth >= lo) & (truth_depth <= hi)
    depth = F.smooth_l1_loss(lp[valid], lt[valid], beta=0.05) if valid.any() else lp.sum() * 0.0
    teacher_loss = F.smooth_l1_loss(lp, lteacher, beta=0.05)
    grad = F.l1_loss(torch.diff(lp, dim=-1), torch.diff(lt, dim=-1)) + F.l1_loss(torch.diff(lp, dim=-2), torch.diff(lt, dim=-2))
    target_clearance = clearance_targets(truth_depth)
    clearance = F.smooth_l1_loss(pred["clearance"], target_clearance, beta=0.05)
    occupied = target_clearance < 1.5
    false_clear = F.relu(pred["clearance"] - target_clearance - 0.05)
    occupancy = (false_clear[occupied] ** 2).mean() if occupied.any() else false_clear.sum() * 0.0
    confidence_target = valid.float().mean(dim=(1, 2)).clamp(0.0, 1.0)
    confidence = F.binary_cross_entropy_with_logits(pred["confidence"].mean(dim=(1, 2)), confidence_target)
    total = depth + 0.25 * teacher_loss + 0.20 * grad + 0.50 * clearance + 0.75 * occupancy + 0.05 * confidence
    parts = {"log_depth": depth, "teacher": teacher_loss, "gradient": grad, "clearance": clearance, "occupancy_false_clear": occupancy, "confidence": confidence, "total": total}
    return total, {key: float(value.detach().cpu()) for key, value in parts.items()}


def build_config() -> ClearanceStudentConfig:
    return ClearanceStudentConfig()


def require_finite_metrics(metrics: dict[str, float]) -> None:
    missing = [key for key, value in metrics.items() if not isinstance(value, (int, float)) or not math.isfinite(value)]
    if missing:
        raise ValueError(f"NON_FINITE_OR_UNDEFINED_METRICS: {sorted(missing)}")
