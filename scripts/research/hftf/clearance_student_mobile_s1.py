"""Clearance-Student Mobile S1: medium-capacity four-scale geometry student."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

ARTIFACT_ROOT = Path("artifacts.local/evidence/hftf/clearance-student-mobile-s1")


@dataclass(frozen=True)
class S1Config:
    input_size: int = 384
    stage: str = "S1-A"
    feature_scales: int = 4
    teacher: str = "canonical_da_v2_518_offline"
    temporal_head: bool = False


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )


class DepthwiseRefine(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
        )


class ClearanceStudentMobileS1(nn.Module):
    """MobileNetV3-Large with four-scale DPT-lite-style decoder."""

    input_size = 384
    tap_channels = (24, 40, 112, 960)
    decoder_channels = 96
    # 2,048-channel bottleneck puts the complete student at ~5.4M params,
    # inside the planned 5--10M exploration band without changing topology.
    context_channels = 2048

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        # A modest high-level context expansion keeps S1 in the intended
        # 5--10M parameter band while retaining HTP-friendly static ops.
        self.context = ConvNormAct(self.tap_channels[-1], self.context_channels, 1)
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, self.decoder_channels, 1) for channels in self.tap_channels[:-1]]
            + [nn.Conv2d(self.context_channels, self.decoder_channels, 1)]
        )
        self.refine = nn.ModuleList([DepthwiseRefine(self.decoder_channels) for _ in range(4)])
        self.depth_head = nn.Sequential(ConvNormAct(self.decoder_channels, 32), nn.Conv2d(32, 1, 1))
        self.confidence_head = nn.Conv2d(self.decoder_channels, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ground_head = nn.Linear(self.decoder_channels, 4)
        self.height_head = nn.Linear(self.decoder_channels, 1)
        self.clearance_head = nn.Linear(self.decoder_channels, 3)
        self.feature_projections = nn.ModuleList([nn.Conv2d(self.decoder_channels, 384, 1) for _ in range(4)])
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.depth_head[-1].bias, math.log(math.expm1(2.0)))
        nn.init.constant_(self.height_head.bias, 1.2)
        nn.init.constant_(self.clearance_head.bias, 2.0)

    def forward(self, image: torch.Tensor, output_size: tuple[int, int] | None = None) -> dict[str, Any]:
        value = image
        taps: list[torch.Tensor] = []
        # These indices correspond to 1/4, 1/8, 1/16 and 1/32 feature maps in
        # torchvision MobileNetV3-Large. Verify channels instead of silently
        # accepting a backbone topology drift.
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in (3, 6, 12, 16):
                taps.append(value)
        if len(taps) != 4 or tuple(t.shape[1] for t in taps) != self.tap_channels:
            raise RuntimeError(f"S1 tap topology drift: {[tuple(t.shape) for t in taps]}")
        features = []
        top = self.refine[3](self.lateral[3](self.context(taps[3])))
        features.append(top)
        for index in (2, 1, 0):
            top = self.refine[index](F.interpolate(top, size=taps[index].shape[-2:], mode="bilinear", align_corners=False) + self.lateral[index](taps[index]))
            features.append(top)
        full = F.interpolate(top, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        size = output_size or (self.input_size, self.input_size)
        depth = F.softplus(self.depth_head(full)) + 0.10
        confidence_logits = self.confidence_head(full)
        if depth.shape[-2:] != size:
            depth = F.interpolate(depth, size=size, mode="bilinear", align_corners=True)
            confidence_logits = F.interpolate(confidence_logits, size=size, mode="bilinear", align_corners=True)
        pooled = self.pool(full).flatten(1)
        return {
            "metric_depth": depth[:, 0],
            "confidence_logits": confidence_logits[:, 0],
            "ground_plane": self.ground_head(pooled),
            "camera_height": F.softplus(self.height_head(pooled))[:, 0] + 0.25,
            "clearance": F.softplus(self.clearance_head(pooled)) + 0.05,
            # Return features shallow-to-deep (1/4, 1/8, 1/16, 1/32), which
            # is the ordering used by teacher intermediate-layer caches.
            "features": tuple(reversed(features)),
        }


def normalize_bgr_batch(images: list[torch.Tensor]) -> torch.Tensor:
    batch = torch.stack(images).float() / 255.0
    batch = batch[:, [2, 1, 0]]
    batch = F.interpolate(batch, size=(S1Config.input_size,) * 2, mode="bilinear", align_corners=False)
    mean = batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (batch - mean) / std


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def feature_distillation_loss(student_features: tuple[torch.Tensor, ...], teacher_features: tuple[torch.Tensor, ...], projections: nn.ModuleList) -> torch.Tensor:
    if len(student_features) != 4 or len(teacher_features) != 4:
        raise ValueError("S1 feature distillation requires exactly four scales")
    losses = []
    for student, teacher, projection in zip(student_features, teacher_features, projections, strict=True):
        projected = projection(student)
        target = F.interpolate(teacher.detach(), size=projected.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(F.l1_loss(projected, target))
    return torch.stack(losses).mean()


def geometry_loss(pred: dict[str, torch.Tensor], truth: torch.Tensor, teacher: torch.Tensor, config: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config or {}
    lo, hi = map(float, cfg.get("depth_clamp_m", (0.25, 6.0)))
    p = pred["metric_depth"].float().clamp(lo, hi)
    t = truth.float().clamp(lo, hi)
    teach = teacher.float().clamp(lo, hi)
    valid = torch.isfinite(truth) & (truth >= lo) & (truth <= hi)
    lp, lt, lteach = torch.log(p), torch.log(t), torch.log(teach)
    depth = F.smooth_l1_loss(lp[valid], lt[valid], beta=0.05) if valid.any() else lp.sum() * 0.0
    teacher_depth = F.smooth_l1_loss(lp, lteach, beta=0.05)
    gradient = F.l1_loss(torch.diff(lp, dim=-1), torch.diff(lt, dim=-1)) + F.l1_loss(torch.diff(lp, dim=-2), torch.diff(lt, dim=-2))
    log_ratio = (lp - lteach).flatten(1)
    scale = torch.median(log_ratio, dim=1).values.abs().mean()
    confidence_target = valid.float().mean(dim=(1, 2))
    return depth + 0.25 * teacher_depth + 0.2 * gradient + 0.15 * scale, {"log_depth": depth, "teacher_depth": teacher_depth, "gradient": gradient, "scale": scale, "confidence_target": confidence_target}


def require_finite_metrics(metrics: dict[str, float]) -> None:
    invalid = [key for key, value in metrics.items() if not isinstance(value, (int, float)) or not math.isfinite(float(value))]
    if invalid:
        raise ValueError(f"NON_FINITE_OR_UNDEFINED_METRICS: {sorted(invalid)}")
