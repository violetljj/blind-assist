"""Clearance-Student Mobile S1.1: mechanism-corrected mobile geometry student.

This module is deliberately independent from S1.  The encoder may be loaded
with ImageNet weights, and only student-owned decoder/head modules are
initialised locally.  The helpers expose binding digests and confidence-aware
metric-depth losses for the E0 preflight.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


@dataclass(frozen=True)
class S11Config:
    input_size: int = 384
    context_channels: int = 512
    decoder_channels: int = 128
    depth_min_m: float = 0.25
    depth_max_m: float = 6.0
    confidence_equals: int = 2


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


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def encoder_binding_digest(encoder: nn.Module) -> str:
    """Return a stable SHA-256 digest of encoder state tensors."""
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest().upper()


def encoder_binding_receipt(pre_digest: str, post_digest: str) -> dict[str, Any]:
    """Build a fail-closed pre/post encoder binding receipt."""
    return {
        "pre_digest_sha256": str(pre_digest),
        "post_digest_sha256": str(post_digest),
        "encoder_unchanged": str(pre_digest) == str(post_digest),
    }


class ClearanceStudentMobileS11(nn.Module):
    """Pretrained MobileNetV3-Large plus a higher-resolution four-scale decoder."""

    input_size = S11Config.input_size
    tap_channels = (24, 40, 112, 960)

    def __init__(
        self,
        pretrained: bool = True,
        *,
        weights: MobileNet_V3_Large_Weights | None = None,
        config: S11Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config or S11Config()
        if weights is None and pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.encoder = mobilenet_v3_large(weights=weights).features
        self.encoder_pre_init_digest = encoder_binding_digest(self.encoder)

        c = self.config.decoder_channels
        context = self.config.context_channels
        if context > 512:
            raise ValueError("S1.1 context_channels must be <= 512")
        # Keep capacity on 1/4 and 1/8 refinements rather than a huge 1/32
        # context expansion.
        self.context = ConvNormAct(self.tap_channels[-1], context, 1)
        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, c, 1) for ch in self.tap_channels[:-1]]
            + [nn.Conv2d(context, c, 1)]
        )
        self.refine = nn.ModuleList([DepthwiseRefine(c) for _ in range(4)])
        self.high_res_refine = nn.Sequential(ConvNormAct(c, c), DepthwiseRefine(c))
        self.depth_head = nn.Sequential(ConvNormAct(c, 32), nn.Conv2d(32, 1, 1))
        self.confidence_head = nn.Conv2d(c, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ground_head = nn.Linear(c, 4)
        self.height_head = nn.Linear(c, 1)
        self.clearance_head = nn.Linear(c, 3)
        self.feature_projections = nn.ModuleList([nn.Conv2d(c, 384, 1) for _ in range(4)])
        self._initialize_student_modules()
        self.encoder_post_init_digest = encoder_binding_digest(self.encoder)
        self.encoder_binding = encoder_binding_receipt(
            self.encoder_pre_init_digest, self.encoder_post_init_digest
        )

    def _initialize_student_modules(self) -> None:
        """Initialise only modules owned by S1.1; never touch encoder weights."""
        owned = (
            self.context,
            self.lateral,
            self.refine,
            self.high_res_refine,
            self.depth_head,
            self.confidence_head,
            self.ground_head,
            self.height_head,
            self.clearance_head,
            self.feature_projections,
        )
        for root in owned:
            for module in root.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.depth_head[-1].bias, math.log(math.expm1(2.0)))
        nn.init.constant_(self.height_head.bias, 1.2)
        nn.init.constant_(self.clearance_head.bias, 2.0)

    def forward(self, image: torch.Tensor, output_size: tuple[int, int] | None = None) -> dict[str, Any]:
        value = image
        taps: list[torch.Tensor] = []
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in (3, 6, 12, 16):
                taps.append(value)
        if len(taps) != 4 or tuple(t.shape[1] for t in taps) != self.tap_channels:
            raise RuntimeError(f"S1.1 tap topology drift: {[tuple(t.shape) for t in taps]}")
        features: list[torch.Tensor] = []
        top = self.refine[3](self.lateral[3](self.context(taps[3])))
        features.append(top)
        for index in (2, 1, 0):
            top = self.refine[index](
                F.interpolate(top, size=taps[index].shape[-2:], mode="bilinear", align_corners=False)
                + self.lateral[index](taps[index])
            )
            features.append(top)
        top = self.high_res_refine(top)
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
            "features": tuple(reversed(features)),
        }


def normalize_bgr_batch(images: list[torch.Tensor]) -> torch.Tensor:
    batch = torch.stack(images).float() / 255.0
    batch = batch[:, [2, 1, 0]]
    batch = F.interpolate(batch, size=(S11Config.input_size,) * 2, mode="bilinear", align_corners=False)
    mean = batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (batch - mean) / std


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _masked_pairwise_gradient(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    if values.shape[-1] > 1:
        mask_x = valid[..., 1:] & valid[..., :-1]
        if mask_x.any():
            losses.append(F.l1_loss(torch.diff(values, dim=-1)[mask_x], torch.zeros_like(torch.diff(values, dim=-1)[mask_x])))
    if values.shape[-2] > 1:
        mask_y = valid[..., 1:, :] & valid[..., :-1, :]
        if mask_y.any():
            losses.append(F.l1_loss(torch.diff(values, dim=-2)[mask_y], torch.zeros_like(torch.diff(values, dim=-2)[mask_y])))
    return torch.stack(losses).mean() if losses else values.sum() * 0.0


def confidence_masked_geometry_loss(
    pred: dict[str, torch.Tensor],
    truth: torch.Tensor,
    confidence: torch.Tensor,
    teacher: torch.Tensor | None = None,
    config: S11Config | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Metric-depth loss with pixel-valid confidence masking.

    ``confidence`` is expected to contain the source confidence labels; only
    ``confidence == confidence_equals`` contributes to metric, gradient, and
    scale terms. Undefined batches fail closed through a zero-valued loss and
    an explicit valid-fraction component.
    """
    cfg = config or S11Config()
    lo, hi = cfg.depth_min_m, cfg.depth_max_m
    p = pred["metric_depth"].float().clamp(lo, hi)
    t = truth.float()
    valid = torch.isfinite(t) & (confidence == cfg.confidence_equals) & (t >= lo) & (t <= hi)
    safe_t = t.clamp(lo, hi)
    lp, lt = torch.log(p), torch.log(safe_t)
    depth = F.smooth_l1_loss(lp[valid], lt[valid], beta=0.05) if valid.any() else lp.sum() * 0.0
    grad_terms: list[torch.Tensor] = []
    for axis in (-1, -2):
        if lp.shape[axis] > 1:
            dp, dt = torch.diff(lp, dim=axis), torch.diff(lt, dim=axis)
            mask = (valid[..., 1:] & valid[..., :-1]) if axis == -1 else (valid[..., 1:, :] & valid[..., :-1, :])
            if mask.any():
                grad_terms.append(F.l1_loss(dp[mask], dt[mask]))
    gradient = torch.stack(grad_terms).mean() if grad_terms else lp.sum() * 0.0
    ratios = (lp - lt)[valid]
    scale = ratios.abs().mean() if ratios.numel() else lp.sum() * 0.0
    teacher_loss = lp.sum() * 0.0
    if teacher is not None:
        teach = teacher.float().clamp(lo, hi)
        teacher_loss = F.smooth_l1_loss(lp, torch.log(teach), beta=0.05)
    valid_fraction = valid.float().mean()
    total = depth + 0.20 * gradient + 0.15 * scale + 0.25 * teacher_loss
    return total, {
        "log_depth": depth,
        "gradient": gradient,
        "scale": scale,
        "teacher_depth": teacher_loss,
        "valid_fraction": valid_fraction,
    }


def _channel_normalize(feature: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = feature.mean(dim=(-2, -1), keepdim=True)
    std = feature.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (feature - mean) / std


def feature_distillation_loss(
    student_features: tuple[torch.Tensor, ...],
    teacher_features: tuple[torch.Tensor, ...],
    projections: nn.ModuleList,
) -> torch.Tensor:
    """Per-scale channel-normalized cosine + L1 feature distillation."""
    if len(student_features) != len(teacher_features) or len(student_features) != len(projections):
        raise ValueError("S1.1 feature distillation requires matching feature scales")
    losses: list[torch.Tensor] = []
    for student, teacher, projection in zip(student_features, teacher_features, projections, strict=True):
        projected = _channel_normalize(projection(student))
        target = _channel_normalize(teacher.detach())
        target = F.interpolate(target, size=projected.shape[-2:], mode="bilinear", align_corners=False)
        cosine = 1.0 - F.cosine_similarity(projected, target, dim=1).mean()
        losses.append(0.5 * cosine + 0.5 * F.l1_loss(projected, target))
    return torch.stack(losses).mean()


def require_finite_metrics(metrics: dict[str, float]) -> None:
    invalid = [k for k, v in metrics.items() if not isinstance(v, (int, float)) or not math.isfinite(float(v))]
    if invalid:
        raise ValueError(f"NON_FINITE_OR_UNDEFINED_METRICS: {sorted(invalid)}")


__all__ = [
    "S11Config",
    "ClearanceStudentMobileS11",
    "encoder_binding_digest",
    "encoder_binding_receipt",
    "confidence_masked_geometry_loss",
    "feature_distillation_loss",
    "normalize_bgr_batch",
    "parameter_count",
    "require_finite_metrics",
]
