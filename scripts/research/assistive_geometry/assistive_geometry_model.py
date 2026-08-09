#!/usr/bin/env python3
"""DepthART-backed Assistive Geometry heads and frozen B1 loss functions."""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


BANDS = 3
HORIZONS = 3


class AssistiveTaskHeads(nn.Module):
    def __init__(self, channels: int = 48, hidden: int = 32) -> None:
        super().__init__()
        self.ground_pre = nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(inplace=False))
        self.ground_out = nn.Conv2d(16, 1, 1)
        self.band_mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=False))
        self.clearance_out = nn.Linear(hidden, 1)
        self.occupancy_out = nn.Linear(hidden, HORIZONS)
        self.confidence_out = nn.Linear(hidden, 1)

    @staticmethod
    def band_pool(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError("shared feature must be BCHW")
        width = feature.shape[-1]
        boundaries = (0, width // 3, (2 * width) // 3, width)
        if not (boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3]):
            raise ValueError("feature width is too small for three bands")
        return torch.stack(
            [feature[..., boundaries[index] : boundaries[index + 1]].mean(dim=(-2, -1)) for index in range(3)],
            dim=1,
        )

    def forward(self, feature: torch.Tensor, output_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        ground = self.ground_pre(feature)
        ground = F.interpolate(ground, output_hw, mode="bilinear", align_corners=True)
        ground_logits = self.ground_out(ground)
        bands = self.band_mlp(self.band_pool(feature))
        return {
            "ground_logits": ground_logits,
            "clearance_m": F.softplus(self.clearance_out(bands)).squeeze(-1),
            "occupancy_logits": self.occupancy_out(bands),
            "confidence_logits": self.confidence_out(bands).squeeze(-1),
        }


class DepthArtAssistiveGeometry(nn.Module):
    """Reuse a loaded metric DepthART decoder and expose its stride-4 feature."""

    def __init__(self, metric_depthart: nn.Module) -> None:
        super().__init__()
        self.metric_depthart = metric_depthart
        self.assistive_heads = AssistiveTaskHeads(channels=48)

    def _decode(self, features: list[torch.Tensor], output_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.metric_depthart.depth_head
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1 = head.scratch.layer1_rn(layer_1)
        layer_2 = head.scratch.layer2_rn(layer_2)
        layer_3 = head.scratch.layer3_rn(layer_3)
        layer_4 = head.scratch.layer4_rn(layer_4)
        path_4 = head.scratch.refinenet4(layer_4, size=layer_3.shape[2:])
        path_3 = head.scratch.refinenet3(path_4, layer_3, size=layer_2.shape[2:])
        path_2 = head.scratch.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
        shared = head.scratch.refinenet1(path_2, layer_1)
        depth = head.scratch.output_conv1(shared)
        depth = F.interpolate(depth, output_hw, mode="bilinear", align_corners=True)
        depth = head.scratch.output_conv2(depth)
        return depth, shared

    def forward(self, image: torch.Tensor, intrinsics: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = image.shape[-2:]
        cameras = self.metric_depthart.cam_embedder(intrinsics, height, width, image.device)
        features = self.metric_depthart.pretrained.forward_with_adapters(
            image,
            adapters=[self.metric_depthart.daa1, self.metric_depthart.daa2, self.metric_depthart.daa3, self.metric_depthart.daa4],
            cams=list(cameras),
        )
        relative_depth, shared = self._decode(list(features), (height, width))
        scale = self.metric_depthart.sfh(features[3], cameras[3])
        depth_m = relative_depth * scale.view(-1, 1, 1, 1) * self.metric_depthart.max_depth
        outputs = self.assistive_heads(shared, (height, width))
        outputs["dense_depth_m"] = depth_m
        outputs["interface_confidence"] = torch.sigmoid(outputs["confidence_logits"])[..., None].expand(-1, -1, HORIZONS)
        return outputs


def near_field_weights(depth_m: torch.Tensor) -> torch.Tensor:
    return torch.where(depth_m <= 2.0, 3.0, torch.where(depth_m <= 5.0, 2.0, 1.0)).to(depth_m.dtype)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    selected = mask.to(torch.bool)
    if not bool(selected.any()):
        return value.sum() * 0.0
    if weight is None:
        return value[selected].mean()
    numerator = (value * weight)[selected].sum()
    return numerator / weight[selected].sum().clamp_min(1e-12)


def confidence_correctness(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_occupied = torch.sigmoid(outputs["occupancy_logits"].detach()) >= 0.5
    truth_occupied = targets["occupancy"] >= 0.5
    occupancy_valid = targets["occupancy_valid"].bool()
    all_occupancy_valid = occupancy_valid.all(dim=-1)
    occupancy_correct = ((predicted_occupied == truth_occupied) | ~occupancy_valid).all(dim=-1)
    observed = targets["clearance_valid"].bool()
    clearance_correct = (outputs["clearance_m"].detach() - targets["clearance_m"]).abs() <= 0.25
    censored_clear = all_occupancy_valid & (~truth_occupied).all(dim=-1)
    correct = occupancy_correct & torch.where(observed, clearance_correct, censored_clear)
    valid = observed | censored_clear
    return correct.to(outputs["confidence_logits"].dtype), valid


def ground_plane_depth_loss(predicted_depth: torch.Tensor, targets: dict[str, torch.Tensor]) -> torch.Tensor:
    batch, _, height, width = predicted_depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=predicted_depth.device, dtype=predicted_depth.dtype),
        torch.arange(width, device=predicted_depth.device, dtype=predicted_depth.dtype),
        indexing="ij",
    )
    intrinsics = targets["intrinsics_tensor"]
    x = (xs[None] - intrinsics[:, 0, 2, None, None]) / intrinsics[:, 0, 0, None, None]
    y = (ys[None] - intrinsics[:, 1, 2, None, None]) / intrinsics[:, 1, 1, None, None]
    z = predicted_depth[:, 0]
    points = torch.stack((x * z, y * z, z), dim=-1)
    up = targets["up_camera"][:, None, None, :]
    heights = (points * up).sum(dim=-1) + targets["camera_height_m"][:, None, None]
    mask = targets["ground_probability"][:, 0] >= 0.5
    mask &= targets["ground_label_valid"][:, 0].bool()
    mask &= targets["ground_plane_valid"][:, None, None].bool()
    return _masked_mean(heights.abs(), mask)


LOSS_LAMBDAS = {
    "masked_log_depth": 1.0,
    "valid_neighbor_log_gradient": 0.5,
    "ground_bce": 0.5,
    "ground_plane_depth": 0.25,
    "clearance_huber": 1.0,
    "occupancy_bce": 1.0,
    "false_clear_extra": 2.0,
    "confidence_bce": 0.5,
}


def compute_b1_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    active_losses: Iterable[str],
) -> dict[str, torch.Tensor]:
    active = tuple(active_losses)
    unknown = set(active) - set(LOSS_LAMBDAS)
    if unknown:
        raise ValueError(f"unknown B1 losses: {sorted(unknown)}")
    predicted = outputs["dense_depth_m"].clamp_min(1e-4)
    truth = targets["dense_depth_m"].clamp_min(1e-4)
    valid = targets["depth_valid"].bool()
    weights = near_field_weights(truth)
    raw: dict[str, torch.Tensor] = {}
    raw["masked_log_depth"] = _masked_mean((predicted.log() - truth.log()).abs(), valid, weights)
    log_error = predicted.log() - truth.log()
    horizontal_valid = valid[..., :, 1:] & valid[..., :, :-1]
    vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
    horizontal = _masked_mean((log_error[..., :, 1:] - log_error[..., :, :-1]).abs(), horizontal_valid)
    vertical = _masked_mean((log_error[..., 1:, :] - log_error[..., :-1, :]).abs(), vertical_valid)
    raw["valid_neighbor_log_gradient"] = 0.5 * (horizontal + vertical)
    raw["ground_bce"] = _masked_mean(
        F.binary_cross_entropy_with_logits(outputs["ground_logits"], targets["ground_probability"], reduction="none"),
        targets["ground_label_valid"].bool(),
    )
    raw["ground_plane_depth"] = ground_plane_depth_loss(predicted, targets)
    raw["clearance_huber"] = _masked_mean(
        F.smooth_l1_loss(outputs["clearance_m"], targets["clearance_m"], reduction="none", beta=0.25),
        targets["clearance_valid"].bool(),
    )
    raw["occupancy_bce"] = _masked_mean(
        F.binary_cross_entropy_with_logits(outputs["occupancy_logits"], targets["occupancy"], reduction="none"),
        targets["occupancy_valid"].bool(),
    )
    raw["false_clear_extra"] = _masked_mean(
        F.softplus(-outputs["occupancy_logits"]),
        targets["occupancy_valid"].bool() & (targets["occupancy"] >= 0.5),
    )
    confidence_target, confidence_valid = confidence_correctness(outputs, targets)
    raw["confidence_bce"] = _masked_mean(
        F.binary_cross_entropy_with_logits(outputs["confidence_logits"], confidence_target, reduction="none"),
        confidence_valid,
    )
    weighted = {name: raw[name] * LOSS_LAMBDAS[name] for name in active}
    total = sum(weighted.values(), predicted.sum() * 0.0)
    return {"total": total, **{f"raw/{name}": raw[name] for name in active}, **{f"weighted/{name}": value for name, value in weighted.items()}}


def horizontal_flip_batch(targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = {key: value.clone() for key, value in targets.items()}
    for key in ("dense_depth_m", "depth_valid", "ground_probability", "ground_label_valid"):
        output[key] = torch.flip(output[key], dims=(-1,))
    output["intrinsics_tensor"][:, 0, 2] = targets["dense_depth_m"].shape[-1] - 1 - targets["intrinsics_tensor"][:, 0, 2]
    output["up_camera"][:, 0] = -targets["up_camera"][:, 0]
    for key in ("clearance_m", "clearance_valid"):
        output[key] = torch.flip(output[key], dims=(1,))
    for key in ("occupancy", "occupancy_valid"):
        output[key] = torch.flip(output[key], dims=(1,))
    return output
