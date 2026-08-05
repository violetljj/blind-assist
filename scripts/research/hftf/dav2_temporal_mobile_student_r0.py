#!/usr/bin/env python3
"""Frozen A3 MobileNetV3-Small temporal metric-depth student."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvNormAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(inplace=True),
        )


class TemporalMobileDepthStudent(nn.Module):
    input_height = 294
    input_width = 392

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights).features
        self.lateral3 = nn.Conv2d(24, 96, kernel_size=1)
        self.lateral4 = nn.Conv2d(48, 96, kernel_size=1)
        self.lateral5 = nn.Conv2d(576, 96, kernel_size=1)
        self.refine5 = ConvNormAct(96, 96)
        self.refine4 = ConvNormAct(96, 96)
        self.refine3 = ConvNormAct(96, 96)
        self.output_refine = nn.Sequential(
            ConvNormAct(96, 32),
            ConvNormAct(32, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        decoder_modules = (
            self.lateral3,
            self.lateral4,
            self.lateral5,
            self.refine5,
            self.refine4,
            self.refine3,
            self.output_refine,
        )
        for module in decoder_modules:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        output = self.output_refine[-1]
        desired_depth = 3.0 - 0.1
        output.bias.data.fill_(math.log(math.expm1(desired_depth)))

    def forward(
        self, image: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        feature3 = None
        feature4 = None
        feature5 = None
        value = image
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index == 2:
                feature3 = value
            elif index == 7:
                feature4 = value
            elif index == 12:
                feature5 = value
        if feature3 is None or feature4 is None or feature5 is None:
            raise RuntimeError("MobileNet feature taps were not produced")
        pyramid5 = self.refine5(self.lateral5(feature5))
        pyramid4 = self.refine4(
            functional.interpolate(
                pyramid5,
                size=feature4.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + self.lateral4(feature4)
        )
        pyramid3 = self.refine3(
            functional.interpolate(
                pyramid4,
                size=feature3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + self.lateral3(feature3)
        )
        full = functional.interpolate(
            pyramid3,
            size=(self.input_height, self.input_width),
            mode="bilinear",
            align_corners=False,
        )
        logits = self.output_refine(full)
        depth = functional.softplus(logits[:, 0]) + 0.1
        return functional.interpolate(
            depth[:, None],
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )[:, 0]


def normalize_bgr_batch(images_bgr: list[torch.Tensor]) -> torch.Tensor:
    """Convert uint8 BGR CHW tensors to normalized RGB student input."""
    batch = torch.stack(images_bgr).float() / 255.0
    batch = batch[:, [2, 1, 0]]
    batch = functional.interpolate(
        batch,
        size=(TemporalMobileDepthStudent.input_height, TemporalMobileDepthStudent.input_width),
        mode="bilinear",
        align_corners=False,
    )
    mean = batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (batch - mean) / std


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_torchvision_encoder_weights(
    model: TemporalMobileDepthStudent, weights_path: Path
) -> None:
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    feature_state = {
        key.removeprefix("features."): value
        for key, value in state.items()
        if key.startswith("features.")
    }
    model.encoder.load_state_dict(feature_state, strict=True)
