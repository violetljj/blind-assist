"""Self-contained DDRNet-23-Slim adapter for the FP-aware successor.

The historical R1 module is archived and cannot be called as a successor
implementation. This file therefore preserves the same official architecture,
raw-RGB normalization, four-class head, and source-checkpoint loading contract
without importing the archived module.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CLASS_COUNT = 4
INPUT_SIZE = 256
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"official DDRNet architecture source is missing: {path}")
    spec = importlib.util.spec_from_file_location("fp_aware_r0_official_ddrnet23_slim", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official DDRNet architecture source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_state(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    raw = payload.get("state_dict", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"checkpoint state_dict must be a mapping: {path}")
    state = {
        str(key).removeprefix("module."): value
        for key, value in raw.items()
        if isinstance(key, str) and isinstance(value, Tensor)
    }
    if not state:
        raise ValueError(f"checkpoint contains no tensors: {path}")
    return state


@dataclass(frozen=True)
class BuildReceipt:
    model_id: str
    implementation_identity: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    architecture_source: str
    architecture_source_sha256: str
    initialization_kind: str
    head_reset: str
    compatible_tensor_count: int
    checkpoint_tensor_count: int
    model_tensor_count: int
    missing_tensor_count: int
    unexpected_tensor_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DDRNet23SlimSegmenter(nn.Module):
    """Official DDRNet-23-Slim graph with the frozen four-class contract."""

    head_prefixes = ("core.spp.", "core.final_layer.")

    def __init__(self, architecture_source: Path, source_checkpoint: Path) -> None:
        super().__init__()
        module = _load_module(architecture_source)
        self.core = module.DualResNet(
            module.BasicBlock,
            [2, 2, 2, 2],
            num_classes=CLASS_COUNT,
            planes=32,
            spp_planes=128,
            head_planes=64,
            augment=False,
        )
        source_state = checkpoint_state(source_checkpoint)
        model_state = self.core.state_dict()
        compatible = {
            key: tensor
            for key, tensor in source_state.items()
            if key in model_state and tuple(tensor.shape) == tuple(model_state[key].shape)
        }
        self.core.load_state_dict(compatible, strict=False)
        self.register_buffer("image_mean", torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(IMAGE_STD).view(1, 3, 1, 1))
        self.build_receipt = BuildReceipt(
            model_id="DDRNet-23-Slim",
            implementation_identity="ddrnet23_slim_fp_aware_r0",
            source_checkpoint=str(source_checkpoint.resolve()),
            source_checkpoint_sha256=sha256_file(source_checkpoint),
            architecture_source=str(architecture_source.resolve()),
            architecture_source_sha256=sha256_file(architecture_source),
            initialization_kind="official_source_attested_ImageNet_FP32_backbone",
            head_reset="new_four_class_final_layer; SPP decoder random where absent from classification checkpoint",
            compatible_tensor_count=len(compatible),
            checkpoint_tensor_count=len(source_state),
            model_tensor_count=len(model_state),
            missing_tensor_count=len(model_state) - len(compatible),
            unexpected_tensor_count=len(source_state) - len(compatible),
        )

    def forward(self, raw_rgb_nchw: Tensor) -> Tensor:
        if raw_rgb_nchw.ndim != 4 or raw_rgb_nchw.shape[1] != 3:
            raise ValueError(f"expected raw RGB NCHW, got {tuple(raw_rgb_nchw.shape)}")
        normalized = (raw_rgb_nchw.float() / 255.0 - self.image_mean) / self.image_std
        logits = self.core(normalized)
        return F.interpolate(logits, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)

    def set_stage_trainability(self, *, head_only: bool) -> dict[str, Any]:
        trainable: list[str] = []
        frozen: list[str] = []
        for name, parameter in self.named_parameters():
            is_head = any(name.startswith(prefix) for prefix in self.head_prefixes)
            parameter.requires_grad = (not head_only) or is_head
            (trainable if parameter.requires_grad else frozen).append(name)
        return {
            "head_only": head_only,
            "trainable_parameter_count": len(trainable),
            "frozen_parameter_count": len(frozen),
            "head_prefixes": list(self.head_prefixes),
        }


def load_exact_checkpoint(model: nn.Module, checkpoint: Path) -> dict[str, Any]:
    state = checkpoint_state(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint tensor identity mismatch: missing={list(missing)[:5]} "
            f"unexpected={list(unexpected)[:5]}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload if isinstance(payload, dict) else {}
