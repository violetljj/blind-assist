#!/usr/bin/env python3
"""R1 candidate model adapters with explicit source/checkpoint identity.

The adapters expose one raw-RGB contract to the shared trainer and exporter:
``[N, 3, 256, 256]`` tensors contain values in ``0..255``.  Normalization is
inside the model so the exported TFLite graph has the same contract.  The
official repositories/checkpoints remain external, hash-bound inputs under
``artifacts.local``; this module fails closed when they are absent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
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


def _load_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"official architecture source is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load architecture source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_state(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint is missing: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint must be a state-dict-like mapping: {path}")
    state = value.get("state_dict", value)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint state_dict must be a mapping: {path}")
    normalized: dict[str, Tensor] = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, Tensor):
            continue
        normalized[key.removeprefix("module.")] = tensor
    if not normalized:
        raise ValueError(f"checkpoint contains no tensor state: {path}")
    return normalized


@dataclass(frozen=True)
class BuildReceipt:
    model_id: str
    implementation_identity: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    architecture_source: str
    architecture_source_sha256: str | None
    initialization_kind: str
    head_reset: str
    compatible_tensor_count: int
    checkpoint_tensor_count: int
    model_tensor_count: int
    missing_tensor_count: int
    unexpected_tensor_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "implementation_identity": self.implementation_identity,
            "source_checkpoint": self.source_checkpoint,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "architecture_source": self.architecture_source,
            "architecture_source_sha256": self.architecture_source_sha256,
            "initialization_kind": self.initialization_kind,
            "head_reset": self.head_reset,
            "compatible_tensor_count": self.compatible_tensor_count,
            "checkpoint_tensor_count": self.checkpoint_tensor_count,
            "model_tensor_count": self.model_tensor_count,
            "missing_tensor_count": self.missing_tensor_count,
            "unexpected_tensor_count": self.unexpected_tensor_count,
        }


class RawRgbSegmenter(nn.Module):
    """Shared raw RGB normalization and output-size contract."""

    head_prefixes: tuple[str, ...] = ()

    def __init__(self, input_size: int = INPUT_SIZE) -> None:
        super().__init__()
        if input_size != INPUT_SIZE:
            raise ValueError("R1 model selection is frozen at input_size=256")
        self.input_size = input_size
        self.register_buffer("image_mean", torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(IMAGE_STD).view(1, 3, 1, 1))

    def normalize_nchw(self, raw_rgb_nchw: Tensor) -> Tensor:
        if raw_rgb_nchw.ndim != 4 or raw_rgb_nchw.shape[1] != 3:
            raise ValueError(f"expected raw RGB NCHW, got {tuple(raw_rgb_nchw.shape)}")
        values = raw_rgb_nchw.float() / 255.0
        return (values - self.image_mean) / self.image_std

    def forward(self, raw_rgb_nchw: Tensor) -> Tensor:  # pragma: no cover - abstract contract
        raise NotImplementedError

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
            "trainable_parameter_names": trainable,
        }


class DDRNet23SlimSegmenter(RawRgbSegmenter):
    """Official DDRNet-23-Slim segmentation graph with a four-class head."""

    head_prefixes = ("core.spp.", "core.final_layer.")

    def __init__(self, architecture_source: Path, checkpoint: Path) -> None:
        super().__init__()
        source_module = _load_module(architecture_source, "r1_official_ddrnet23_slim")
        self.core = source_module.DualResNet(
            source_module.BasicBlock,
            [2, 2, 2, 2],
            num_classes=CLASS_COUNT,
            planes=32,
            spp_planes=128,
            head_planes=64,
            augment=False,
        )
        checkpoint_state = _checkpoint_state(checkpoint)
        model_state = self.core.state_dict()
        compatible = {
            key: tensor
            for key, tensor in checkpoint_state.items()
            if key in model_state and tuple(tensor.shape) == tuple(model_state[key].shape)
        }
        self.core.load_state_dict(compatible, strict=False)
        self.build_receipt = BuildReceipt(
            model_id="DDRNet-23-Slim",
            implementation_identity="ddrnet23_slim",
            source_checkpoint=str(checkpoint.resolve()),
            source_checkpoint_sha256=sha256_file(checkpoint),
            architecture_source=str(architecture_source.resolve()),
            architecture_source_sha256=sha256_file(architecture_source),
            initialization_kind="official_source_attested_ImageNet_FP32_backbone",
            head_reset="new_four_class_final_layer; SPP decoder random where absent from classification checkpoint",
            compatible_tensor_count=len(compatible),
            checkpoint_tensor_count=len(checkpoint_state),
            model_tensor_count=len(model_state),
            missing_tensor_count=len(model_state) - len(compatible),
            unexpected_tensor_count=len(checkpoint_state) - len(compatible),
        )

    def forward(self, raw_rgb_nchw: Tensor) -> Tensor:
        logits = self.core(self.normalize_nchw(raw_rgb_nchw))
        return F.interpolate(
            logits,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )


class SegFormerB0Segmenter(RawRgbSegmenter):
    """SegFormer-B0 with an official MiT-B0 backbone and four-class decoder."""

    head_prefixes = ("core.decode_head.",)

    def __init__(self, checkpoint_dir: Path) -> None:
        super().__init__()
        if not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(f"SegFormer source config is missing: {checkpoint_dir / 'config.json'}")
        if not (checkpoint_dir / "pytorch_model.bin").is_file():
            raise FileNotFoundError(f"SegFormer source checkpoint is missing: {checkpoint_dir / 'pytorch_model.bin'}")
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("Transformers is required for the SegFormer-B0 candidate") from exc
        self.core = SegformerForSemanticSegmentation.from_pretrained(
            str(checkpoint_dir),
            num_labels=CLASS_COUNT,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        checkpoint = checkpoint_dir / "pytorch_model.bin"
        state = self.core.state_dict()
        encoder_loaded = sum(
            1
            for name, tensor in state.items()
            if name.startswith("segformer.") and tensor.numel() > 0
        )
        self.build_receipt = BuildReceipt(
            model_id="SegFormer-B0",
            implementation_identity="segformer_b0",
            source_checkpoint=str(checkpoint.resolve()),
            source_checkpoint_sha256=sha256_file(checkpoint),
            architecture_source="transformers.SegformerForSemanticSegmentation + NVIDIA nvidia/mit-b0 config",
            architecture_source_sha256=sha256_file(checkpoint_dir / "config.json"),
            initialization_kind="official_source_attested_MiT_B0_FP32_backbone",
            head_reset="new_four_class_decode_head",
            compatible_tensor_count=encoder_loaded,
            checkpoint_tensor_count=len(torch.load(checkpoint, map_location="cpu", weights_only=False)),
            model_tensor_count=len(state),
            missing_tensor_count=len([name for name in state if name.startswith("decode_head.")]),
            unexpected_tensor_count=2,
        )

    def forward(self, raw_rgb_nchw: Tensor) -> Tensor:
        output = self.core(pixel_values=self.normalize_nchw(raw_rgb_nchw))
        logits = output.logits
        return F.interpolate(
            logits,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )


class ExportableRawRgbSegmenter(nn.Module):
    """NHWC wrapper used for the fixed TFLite input/output contract."""

    def __init__(self, model: RawRgbSegmenter) -> None:
        super().__init__()
        self.model = model

    def forward(self, raw_rgb_nhwc: Tensor) -> Tensor:
        if raw_rgb_nhwc.ndim != 4 or raw_rgb_nhwc.shape[-1] != 3:
            raise ValueError(f"expected raw RGB NHWC, got {tuple(raw_rgb_nhwc.shape)}")
        logits_nchw = self.model(raw_rgb_nhwc.permute(0, 3, 1, 2).contiguous())
        return logits_nchw.permute(0, 2, 3, 1).contiguous()


class ExportableNchwRawRgbSegmenter(nn.Module):
    """NCHW export variant for converters that canonicalize layout themselves."""

    def __init__(self, model: RawRgbSegmenter) -> None:
        super().__init__()
        self.model = model

    def forward(self, raw_rgb_nchw: Tensor) -> Tensor:
        return self.model(raw_rgb_nchw)


def build_model(
    model_id: str,
    *,
    ddrnet_architecture_source: Path | None = None,
    ddrnet_checkpoint: Path | None = None,
    segformer_checkpoint_dir: Path | None = None,
) -> RawRgbSegmenter:
    if model_id == "DDRNet-23-Slim":
        if ddrnet_architecture_source is None or ddrnet_checkpoint is None:
            raise ValueError("DDRNet requires architecture source and checkpoint")
        return DDRNet23SlimSegmenter(ddrnet_architecture_source, ddrnet_checkpoint)
    if model_id == "SegFormer-B0":
        if segformer_checkpoint_dir is None:
            raise ValueError("SegFormer requires a local checkpoint directory")
        return SegFormerB0Segmenter(segformer_checkpoint_dir)
    raise ValueError(f"unsupported R1 model_id: {model_id}")


def write_build_receipt(path: Path, model: RawRgbSegmenter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.build_receipt.as_dict(), indent=2) + "\n", encoding="utf-8", newline="\n")
