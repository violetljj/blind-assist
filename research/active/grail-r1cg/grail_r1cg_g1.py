#!/usr/bin/env python3
"""Matched single/multiview RGB-mask model for R1C-G1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


IMAGE_SIZE = 224
PATCH_SIZE = 14
MODES = ("PRESERVE", "FLIP")
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _letterbox(image: Image.Image, fill: int | tuple[int, int, int]) -> Image.Image:
    image = image.copy()
    resampling = Image.Resampling.BICUBIC if image.mode == "RGB" else Image.Resampling.NEAREST
    image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), resampling)
    canvas = Image.new(image.mode, (IMAGE_SIZE, IMAGE_SIZE), fill)
    canvas.paste(image, ((IMAGE_SIZE - image.width) // 2, (IMAGE_SIZE - image.height) // 2))
    return canvas


def load_view(root: Path, view: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    rgb_image = _letterbox(Image.open(root / view["rgb"]).convert("RGB"), (255, 255, 255))
    rgb = torch.from_numpy(np.asarray(rgb_image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    rgb = (rgb - DINO_MEAN) / DINO_STD
    masks = []
    for key in ("owner_union_mask", "sibling_centroid_mask"):
        mask_image = _letterbox(Image.open(root / view[key]).convert("L"), 0)
        masks.append(torch.from_numpy(np.asarray(mask_image, dtype=np.float32).copy()) / 255.0)
    return rgb, torch.stack(masks)


class G1Collection(Dataset[dict[str, Any]]):
    def __init__(self, collection_path: Path, root: Path, arm: str, discriminative_only: bool):
        if arm not in ("b1_single", "g1_triplet"):
            raise ValueError(f"Unknown R1C-G1 arm: {arm}")
        self.collection_path = collection_path
        self.root = root
        self.arm = arm
        self.collection = json.loads(collection_path.read_text(encoding="utf-8"))
        if self.collection.get("schema") != "blindassist_grail_r1c_g1_collection_v1":
            raise ValueError("R1C-G1 collection schema mismatch")
        self.views = {row["view_id"]: row for row in self.collection["views"]}
        samples = self.collection["samples"]
        self.samples = [row for row in samples if len(row["valid_slot_modes"]) == 1] \
            if discriminative_only else samples
        for sample in self.samples:
            references = [sample["anchor_view_id"]] if arm == "b1_single" else sample["reference_view_ids"]
            missing = [view_id for view_id in references + [sample["query_view_id"]] if view_id not in self.views]
            if missing:
                raise ValueError(f"R1C-G1 sample references missing views: {sample['sample_id']} {missing}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        reference_ids = [sample["anchor_view_id"]] if self.arm == "b1_single" else sample["reference_view_ids"]
        loaded = [load_view(self.root, self.views[view_id]) for view_id in reference_ids]
        query_rgb, query_masks = load_view(self.root, self.views[sample["query_view_id"]])
        valid = torch.tensor([mode in sample["valid_slot_modes"] for mode in MODES], dtype=torch.bool)
        return {
            "reference_rgb": torch.stack([row[0] for row in loaded]),
            "reference_masks": torch.stack([row[1] for row in loaded]),
            "query_rgb": query_rgb,
            "query_masks": query_masks,
            "valid_modes": valid,
            "object_type": sample["object_type"],
            "group_id": sample["group_id"],
            "house_index": int(sample["house_index"]),
            "sample_id": sample["sample_id"],
        }


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, hidden_size: int = 384, heads: int = 6):
        super().__init__()
        self.reference_attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.query_attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.reference_norm1 = nn.LayerNorm(hidden_size)
        self.query_norm1 = nn.LayerNorm(hidden_size)
        self.reference_norm2 = nn.LayerNorm(hidden_size)
        self.query_norm2 = nn.LayerNorm(hidden_size)
        self.reference_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size)
        )
        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, reference: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reference_delta = self.reference_attention(reference, query, query, need_weights=False)[0]
        query_delta = self.query_attention(query, reference, reference, need_weights=False)[0]
        reference = self.reference_norm1(reference + reference_delta)
        query = self.query_norm1(query + query_delta)
        reference = self.reference_norm2(reference + self.reference_ffn(reference))
        query = self.query_norm2(query + self.query_ffn(query))
        return reference, query


class ActiveMultiviewPermutation(nn.Module):
    """The parameterization is identical for V=1 and V=3; only observations differ."""

    def __init__(self, backbone_path: Path):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(backbone_path, local_files_only=True)
        hidden = int(self.backbone.config.hidden_size)
        if hidden != 384 or int(self.backbone.config.patch_size) != PATCH_SIZE:
            raise ValueError("R1C-G1 requires the pinned DINOv2-S hidden=384 patch=14 backbone")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for layer in self.backbone.encoder.layer[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.mask_adapter = nn.Conv2d(2, hidden, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.input_norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.ModuleList([BidirectionalCrossAttention(hidden, 6) for _ in range(2)])
        evidence_size = hidden * 4
        self.head = nn.Sequential(
            nn.LayerNorm(evidence_size * 2),
            nn.Linear(evidence_size * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(MODES)),
        )

    def encode(self, rgb: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(pixel_values=rgb, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
        mask_tokens = self.mask_adapter(masks).flatten(2).transpose(1, 2)
        if tokens.shape != mask_tokens.shape:
            raise ValueError(f"R1C-G1 mask/RGB token mismatch {mask_tokens.shape} != {tokens.shape}")
        return self.input_norm(tokens + mask_tokens)

    def pair_evidence(self, reference: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        for block in self.cross_attention:
            reference, query = block(reference, query)
        reference_pool, query_pool = reference.mean(dim=1), query.mean(dim=1)
        return torch.cat(
            [reference_pool, query_pool, torch.abs(reference_pool - query_pool), reference_pool * query_pool],
            dim=-1,
        )

    def forward(self, reference_rgb: torch.Tensor, reference_masks: torch.Tensor,
                query_rgb: torch.Tensor, query_masks: torch.Tensor) -> torch.Tensor:
        batch, views = reference_rgb.shape[:2]
        reference_tokens = self.encode(
            reference_rgb.flatten(0, 1), reference_masks.flatten(0, 1)
        )
        query_tokens = self.encode(query_rgb, query_masks)
        repeated_query = query_tokens[:, None].expand(-1, views, -1, -1).flatten(0, 1)
        evidence = self.pair_evidence(reference_tokens, repeated_query).reshape(batch, views, -1)
        aggregated = torch.cat([evidence.mean(dim=1), evidence.max(dim=1).values], dim=-1)
        return self.head(aggregated)


def set_valued_cross_entropy(logits: torch.Tensor, valid_modes: torch.Tensor) -> torch.Tensor:
    if not torch.all(valid_modes.any(dim=1)):
        raise ValueError("R1C-G1 sample has no valid slot mode")
    selected = F.log_softmax(logits.float(), dim=-1).masked_fill(~valid_modes, float("-inf"))
    return -torch.logsumexp(selected, dim=-1).mean()


def predicted_mode_indices(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().argmax(dim=-1)


def prediction_correct(logits: torch.Tensor, valid_modes: torch.Tensor) -> torch.Tensor:
    return valid_modes.gather(1, predicted_mode_indices(logits)[:, None]).squeeze(1)
