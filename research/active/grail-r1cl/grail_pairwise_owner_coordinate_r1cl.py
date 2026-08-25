#!/usr/bin/env python3
"""Model, inputs, and symmetry-marginalized loss for GRAIL-R1C-L."""

from __future__ import annotations

import json
import math
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
YAW_BINS = 36
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _letterbox(image: Image.Image, fill: int | tuple[int, int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC if image.mode == "RGB" else Image.Resampling.NEAREST)
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


class PairCollection(Dataset[dict[str, Any]]):
    def __init__(self, collection_path: Path, root: Path):
        self.collection_path = collection_path
        self.root = root
        self.collection = json.loads(collection_path.read_text(encoding="utf-8"))
        self.views = {row["view_id"]: row for row in self.collection["views"]}
        self.pairs = self.collection["pairs"]
        missing = [row["pair_id"] for row in self.pairs
                   if row["reference_view_id"] not in self.views or row["query_view_id"] not in self.views]
        if missing:
            raise ValueError(f"R1C-L pair references missing views: {missing[:3]}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        reference_rgb, reference_masks = load_view(self.root, self.views[pair["reference_view_id"]])
        query_rgb, query_masks = load_view(self.root, self.views[pair["query_view_id"]])
        valid = torch.zeros(YAW_BINS, dtype=torch.bool)
        valid[pair["valid_bins"]] = True
        return {
            "reference_rgb": reference_rgb, "reference_masks": reference_masks,
            "query_rgb": query_rgb, "query_masks": query_masks, "valid_bins": valid,
            "object_type": pair["object_type"], "pair_id": pair["pair_id"],
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
        self.reference_ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
                                           nn.Linear(hidden_size * 4, hidden_size))
        self.query_ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
                                       nn.Linear(hidden_size * 4, hidden_size))

    def forward(self, reference: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reference_delta = self.reference_attention(reference, query, query, need_weights=False)[0]
        query_delta = self.query_attention(query, reference, reference, need_weights=False)[0]
        reference = self.reference_norm1(reference + reference_delta)
        query = self.query_norm1(query + query_delta)
        reference = self.reference_norm2(reference + self.reference_ffn(reference))
        query = self.query_norm2(query + self.query_ffn(query))
        return reference, query


class PairwiseOwnerCoordinate(nn.Module):
    def __init__(self, backbone_path: Path):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(backbone_path, local_files_only=True)
        hidden = int(self.backbone.config.hidden_size)
        if hidden != 384 or int(self.backbone.config.patch_size) != PATCH_SIZE:
            raise ValueError("R1C-L frozen architecture requires DINOv2-S hidden=384 patch=14")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for layer in self.backbone.encoder.layer[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.mask_adapter = nn.Conv2d(2, hidden, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.input_norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.ModuleList([BidirectionalCrossAttention(hidden, 6) for _ in range(2)])
        self.yaw_head = nn.Sequential(
            nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Linear(hidden, YAW_BINS)
        )

    def encode(self, rgb: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(pixel_values=rgb, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
        mask_tokens = self.mask_adapter(masks).flatten(2).transpose(1, 2)
        if mask_tokens.shape != tokens.shape:
            raise ValueError(f"R1C-L mask/RGB token mismatch {mask_tokens.shape} != {tokens.shape}")
        return self.input_norm(tokens + mask_tokens)

    def predict_tokens(self, reference: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        for block in self.cross_attention:
            reference, query = block(reference, query)
        reference_pool, query_pool = reference.mean(dim=1), query.mean(dim=1)
        features = torch.cat(
            [reference_pool, query_pool, torch.abs(reference_pool - query_pool), reference_pool * query_pool], dim=-1
        )
        return self.yaw_head(features)

    def forward(self, reference_rgb: torch.Tensor, reference_masks: torch.Tensor,
                query_rgb: torch.Tensor, query_masks: torch.Tensor,
                return_reverse: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        reference = self.encode(reference_rgb, reference_masks)
        query = self.encode(query_rgb, query_masks)
        forward = self.predict_tokens(reference, query)
        if not return_reverse:
            return forward
        return forward, self.predict_tokens(query, reference)


def slot_marginalized_loss(logits: torch.Tensor, valid_bins_mask: torch.Tensor) -> torch.Tensor:
    if not torch.all(valid_bins_mask.any(dim=1)):
        raise ValueError("R1C-L sample has no correct permutation bins")
    log_probability = F.log_softmax(logits.float(), dim=-1)
    selected = log_probability.masked_fill(~valid_bins_mask, float("-inf"))
    return -torch.logsumexp(selected, dim=-1).mean()


def exchange_consistency_loss(forward_logits: torch.Tensor, reverse_logits: torch.Tensor) -> torch.Tensor:
    inverse = torch.tensor([(-index) % YAW_BINS for index in range(YAW_BINS)], device=forward_logits.device)
    forward_log = F.log_softmax(forward_logits.float(), dim=-1)
    reverse_log = F.log_softmax(reverse_logits.float(), dim=-1).index_select(-1, inverse)
    forward_probability, reverse_probability = forward_log.exp(), reverse_log.exp()
    return 0.5 * (
        F.kl_div(forward_log, reverse_probability, reduction="batchmean")
        + F.kl_div(reverse_log, forward_probability, reduction="batchmean")
    )


def predicted_slot_modes(logits: torch.Tensor) -> torch.Tensor:
    probability = F.softmax(logits.float(), dim=-1)
    cosine = torch.tensor([math.cos(math.radians(index * 10.0)) for index in range(YAW_BINS)],
                          device=logits.device)
    preserve = probability[:, cosine > 1e-8].sum(dim=-1)
    flip = probability[:, cosine < -1e-8].sum(dim=-1)
    return preserve >= flip  # True=PRESERVE, False=FLIP


def slot_mode_correct(logits: torch.Tensor, valid_bins_mask: torch.Tensor) -> torch.Tensor:
    modes = predicted_slot_modes(logits)
    cosine = torch.tensor([math.cos(math.radians(index * 10.0)) for index in range(YAW_BINS)],
                          device=logits.device)
    preserve_valid = valid_bins_mask[:, cosine > 1e-8].any(dim=-1)
    flip_valid = valid_bins_mask[:, cosine < -1e-8].any(dim=-1)
    return torch.where(modes, preserve_valid, flip_valid)
