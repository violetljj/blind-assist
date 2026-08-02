#!/usr/bin/env python3
"""Train repairable single-frame/history HFTF Development students."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


ARMS = ("single", "history")
ARCHITECTURES = ("pooled", "directional", "grid")
TEMPORAL_MODES = (
    "joint",
    "current_residual",
    "current_spatial_residual",
    "early_pair",
)
OPTIMIZATION_MODES = (
    "all",
    "temporal_residual_only",
    "early_pair_only",
)
PAIR_CONSTRAINT_MODES = ("none", "future_body_head_recall")
KNOWN_LOSS_MODES = ("plain", "balanced", "sqrt_balanced")
SELECTION_MODES = ("aggregate", "environment_macro")
HORIZONS = ("current", "near", "far")
HEIGHTS = ("foot", "body", "head")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def decode_labels(
    record: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    known_rows = []
    risk_rows = []
    for horizon in HORIZONS:
        label = record["labels"][horizon]
        known = np.asarray(label["known_target"], dtype=np.float32)
        risk_object = np.asarray(
            label["risk_score_target_nullable"],
            dtype=object,
        )
        if known.shape != (3, 6, 6) or risk_object.shape != (3, 6, 6):
            raise ValueError("Each horizon label must have shape [3,6,6]")
        numeric = np.zeros(risk_object.shape, dtype=bool)
        risk = np.zeros(risk_object.shape, dtype=np.float32)
        for index in np.ndindex(risk_object.shape):
            value = risk_object[index]
            if value is None:
                continue
            number = float(value)
            if not 0.0 <= number <= 1.0:
                raise ValueError("Risk score must be in [0,1]")
            numeric[index] = True
            risk[index] = number
        if not np.array_equal(numeric, known.astype(bool)):
            raise ValueError("Risk score must be numeric iff cell is known")
        known_rows.append(known)
        risk_rows.append(risk)
    return (
        torch.from_numpy(np.stack(risk_rows)),
        torch.from_numpy(np.stack(known_rows)),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def augmentation(seed: int, epoch: int, sample_id: str) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{seed}:{epoch}:{sample_id}".encode("utf-8")
    ).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    operations = ["brightness", "contrast", "saturation", "hue"]
    generator.shuffle(operations)
    return {
        "horizontal_flip": generator.random() < 0.5,
        "brightness": generator.uniform(0.8, 1.2),
        "contrast": generator.uniform(0.8, 1.2),
        "saturation": generator.uniform(0.85, 1.15),
        "hue": generator.uniform(-0.02, 0.02),
        "operation_order": operations,
    }


def transform_image(
    image: Image.Image,
    parameters: dict[str, Any] | None,
) -> torch.Tensor:
    value = image
    if parameters is not None:
        if parameters["horizontal_flip"]:
            value = tvf.hflip(value)
        for operation in parameters["operation_order"]:
            value = getattr(tvf, f"adjust_{operation}")(
                value,
                parameters[operation],
            )
    tensor = tvf.pil_to_tensor(value).float().div_(255.0)
    return tvf.normalize(tensor, MEAN, STD)


class HftfDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        records: list[dict[str, Any]],
        arm: str,
        *,
        train: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.arm = arm
        self.train_mode = train
        self.seed = seed
        self.epoch = 0
        self.cache: dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def load_image(self, path: str) -> Image.Image:
        value = self.cache.get(path)
        if value is None:
            with Image.open(path) as source:
                value = tvf.resize(
                    source.convert("RGB"),
                    [128, 224],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
            self.cache[path] = value
        return value

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        history = record["history_rgb"]
        if len(history) != 5:
            raise ValueError("Five history frames are required")
        selected = [history[-1]] * 5 if self.arm == "single" else history
        parameters = (
            augmentation(self.seed, self.epoch, record["sample_id"])
            if self.train_mode
            else None
        )
        frames = [
            transform_image(self.load_image(item["image_path"]), parameters)
            for item in selected
        ]
        risk, known = decode_labels(record)
        if parameters is not None and parameters["horizontal_flip"]:
            risk = torch.flip(risk, dims=(2,))
            known = torch.flip(known, dims=(2,))
        return torch.stack(frames), risk, known


class DepthwiseSeparableDownsample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = 8 if output_channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(groups, output_channels),
            nn.Hardswish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class EarlyPairStem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                12,
                24,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(6, 24),
            nn.Hardswish(),
            DepthwiseSeparableDownsample(24, 32),
            DepthwiseSeparableDownsample(32, 64),
            DepthwiseSeparableDownsample(64, 128),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class TemporalStudent(nn.Module):
    def __init__(
        self,
        pretrained_path: Path,
        architecture: str = "pooled",
        temporal_mode: str = "joint",
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        if temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"Unknown temporal mode: {temporal_mode}")
        self.architecture = architecture
        self.temporal_mode = temporal_mode
        backbone = mobilenet_v3_small(weights=None)
        backbone.load_state_dict(
            torch.load(
                pretrained_path,
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        self.encoder = backbone.features
        self.temporal_depthwise = nn.Conv3d(
            576,
            576,
            kernel_size=(5, 1, 1),
            groups=576,
            bias=False,
        )
        if temporal_mode in {
            "current_residual",
            "current_spatial_residual",
        }:
            spatial_kernel = (
                (1, 1)
                if temporal_mode == "current_residual"
                else (3, 3)
            )
            spatial_padding = (
                (0, 0)
                if temporal_mode == "current_residual"
                else (1, 1)
            )
            self.temporal_residual = nn.Conv3d(
                576,
                576,
                kernel_size=(4, *spatial_kernel),
                padding=(0, *spatial_padding),
                groups=576,
                bias=False,
            )
            nn.init.zeros_(self.temporal_residual.weight)
        elif temporal_mode == "early_pair":
            self.early_pair_stem = EarlyPairStem()
            self.early_pair_output = nn.Conv2d(
                128,
                128,
                kernel_size=1,
                bias=False,
            )
            nn.init.zeros_(self.early_pair_output.weight)
        self.pointwise = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.Hardswish(),
        )
        self.dropout = nn.Dropout(0.2)
        if architecture == "pooled":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(128, 2 * 3 * 3 * 6 * 6)
        elif architecture == "directional":
            self.pool = nn.AdaptiveAvgPool2d((1, 6))
            self.head = nn.Conv1d(
                128,
                2 * 3 * 3 * 6,
                kernel_size=1,
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d((3, 6))
            self.head = nn.Conv2d(
                128,
                2 * 3 * 6,
                kernel_size=1,
            )

    def train(self, mode: bool = True) -> "TemporalStudent":
        super().train(mode)
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(
        self,
        frames: torch.Tensor,
        *,
        return_reference: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3):
            raise ValueError("Expected Bx5x3xHxW input")
        batch, time, channels, height, width = frames.shape
        temporal_mode = getattr(self, "temporal_mode", "joint")
        if temporal_mode == "early_pair":
            current = frames[:, -1]
            baseline = frames[:, :-1].mean(dim=1)
            delta = current - baseline
            pair = torch.cat(
                (current, baseline, delta, delta.abs()),
                dim=1,
            )
            repeated_current = current[:, None].expand(
                -1,
                time,
                -1,
                -1,
                -1,
            ).contiguous()
            encoded_current = self.encoder(
                repeated_current.reshape(
                    batch * time,
                    channels,
                    height,
                    width,
                )
            )
            (
                _,
                feature_channels,
                feature_height,
                feature_width,
            ) = encoded_current.shape
            encoded_current = encoded_current.reshape(
                batch,
                time,
                feature_channels,
                feature_height,
                feature_width,
            ).permute(0, 2, 1, 3, 4)
            current_fused = self.temporal_depthwise(
                encoded_current
            ).squeeze(2)
            fused = self.pointwise(current_fused)
            pair_residual = self.early_pair_output(
                self.early_pair_stem(pair)
            )
            pair_residual = nnf.interpolate(
                pair_residual,
                size=fused.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            reference_fused = fused
            fused = reference_fused + pair_residual
        else:
            encoded = self.encoder(
                frames.reshape(
                    batch * time,
                    channels,
                    height,
                    width,
                )
            )
            (
                _,
                feature_channels,
                feature_height,
                feature_width,
            ) = encoded.shape
            encoded = encoded.reshape(
                batch,
                time,
                feature_channels,
                feature_height,
                feature_width,
            ).permute(0, 2, 1, 3, 4)
            if temporal_mode == "joint":
                fused = self.temporal_depthwise(encoded).squeeze(2)
            else:
                current = encoded[:, :, -1]
                current_kernel = self.temporal_depthwise.weight.sum(
                    dim=2
                )
                current_fused = nnf.conv2d(
                    current,
                    current_kernel,
                    groups=current.shape[1],
                )
                history_delta = (
                    encoded[:, :, :-1] - encoded[:, :, -1:]
                )
                fused = current_fused + self.temporal_residual(
                    history_delta
                ).squeeze(2)
            fused = self.pointwise(fused)
        if return_reference:
            if temporal_mode != "early_pair":
                raise ValueError(
                    "Reference output is only available for early_pair"
                )
            projected = self._project_fused(fused)
            reference_projected = self._project_fused(reference_fused)
            if isinstance(self.dropout, nn.Dropout) and self.training:
                keep_probability = 1.0 - self.dropout.p
                dropout_mask = torch.empty_like(projected).bernoulli_(
                    keep_probability
                )
                dropout_mask.div_(keep_probability)
                projected = projected * dropout_mask
                reference_projected = reference_projected * dropout_mask
            else:
                projected = self.dropout(projected)
                reference_projected = self.dropout(reference_projected)
            risk_logits, known_logits = self._decode_projected(
                projected,
                batch,
            )
            reference_risk_logits, reference_known_logits = (
                self._decode_projected(reference_projected, batch)
            )
            return (
                risk_logits,
                known_logits,
                reference_risk_logits,
                reference_known_logits,
            )
        return self._decode_projected(
            self.dropout(self._project_fused(fused)),
            batch,
        )

    def _project_fused(self, fused: torch.Tensor) -> torch.Tensor:
        if getattr(self, "architecture", "pooled") == "pooled":
            return self.pool(fused).flatten(1)
        if self.architecture == "directional":
            return self.pool(fused).squeeze(2)
        return self.pool(fused)

    def _decode_projected(
        self,
        projected: torch.Tensor,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "architecture", "pooled") == "pooled":
            output = self.head(projected)
            output = output.reshape(batch, 2, 3, 3, 6, 6)
        elif self.architecture == "directional":
            output = self.head(projected)
            output = output.reshape(batch, 2, 3, 3, 6, 6)
            output = output.permute(0, 1, 2, 3, 5, 4)
        else:
            output = self.head(projected)
            output = output.reshape(batch, 2, 3, 6, 3, 6)
            output = output.permute(0, 1, 2, 4, 5, 3)
        return output[:, 0], output[:, 1]


def positive_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    positive = np.zeros((3, 3), dtype=np.float64)
    negative = np.zeros((3, 3), dtype=np.float64)
    for record in records:
        risk, known = decode_labels(record)
        risk_array = risk.numpy()
        known_array = known.numpy()
        positive += (risk_array * known_array).sum(axis=(2, 3))
        negative += ((1.0 - risk_array) * known_array).sum(axis=(2, 3))
    if np.any(positive <= 0.0) or np.any(negative <= 0.0):
        raise ValueError("Every horizon/height needs positive and negative mass")
    return torch.from_numpy(
        np.clip(negative / positive, 0.25, 20.0).astype(np.float32)
    )


def known_positive_weights(
    records: list[dict[str, Any]],
    power: float = 1.0,
) -> torch.Tensor:
    if not 0.0 < power <= 1.0:
        raise ValueError("Known positive-weight power must be in (0, 1]")
    positive = np.zeros((3, 3), dtype=np.float64)
    total = np.zeros((3, 3), dtype=np.float64)
    for record in records:
        _, known = decode_labels(record)
        known_array = known.numpy()
        positive += known_array.sum(axis=(2, 3))
        total += np.prod(known_array.shape[2:])
    negative = total - positive
    if np.any(positive <= 0.0) or np.any(negative <= 0.0):
        raise ValueError(
            "Every horizon/height needs known and unknown cells"
        )
    return torch.from_numpy(
        np.clip((negative / positive) ** power, 0.25, 20.0).astype(
            np.float32
        )
    )


def losses(
    risk_logits: torch.Tensor,
    known_logits: torch.Tensor,
    risk: torch.Tensor,
    known: torch.Tensor,
    weights: torch.Tensor,
    known_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_risk = nnf.binary_cross_entropy_with_logits(
        risk_logits,
        risk,
        pos_weight=weights.view(1, 3, 3, 1, 1),
        reduction="none",
    )
    risk_loss = (raw_risk * known).sum() / known.sum().clamp_min(1.0)
    known_loss = nnf.binary_cross_entropy_with_logits(
        known_logits,
        known,
        pos_weight=(
            known_weights.view(1, 3, 3, 1, 1)
            if known_weights is not None
            else None
        ),
    )
    return risk_loss + known_loss, risk_loss, known_loss


def future_body_head_recall_preservation_loss(
    candidate_risk_logits: torch.Tensor,
    reference_risk_logits: torch.Tensor,
    risk: torch.Tensor,
    known: torch.Tensor,
) -> torch.Tensor:
    """Penalize pair-induced risk-logit decreases on critical positives."""
    critical_positive = (
        (known[:, 1:, 1:] > 0.5)
        & (risk[:, 1:, 1:] >= 0.5)
    )
    if not torch.any(critical_positive):
        return candidate_risk_logits.sum() * 0.0
    decreases = nnf.relu(
        reference_risk_logits[:, 1:, 1:].detach()
        - candidate_risk_logits[:, 1:, 1:]
    )
    return decreases[critical_positive].mean()


def binary_metrics(
    probability: np.ndarray,
    truth_score: np.ndarray,
    known: np.ndarray,
) -> dict[str, float | int | None]:
    mask = known.astype(bool)
    scores = probability[mask]
    prediction = scores >= 0.5
    truth = truth_score[mask] >= 0.5
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    tn = int(np.sum(~prediction & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    positive_count = int(truth.sum())
    negative_count = int((~truth).sum())
    auroc = None
    average_precision = None
    if positive_count and negative_count:
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        ranks = np.empty(scores.shape[0], dtype=np.float64)
        start = 0
        while start < scores.shape[0]:
            end = start + 1
            while (
                end < scores.shape[0]
                and sorted_scores[end] == sorted_scores[start]
            ):
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        auroc = float(
            (
                ranks[truth].sum()
                - positive_count * (positive_count + 1) / 2.0
            )
            / (positive_count * negative_count)
        )

        descending = np.argsort(-scores, kind="stable")
        sorted_scores = scores[descending]
        sorted_truth = truth[descending]
        cumulative_tp = 0
        cumulative_fp = 0
        average_precision_value = 0.0
        start = 0
        while start < scores.shape[0]:
            end = start + 1
            while (
                end < scores.shape[0]
                and sorted_scores[end] == sorted_scores[start]
            ):
                end += 1
            group_positive = int(sorted_truth[start:end].sum())
            group_negative = end - start - group_positive
            cumulative_tp += group_positive
            cumulative_fp += group_negative
            if group_positive:
                average_precision_value += (
                    group_positive
                    / positive_count
                    * cumulative_tp
                    / (cumulative_tp + cumulative_fp)
                )
            start = end
        average_precision = float(average_precision_value)
    return {
        "known_cells": int(mask.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "auroc": auroc,
        "average_precision": average_precision,
        "risk_score_mae": (
            float(np.mean(np.abs(scores - truth_score[mask])))
            if np.any(mask)
            else 0.0
        ),
    }


def summarize_metrics(
    probability: np.ndarray,
    known_probability: np.ndarray,
    truth: np.ndarray,
    known: np.ndarray,
) -> dict[str, Any]:
    metrics = {
        "risk_all": binary_metrics(probability, truth, known),
        "risk_future": binary_metrics(
            probability[:, 1:],
            truth[:, 1:],
            known[:, 1:],
        ),
        "risk_future_body_head": binary_metrics(
            probability[:, 1:, 1:],
            truth[:, 1:, 1:],
            known[:, 1:, 1:],
        ),
        "known_accuracy": float(
            np.mean((known_probability >= 0.5) == (known >= 0.5))
        ),
        "by_horizon": {},
        "by_height": {},
        "by_horizon_height": {},
    }
    for index, name in enumerate(HORIZONS):
        metrics["by_horizon"][name] = binary_metrics(
            probability[:, index],
            truth[:, index],
            known[:, index],
        )
        metrics["by_horizon_height"][name] = {}
        for height_index, height_name in enumerate(HEIGHTS):
            metrics["by_horizon_height"][name][height_name] = binary_metrics(
                probability[:, index, height_index],
                truth[:, index, height_index],
                known[:, index, height_index],
            )
    for index, name in enumerate(HEIGHTS):
        metrics["by_height"][name] = binary_metrics(
            probability[:, :, index],
            truth[:, :, index],
            known[:, :, index],
        )
    metrics["future_body_head_macro_f1"] = float(
        np.mean(
            [
                metrics["by_horizon_height"][horizon][height]["f1"]
                for horizon in ("near", "far")
                for height in ("body", "head")
            ]
        )
    )
    return metrics


def train_prior_metrics(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
) -> dict[str, Any]:
    train_risk = []
    train_known = []
    for record in train_records:
        risk, known = decode_labels(record)
        train_risk.append(risk.numpy())
        train_known.append(known.numpy())
    risk_array = np.stack(train_risk)
    known_array = np.stack(train_known)
    numerator = (risk_array * known_array).sum(axis=0)
    denominator = known_array.sum(axis=0)
    prior = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    for horizon_index in range(3):
        for height_index in range(3):
            missing = denominator[horizon_index, height_index] <= 0
            if not np.any(missing):
                continue
            known_mass = denominator[horizon_index, height_index].sum()
            fallback = (
                numerator[horizon_index, height_index].sum() / known_mass
                if known_mass > 0
                else 0.5
            )
            prior[horizon_index, height_index][missing] = fallback
    known_prior = known_array.mean(axis=0)

    dev_risk = []
    dev_known = []
    for record in dev_records:
        risk, known = decode_labels(record)
        dev_risk.append(risk.numpy())
        dev_known.append(known.numpy())
    truth = np.stack(dev_risk)
    known = np.stack(dev_known)
    probability = np.broadcast_to(prior, truth.shape)
    known_probability = np.broadcast_to(known_prior, known.shape)
    return summarize_metrics(
        probability,
        known_probability,
        truth,
        known,
    )


@torch.no_grad()
def evaluate(
    model: TemporalStudent,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    risk_probabilities = []
    known_probabilities = []
    risk_targets = []
    known_targets = []
    for frames, risk, known in loader:
        risk_logits, known_logits = model(frames.to(device))
        risk_probabilities.append(torch.sigmoid(risk_logits).cpu().numpy())
        known_probabilities.append(torch.sigmoid(known_logits).cpu().numpy())
        risk_targets.append(risk.numpy())
        known_targets.append(known.numpy())
    probability = np.concatenate(risk_probabilities)
    known_probability = np.concatenate(known_probabilities)
    truth = np.concatenate(risk_targets)
    known = np.concatenate(known_targets)
    return summarize_metrics(
        probability,
        known_probability,
        truth,
        known,
    )


def train(
    samples_path: Path,
    pretrained_path: Path,
    output_root: Path,
    arm: str,
    seed: int,
    epochs: int,
    *,
    initial_checkpoint_path: Path | None = None,
    encoder_lr: float = 3e-5,
    head_lr: float = 3e-4,
    architecture: str = "pooled",
    temporal_mode: str = "joint",
    optimization_mode: str = "all",
    pair_constraint_mode: str = "none",
    known_loss_mode: str = "plain",
    selection_mode: str = "aggregate",
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if encoder_lr <= 0.0 or head_lr <= 0.0:
        raise ValueError("Learning rates must be positive")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    if temporal_mode not in TEMPORAL_MODES:
        raise ValueError(f"Unknown temporal mode: {temporal_mode}")
    if optimization_mode not in OPTIMIZATION_MODES:
        raise ValueError(f"Unknown optimization mode: {optimization_mode}")
    if pair_constraint_mode not in PAIR_CONSTRAINT_MODES:
        raise ValueError(
            f"Unknown pair constraint mode: {pair_constraint_mode}"
        )
    if known_loss_mode not in KNOWN_LOSS_MODES:
        raise ValueError(f"Unknown known loss mode: {known_loss_mode}")
    if optimization_mode == "temporal_residual_only" and (
        temporal_mode
        not in {"current_residual", "current_spatial_residual"}
        or initial_checkpoint_path is None
    ):
        raise ValueError(
            "temporal_residual_only requires a residual temporal mode "
            "and an initial checkpoint"
        )
    if optimization_mode == "early_pair_only" and (
        temporal_mode != "early_pair"
        or initial_checkpoint_path is None
    ):
        raise ValueError(
            "early_pair_only requires early_pair mode "
            "and an initial checkpoint"
        )
    if pair_constraint_mode != "none" and (
        temporal_mode != "early_pair"
        or optimization_mode != "early_pair_only"
    ):
        raise ValueError(
            "Pair constraints require early_pair mode and "
            "early_pair_only optimization"
        )
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode: {selection_mode}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Development run")
    seed_everything(seed)
    records = load_jsonl(samples_path)
    train_records = [row for row in records if row["role"] == "train"]
    dev_records = [row for row in records if row["role"] == "dev"]
    train_dataset = HftfDataset(
        train_records,
        arm,
        train=True,
        seed=seed,
    )
    dev_dataset = HftfDataset(
        dev_records,
        arm,
        train=False,
        seed=seed,
    )
    weights = positive_weights(train_records)
    known_weight_power = {
        "plain": None,
        "balanced": 1.0,
        "sqrt_balanced": 0.5,
    }[known_loss_mode]
    known_weights = (
        known_positive_weights(train_records, power=known_weight_power)
        if known_weight_power is not None
        else None
    )
    prior_dev_metrics = train_prior_metrics(train_records, dev_records)
    device = torch.device("cuda")
    model = TemporalStudent(
        pretrained_path,
        architecture=architecture,
        temporal_mode=temporal_mode,
    ).to(device)
    initial_checkpoint_sha256 = None
    if initial_checkpoint_path is not None:
        initial_checkpoint = torch.load(
            initial_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        initial_architecture = initial_checkpoint.get(
            "architecture",
            "pooled",
        )
        if initial_architecture != architecture:
            raise ValueError(
                "Initial checkpoint architecture mismatch: "
                f"{initial_architecture} != {architecture}"
            )
        initial_temporal_mode = initial_checkpoint.get(
            "temporal_mode",
            "joint",
        )
        if initial_temporal_mode == temporal_mode:
            model.load_state_dict(initial_checkpoint["model_state_dict"])
        elif (
            initial_temporal_mode == "joint"
            and temporal_mode
            in {"current_residual", "current_spatial_residual"}
        ):
            incompatible = model.load_state_dict(
                initial_checkpoint["model_state_dict"],
                strict=False,
            )
            if incompatible.missing_keys != ["temporal_residual.weight"]:
                raise ValueError(
                    "Unexpected missing keys during joint-to-residual "
                    f"initialization: {incompatible.missing_keys}"
                )
            if incompatible.unexpected_keys:
                raise ValueError(
                    "Unexpected keys during joint-to-residual "
                    f"initialization: {incompatible.unexpected_keys}"
                )
        elif (
            initial_temporal_mode == "joint"
            and temporal_mode == "early_pair"
        ):
            incompatible = model.load_state_dict(
                initial_checkpoint["model_state_dict"],
                strict=False,
            )
            if (
                not incompatible.missing_keys
                or any(
                    not key.startswith(
                        ("early_pair_stem.", "early_pair_output.")
                    )
                    for key in incompatible.missing_keys
                )
            ):
                raise ValueError(
                    "Unexpected missing keys during joint-to-early-pair "
                    f"initialization: {incompatible.missing_keys}"
                )
            if incompatible.unexpected_keys:
                raise ValueError(
                    "Unexpected keys during joint-to-early-pair "
                    f"initialization: {incompatible.unexpected_keys}"
                )
        else:
            raise ValueError(
                "Initial checkpoint temporal mode mismatch: "
                f"{initial_temporal_mode} != {temporal_mode}"
            )
        initial_checkpoint_sha256 = sha256(initial_checkpoint_path)
    if optimization_mode == "temporal_residual_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("temporal_residual.")
        optimizer_groups = [
            {
                "params": model.temporal_residual.parameters(),
                "lr": head_lr,
            }
        ]
    elif optimization_mode == "early_pair_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(
                ("early_pair_stem.", "early_pair_output.")
            )
        optimizer_groups = [
            {
                "params": [
                    *model.early_pair_stem.parameters(),
                    *model.early_pair_output.parameters(),
                ],
                "lr": head_lr,
            }
        ]
    else:
        optimizer_groups = [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("encoder.")
                ],
                "lr": head_lr,
            },
        ]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=1e-4,
    )
    weights = weights.to(device)
    if known_weights is not None:
        known_weights = known_weights.to(device)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    dev_environment_loaders = {
        environment: DataLoader(
            HftfDataset(
                [
                    record
                    for record in dev_records
                    if record["environment"] == environment
                ],
                arm,
                train=False,
                seed=seed,
            ),
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for environment in sorted(
            {record["environment"] for record in dev_records}
        )
    }

    def score(
        aggregate_metrics: dict[str, Any],
        environment_rows: dict[str, dict[str, Any]],
    ) -> float:
        if selection_mode == "aggregate":
            return float(
                aggregate_metrics["future_body_head_macro_f1"]
            )
        return float(
            np.mean(
                [
                    metrics["future_body_head_macro_f1"]
                    for metrics in environment_rows.values()
                ]
            )
        )

    best_f1 = -1.0
    best_epoch = -1
    best_metrics = None
    best_state = None
    history = []
    if initial_checkpoint_path is not None:
        initial_metrics = evaluate(model, dev_loader, device)
        initial_environment_metrics = (
            {
                environment: evaluate(model, loader, device)
                for environment, loader in dev_environment_loaders.items()
            }
            if selection_mode == "environment_macro"
            else {}
        )
        best_f1 = score(
            initial_metrics,
            initial_environment_metrics,
        )
        best_epoch = 0
        best_metrics = copy.deepcopy(initial_metrics)
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        history.append(
            {
                "epoch": 0,
                "train_loss": None,
                "train_risk_loss": None,
                "train_known_loss": None,
                "train_pair_constraint_loss": None,
                "dev": initial_metrics,
                "dev_by_environment": initial_environment_metrics,
            }
        )
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=True,
        )
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        batches = 0
        for frames, risk, known in train_loader:
            frames = frames.to(device, non_blocking=True)
            risk = risk.to(device, non_blocking=True)
            known = known.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if pair_constraint_mode == "future_body_head_recall":
                (
                    risk_logits,
                    known_logits,
                    reference_risk_logits,
                    _,
                ) = model(frames, return_reference=True)
            else:
                risk_logits, known_logits = model(frames)
            loss, risk_loss, known_loss = losses(
                risk_logits,
                known_logits,
                risk,
                known,
                weights,
                known_weights,
            )
            if pair_constraint_mode == "future_body_head_recall":
                pair_constraint_loss = (
                    future_body_head_recall_preservation_loss(
                        risk_logits,
                        reference_risk_logits,
                        risk,
                        known,
                    )
                )
                loss = loss + pair_constraint_loss
            else:
                pair_constraint_loss = loss.new_zeros(())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            totals += [
                float(loss.detach()),
                float(risk_loss.detach()),
                float(known_loss.detach()),
                float(pair_constraint_loss.detach()),
            ]
            batches += 1
        metrics = evaluate(model, dev_loader, device)
        epoch_environment_metrics = (
            {
                environment: evaluate(model, loader, device)
                for environment, loader in dev_environment_loaders.items()
            }
            if selection_mode == "environment_macro"
            else {}
        )
        row = {
            "epoch": epoch,
            "train_loss": float(totals[0] / batches),
            "train_risk_loss": float(totals[1] / batches),
            "train_known_loss": float(totals[2] / batches),
            "train_pair_constraint_loss": float(totals[3] / batches),
            "dev": metrics,
            "dev_by_environment": epoch_environment_metrics,
        }
        history.append(row)
        future_f1 = score(metrics, epoch_environment_metrics)
        if future_f1 > best_f1:
            best_f1 = future_f1
            best_epoch = epoch
            best_metrics = copy.deepcopy(metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            json.dumps(
                {
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    "selection_metric": (
                        "environment_macro_future_body_head_macro_f1"
                        if selection_mode == "environment_macro"
                        else "future_body_head_macro_f1"
                    ),
                    "dev_selection_f1": future_f1,
                    "best_epoch": best_epoch,
                    "best_selection_f1": best_f1,
                }
            ),
            flush=True,
        )
    if best_state is None or best_metrics is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    environment_metrics = {}
    prior_environment_metrics = {}
    for environment in sorted(
        {record["environment"] for record in dev_records}
    ):
        environment_records = [
            record
            for record in dev_records
            if record["environment"] == environment
        ]
        environment_loader = DataLoader(
            HftfDataset(
                environment_records,
                arm,
                train=False,
                seed=seed,
            ),
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        environment_metrics[environment] = evaluate(
            model,
            environment_loader,
            device,
        )
        prior_environment_metrics[environment] = train_prior_metrics(
            train_records,
            environment_records,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "blindassist_hftf_tartanground_development_checkpoint",
        "arm": arm,
        "architecture": architecture,
        "temporal_mode": temporal_mode,
        "optimization_mode": optimization_mode,
        "pair_constraint_mode": pair_constraint_mode,
        "known_loss_mode": known_loss_mode,
        "selection_mode": selection_mode,
        "seed": seed,
        "selected_epoch": best_epoch,
        "selected_selection_score": best_f1,
        "selected_dev_metrics": best_metrics,
        "selected_dev_metrics_by_environment": environment_metrics,
        "model_state_dict": best_state,
    }
    checkpoint_path = output_root / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": "blindassist_hftf_stage_c_d5_tartanground_development_student",
        "status": "DEVELOPMENT_TRAINING_COMPLETE",
        "policy": {
            "outcome_open": True,
            "repairable": True,
            "one_shot": False,
            "promotion_evidence": False,
        },
        "arm": arm,
        "architecture": architecture,
        "temporal_mode": temporal_mode,
        "seed": seed,
        "epochs": epochs,
        "selection_metric": (
            "environment_macro_future_body_head_macro_f1"
            if selection_mode == "environment_macro"
            else "future_body_head_macro_f1"
        ),
        "selection_mode": selection_mode,
        "optimization": {
            "mode": optimization_mode,
            "pair_constraint_mode": pair_constraint_mode,
            "pair_constraint_weight": (
                1.0 if pair_constraint_mode != "none" else None
            ),
            "known_loss_mode": known_loss_mode,
            "known_positive_weights": (
                known_weights.detach().cpu().tolist()
                if known_weights is not None
                else None
            ),
            "encoder_lr": encoder_lr,
            "head_lr": head_lr,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "initial_checkpoint_path": (
                str(initial_checkpoint_path.resolve())
                if initial_checkpoint_path is not None
                else None
            ),
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
        },
        "train_prior_dev_metrics": prior_dev_metrics,
        "train_prior_dev_metrics_by_environment": prior_environment_metrics,
        "train_sample_count": len(train_records),
        "dev_sample_count": len(dev_records),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "samples_sha256": sha256(samples_path),
        "pretrained_sha256": sha256(pretrained_path),
        "selected_epoch": best_epoch,
        "selected_selection_score": best_f1,
        "selected_dev_metrics": best_metrics,
        "selected_dev_metrics_by_environment": environment_metrics,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
        },
        "history": history,
    }
    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="pooled",
    )
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default="aggregate",
    )
    parser.add_argument(
        "--temporal-mode",
        choices=TEMPORAL_MODES,
        default="joint",
    )
    parser.add_argument(
        "--optimization-mode",
        choices=OPTIMIZATION_MODES,
        default="all",
    )
    parser.add_argument(
        "--pair-constraint-mode",
        choices=PAIR_CONSTRAINT_MODES,
        default="none",
    )
    parser.add_argument(
        "--known-loss-mode",
        choices=KNOWN_LOSS_MODES,
        default="plain",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    args = parser.parse_args()
    report = train(
        args.samples,
        args.pretrained,
        args.output_root,
        args.arm,
        args.seed,
        args.epochs,
        initial_checkpoint_path=args.initial_checkpoint,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        architecture=args.architecture,
        temporal_mode=args.temporal_mode,
        optimization_mode=args.optimization_mode,
        pair_constraint_mode=args.pair_constraint_mode,
        known_loss_mode=args.known_loss_mode,
        selection_mode=args.selection_mode,
    )
    print(
        json.dumps(
            {
                "arm": report["arm"],
                "seed": report["seed"],
                "selected_epoch": report["selected_epoch"],
                "selected_selection_score": report[
                    "selected_selection_score"
                ],
                "dev_future_micro_f1": report["selected_dev_metrics"][
                    "risk_future"
                ]["f1"],
                "dev_future_body_head_macro_f1": report[
                    "selected_dev_metrics"
                ]["future_body_head_macro_f1"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
