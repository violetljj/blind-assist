#!/usr/bin/env python3
"""Frozen model, features, training losses, baselines, and metrics for R1."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BANDS = ("left", "center", "right")
HORIZONS = np.asarray([1.0, 1.5, 2.0], dtype=np.float64)


class SpatialCalibrationHead(nn.Module):
    def __init__(self, input_dimension: int = 781, hidden_dimension: int = 12) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dimension, hidden_dimension)
        self.output = nn.Linear(hidden_dimension, 3)

    def forward(self, region_inputs: torch.Tensor, raw_clearance: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.output(F.silu(self.hidden(region_inputs)))
        scale = torch.exp(math.log(4.0) * torch.tanh(raw[..., 0]))
        offset = 3.0 * torch.tanh(raw[..., 1])
        confidence = torch.sigmoid(raw[..., 2])
        calibrated = torch.clamp(scale * raw_clearance + offset, 0.0, 20.0)
        return {"scale": scale, "offset": offset, "confidence": confidence, "clearance": calibrated}


class SpatialCalibrationNoConfidence(nn.Module):
    def __init__(self, input_dimension: int = 781, hidden_dimension: int = 12) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dimension, hidden_dimension)
        self.output = nn.Linear(hidden_dimension, 2)

    def forward(self, region_inputs: torch.Tensor, raw_clearance: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.output(F.silu(self.hidden(region_inputs)))
        scale = torch.exp(math.log(4.0) * torch.tanh(raw[..., 0]))
        offset = 3.0 * torch.tanh(raw[..., 1])
        calibrated = torch.clamp(scale * raw_clearance + offset, 0.0, 20.0)
        return {"scale": scale, "offset": offset, "confidence": torch.ones_like(calibrated), "clearance": calibrated}


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def regional_feature_inputs(
    patch_tokens: np.ndarray,
    depth_m: np.ndarray,
    intrinsics_fx_fy_cx_cy: Iterable[float],
    valid_depth_range_m: tuple[float, float] = (0.2, 20.0),
) -> np.ndarray:
    """Build the exact 3x781 frozen regional inputs for one frame."""
    tokens = np.asarray(patch_tokens, dtype=np.float32)
    depth = np.asarray(depth_m, dtype=np.float32)
    if tokens.ndim != 3 or tokens.shape[2] != 384:
        raise ValueError("patch tokens must be HxWx384")
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    fx, fy, cx, cy = (float(value) for value in intrinsics_fx_fy_cx_cy)
    height, width = depth.shape
    gradient_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gradient_x, gradient_y)
    y_coordinates = (np.arange(height, dtype=np.float32) + 0.5) / height
    token_width = tokens.shape[1]
    token_x = (np.arange(token_width, dtype=np.float32) + 0.5) / token_width
    boundaries = ((0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0))
    outputs = []
    low_depth, high_depth = valid_depth_range_m
    for band_index, (minimum, maximum) in enumerate(boundaries):
        token_mask = (token_x >= minimum) & (token_x < maximum if band_index < 2 else token_x <= maximum)
        selected_tokens = tokens[:, token_mask, :].reshape(-1, 384)
        if not len(selected_tokens):
            raise ValueError("empty token region")
        token_stats = np.concatenate((selected_tokens.mean(axis=0), selected_tokens.std(axis=0)))

        x0 = int(round(minimum * width))
        x1 = int(round(maximum * width)) if band_index < 2 else width
        region_depth = depth[:, x0:x1]
        region_gradient = gradient[:, x0:x1]
        valid = np.isfinite(region_depth) & (region_depth >= low_depth) & (region_depth <= high_depth)
        if not np.any(valid):
            raise ValueError("region has no valid DA depth")
        values = region_depth[valid]
        p10, p25, median, p75 = np.quantile(values, [0.10, 0.25, 0.50, 0.75])
        gradient_values = region_gradient[valid]
        near_mask = valid & (region_depth <= p10)
        near_y = np.broadcast_to(y_coordinates[:, None], region_depth.shape)[near_mask]
        scalars = np.asarray(
            [
                median, p10, p25, p75,
                np.median(gradient_values), np.quantile(gradient_values, 0.90),
                np.mean(valid), np.mean(near_y), (minimum + maximum) / 2.0,
                fx / width, fy / height, cx / width, cy / height,
            ],
            dtype=np.float32,
        )
        outputs.append(np.concatenate((token_stats.astype(np.float32), scalars)))
    result = np.stack(outputs)
    if result.shape != (3, 781) or not np.all(np.isfinite(result)):
        raise ValueError("invalid regional feature result")
    return result


def spatial_loss(
    output: dict[str, torch.Tensor],
    truth_clearance: torch.Tensor,
    truth_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = truth_valid.bool() & torch.isfinite(truth_clearance) & torch.isfinite(output["clearance"])
    if not torch.any(valid):
        raise ValueError("batch has no truth-valid regions")
    prediction = output["clearance"][valid]
    truth = truth_clearance[valid]
    regression = F.huber_loss(prediction, truth, reduction="mean", delta=0.25)

    horizons = torch.tensor([1.0, 1.5, 2.0], device=prediction.device, dtype=prediction.dtype)
    logits = (horizons[None, :] - prediction[:, None]) / 0.10
    occupied = (truth[:, None] <= horizons[None, :]).to(prediction.dtype)
    occupancy = F.binary_cross_entropy_with_logits(
        logits, occupied, pos_weight=torch.tensor(3.0, device=prediction.device, dtype=prediction.dtype)
    )
    predicted_occupied = logits.detach() >= 0
    correct = ((torch.abs(prediction.detach() - truth) <= 0.25) & torch.all(predicted_occupied == occupied.bool(), dim=1)).to(prediction.dtype)
    confidence = output["confidence"][valid]
    confidence_bce = F.binary_cross_entropy(confidence, correct)
    coverage_hinge = torch.square(torch.clamp(0.90 - torch.mean(confidence), min=0.0))
    total = regression + occupancy + 0.25 * confidence_bce + 2.0 * coverage_hinge
    return {
        "total": total,
        "regression": regression,
        "occupancy": occupancy,
        "confidence": confidence_bce,
        "coverage_hinge": coverage_hinge,
    }


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        return cls(mean=mean, std=np.where(std > 1e-8, std, 1.0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def fit_ridge(features: np.ndarray, targets: np.ndarray, ridge_lambda: float = 10.0) -> dict[str, np.ndarray]:
    standardizer = Standardizer.fit(features)
    x = standardizer.transform(features)
    target_mean = np.mean(targets, axis=0)
    kernel = np.linalg.solve(x.T @ x + ridge_lambda * np.eye(x.shape[1]), x.T @ (targets - target_mean))
    return {"mean": standardizer.mean, "std": standardizer.std, "target_mean": target_mean, "kernel": kernel}


def predict_ridge(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    return model["target_mean"] + ((features - model["mean"]) / model["std"]) @ model["kernel"]


def apply_global_affine(raw_clearance: np.ndarray, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim == 1:
        parameters = np.broadcast_to(parameters, (len(raw_clearance), 2))
    valid = (
        np.all(np.isfinite(parameters), axis=1)
        & (parameters[:, 0] >= 0.25) & (parameters[:, 0] <= 4.0)
        & (parameters[:, 1] >= -3.0) & (parameters[:, 1] <= 3.0)
    )
    predicted = np.clip(parameters[:, 0, None] * raw_clearance + parameters[:, 1, None], 0.0, 20.0)
    return predicted, np.broadcast_to(valid[:, None], predicted.shape).copy()


def train_spatial_model(
    region_inputs: np.ndarray,
    raw_clearance: np.ndarray,
    truth_clearance: np.ndarray,
    truth_valid: np.ndarray,
    train_indices: np.ndarray,
    training: dict[str, Any],
) -> tuple[SpatialCalibrationHead, Standardizer, list[float]]:
    torch.manual_seed(int(training["seed"]))
    np.random.seed(int(training["seed"]))
    torch.use_deterministic_algorithms(True)
    selected_features = region_inputs[train_indices]
    standardizer = Standardizer.fit(selected_features.reshape(-1, selected_features.shape[-1]))
    normalized = standardizer.transform(region_inputs).astype(np.float32)
    model = SpatialCalibrationHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    generator = torch.Generator().manual_seed(int(training["seed"]))
    batch_size = int(training["batch_size_frames"])
    losses: list[float] = []
    model.train()
    for _epoch in range(int(training["epochs"])):
        order = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            supervised = (
                truth_valid[batch].astype(bool)
                & np.isfinite(truth_clearance[batch])
                & np.isfinite(raw_clearance[batch])
            )
            if not np.any(supervised):
                continue
            features = torch.from_numpy(normalized[batch])
            raw_values = np.nan_to_num(
                raw_clearance[batch].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            raw = torch.from_numpy(raw_values)
            truth = torch.from_numpy(truth_clearance[batch].astype(np.float32))
            valid = torch.from_numpy(truth_valid[batch].astype(bool))
            optimizer.zero_grad(set_to_none=True)
            components = spatial_loss(model(features, raw), truth, valid)
            components["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            epoch_losses.append(float(components["total"].detach()))
        if not epoch_losses:
            raise ValueError("epoch has no supervised training batches")
        losses.append(float(np.mean(epoch_losses)))
    return model.eval(), standardizer, losses


def predict_spatial(model: SpatialCalibrationHead, standardizer: Standardizer, region_inputs: np.ndarray, raw_clearance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = torch.from_numpy(standardizer.transform(region_inputs).astype(np.float32))
    raw = torch.from_numpy(
        np.nan_to_num(raw_clearance.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    )
    with torch.inference_mode():
        output = model(features, raw)
    confidence = output["confidence"].numpy()
    clearance = output["clearance"].numpy()
    known = np.isfinite(raw_clearance) & np.all(np.isfinite(region_inputs), axis=2) & (confidence >= 0.5)
    return clearance, known, confidence


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_predictions(
    records: list[dict[str, Any]],
    prediction: np.ndarray,
    known: np.ndarray,
    truth: np.ndarray,
    truth_valid: np.ndarray,
    confidence: np.ndarray | None = None,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    known = np.asarray(known, dtype=bool) & np.isfinite(prediction)
    truth_valid = np.asarray(truth_valid, dtype=bool) & np.isfinite(truth)
    if prediction.shape != truth.shape or prediction.shape != known.shape or prediction.shape != truth_valid.shape:
        raise ValueError("prediction metric shapes differ")
    parent_results = {}
    for parent in sorted({str(row["parent_id"]) for row in records}):
        indices = [index for index, row in enumerate(records) if str(row["parent_id"]) == parent]
        valid_count = int(np.sum(truth_valid[indices]))
        paired = truth_valid[indices] & known[indices]
        errors = np.abs(prediction[indices][paired] - truth[indices][paired])
        truth_occupied = truth[indices, :, None] <= HORIZONS[None, None, :]
        predicted_occupied = prediction[indices, :, None] <= HORIZONS[None, None, :]
        paired_decisions = paired[:, :, None]
        decisions = (truth_occupied == predicted_occupied)[paired_decisions.repeat(3, axis=2)]
        false_clear = (truth_occupied & ~predicted_occupied)[paired_decisions.repeat(3, axis=2)]
        occupied_paired = truth_occupied & paired_decisions
        conditional_false = (truth_occupied & ~predicted_occupied & paired_decisions)
        delta_errors = []
        groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index in indices:
            for band in range(3):
                groups[(str(records[index]["video_id"]), band)].append(index)
        for (_video, band), group in groups.items():
            ordered = sorted(group, key=lambda idx: float(records[idx]["timestamp"]))
            previous = None
            for index in ordered:
                if not (truth_valid[index, band] and known[index, band]):
                    previous = None
                    continue
                timestamp = float(records[index]["timestamp"])
                if previous is not None:
                    prior_index, prior_time = previous
                    if timestamp > prior_time:
                        delta_errors.append(abs((prediction[index, band] - prediction[prior_index, band]) - (truth[index, band] - truth[prior_index, band])))
                previous = (index, timestamp)
        ece = None
        if confidence is not None and np.any(paired):
            conf = np.asarray(confidence)[indices][paired]
            paired_prediction = prediction[indices][paired]
            paired_truth = truth[indices][paired]
            occupancy_correct = np.all((paired_prediction[:, None] <= HORIZONS) == (paired_truth[:, None] <= HORIZONS), axis=1)
            correctness = ((np.abs(paired_prediction - paired_truth) <= 0.25) & occupancy_correct).astype(float)
            ece_value = 0.0
            for bin_index in range(10):
                low, high = bin_index / 10, (bin_index + 1) / 10
                mask = (conf >= low) & (conf < high if bin_index < 9 else conf <= high)
                if np.any(mask):
                    ece_value += np.mean(mask) * abs(float(np.mean(conf[mask])) - float(np.mean(correctness[mask])))
            ece = float(ece_value)
        parent_results[parent] = {
            "truth_valid_regions": valid_count,
            "known_coverage": float(np.sum(paired) / valid_count) if valid_count else None,
            "clearance_mae_m": _mean(errors.tolist()),
            "envelope_agreement": _mean(decisions.astype(float).tolist()),
            "false_clear_rate": _mean(false_clear.astype(float).tolist()),
            "conditional_false_clear_rate": float(np.sum(conditional_false) / np.sum(occupied_paired)) if np.any(occupied_paired) else None,
            "temporal_delta_mae_m": _mean(delta_errors),
            "confidence_ece": ece,
        }
    metric_names = ("known_coverage", "clearance_mae_m", "envelope_agreement", "false_clear_rate", "temporal_delta_mae_m", "confidence_ece")
    parent_macro = {
        name: _mean([row[name] for row in parent_results.values() if row[name] is not None])
        for name in metric_names
    }
    gates = {
        "known_coverage": parent_macro["known_coverage"] is not None and parent_macro["known_coverage"] >= 0.90,
        "clearance_mae": parent_macro["clearance_mae_m"] is not None and parent_macro["clearance_mae_m"] <= 0.25,
        "envelope_agreement": parent_macro["envelope_agreement"] is not None and parent_macro["envelope_agreement"] >= 0.90,
        "false_clear": parent_macro["false_clear_rate"] is not None and parent_macro["false_clear_rate"] <= 0.05,
        "temporal_delta_mae": parent_macro["temporal_delta_mae_m"] is not None and parent_macro["temporal_delta_mae_m"] <= 0.15,
    }
    if confidence is not None:
        gates["confidence_ece"] = parent_macro["confidence_ece"] is not None and parent_macro["confidence_ece"] <= 0.10
    return {"parents": parent_results, "parent_macro": parent_macro, "gates": gates}
