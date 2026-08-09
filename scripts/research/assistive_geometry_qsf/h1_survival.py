#!/usr/bin/env python3
"""H1-only queryable survival geometry mechanics.

The H1 representation predicts four discrete robust q-contact hazards per
left/centre/right band.  Occupancy at 1.0/1.5/2.0 m and a horizon-capped
clearance estimate are derived from that one distribution.  UNKNOWN support is
kept outside the likelihood; it is never encoded as a negative observation.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.research.assistive_geometry.assistive_geometry_model import (
    AssistiveTaskHeads,
    DepthArtAssistiveGeometry,
    _masked_mean,
    ground_plane_depth_loss,
    near_field_weights,
)


BANDS = 3
HAZARD_BIN_EDGES_M = (0.5, 1.0, 1.5, 2.0)
HAZARD_BIN_MIDPOINTS_M = (0.25, 0.75, 1.25, 1.75)
OCCUPANCY_HORIZONS_M = (1.0, 1.5, 2.0)
OCCUPANCY_CDF_INDICES = (1, 2, 3)
HAZARD_BINS = len(HAZARD_BIN_EDGES_M)


def _require_shape(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if value.ndim != len(shape) or any(
        expected >= 0 and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def compile_h1_targets(targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compile B1 sensor-geometry targets into H1 event/censor states.

    A robust q-contact at or before 2 m is an event.  A q-contact beyond 2 m,
    or fully known and clear occupancy through 2 m, is a right-censored
    observation.  Every other band remains UNKNOWN and contributes zero
    survival likelihood.
    """

    clearance = targets["clearance_m"]
    clearance_valid = targets["clearance_valid"].bool()
    occupancy = targets["occupancy"]
    occupancy_valid = targets["occupancy_valid"].bool()
    _require_shape(clearance, (-1, BANDS), "clearance_m")
    _require_shape(clearance_valid, (-1, BANDS), "clearance_valid")
    _require_shape(occupancy, (-1, BANDS, len(OCCUPANCY_HORIZONS_M)), "occupancy")
    _require_shape(
        occupancy_valid,
        (-1, BANDS, len(OCCUPANCY_HORIZONS_M)),
        "occupancy_valid",
    )
    if clearance.shape[0] != occupancy.shape[0]:
        raise ValueError("clearance and occupancy batch sizes differ")
    if not bool(torch.isfinite(clearance[clearance_valid]).all()):
        raise ValueError("known clearance must be finite")
    if bool((clearance[clearance_valid] < 0.0).any()):
        raise ValueError("known clearance must be non-negative")
    if not bool(torch.isfinite(occupancy[occupancy_valid]).all()):
        raise ValueError("known occupancy must be finite")
    if bool(((occupancy[occupancy_valid] != 0.0) & (occupancy[occupancy_valid] != 1.0)).any()):
        raise ValueError("known occupancy must be binary")
    if bool(clearance[~clearance_valid].ne(0.0).any()):
        raise ValueError("UNKNOWN clearance payload must be neutral zero")
    if bool(occupancy[~occupancy_valid].ne(0.0).any()):
        raise ValueError("UNKNOWN occupancy payload must be neutral zero")

    known_pairs = occupancy_valid[..., 1:] & occupancy_valid[..., :-1]
    decreasing_known = known_pairs & (occupancy[..., 1:] < occupancy[..., :-1])
    if bool(decreasing_known.any()):
        raise ValueError("known occupancy truth must be horizon-monotone")

    horizons = occupancy.new_tensor(OCCUPANCY_HORIZONS_M)
    expected_from_clearance = clearance[..., None] <= horizons
    comparable = clearance_valid[..., None] & occupancy_valid
    observed_occupied = occupancy >= 0.5
    if bool((comparable & (expected_from_clearance != observed_occupied)).any()):
        raise ValueError("clearance and known occupancy truth disagree")

    maximum = HAZARD_BIN_EDGES_M[-1]
    event_observed = clearance_valid & (clearance <= maximum)
    event_beyond_grid = clearance_valid & (clearance > maximum)
    fully_known = occupancy_valid.all(dim=-1)
    fully_clear = fully_known & (~observed_occupied).all(dim=-1)
    right_censored = event_beyond_grid | ((~clearance_valid) & fully_clear)
    distribution_valid = event_observed | right_censored

    edges = clearance.new_tensor(HAZARD_BIN_EDGES_M)
    event_bin = torch.bucketize(clearance.contiguous(), edges, right=False).to(torch.long)
    event_bin = torch.where(event_observed, event_bin, torch.full_like(event_bin, -1))
    if bool((event_bin[event_observed] >= HAZARD_BINS).any()):
        raise ValueError("event bin escaped the frozen hazard grid")

    return {
        "event_bin": event_bin,
        "event_observed": event_observed,
        "right_censored": right_censored,
        "distribution_valid": distribution_valid,
        "unknown": ~distribution_valid,
        "clearance_m": clearance,
        "clearance_valid": clearance_valid,
        "occupancy": occupancy,
        "occupancy_valid": occupancy_valid,
    }


def hazard_distribution(hazard_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    _require_shape(hazard_logits, (-1, BANDS, HAZARD_BINS), "hazard_logits")
    hazards = torch.sigmoid(hazard_logits)
    survival = torch.cumprod(1.0 - hazards, dim=-1)
    prefix = torch.cat((torch.ones_like(survival[..., :1]), survival[..., :-1]), dim=-1)
    event_probability = prefix * hazards
    tail_probability = survival[..., -1]
    occupancy_cdf = 1.0 - survival
    return {
        "hazard_probability": hazards,
        "survival_probability": survival,
        "event_probability": event_probability,
        "tail_probability": tail_probability,
        "occupancy_cdf": occupancy_cdf,
    }


def decode_h1_outputs(hazard_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    distribution = hazard_distribution(hazard_logits)
    midpoint = hazard_logits.new_tensor(HAZARD_BIN_MIDPOINTS_M)
    capped_clearance = (
        distribution["event_probability"] * midpoint
    ).sum(dim=-1) + distribution["tail_probability"] * HAZARD_BIN_EDGES_M[-1]
    indices = torch.as_tensor(OCCUPANCY_CDF_INDICES, device=hazard_logits.device)
    occupancy = distribution["occupancy_cdf"].index_select(-1, indices)
    return {
        **distribution,
        "clearance_m": capped_clearance,
        "occupancy_probability": occupancy,
    }


def discrete_survival_nll(
    hazard_logits: torch.Tensor,
    compiled: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean and per-band right-censored discrete survival NLL."""

    _require_shape(hazard_logits, (-1, BANDS, HAZARD_BINS), "hazard_logits")
    event_bin = compiled["event_bin"]
    event_observed = compiled["event_observed"].bool()
    right_censored = compiled["right_censored"].bool()
    distribution_valid = compiled["distribution_valid"].bool()
    _require_shape(event_bin, (-1, BANDS), "event_bin")
    if hazard_logits.shape[:2] != event_bin.shape:
        raise ValueError("hazard logits and compiled targets differ in batch/band shape")

    bins = torch.arange(HAZARD_BINS, device=hazard_logits.device)
    survived = right_censored[..., None] | (
        event_observed[..., None] & (bins < event_bin[..., None])
    )
    event = event_observed[..., None] & (bins == event_bin[..., None])
    per_bin = F.softplus(hazard_logits) * survived + F.softplus(-hazard_logits) * event
    per_band = per_bin.sum(dim=-1)
    mean = _masked_mean(per_band, distribution_valid)
    return mean, per_band


class QsfH1TaskHeads(nn.Module):
    """Parameter-matched replacement for the direct clearance/occupancy heads."""

    def __init__(self, channels: int = 48, hidden: int = 32) -> None:
        super().__init__()
        self.ground_pre = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.ground_out = nn.Conv2d(16, 1, 1)
        self.band_mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=False))
        self.hazard_out = nn.Linear(hidden, HAZARD_BINS)
        self.confidence_out = nn.Linear(hidden, 1)

    @staticmethod
    def band_pool(feature: torch.Tensor) -> torch.Tensor:
        return AssistiveTaskHeads.band_pool(feature)

    def forward_bands(self, bands: torch.Tensor) -> dict[str, torch.Tensor]:
        _require_shape(bands, (-1, BANDS, self.band_mlp[0].in_features), "band_features")
        hidden = self.band_mlp(bands)
        hazard_logits = self.hazard_out(hidden)
        return {
            "hazard_logits": hazard_logits,
            "confidence_logits": self.confidence_out(hidden).squeeze(-1),
            **decode_h1_outputs(hazard_logits),
        }

    def forward(self, feature: torch.Tensor, output_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        ground = self.ground_pre(feature)
        ground = F.interpolate(ground, output_hw, mode="bilinear", align_corners=True)
        return {
            "ground_logits": self.ground_out(ground),
            **self.forward_bands(self.band_pool(feature)),
        }


class DepthArtAssistiveGeometryH1(DepthArtAssistiveGeometry):
    """DepthART feature extractor with the isolated H1 survival head."""

    def __init__(self, metric_depthart: nn.Module) -> None:
        super().__init__(metric_depthart)
        self.assistive_heads = QsfH1TaskHeads(channels=48)

    def extract_band_features(
        self,
        image: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the frozen shared DPT feature and pool it into three bands."""

        height, width = image.shape[-2:]
        cameras = self.metric_depthart.cam_embedder(
            intrinsics,
            height,
            width,
            image.device,
        )
        features = self.metric_depthart.pretrained.forward_with_adapters(
            image,
            adapters=[
                self.metric_depthart.daa1,
                self.metric_depthart.daa2,
                self.metric_depthart.daa3,
                self.metric_depthart.daa4,
            ],
            cams=list(cameras),
        )
        _, shared = self._decode(list(features), (height, width))
        return self.assistive_heads.band_pool(shared)


H1_LOSS_LAMBDAS = {
    "masked_log_depth": 1.0,
    "valid_neighbor_log_gradient": 0.5,
    "ground_bce": 0.5,
    "ground_plane_depth": 0.25,
    "survival_nll": 1.0,
    "false_clear_extra": 2.0,
    "confidence_bce": 0.5,
}


def h1_confidence_correctness(
    outputs: dict[str, torch.Tensor],
    compiled: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_occupied = outputs["occupancy_probability"].detach() >= 0.5
    truth_occupied = compiled["occupancy"] >= 0.5
    occupancy_valid = compiled["occupancy_valid"].bool()
    occupancy_correct = ((predicted_occupied == truth_occupied) | ~occupancy_valid).all(dim=-1)
    event_correct = (
        outputs["clearance_m"].detach() - compiled["clearance_m"]
    ).abs() <= 0.25
    censor_correct = outputs["tail_probability"].detach() >= 0.5
    distribution_correct = torch.where(
        compiled["event_observed"].bool(),
        event_correct,
        censor_correct,
    )
    correct = occupancy_correct & distribution_correct
    valid = compiled["distribution_valid"].bool()
    return correct.to(outputs["confidence_logits"].dtype), valid


def compute_h1_band_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    active_losses: Iterable[str] = (
        "survival_nll",
        "false_clear_extra",
        "confidence_bce",
    ),
) -> dict[str, torch.Tensor]:
    active = tuple(active_losses)
    allowed = {"survival_nll", "false_clear_extra", "confidence_bce"}
    unknown = set(active) - allowed
    if unknown:
        raise ValueError(f"unknown H1 band losses: {sorted(unknown)}")
    compiled = compile_h1_targets(targets)
    raw: dict[str, torch.Tensor] = {}
    raw["survival_nll"], _ = discrete_survival_nll(outputs["hazard_logits"], compiled)
    raw["false_clear_extra"] = _masked_mean(
        -outputs["occupancy_probability"].clamp_min(1e-7).log(),
        compiled["occupancy_valid"].bool() & (compiled["occupancy"] >= 0.5),
    )
    confidence_target, confidence_valid = h1_confidence_correctness(outputs, compiled)
    raw["confidence_bce"] = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["confidence_logits"],
            confidence_target,
            reduction="none",
        ),
        confidence_valid,
    )
    weighted = {name: raw[name] * H1_LOSS_LAMBDAS[name] for name in active}
    total = sum(weighted.values(), outputs["hazard_logits"].sum() * 0.0)
    return {
        "total": total,
        **{f"raw/{name}": raw[name] for name in active},
        **{f"weighted/{name}": value for name, value in weighted.items()},
        "compiled/distribution_valid_count": compiled["distribution_valid"].sum(),
        "compiled/unknown_count": compiled["unknown"].sum(),
    }


def compute_h1_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    active_losses: Iterable[str],
) -> dict[str, torch.Tensor]:
    active = tuple(active_losses)
    unknown = set(active) - set(H1_LOSS_LAMBDAS)
    if unknown:
        raise ValueError(f"unknown H1 losses: {sorted(unknown)}")
    compiled = compile_h1_targets(targets)
    predicted = outputs["dense_depth_m"].clamp_min(1e-4)
    truth = targets["dense_depth_m"].clamp_min(1e-4)
    valid = targets["depth_valid"].bool()
    weights = near_field_weights(truth)
    raw: dict[str, torch.Tensor] = {}
    raw["masked_log_depth"] = _masked_mean(
        (predicted.log() - truth.log()).abs(),
        valid,
        weights,
    )
    log_error = predicted.log() - truth.log()
    horizontal_valid = valid[..., :, 1:] & valid[..., :, :-1]
    vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
    horizontal = _masked_mean(
        (log_error[..., :, 1:] - log_error[..., :, :-1]).abs(),
        horizontal_valid,
    )
    vertical = _masked_mean(
        (log_error[..., 1:, :] - log_error[..., :-1, :]).abs(),
        vertical_valid,
    )
    raw["valid_neighbor_log_gradient"] = 0.5 * (horizontal + vertical)
    raw["ground_bce"] = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["ground_logits"],
            targets["ground_probability"],
            reduction="none",
        ),
        targets["ground_label_valid"].bool(),
    )
    raw["ground_plane_depth"] = ground_plane_depth_loss(predicted, targets)
    band_losses = compute_h1_band_losses(
        outputs,
        targets,
        active_losses=[
            name
            for name in active
            if name in {"survival_nll", "false_clear_extra", "confidence_bce"}
        ],
    )
    for name in ("survival_nll", "false_clear_extra", "confidence_bce"):
        if name in active:
            raw[name] = band_losses[f"raw/{name}"]
    weighted = {name: raw[name] * H1_LOSS_LAMBDAS[name] for name in active}
    total = sum(weighted.values(), predicted.sum() * 0.0)
    return {
        "total": total,
        **{f"raw/{name}": raw[name] for name in active},
        **{f"weighted/{name}": value for name, value in weighted.items()},
        "compiled/distribution_valid_count": compiled["distribution_valid"].sum(),
        "compiled/unknown_count": compiled["unknown"].sum(),
    }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def h1_parameter_budget() -> dict[str, Any]:
    direct = AssistiveTaskHeads(channels=48, hidden=32)
    h1 = QsfH1TaskHeads(channels=48, hidden=32)
    return {
        "direct_task_head_parameters": parameter_count(direct),
        "h1_task_head_parameters": parameter_count(h1),
        "exact_match": parameter_count(direct) == parameter_count(h1),
        "hazard_bins": list(HAZARD_BIN_EDGES_M),
    }
