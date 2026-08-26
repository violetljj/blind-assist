"""Frozen fusion arms for candidate-aligned evidence scores."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FusionOutput:
    candidate_scores: Tensor
    modality_weights: Tensor


def _validate(scores: Tensor, available: Tensor) -> None:
    if scores.ndim != 3:
        raise ValueError("scores must have shape [batch, candidates, modalities]")
    if available.shape != scores.shape or available.dtype != torch.bool:
        raise ValueError("available must be a bool tensor matching scores")
    if not torch.all(available.any(dim=-1)):
        raise ValueError("every candidate needs at least one available evidence source")


def fixed_equal_available_fusion(scores: Tensor, available: Tensor) -> FusionOutput:
    _validate(scores, available)
    weights = available.to(scores.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return FusionOutput((scores * weights).sum(dim=-1), weights)


class StaticLearnedFusion(nn.Module):
    """Strong learned fusion without observation-quality inputs.

    Missingness is observable and therefore included, but the arm cannot inspect
    blur, OCR confidence, GPS uncertainty, or any other quality measurement.
    """

    def __init__(self, modalities: int, hidden_dim: int = 32):
        super().__init__()
        self.modalities = modalities
        self.network = nn.Sequential(
            nn.Linear(modalities * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, scores: Tensor, available: Tensor) -> FusionOutput:
        _validate(scores, available)
        if scores.shape[-1] != self.modalities:
            raise ValueError("unexpected modality count")
        masked_scores = torch.where(available, scores, torch.zeros_like(scores))
        features = torch.cat((masked_scores, available.to(scores.dtype)), dim=-1)
        candidate_scores = self.network(features).squeeze(-1)
        # Static arm has no per-observation routing claim. These normalized
        # availability weights are diagnostic placeholders, not learned weights.
        weights = available.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return FusionOutput(candidate_scores, weights)


class QualityConditionedEvidenceRouter(nn.Module):
    """Route among evidence sources using only measured observation quality."""

    def __init__(self, modalities: int, quality_dim: int, hidden_dim: int = 32):
        super().__init__()
        if modalities < 2 or quality_dim < 1:
            raise ValueError("router needs at least two modalities and one quality field")
        self.modalities = modalities
        self.quality_dim = quality_dim
        self.modality_embedding = nn.Parameter(torch.zeros(modalities, hidden_dim))
        self.quality_encoder = nn.Sequential(
            nn.Linear(quality_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.route_head = nn.Linear(hidden_dim, 1)
        self.score_calibrator = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, scores: Tensor, available: Tensor, quality: Tensor) -> FusionOutput:
        _validate(scores, available)
        if scores.shape[-1] != self.modalities:
            raise ValueError("unexpected modality count")
        expected = (*scores.shape, self.quality_dim)
        if quality.shape != expected:
            raise ValueError(f"quality must have shape {expected}")
        if not torch.isfinite(quality).all():
            raise ValueError("quality must be finite; encode missingness in available")

        encoded = self.quality_encoder(quality)
        encoded = encoded + self.modality_embedding.view(1, 1, self.modalities, -1)
        route_logits = self.route_head(encoded).squeeze(-1)
        route_logits = route_logits.masked_fill(~available, torch.finfo(scores.dtype).min)
        weights = torch.softmax(route_logits, dim=-1)
        fused = (weights * scores).sum(dim=-1)

        available_scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
        strongest = available_scores.max(dim=-1).values
        count = available.sum(dim=-1).to(scores.dtype) / self.modalities
        calibration = self.score_calibrator(torch.stack((fused, strongest, count), dim=-1)).squeeze(-1)
        return FusionOutput(calibration, weights)
