#!/usr/bin/env python3
"""Outcome-blind causal temporal candidates for Assistive Geometry phase D."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


BANDS = 3
DISTANCE_HORIZONS = 3
FUTURE_OFFSETS = 3
FEATURES_PER_BAND = 12
FRAME_FEATURES = BANDS * FEATURES_PER_BAND + 1
PARAMETER_BUDGET = 50_000


def _require_shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {expected}")


def encode_geometry_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode GeometryState without allowing UNKNOWN payloads to behave as clear."""

    clearance = state["clearance_m"]
    if clearance.ndim != 3 or clearance.shape[-1] != BANDS:
        raise ValueError("clearance_m must be [batch,time,band]")
    batch, steps, _ = clearance.shape
    band_shape = (batch, steps, BANDS)
    horizon_shape = (batch, steps, BANDS, DISTANCE_HORIZONS)
    clearance_valid = state["clearance_valid"].bool()
    occupancy = state["occupancy_probability"]
    confidence = state["task_confidence"]
    known = state["state_known"].bool()
    ground = state["ground_support"]
    captured = state["captured_at_s"]
    _require_shape("clearance_valid", clearance_valid, band_shape)
    _require_shape("occupancy_probability", occupancy, horizon_shape)
    _require_shape("task_confidence", confidence, horizon_shape)
    _require_shape("state_known", known, horizon_shape)
    _require_shape("ground_support", ground, band_shape)
    _require_shape("captured_at_s", captured, (batch, steps))
    if not bool(torch.isfinite(captured).all()):
        raise ValueError("captured_at_s contains non-finite values")
    if steps > 1 and not bool(((captured[:, 1:] - captured[:, :-1]) > 0).all()):
        raise ValueError("captured_at_s must be strictly increasing within each session")

    clearance_feature = torch.where(clearance_valid, clearance.clamp(0.0, 2.0) / 2.0, 0.0)
    occupancy_feature = torch.where(known, occupancy.clamp(0.0, 1.0), 0.0)
    confidence_feature = confidence.clamp(0.0, 1.0)
    per_band = torch.cat(
        (
            clearance_feature.unsqueeze(-1),
            clearance_valid.to(clearance.dtype).unsqueeze(-1),
            occupancy_feature,
            confidence_feature,
            known.to(clearance.dtype),
            ground.clamp(0.0, 1.0).unsqueeze(-1),
        ),
        dim=-1,
    )
    if per_band.shape[-1] != FEATURES_PER_BAND:
        raise RuntimeError("temporal feature contract drift")
    delta = torch.zeros_like(captured)
    if steps > 1:
        delta[:, 1:] = (captured[:, 1:] - captured[:, :-1]).clamp(max=1.0)
    return torch.cat((per_band.reshape(batch, steps, -1), delta.unsqueeze(-1)), dim=-1)


class TemporalOutputHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.future_clearance = nn.Linear(hidden, BANDS * FUTURE_OFFSETS)
        self.ttc = nn.Linear(hidden, BANDS)
        self.compute_gate = nn.Linear(hidden, 1)

    def forward(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, steps, _ = hidden_state.shape
        future = 2.0 * torch.tanh(self.future_clearance(hidden_state))
        return {
            "future_clearance_delta_m": future.reshape(batch, steps, BANDS, FUTURE_OFFSETS),
            "ttc_s": F.softplus(self.ttc(hidden_state)) + 1e-3,
            "compute_gate_logits": self.compute_gate(hidden_state),
        }


class GruTemporalCandidate(nn.Module):
    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.backbone = nn.GRU(FRAME_FEATURES, hidden, batch_first=True)
        self.head = TemporalOutputHead(hidden)

    def forward(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = encode_geometry_state(state)
        hidden, _ = self.backbone(encoded)
        return self.head(hidden)


class CausalTemporalBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, dilation=dilation)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        filtered = self.conv(F.pad(value, (self.left_padding, 0)))
        normalized = self.norm(filtered.transpose(1, 2)).transpose(1, 2)
        return value + F.silu(normalized)


class TcnTemporalCandidate(nn.Module):
    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(FRAME_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(CausalTemporalBlock(hidden, dilation) for dilation in (1, 2, 4))
        self.head = TemporalOutputHead(hidden)

    def forward(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        value = self.input_projection(encode_geometry_state(state).transpose(1, 2))
        for block in self.blocks:
            value = block(value)
        return self.head(value.transpose(1, 2))


class DiagonalSsmTemporalCandidate(nn.Module):
    """Small stable diagonal recurrence; this is not a Mamba claim."""

    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.input_projection = nn.Linear(FRAME_FEATURES, hidden)
        self.decay_logits = nn.Parameter(torch.zeros(hidden))
        self.norm = nn.LayerNorm(hidden)
        self.head = TemporalOutputHead(hidden)

    def forward(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = torch.tanh(self.input_projection(encode_geometry_state(state)))
        decay = torch.sigmoid(self.decay_logits).view(1, -1)
        recurrent = torch.zeros(inputs.shape[0], inputs.shape[-1], device=inputs.device, dtype=inputs.dtype)
        history: list[torch.Tensor] = []
        for index in range(inputs.shape[1]):
            recurrent = decay * recurrent + (1.0 - decay) * inputs[:, index]
            history.append(self.norm(recurrent))
        return self.head(torch.stack(history, dim=1))


CANDIDATES: dict[str, type[nn.Module]] = {
    "GRU": GruTemporalCandidate,
    "TCN": TcnTemporalCandidate,
    "DIAGONAL_SSM": DiagonalSsmTemporalCandidate,
}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_temporal_candidate(name: str, hidden: int = 48) -> nn.Module:
    if name not in CANDIDATES:
        raise ValueError(f"unsupported temporal candidate: {name}")
    model = CANDIDATES[name](hidden=hidden)
    count = parameter_count(model)
    if count > PARAMETER_BUDGET:
        raise ValueError(f"{name} parameter budget exceeded: {count} > {PARAMETER_BUDGET}")
    return model


def validate_output_contract(output: dict[str, torch.Tensor], batch: int, steps: int) -> None:
    expected: dict[str, tuple[int, ...]] = {
        "future_clearance_delta_m": (batch, steps, BANDS, FUTURE_OFFSETS),
        "ttc_s": (batch, steps, BANDS),
        "compute_gate_logits": (batch, steps, 1),
    }
    if set(output) != set(expected):
        raise ValueError(f"temporal output keys drift: {sorted(output)}")
    for name, shape in expected.items():
        _require_shape(name, output[name], shape)
        if not bool(torch.isfinite(output[name]).all()):
            raise ValueError(f"{name} contains non-finite values")


def candidate_receipt(name: str, model: nn.Module) -> dict[str, Any]:
    return {
        "candidate": name,
        "parameter_count": parameter_count(model),
        "parameter_budget": PARAMETER_BUDGET,
        "causal": True,
        "predicts_final_tristate": False,
        "unknown_remains_host_postprocess_authority": True,
    }
