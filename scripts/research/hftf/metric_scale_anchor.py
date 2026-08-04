"""Causal sparse metric-scale anchors for a fast clearance observer."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

BANDS = ("left", "center", "right")


@dataclass(frozen=True)
class MetricScaleAnchor:
    timestamp_ns: int
    scale: float
    pair_count: int
    median_abs_ratio_residual: float
    source: str


def estimate_scale_anchor(
    timestamp_ns: int,
    candidate_frames: Sequence[Mapping[str, float | None]],
    metric_frames: Sequence[Mapping[str, float | None]],
    source: str,
) -> MetricScaleAnchor:
    if len(candidate_frames) != len(metric_frames) or not candidate_frames:
        raise ValueError("candidate and metric anchor windows must be non-empty and aligned")
    ratios = []
    for candidate, metric in zip(candidate_frames, metric_frames, strict=True):
        for band in BANDS:
            predicted = candidate.get(band)
            observed = metric.get(band)
            if predicted is None or observed is None:
                continue
            predicted_value = float(predicted)
            observed_value = float(observed)
            if (
                math.isfinite(predicted_value)
                and math.isfinite(observed_value)
                and predicted_value > 0
                and observed_value > 0
            ):
                ratios.append(observed_value / predicted_value)
    if len(ratios) < len(BANDS):
        raise ValueError("a scale anchor requires at least one valid sample per band")
    scale = float(statistics.median(ratios))
    residual = float(statistics.median(abs(value - scale) for value in ratios))
    return MetricScaleAnchor(
        timestamp_ns=int(timestamp_ns),
        scale=scale,
        pair_count=len(ratios),
        median_abs_ratio_residual=residual,
        source=source,
    )


class MetricScaleTracker:
    def __init__(self, max_age_ns: int) -> None:
        if max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive")
        self.max_age_ns = int(max_age_ns)
        self.anchor: MetricScaleAnchor | None = None

    def update(self, anchor: MetricScaleAnchor) -> None:
        if self.anchor is not None and anchor.timestamp_ns <= self.anchor.timestamp_ns:
            raise ValueError("scale anchors must be strictly time ordered")
        self.anchor = anchor

    def apply(
        self, timestamp_ns: int, clearance: Mapping[str, float | None]
    ) -> dict[str, object]:
        receipt = self.resolve(timestamp_ns)
        if receipt["status"] != "VALID":
            return receipt
        assert self.anchor is not None
        scaled = {
            band: (
                None
                if clearance.get(band) is None
                else float(clearance[band]) * self.anchor.scale
            )
            for band in BANDS
        }
        if all(value is None for value in scaled.values()):
            return {"status": "UNKNOWN_NO_CLEARANCE_BANDS"}
        return {**receipt, "bands_m": scaled}

    def resolve(self, timestamp_ns: int) -> dict[str, object]:
        """Resolve the causal scale receipt without depending on legacy bands."""

        if self.anchor is None:
            return {"status": "UNKNOWN_NO_METRIC_SCALE_ANCHOR"}
        age_ns = int(timestamp_ns) - self.anchor.timestamp_ns
        if age_ns < 0:
            return {"status": "UNKNOWN_FUTURE_METRIC_SCALE_ANCHOR"}
        if age_ns > self.max_age_ns:
            return {
                "status": "UNKNOWN_STALE_METRIC_SCALE_ANCHOR",
                "anchor_age_ns": age_ns,
            }
        return {
            "status": "VALID",
            "scale": self.anchor.scale,
            "anchor_age_ns": age_ns,
            "anchor_pair_count": self.anchor.pair_count,
            "anchor_source": self.anchor.source,
        }
