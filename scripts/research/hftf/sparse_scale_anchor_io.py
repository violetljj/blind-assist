"""Validated JSONL transport for causal sparse metric-scale anchors."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path

from metric_scale_anchor import MetricScaleAnchor

SCHEMA = "hftf_sparse_metric_scale_anchor_r0"


def load_scale_anchors(path: Path) -> dict[str, list[MetricScaleAnchor]]:
    grouped: dict[str, list[MetricScaleAnchor]] = defaultdict(list)
    previous: dict[str, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != SCHEMA:
            raise ValueError(f"line {line_number}: unsupported anchor schema")
        sequence = str(row["sequence_id"])
        timestamp_ns = int(row["timestamp_ns"])
        scale = float(row["scale"])
        pair_count = int(row["pair_count"])
        residual = float(row["median_abs_ratio_residual"])
        source = str(row["source"])
        if previous.get(sequence, -1) >= timestamp_ns:
            raise ValueError(f"line {line_number}: anchor timestamps must increase")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"line {line_number}: scale must be finite and positive")
        if pair_count < 3:
            raise ValueError(f"line {line_number}: at least three pairs are required")
        if not math.isfinite(residual) or residual < 0:
            raise ValueError(f"line {line_number}: residual must be finite and nonnegative")
        if not source:
            raise ValueError(f"line {line_number}: source must be non-empty")
        grouped[sequence].append(
            MetricScaleAnchor(
                timestamp_ns=timestamp_ns,
                scale=scale,
                pair_count=pair_count,
                median_abs_ratio_residual=residual,
                source=source,
            )
        )
        previous[sequence] = timestamp_ns
    if not grouped:
        raise ValueError("anchor file contains no records")
    return dict(grouped)


class ScaleAnchorStream:
    def __init__(self, anchors: dict[str, list[MetricScaleAnchor]]) -> None:
        self.pending = {
            sequence: deque(values) for sequence, values in anchors.items()
        }

    def take_available(
        self, sequence_id: str, timestamp_ns: int
    ) -> list[MetricScaleAnchor]:
        queue = self.pending.get(sequence_id)
        available = []
        while queue and queue[0].timestamp_ns <= timestamp_ns:
            available.append(queue.popleft())
        return available
