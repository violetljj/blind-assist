"""Fresh B5-A cohort over the unchanged B4 finite benchmark mechanics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.research.l10m_b1.policy_space import PolicySpec

from .hard_benchmark import MOTIF_KEYS, build_instance, legal_neighbors
from .hard_benchmark import evaluate_instance as _evaluate_instance


BENCHMARK_PATH = Path(__file__).with_name("fresh_benchmark_v1.json")
EXPECTED_BENCHMARK_ID = "L10M-B5-FRESH-HARDER-COHORT-V1"


def load_fresh_benchmark(path: Path = BENCHMARK_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
        raise ValueError("unexpected B5-A benchmark identity")
    instances = payload.get("instances")
    if not isinstance(instances, list) or len(instances) != 3:
        raise ValueError("B5-A fresh cohort requires exactly three instances")
    ids = [row.get("instance_id") for row in instances]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("B5-A instance identities must be unique and non-empty")
    for row in instances:
        motifs = row.get("motifs")
        if not isinstance(motifs, dict) or set(motifs) != MOTIF_KEYS:
            raise ValueError(f"unexpected motif schema for {row.get('instance_id')}")
        if any(not isinstance(value, int) or value < 0 for value in motifs.values()):
            raise ValueError("motif counts must be non-negative integers")
    return payload


def evaluate_fresh_instance(spec: PolicySpec, instance: dict[str, Any]) -> dict[str, object]:
    return _evaluate_instance(spec, instance)


__all__ = [
    "BENCHMARK_PATH",
    "build_instance",
    "evaluate_fresh_instance",
    "legal_neighbors",
    "load_fresh_benchmark",
]
