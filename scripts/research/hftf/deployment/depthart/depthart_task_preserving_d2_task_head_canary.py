#!/usr/bin/env python3
"""Mechanics-only canary for the DepthART D2 task-evidence head contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "blindassist_depthart_task_preserving_d2_task_head_canary_v1"


@dataclass(frozen=True)
class TaskHeadPolicy:
    horizons_m: tuple[float, ...] = (1.0, 1.5, 2.0)
    occupancy_threshold: float = 0.5
    known_probability_threshold: float = 0.8
    minimum_valid_depth_fraction: float = 0.05
    minimum_ground_support_fraction: float = 0.02
    minimum_band_support_points: int = 20
    maximum_clearance_residual_m: float = 0.5

    def validate(self) -> None:
        if tuple(sorted(set(self.horizons_m))) != self.horizons_m:
            raise ValueError("horizons must be strictly increasing")
        if not all(value > 0 for value in self.horizons_m):
            raise ValueError("horizons must be positive")
        for value, name in (
            (self.occupancy_threshold, "occupancy_threshold"),
            (self.known_probability_threshold, "known_probability_threshold"),
            (self.minimum_valid_depth_fraction, "minimum_valid_depth_fraction"),
            (self.minimum_ground_support_fraction, "minimum_ground_support_fraction"),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must be inside (0, 1)")
        if self.minimum_band_support_points <= 0:
            raise ValueError("minimum_band_support_points must be positive")
        if self.maximum_clearance_residual_m <= 0:
            raise ValueError("maximum_clearance_residual_m must be positive")


def sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def monotone_occupancy_probabilities(logits: Iterable[float]) -> list[float]:
    probabilities = [sigmoid(float(value)) for value in logits]
    result: list[float] = []
    running = 0.0
    for value in probabilities:
        running = max(running, value)
        result.append(running)
    return result


def bounded_clearance(
    raw_clearance_m: float | None,
    residual_logit: float,
    policy: TaskHeadPolicy,
) -> float | None:
    if raw_clearance_m is None:
        return None
    if not math.isfinite(raw_clearance_m) or raw_clearance_m < 0:
        raise ValueError("raw clearance must be finite and non-negative")
    residual = policy.maximum_clearance_residual_m * math.tanh(float(residual_logit))
    return max(0.0, raw_clearance_m + residual)


def compose_band(
    *,
    occupancy_logits: Iterable[float],
    known_probability: float,
    raw_clearance_m: float | None,
    residual_logit: float,
    valid_depth_fraction: float,
    ground_support_fraction: float,
    band_support_points: int,
    ground_plane_available: bool,
    policy: TaskHeadPolicy = TaskHeadPolicy(),
) -> dict[str, Any]:
    policy.validate()
    logits = [float(value) for value in occupancy_logits]
    if len(logits) != len(policy.horizons_m) or not all(math.isfinite(value) for value in logits):
        raise ValueError("one finite occupancy logit is required per horizon")
    if not math.isfinite(known_probability) or not 0 <= known_probability <= 1:
        raise ValueError("known_probability must be inside [0, 1]")
    hard_evidence = (
        ground_plane_available
        and valid_depth_fraction >= policy.minimum_valid_depth_fraction
        and ground_support_fraction >= policy.minimum_ground_support_fraction
        and band_support_points >= policy.minimum_band_support_points
    )
    probabilities = monotone_occupancy_probabilities(logits)
    known = hard_evidence and known_probability >= policy.known_probability_threshold
    states = [
        ("OCCUPIED" if probability >= policy.occupancy_threshold else "CLEAR")
        if known
        else "UNKNOWN_GROUND"
        for probability in probabilities
    ]
    return {
        "horizons_m": list(policy.horizons_m),
        "occupancy_probabilities": probabilities,
        "states": states,
        "known": known,
        "hard_evidence": hard_evidence,
        "clearance_m": bounded_clearance(raw_clearance_m, residual_logit, policy),
    }


def run_canary() -> dict[str, Any]:
    policy = TaskHeadPolicy()
    base = {
        "known_probability": 0.95,
        "raw_clearance_m": 1.2,
        "residual_logit": 0.0,
        "valid_depth_fraction": 0.8,
        "ground_support_fraction": 0.2,
        "band_support_points": 200,
        "ground_plane_available": True,
        "policy": policy,
    }
    cases = {
        "far_obstacle_monotone": compose_band(occupancy_logits=(-3.0, -1.0, 2.0), **base),
        "nonmonotone_logits_fail_safe": compose_band(occupancy_logits=(3.0, -3.0, -3.0), **base),
        "missing_ground_is_unknown": compose_band(
            occupancy_logits=(3.0, 3.0, 3.0), **{**base, "ground_plane_available": False}
        ),
        "low_support_is_unknown": compose_band(
            occupancy_logits=(-3.0, -3.0, -3.0), **{**base, "band_support_points": 0}
        ),
        "bounded_positive_residual": compose_band(
            occupancy_logits=(-3.0, -1.0, 2.0), **{**base, "residual_logit": 100.0}
        ),
        "bounded_negative_residual": compose_band(
            occupancy_logits=(-3.0, -1.0, 2.0), **{**base, "raw_clearance_m": 0.1, "residual_logit": -100.0}
        ),
    }
    checks = {
        "far_obstacle_states": cases["far_obstacle_monotone"]["states"] == ["CLEAR", "CLEAR", "OCCUPIED"],
        "horizon_monotonicity": cases["nonmonotone_logits_fail_safe"]["states"]
        == ["OCCUPIED", "OCCUPIED", "OCCUPIED"],
        "missing_ground_unknown": all(
            value == "UNKNOWN_GROUND" for value in cases["missing_ground_is_unknown"]["states"]
        ),
        "low_support_unknown": all(
            value == "UNKNOWN_GROUND" for value in cases["low_support_is_unknown"]["states"]
        ),
        "positive_residual_bounded": math.isclose(
            cases["bounded_positive_residual"]["clearance_m"], 1.7, abs_tol=1e-12
        ),
        "negative_residual_nonnegative": cases["bounded_negative_residual"]["clearance_m"] == 0.0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema": SCHEMA,
        "status": status,
        "authority": "SYNTHETIC_MECHANICS_ONLY_NO_ACCURACY_OR_CANDIDATE_AUTHORITY",
        "policy": {**asdict(policy), "horizons_m": list(policy.horizons_m)},
        "cases": cases,
        "checks": checks,
        "stop": status != "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_canary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, indent=2) + "\n").encode("utf-8")
    args.output.write_bytes(encoded)
    print(json.dumps({"status": result["status"], "sha256": hashlib.sha256(encoded).hexdigest().upper()}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
