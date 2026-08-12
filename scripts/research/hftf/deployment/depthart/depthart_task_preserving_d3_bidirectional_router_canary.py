#!/usr/bin/env python3
"""Synthetic mechanics canary for the DepthART D3 correction router."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "blindassist_depthart_task_preserving_d3_bidirectional_router_canary_v1"
KNOWN_STATES = frozenset({"CLEAR", "OCCUPIED"})
VALID_STATES = KNOWN_STATES | {"UNKNOWN_GROUND"}


@dataclass(frozen=True)
class RouterPolicy:
    """Frozen decision mechanics; these values are not learned thresholds."""

    horizons_m: tuple[float, ...] = (1.0, 1.5, 2.0)
    strong_certificate_threshold: float = 0.9
    opposite_certificate_max: float = 0.1
    contradiction_threshold: float = 0.9
    hard_evidence_required_for_override: bool = True

    def validate(self) -> None:
        if tuple(sorted(set(self.horizons_m))) != self.horizons_m:
            raise ValueError("horizons must be strictly increasing")
        if not all(math.isfinite(value) and value > 0 for value in self.horizons_m):
            raise ValueError("horizons must be finite and positive")
        for value, name in (
            (self.strong_certificate_threshold, "strong_certificate_threshold"),
            (self.opposite_certificate_max, "opposite_certificate_max"),
            (self.contradiction_threshold, "contradiction_threshold"),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be inside [0, 1]")
        if self.opposite_certificate_max >= self.strong_certificate_threshold:
            raise ValueError("opposite_certificate_max must be below the strong threshold")
        if self.contradiction_threshold < self.strong_certificate_threshold:
            raise ValueError("contradiction_threshold must not be below the strong threshold")
        if not self.hard_evidence_required_for_override:
            raise ValueError("D3 requires hard evidence for every override")


def _finite_probabilities(values: Iterable[float], expected: int, name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != expected:
        raise ValueError(f"{name} requires one probability per horizon")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in result):
        raise ValueError(f"{name} values must be finite and inside [0, 1]")
    return result


def monotone_occupied_certificates(values: Iterable[float]) -> list[float]:
    """An obstacle seen at a nearer horizon remains relevant at later horizons."""

    result: list[float] = []
    running = 0.0
    for value in values:
        running = max(running, float(value))
        result.append(running)
    return result


def monotone_clear_certificates(values: Iterable[float]) -> list[float]:
    """Clear-through confidence cannot increase with a longer horizon."""

    result: list[float] = []
    running = 1.0
    for value in values:
        running = min(running, float(value))
        result.append(running)
    return result


def _validate_baseline(states: Sequence[str], expected: int) -> list[str]:
    result = [str(value) for value in states]
    if len(result) != expected or any(value not in VALID_STATES for value in result):
        raise ValueError("baseline requires one valid state per horizon")
    unknown_count = sum(value == "UNKNOWN_GROUND" for value in result)
    if unknown_count not in (0, expected):
        raise ValueError("baseline knownness is band-level and must be uniform across horizons")
    seen_occupied = False
    for value in result:
        if value == "OCCUPIED":
            seen_occupied = True
        elif value == "CLEAR" and seen_occupied:
            raise ValueError("baseline cannot become CLEAR after OCCUPIED")
    return result


def _validate_hard_evidence(values: Sequence[bool], expected: int) -> list[bool]:
    if len(values) != expected or any(type(value) is not bool for value in values):
        raise ValueError("hard_evidence requires one boolean per horizon")
    return list(values)


def compose_route(
    *,
    baseline_states: Sequence[str],
    clear_certificates: Iterable[float],
    occupied_certificates: Iterable[float],
    hard_evidence: Sequence[bool],
    policy: RouterPolicy = RouterPolicy(),
) -> dict[str, Any]:
    """Apply selective RELEASE/VETO certificates to a frozen baseline.

    Weak or absent evidence preserves a known baseline. Contradictory strong
    certificates return UNKNOWN rather than forcing either decision.
    """

    policy.validate()
    count = len(policy.horizons_m)
    baseline = _validate_baseline(baseline_states, count)
    evidence = _validate_hard_evidence(hard_evidence, count)
    clear_raw = _finite_probabilities(clear_certificates, count, "clear_certificates")
    occupied_raw = _finite_probabilities(occupied_certificates, count, "occupied_certificates")
    clear = monotone_clear_certificates(clear_raw)
    occupied = monotone_occupied_certificates(occupied_raw)

    raw_states: list[str] = []
    actions: list[str] = []
    for base, clear_value, occupied_value, evidence_present in zip(
        baseline, clear, occupied, evidence
    ):
        if not evidence_present:
            raw_states.append(base if base in KNOWN_STATES else "UNKNOWN_GROUND")
            actions.append("KEEP_BASELINE_NO_HARD_EVIDENCE")
            continue
        if (
            clear_value >= policy.contradiction_threshold
            and occupied_value >= policy.contradiction_threshold
        ):
            raw_states.append("UNKNOWN_GROUND")
            actions.append("CONTRADICTION_TO_UNKNOWN")
            continue
        if (
            clear_value >= policy.strong_certificate_threshold
            and occupied_value <= policy.opposite_certificate_max
        ):
            raw_states.append("CLEAR")
            actions.append("RELEASE_TO_CLEAR" if base != "CLEAR" else "KEEP_BASELINE_CLEAR_CERTIFIED")
            continue
        if (
            occupied_value >= policy.strong_certificate_threshold
            and clear_value <= policy.opposite_certificate_max
        ):
            raw_states.append("OCCUPIED")
            actions.append(
                "VETO_TO_OCCUPIED" if base != "OCCUPIED" else "KEEP_BASELINE_OCCUPIED_CERTIFIED"
            )
            continue
        raw_states.append(base if base in KNOWN_STATES else "UNKNOWN_GROUND")
        actions.append("KEEP_BASELINE_WEAK_OR_AMBIGUOUS_CERTIFICATE")

    final_states: list[str] = []
    projection: list[str] = []
    blocked_clear = False
    for raw_state in raw_states:
        if raw_state == "OCCUPIED":
            final_states.append(raw_state)
            projection.append("AS_COMPOSED")
            blocked_clear = True
        elif raw_state == "UNKNOWN_GROUND":
            final_states.append(raw_state)
            projection.append("AS_COMPOSED")
            blocked_clear = True
        elif blocked_clear:
            final_states.append("UNKNOWN_GROUND")
            projection.append("CLEAR_BLOCKED_BY_EARLIER_NONCLEAR")
        else:
            final_states.append(raw_state)
            projection.append("AS_COMPOSED")

    return {
        "horizons_m": list(policy.horizons_m),
        "baseline_states": baseline,
        "hard_evidence": evidence,
        "clear_certificates_raw": clear_raw,
        "clear_certificates_monotone": clear,
        "occupied_certificates_raw": occupied_raw,
        "occupied_certificates_monotone": occupied,
        "actions": actions,
        "raw_states": raw_states,
        "final_states": final_states,
        "horizon_projection": projection,
    }


def run_canary() -> dict[str, Any]:
    policy = RouterPolicy()
    cases = {
        "neutral_preserves_baseline": compose_route(
            baseline_states=("CLEAR", "CLEAR", "OCCUPIED"),
            clear_certificates=(0.5, 0.5, 0.5),
            occupied_certificates=(0.5, 0.5, 0.5),
            hard_evidence=(True, True, True),
            policy=policy,
        ),
        "clear_release_corrects_near_false_block": compose_route(
            baseline_states=("OCCUPIED", "OCCUPIED", "OCCUPIED"),
            clear_certificates=(0.95, 0.5, 0.5),
            occupied_certificates=(0.05, 0.05, 0.05),
            hard_evidence=(True, True, True),
            policy=policy,
        ),
        "occupied_veto_corrects_far_false_clear": compose_route(
            baseline_states=("CLEAR", "CLEAR", "CLEAR"),
            clear_certificates=(0.05, 0.05, 0.05),
            occupied_certificates=(0.05, 0.05, 0.95),
            hard_evidence=(True, True, True),
            policy=policy,
        ),
        "contradiction_becomes_unknown": compose_route(
            baseline_states=("CLEAR", "CLEAR", "CLEAR"),
            clear_certificates=(0.95, 0.95, 0.95),
            occupied_certificates=(0.95, 0.95, 0.95),
            hard_evidence=(True, True, True),
            policy=policy,
        ),
        "missing_evidence_cannot_override": compose_route(
            baseline_states=("CLEAR", "CLEAR", "CLEAR"),
            clear_certificates=(0.05, 0.05, 0.05),
            occupied_certificates=(0.95, 0.95, 0.95),
            hard_evidence=(False, False, False),
            policy=policy,
        ),
        "unknown_promoted_by_strong_clear_certificate": compose_route(
            baseline_states=("UNKNOWN_GROUND",) * 3,
            clear_certificates=(0.95, 0.95, 0.95),
            occupied_certificates=(0.05, 0.05, 0.05),
            hard_evidence=(True, True, True),
            policy=policy,
        ),
        "unknown_without_evidence_stays_unknown": compose_route(
            baseline_states=("UNKNOWN_GROUND",) * 3,
            clear_certificates=(0.95, 0.95, 0.95),
            occupied_certificates=(0.05, 0.05, 0.05),
            hard_evidence=(False, False, False),
            policy=policy,
        ),
        "certificate_projection_is_horizon_safe": compose_route(
            baseline_states=("CLEAR", "CLEAR", "CLEAR"),
            clear_certificates=(0.95, 0.95, 0.05),
            occupied_certificates=(0.95, 0.05, 0.05),
            hard_evidence=(True, True, True),
            policy=policy,
        ),
    }
    deterministic_replay = compose_route(
        baseline_states=("CLEAR", "CLEAR", "OCCUPIED"),
        clear_certificates=(0.5, 0.5, 0.5),
        occupied_certificates=(0.5, 0.5, 0.5),
        hard_evidence=(True, True, True),
        policy=policy,
    )
    checks = {
        "neutral_preserves_baseline_exactly": cases["neutral_preserves_baseline"]["final_states"]
        == ["CLEAR", "CLEAR", "OCCUPIED"],
        "clear_release_is_selective": cases["clear_release_corrects_near_false_block"]["final_states"]
        == ["CLEAR", "OCCUPIED", "OCCUPIED"],
        "occupied_veto_is_selective": cases["occupied_veto_corrects_far_false_clear"]["final_states"]
        == ["CLEAR", "CLEAR", "OCCUPIED"],
        "contradiction_fails_to_unknown": cases["contradiction_becomes_unknown"]["final_states"]
        == ["UNKNOWN_GROUND"] * 3,
        "missing_evidence_preserves_baseline": cases["missing_evidence_cannot_override"]["final_states"]
        == ["CLEAR"] * 3,
        "strong_evidence_can_promote_unknown": cases["unknown_promoted_by_strong_clear_certificate"][
            "final_states"
        ]
        == ["CLEAR"] * 3,
        "unknown_without_evidence_is_preserved": cases["unknown_without_evidence_stays_unknown"][
            "final_states"
        ]
        == ["UNKNOWN_GROUND"] * 3,
        "occupied_certificate_is_cumulative_max": cases["certificate_projection_is_horizon_safe"][
            "occupied_certificates_monotone"
        ]
        == [0.95, 0.95, 0.95],
        "clear_certificate_is_cumulative_min": cases["certificate_projection_is_horizon_safe"][
            "clear_certificates_monotone"
        ]
        == [0.95, 0.95, 0.05],
        "deterministic_replay_is_exact": deterministic_replay == cases["neutral_preserves_baseline"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema": SCHEMA,
        "status": status,
        "authority": "SYNTHETIC_MECHANICS_ONLY_NO_DATA_ACCURACY_MODEL_OR_CANDIDATE_AUTHORITY",
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
