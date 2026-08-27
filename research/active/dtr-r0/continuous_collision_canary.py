"""Controlled falsifier for between-sample static route collisions.

The canary keeps the production S4 collision primitive and frozen 0.1-second
route cadence, but deliberately constructs long chords whose endpoints are
clear while their interiors contact an obstacle.  It is geometry evidence,
not a natural-motion or detector result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from coda_static_ceiling import (
    ROUTE_HALF_WIDTH_M,
    ROUTE_SAMPLE_STEP_S,
    point_to_box_clearance,
    segment_to_box_entry_fraction,
)


SCHEMA = "dtr-static-continuous-collision-canary-v1"
DENSE_STEPS_PER_CHORD = 24_000


@dataclass(frozen=True)
class CanaryCase:
    name: str
    route_points: tuple[tuple[float, float, float], ...]
    box_x: float
    box_y: float
    box_yaw: float
    length_m: float
    width_m: float


def sampled_entry_s(case: CanaryCase) -> float | None:
    for time_s, point_x, point_y in case.route_points:
        if point_to_box_clearance(
            point_x,
            point_y,
            case.box_x,
            case.box_y,
            case.box_yaw,
            case.length_m,
            case.width_m,
        ) <= ROUTE_HALF_WIDTH_M:
            return time_s
    return None


def continuous_entry_s(case: CanaryCase) -> float | None:
    for left, right in zip(case.route_points, case.route_points[1:]):
        fraction = segment_to_box_entry_fraction(
            left[1],
            left[2],
            right[1],
            right[2],
            case.box_x,
            case.box_y,
            case.box_yaw,
            case.length_m,
            case.width_m,
            ROUTE_HALF_WIDTH_M,
        )
        if fraction is not None:
            return left[0] + fraction * (right[0] - left[0])
    return None


def dense_oracle_entry_s(case: CanaryCase) -> float | None:
    for left, right in zip(case.route_points, case.route_points[1:]):
        for step in range(DENSE_STEPS_PER_CHORD + 1):
            fraction = step / DENSE_STEPS_PER_CHORD
            point_x = left[1] + fraction * (right[1] - left[1])
            point_y = left[2] + fraction * (right[2] - left[2])
            if point_to_box_clearance(
                point_x,
                point_y,
                case.box_x,
                case.box_y,
                case.box_yaw,
                case.length_m,
                case.width_m,
            ) <= ROUTE_HALF_WIDTH_M + 1e-12:
                return left[0] + fraction * (right[0] - left[0])
    return None


def cases() -> Sequence[CanaryCase]:
    return (
        CanaryCase(
            name="thin_crossing",
            route_points=((0.0, -1.2, 0.0), (0.1, 1.2, 0.0)),
            box_x=0.0,
            box_y=0.0,
            box_yaw=0.0,
            length_m=0.02,
            width_m=0.10,
        ),
        CanaryCase(
            name="grazing_contact",
            route_points=((0.0, -1.2, 0.70), (0.1, 1.2, 0.70)),
            box_x=0.0,
            box_y=0.0,
            box_yaw=0.0,
            length_m=0.10,
            width_m=0.10,
        ),
        CanaryCase(
            name="turn_entry_contact",
            route_points=(
                (0.0, -2.0, -2.0),
                (0.1, 0.0, -2.0),
                (0.2, 2.0, 0.0),
            ),
            box_x=1.0,
            box_y=-1.0,
            box_yaw=0.0,
            length_m=0.10,
            width_m=0.10,
        ),
        CanaryCase(
            name="fast_transverse_crossing",
            route_points=((0.0, 0.0, -2.0), (0.1, 0.0, 2.0)),
            box_x=0.0,
            box_y=0.0,
            box_yaw=0.0,
            length_m=0.10,
            width_m=0.10,
        ),
    )


def evaluate() -> dict[str, object]:
    case_results: list[dict[str, object]] = []
    for case in cases():
        sampled = sampled_entry_s(case)
        continuous = continuous_entry_s(case)
        dense = dense_oracle_entry_s(case)
        dense_resolution_s = max(
            right[0] - left[0]
            for left, right in zip(case.route_points, case.route_points[1:])
        ) / DENSE_STEPS_PER_CHORD
        case_results.append(
            {
                "name": case.name,
                "sampled_entry_s": sampled,
                "continuous_entry_s": continuous,
                "dense_oracle_entry_s": dense,
                "sampled_detected": sampled is not None,
                "continuous_detected": continuous is not None,
                "dense_oracle_detected": dense is not None,
                "continuous_dense_entry_agreement": bool(
                    continuous is not None
                    and dense is not None
                    and abs(continuous - dense) <= dense_resolution_s + 1e-12
                ),
            }
        )

    sampled_recalled = sum(bool(item["sampled_detected"]) for item in case_results)
    continuous_recalled = sum(bool(item["continuous_detected"]) for item in case_results)
    dense_recalled = sum(bool(item["dense_oracle_detected"]) for item in case_results)
    passed = bool(
        sampled_recalled == 0
        and continuous_recalled == len(case_results)
        and dense_recalled == len(case_results)
        and all(bool(item["continuous_dense_entry_agreement"]) for item in case_results)
    )
    return {
        "schema": SCHEMA,
        "claim_ceiling": "CONTROLLED_GEOMETRY_CANARY_ONLY",
        "protocol": {
            "route_sample_step_s": ROUTE_SAMPLE_STEP_S,
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "sampled_arm": "collision checked only at route sample points",
            "continuous_arm": "analytic segment time-of-impact against an oriented box expanded by route half width",
            "dense_oracle_steps_per_chord": DENSE_STEPS_PER_CHORD,
        },
        "cases": case_results,
        "summary": {
            "positive_cases": len(case_results),
            "sampled_recalled": sampled_recalled,
            "continuous_recalled": continuous_recalled,
            "dense_oracle_recalled": dense_recalled,
            "interpolation_misses_recovered": continuous_recalled - sampled_recalled,
            "gate": "PASS" if passed else "FAIL",
        },
        "limitations": [
            "The chords are deliberate geometric stress cases, not a natural wearer-speed distribution.",
            "The canary validates collision geometry only; public replay supplies false-alert evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2))
    if result["summary"]["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
