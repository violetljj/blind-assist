"""Matched counterfactuals for localizing the L10M-B0 state transition bug."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .evaluation import Arm, Evidence, Hazard, ProgressStatus, Truth, run_episode

PROTOCOL_ID = "L10M-B0-B-MATCHED-COUNTERFACTUALS-V1"
SCENARIOS = ("transient_no_progress", "true_stuck", "recovery_then_progress", "recovery_plus_arrival", "reactive_solvable", "uncertain_progress")


def _rows(name: str, specs: list[dict]) -> tuple[list[Evidence], list[Truth]]:
    evidence, truth = [], []
    for step, spec in enumerate(specs):
        evidence.append(Evidence(name, step, **{key: spec[key] for key in ("alignment", "center_hazard", "quality", "stale", "conflict", "target_visible", "progress_signal")}))
        truth.append(Truth(name, step, spec["truth_progress"], arrived=spec.get("arrived", False), unsafe_forward=spec.get("unsafe_forward", False)))
    return evidence, truth


def _s(*, signal: float | None, truth_progress: float, arrived: bool = False, quality: float = 0.95, stale: bool = False, conflict: bool = False, hazard: Hazard = Hazard.LOW):
    return {"alignment": 0.0, "center_hazard": hazard, "quality": quality, "stale": stale, "conflict": conflict, "target_visible": True, "progress_signal": signal, "truth_progress": truth_progress, "arrived": arrived}


def build_matrix() -> list[tuple[str, list[Evidence], list[Truth], bool]]:
    return [
        ("transient_no_progress", *_rows("transient_no_progress", [_s(signal=0.0, truth_progress=0.0), _s(signal=1.0, truth_progress=1.0, arrived=True)]), True),
        ("true_stuck", *_rows("true_stuck", [_s(signal=0.0, truth_progress=0.0), _s(signal=0.0, truth_progress=0.0), _s(signal=0.0, truth_progress=0.0), _s(signal=1.0, truth_progress=1.0, arrived=True)]), False),
        ("recovery_then_progress", *_rows("recovery_then_progress", [_s(signal=0.0, truth_progress=0.0), _s(signal=0.0, truth_progress=0.0), _s(signal=1.0, truth_progress=1.0), _s(signal=1.0, truth_progress=1.0, arrived=True)]), False),
        ("recovery_plus_arrival", *_rows("recovery_plus_arrival", [_s(signal=0.0, truth_progress=0.0), _s(signal=0.0, truth_progress=0.0), _s(signal=1.0, truth_progress=1.0, arrived=True)]), False),
        ("reactive_solvable", *_rows("reactive_solvable", [_s(signal=0.5, truth_progress=0.5), _s(signal=1.0, truth_progress=1.0, arrived=True)]), True),
        ("uncertain_progress", *_rows("uncertain_progress", [_s(signal=None, truth_progress=0.0, quality=0.2, stale=True), _s(signal=None, truth_progress=0.0, quality=0.2, stale=True), _s(signal=1.0, truth_progress=1.0, arrived=True)]), False),
    ]


def run_matrix() -> dict:
    results = []
    for name, evidence, truth, reactive_solvable in build_matrix():
        arms = {}
        for arm in Arm:
            stats = run_episode(arm, evidence, truth)
            arms[arm.value] = {"success": stats.success, "unsafe_actions": stats.unsafe_actions, "actions": [action.value for action in stats.actions], "stuck_detection_step": stats.stuck_detection_step, "recovery_success": stats.recovery_success, "oscillations": stats.oscillations}
        results.append({"scenario": name, "reactive_solvable": reactive_solvable, "arms": arms})
    canonical = json.dumps(results, sort_keys=True, separators=(",", ":"))
    preservation = [row for row in results if row["reactive_solvable"] and row["arms"][Arm.REACTIVE.value]["success"]]
    return {"protocol_id": PROTOCOL_ID, "claim_ceiling": "matched mechanism localization only; not representation admission or end-to-end evidence", "scenario_count": len(results), "matrix_sha256": hashlib.sha256(canonical.encode()).hexdigest(), "progress_states": [status.value for status in ProgressStatus], "reactive_solvable_preservation": {"denominator": len(preservation), "stateful_rate": sum(row["arms"][Arm.STATEFUL.value]["success"] for row in preservation) / len(preservation)}, "scenarios": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_matrix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
