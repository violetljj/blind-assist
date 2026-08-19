"""B0-D orthogonal arrival/stuck semantics canary.

B0-D does not modify the consumed B0-B matrix or the B0-C intervention.  It
uses a new synthetic cohort to separate stuck/recovery history from final
confirmed-arrival truth and checks the B0-C terminal invariant across those
states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .b0c_precedence import TERMINAL_ACTIONS, run_episode_b0c, run_matrix_b0c
from .evaluation import Action, Arm, Evidence, Hazard, Truth, run_episode
from .scenario_matrix import run_matrix


PROTOCOL_ID = "L10M-B0-D-ARRIVAL-STUCK-ORTHOGONALITY-V1"
PARENT_PROTOCOL_ID = "L10M-B0-C-RECOVERY-PRECEDENCE-V1"
B0C_FROZEN_VERDICT = "B0_C_TERMINAL_PRECEDENCE_MECHANISM_CONFIRMED_CAUSAL_SELECTIVITY_NOT_IDENTIFIED"
B0D_CONFIRMED_VERDICT = "B0_D_TERMINAL_SEMANTICS_STATE_INDEPENDENT_INVARIANT_CONFIRMED"
B0D_NOT_CONFIRMED_VERDICT = "B0_D_TERMINAL_SEMANTICS_STATE_INDEPENDENT_INVARIANT_NOT_CONFIRMED"
B0B_FROZEN_MATRIX_SHA256 = "72b10f42992e21561f23d5d4fb2a0a681949be10fca13b41a1b2eebefc9cc700"
B0C_FROZEN_MATRIX_SHA256 = "5674fd8834c095b8a8331993138b0a77dac1a8a9863af0d735957045172bba34"


def _step(
    *,
    progress_signal: float | None,
    arrived: bool = False,
    quality: float = 0.95,
    stale: bool = False,
) -> dict:
    return {
        "progress_signal": progress_signal,
        "truth_progress": 1.0 if arrived else 0.0,
        "arrived": arrived,
        "quality": quality,
        "stale": stale,
    }


def _episode(name: str, specs: list[dict]) -> tuple[list[Evidence], list[Truth]]:
    evidence_rows: list[Evidence] = []
    truth_rows: list[Truth] = []
    for index, spec in enumerate(specs):
        evidence_rows.append(
            Evidence(
                episode_id=name,
                step=index,
                alignment=0.0,
                center_hazard=Hazard.LOW,
                quality=spec["quality"],
                stale=spec["stale"],
                conflict=False,
                target_visible=True,
                progress_signal=spec["progress_signal"],
            )
        )
        truth_rows.append(
            Truth(
                episode_id=name,
                step=index,
                progress=spec["truth_progress"],
                arrived=spec["arrived"],
            )
        )
    return evidence_rows, truth_rows


def build_b0d_matrix() -> list[tuple[str, list[Evidence], list[Truth], bool, Action]]:
    """Return the four preregistered stuck/recovery x arrival cases."""
    confirmed_none = _step(progress_signal=0.0)
    confirmed_arrival = _step(progress_signal=1.0, arrived=True)
    unknown_no_arrival = _step(progress_signal=None, quality=0.40, stale=True)
    return [
        (
            "stuck_without_confirmed_arrival",
            *_episode("stuck_without_confirmed_arrival", [confirmed_none, confirmed_none, confirmed_none]),
            False,
            Action.RECOVER,
        ),
        (
            "stuck_with_confirmed_arrival",
            *_episode("stuck_with_confirmed_arrival", [confirmed_none, confirmed_none, confirmed_arrival]),
            True,
            Action.STOP,
        ),
        (
            "recovery_with_confirmed_arrival",
            *_episode(
                "recovery_with_confirmed_arrival",
                [confirmed_none, confirmed_none, confirmed_none, confirmed_arrival],
            ),
            True,
            Action.STOP,
        ),
        (
            "recovery_unknown_without_arrival",
            *_episode(
                "recovery_unknown_without_arrival",
                [confirmed_none, confirmed_none, confirmed_none, unknown_no_arrival],
            ),
            False,
            Action.RECOVER,
        ),
    ]


def _arrival_probe(prefix_length: int) -> dict:
    name = f"arrival_after_{prefix_length}_confirmed_no_progress_steps"
    specs = [_step(progress_signal=0.0) for _ in range(prefix_length)]
    specs.append(_step(progress_signal=1.0, arrived=True))
    evidence, truth = _episode(name, specs)
    parent = run_episode(Arm.STATEFUL, evidence, truth)
    candidate = run_episode_b0c(Arm.STATEFUL, evidence, truth)
    return {
        "probe": name,
        "stuck_evidence_count_before_arrival": prefix_length,
        "recovery_attempts_before_arrival": sum(action == Action.RECOVER for action in candidate.actions[:-1]),
        "parent_prepared_action": parent.actions[-1].value,
        "terminal_action": candidate.actions[-1].value,
        "success": candidate.success,
    }


def _b0c_frozen_observation() -> dict:
    before_result = run_matrix()
    before = {row["scenario"]: row for row in before_result["scenarios"]}
    after_result = run_matrix_b0c()
    after = {row["scenario"]: row for row in after_result["scenarios"]}
    flipped = {}
    for arm in (Arm.STATEFUL, Arm.STATEFUL_SAFETY):
        flipped[arm.value] = [
            scenario
            for scenario in before
            if before[scenario]["arms"][arm.value]["success"]
            != after[scenario]["arms"][arm.value]["success"]
        ]
    return {
        "verdict": B0C_FROZEN_VERDICT,
        "flipped_scenarios": flipped,
        "shared_flip_cause": "final CONFIRMED_ARRIVAL obeys ARRIVAL_IMPLIES_TERMINAL",
        "reactive_solvable_preservation": after_result["reactive_solvable_preservation"],
        "unknown_progress_accumulates_stuck": any(
            after["uncertain_progress"]["arms"][arm.value]["stuck_detection_step"] is not None
            or Action.RECOVER.value in after["uncertain_progress"]["arms"][arm.value]["actions"]
            for arm in (Arm.STATEFUL, Arm.STATEFUL_SAFETY)
        ),
        "recovery_then_progress_exits_recovery": all(
            after["recovery_then_progress"]["arms"][arm.value]["actions"]
            == [Action.FORWARD.value] * 4
            for arm in (Arm.STATEFUL, Arm.STATEFUL_SAFETY)
        ),
        "b0c_inputs_modified": before_result["matrix_sha256"] != B0B_FROZEN_MATRIX_SHA256,
        "b0c_result_modified": after_result["matrix_sha256"] != B0C_FROZEN_MATRIX_SHA256,
    }


def run_matrix_b0d() -> dict:
    rows = []
    for name, evidence, truth, expected_success, expected_final_action in build_b0d_matrix():
        arms = {}
        for arm in (Arm.STATEFUL, Arm.STATEFUL_SAFETY):
            stats = run_episode_b0c(arm, evidence, truth)
            arms[arm.value] = {
                "success": stats.success,
                "actions": [action.value for action in stats.actions],
                "expected_success": expected_success,
                "expected_final_action": expected_final_action.value,
                "matches_expected": stats.success == expected_success
                and stats.actions[-1] == expected_final_action,
            }
        rows.append({"scenario": name, "expected_success": expected_success, "arms": arms})

    probes = [_arrival_probe(prefix_length) for prefix_length in (0, 2, 3, 4)]
    terminal_values = {action.value for action in TERMINAL_ACTIONS}
    invariants = {
        "four_case_matrix_matches_expected": all(
            arm_result["matches_expected"]
            for row in rows
            for arm_result in row["arms"].values()
        ),
        "confirmed_arrival_always_succeeds": all(probe["success"] for probe in probes),
        "confirmed_arrival_always_uses_terminal_action": all(
            probe["terminal_action"] in terminal_values for probe in probes
        ),
        "prepared_action_variation_covered": {probe["parent_prepared_action"] for probe in probes}
        == {Action.FORWARD.value, Action.RECOVER.value},
        "recovery_attempt_variation_covered": {probe["recovery_attempts_before_arrival"] for probe in probes}
        >= {0, 1, 2},
        "no_arrival_never_fabricates_success": all(
            not arm_result["success"]
            for row in rows
            if not row["expected_success"]
            for arm_result in row["arms"].values()
        ),
    }
    confirmed = all(invariants.values())
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "research_mode": "REVERSIBLE_EXPLORATION/CANARY_LITE",
        "claim_ceiling": "state-independent terminal semantics on synthetic policy mechanics only; not causal attribution between recovery and arrival, end-to-end, device, or safety evidence",
        "question": "After orthogonalizing stuck/recovery history from final arrival truth, does confirmed arrival remain terminal independent of state?",
        "scenario_count": len(rows),
        "matrix_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "b0c_frozen_observation": _b0c_frozen_observation(),
        "state_independence_probes": probes,
        "invariants": invariants,
        "verdict": B0D_CONFIRMED_VERDICT if confirmed else B0D_NOT_CONFIRMED_VERDICT,
        "execution_boundary": {"b1_started": False, "structured_search_started": False},
        "scenarios": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_matrix_b0d()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
