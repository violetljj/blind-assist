"""B0-C minimal recovery precedence intervention.

This is a new protocol layered on the consumed B0-B scenarios.  It changes
only control ordering: confirmed terminal evidence wins over recovery, and a
credible positive-progress observation exits recovery before action selection.
Truth, evidence validation, and the B0-B matrix are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .evaluation import (
    Action,
    Arm,
    Belief,
    EpisodeStats,
    Evidence,
    ProgressStatus,
    Truth,
    _proposal,
    _shield,
    _update_belief,
)
from .scenario_matrix import SCENARIOS, build_matrix

PROTOCOL_ID = "L10M-B0-C-RECOVERY-PRECEDENCE-V1"
TERMINAL_ACTIONS = frozenset({Action.STOP, Action.FORWARD})


def _arrival_supported(evidence: Evidence, truth: Truth) -> bool:
    return bool(
        truth.arrived
        and evidence.quality >= 0.50
        and not evidence.stale
        and not evidence.conflict
        and evidence.target_visible
        and abs(evidence.alignment) <= 0.10
    )


def run_episode_b0c(arm: Arm, evidence_rows: list[Evidence], truth_rows: list[Truth]) -> EpisodeStats:
    """Evaluate one episode with terminal/progress/recovery precedence."""
    if len(evidence_rows) != len(truth_rows) or not evidence_rows:
        raise ValueError("evidence and truth must be non-empty and aligned")
    belief = Belief()
    stats = EpisodeStats()
    previous_direction: Action | None = None
    for evidence, truth in zip(evidence_rows, truth_rows):
        evidence.validate()
        if evidence.episode_id != truth.episode_id or evidence.step != truth.step:
            raise ValueError("evidence/truth identity mismatch")

        # B0-C ordering: terminal evidence, then progress/recovery transition,
        # then policy action.  Recovery can never overwrite a terminal action.
        terminal = _arrival_supported(evidence, truth)
        if arm == Arm.REACTIVE:
            action = _proposal(evidence, None)
        else:
            was_recovery = belief.stuck_count >= 2
            # Terminal arbitration is evaluated before recovery exit.  This
            # preserves a terminal FORWARD in ordinary trajectories while
            # preventing an already-active RECOVER state from masking arrival.
            if terminal and was_recovery:
                action = Action.STOP
            else:
                action = None
            if evidence.progress_status == ProgressStatus.POSITIVE:
                belief.stuck_count = 0
            if action is None:
                action = _proposal(evidence, belief)
            # The intervention is deliberately narrow: terminal evidence
            # preempts only a recovery action; ordinary terminal FORWARD is
            # left unchanged to preserve the matched counterfactuals.
            if terminal and action == Action.RECOVER:
                action = Action.STOP
            if arm == Arm.STATEFUL_SAFETY and action != Action.STOP:
                action = _shield(action, evidence, belief)

        if action == Action.STOP and evidence.quality < 0.50:
            stats.unknown_steps += 1
        stats.unsafe_actions += int(action == Action.FORWARD and truth.unsafe_forward)
        if (
            action == Action.STOP and not truth.arrived and evidence.quality >= 0.50
            and evidence.center_hazard.value == "LOW" and abs(evidence.alignment) <= 0.10
        ):
            stats.false_arrivals += 1
        if previous_direction in {Action.LEFT, Action.RIGHT} and action in {Action.LEFT, Action.RIGHT} and action != previous_direction:
            stats.oscillations += 1
        previous_direction = action
        stats.actions.append(action)
        before_stuck = belief.stuck_count
        if arm != Arm.REACTIVE:
            _update_belief(belief, evidence, action)
            if belief.stuck_count >= 2 and before_stuck < 2 and stats.stuck_detection_step is None:
                stats.stuck_detection_step = evidence.step
            if action == Action.RECOVER and truth.progress > 0:
                stats.recovery_success += 1
        if terminal and action in TERMINAL_ACTIONS:
            stats.success = True
    stats.excess_actions = max(0, len(stats.actions) - 1)
    return stats


def run_matrix_b0c() -> dict:
    rows = []
    for name, evidence, truth, reactive_solvable in build_matrix():
        arms = {}
        for arm in Arm:
            stats = run_episode_b0c(arm, evidence, truth)
            arms[arm.value] = {
                "success": stats.success,
                "unsafe_actions": stats.unsafe_actions,
                "actions": [action.value for action in stats.actions],
                "stuck_detection_step": stats.stuck_detection_step,
                "recovery_success": stats.recovery_success,
                "oscillations": stats.oscillations,
            }
        rows.append({"scenario": name, "reactive_solvable": reactive_solvable, "arms": arms})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    preservation = [r for r in rows if r["reactive_solvable"] and r["arms"][Arm.REACTIVE.value]["success"]]
    return {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": "L10M-B0-B-MATCHED-COUNTERFACTUALS-V1",
        "claim_ceiling": "causal recovery precedence repair on matched synthetic mechanics only; not end-to-end evidence",
        "intervention": {"terminal_precedence": True, "positive_progress_exits_recovery": True, "thresholds_changed": False},
        "scenario_count": len(rows),
        "matrix_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "reactive_solvable_preservation": {"denominator": len(preservation), "stateful_rate": sum(r["arms"][Arm.STATEFUL.value]["success"] for r in preservation) / len(preservation)},
        "invariants": {"arrival_implies_terminal_action": all((not r["arms"][Arm.STATEFUL.value]["success"] or r["arms"][Arm.STATEFUL.value]["actions"][-1] in {a.value for a in TERMINAL_ACTIONS}) for r in rows)},
        "scenarios": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_matrix_b0c()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
