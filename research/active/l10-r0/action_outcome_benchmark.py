from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark import GoalLockWorld, make_specs
from l10_r0 import Action, Decision, GoalLockController, State


SEARCH_STATES = {State.SEARCH, State.LOST}


class ActionOutcomeRepairController:
    """GoalLock plus one-step causal repair; it never reads simulator truth."""

    name = "L10_R0_action_outcome_repair"

    def __init__(self) -> None:
        self.base = GoalLockController()
        self.last_search_action: Action | None = None

    def reset(self) -> None:
        self.base.reset()
        self.last_search_action = None

    def step(self, candidates: list[Any]) -> Decision:
        decision = self.base.step(candidates)
        action = decision.action
        no_gain = self.last_search_action is not None and decision.state in SEARCH_STATES
        if no_gain and action is self.last_search_action:
            if action is Action.LEFT:
                action = Action.RIGHT
            elif action is Action.RIGHT:
                action = Action.LEFT
            elif action is Action.FORWARD:
                action = Action.RIGHT
        repaired = Decision(decision.state, action, decision.selected_id, decision.belief)
        self.last_search_action = action if decision.state in SEARCH_STATES else None
        return repaired


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def run_episode(controller: Any, spec: Any, max_steps: int) -> dict[str, Any]:
    controller.reset()
    world = GoalLockWorld(spec)
    completed = False
    false_complete = False
    true_lock_before_occlusion = False
    reacquired_step: int | None = None
    previous_search_action: Action | None = None
    no_gain_opportunities = 0
    repeated_no_gain_actions = 0

    for step in range(max_steps):
        candidates, truth_id, _ = world.observe()
        decision = controller.step(candidates)
        selected_true = truth_id is not None and decision.selected_id == truth_id
        locked = decision.state in {State.LOCKED, State.NEAR, State.TASK_COMPLETE}
        if step < spec.occlusion_start and selected_true and locked:
            true_lock_before_occlusion = True
        if (
            step >= spec.occlusion_end
            and true_lock_before_occlusion
            and reacquired_step is None
            and selected_true
            and locked
        ):
            reacquired_step = step

        if previous_search_action is not None and decision.state in SEARCH_STATES:
            no_gain_opportunities += 1
            repeated_no_gain_actions += int(decision.action is previous_search_action)

        if decision.action is Action.COMPLETE:
            completed = world.completion_is_true(decision, truth_id)
            false_complete = not completed
            break

        previous_search_action = decision.action if decision.state in SEARCH_STATES else None
        world.advance(decision.action)

    reacquire_eligible = spec.target_present and true_lock_before_occlusion
    reacquired = reacquired_step is not None and reacquired_step - spec.occlusion_end <= 12
    return {
        "target_present": spec.target_present,
        "completed": completed,
        "false_complete": false_complete,
        "reacquire_eligible": reacquire_eligible,
        "reacquired": reacquired,
        "no_gain_opportunities": no_gain_opportunities,
        "repeated_no_gain_actions": repeated_no_gain_actions,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row for row in rows if row["target_present"]]
    absent = [row for row in rows if not row["target_present"]]
    reacquire = [row for row in rows if row["reacquire_eligible"]]
    opportunities = sum(row["no_gain_opportunities"] for row in rows)
    repeats = sum(row["repeated_no_gain_actions"] for row in rows)
    return {
        "present_episodes": len(present),
        "absent_episodes": len(absent),
        "task_success_rate": _rate(sum(row["completed"] for row in present), len(present)),
        "absent_false_complete_rate": _rate(
            sum(row["false_complete"] for row in absent), len(absent)
        ),
        "reacquire_eligible_episodes": len(reacquire),
        "reacquire_success_rate": _rate(sum(row["reacquired"] for row in reacquire), len(reacquire)),
        "no_gain_opportunities": opportunities,
        "repeated_no_gain_actions": repeats,
        "repeated_no_gain_action_rate": _rate(repeats, opportunities),
        "mean_no_gain_opportunities": round(
            statistics.mean(row["no_gain_opportunities"] for row in rows), 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    specs = make_specs(protocol["episode_count"], protocol["seed"])
    controllers = (GoalLockController(), ActionOutcomeRepairController())
    arms = {
        controller.name: summarize(
            [run_episode(controller, spec, protocol["max_steps"]) for spec in specs]
        )
        for controller in controllers
    }
    baseline = arms[protocol["baseline"]]
    candidate = arms[protocol["candidate"]]
    base_repeat = baseline["repeated_no_gain_action_rate"]
    repeat_reduction = (
        0.0 if base_repeat == 0.0
        else round(1.0 - candidate["repeated_no_gain_action_rate"] / base_repeat, 4)
    )
    gate = protocol["development_gate"]
    checks = {
        "repeat_reduction": repeat_reduction
        >= gate["minimum_relative_reduction_repeated_no_gain_action_rate"],
        "task_success_noninferiority": candidate["task_success_rate"]
        >= baseline["task_success_rate"] - gate["maximum_task_success_rate_drop"],
        "reacquire_noninferiority": candidate["reacquire_success_rate"]
        >= baseline["reacquire_success_rate"] - gate["maximum_reacquire_success_rate_drop"],
        "absent_false_complete": candidate["absent_false_complete_rate"]
        <= gate["maximum_absent_false_complete_rate"],
    }
    passed = all(checks.values())
    result = {
        "protocol_id": protocol["protocol_id"],
        "claim_ceiling": protocol["claim_ceiling"],
        "status": "DEVELOPMENT_GATE_MET" if passed else "DEVELOPMENT_GATE_NOT_MET",
        "arms": arms,
        "comparisons": {"relative_repeat_reduction": repeat_reduction},
        "checks": checks,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
