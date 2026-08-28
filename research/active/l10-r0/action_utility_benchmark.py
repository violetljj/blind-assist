from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


class DeficitConditionedUcb:
    def __init__(self, exploration_strength: float):
        self.exploration_strength = exploration_strength
        self.counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    def select(self, context: dict[str, Any]) -> str:
        context_id = context["id"]
        total = sum(
            evaluated for (candidate_context, _), (_, evaluated) in self.counts.items()
            if candidate_context == context_id
        )
        ranked = []
        for priority, action in enumerate(context["allowed_actions"]):
            improved, evaluated = self.counts[(context_id, action)]
            posterior_mean = (improved + 1.0) / (evaluated + 2.0)
            bonus = self.exploration_strength * math.sqrt(
                math.log(total + 2.0) / (evaluated + 1.0)
            )
            ranked.append((posterior_mean + bonus, -priority, action))
        return max(ranked)[2]

    def observe(self, context_id: str, action: str, improved: bool, authoritative: bool) -> None:
        if not authoritative:
            return
        counts = self.counts[(context_id, action)]
        counts[0] += int(improved)
        counts[1] += 1


def build_schedule(protocol: dict[str, Any]) -> list[tuple[dict[str, Any], float, bool]]:
    rng = random.Random(protocol["seed"])
    schedule = []
    contexts = protocol["contexts"]
    for _ in range(protocol["trials_per_context"]):
        cycle = list(contexts)
        rng.shuffle(cycle)
        for context in cycle:
            authoritative = rng.random() >= protocol["unknown_outcome_rate"]
            schedule.append((context, rng.random(), authoritative))
    return schedule


def run_arm(protocol: dict[str, Any], schedule: list[Any], learned: bool) -> dict[str, Any]:
    policy = DeficitConditionedUcb(protocol["exploration_strength"])
    rows = []
    for context, outcome_draw, authoritative in schedule:
        action = policy.select(context) if learned else context["fixed_action"]
        probability = context["improvement_probability"][action]
        improved = authoritative and outcome_draw < probability
        policy.observe(context["id"], action, improved, authoritative)
        safe_oracle = max(context["improvement_probability"][item] for item in context["allowed_actions"])
        rows.append(
            {
                "context": context["id"],
                "action": action,
                "authoritative": authoritative,
                "improved": improved,
                "probability": probability,
                "expected_regret": safe_oracle - probability,
                "optimal": action in {
                    item for item in context["allowed_actions"]
                    if context["improvement_probability"][item] == safe_oracle
                },
                "unsafe_approach": "AMBIGUOUS" in context["id"]
                and action == "APPROACH_FOR_IDENTITY",
            }
        )
    authoritative_rows = [row for row in rows if row["authoritative"]]
    final_rows = []
    window = protocol["final_window_per_context"]
    for context in protocol["contexts"]:
        context_rows = [row for row in rows if row["context"] == context["id"]]
        final_rows.extend(context_rows[-window:])
    ambiguous_rows = [row for row in rows if "AMBIGUOUS" in row["context"]]
    return {
        "trials": len(rows),
        "authoritative_outcomes": len(authoritative_rows),
        "unknown_outcomes": len(rows) - len(authoritative_rows),
        "improvements": sum(row["improved"] for row in authoritative_rows),
        "improvement_rate": _rate(
            sum(row["improved"] for row in authoritative_rows), len(authoritative_rows)
        ),
        "expected_cumulative_regret": round(sum(row["expected_regret"] for row in rows), 4),
        "final_window_optimal_action_rate": _rate(
            sum(row["optimal"] for row in final_rows), len(final_rows)
        ),
        "ambiguous_unsafe_approach_rate": _rate(
            sum(row["unsafe_approach"] for row in ambiguous_rows), len(ambiguous_rows)
        ),
        "per_context": {
            context["id"]: {
                "improvement_rate": _rate(
                    sum(row["improved"] for row in authoritative_rows if row["context"] == context["id"]),
                    sum(1 for row in authoritative_rows if row["context"] == context["id"]),
                ),
                "final_optimal_action_rate": _rate(
                    sum(row["optimal"] for row in final_rows if row["context"] == context["id"]),
                    sum(1 for row in final_rows if row["context"] == context["id"]),
                ),
            }
            for context in protocol["contexts"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    schedule = build_schedule(protocol)
    baseline = run_arm(protocol, schedule, learned=False)
    candidate = run_arm(protocol, schedule, learned=True)
    improvement_delta = round(
        candidate["improvement_rate"] - baseline["improvement_rate"], 4
    )
    baseline_regret = baseline["expected_cumulative_regret"]
    regret_reduction = round(
        0.0 if baseline_regret == 0.0
        else 1.0 - candidate["expected_cumulative_regret"] / baseline_regret,
        4,
    )
    gate = protocol["development_gate"]
    checks = {
        "improvement_rate_delta": improvement_delta >= gate["minimum_improvement_rate_delta"],
        "expected_regret_reduction": regret_reduction
        >= gate["minimum_expected_regret_reduction"],
        "final_window_optimal_action_rate": candidate["final_window_optimal_action_rate"]
        >= gate["minimum_final_window_optimal_action_rate"],
        "ambiguous_unsafe_approach_rate": candidate["ambiguous_unsafe_approach_rate"]
        <= gate["maximum_unsafe_approach_rate_in_ambiguous_contexts"],
    }
    result = {
        "protocol_id": protocol["protocol_id"],
        "claim_ceiling": protocol["claim_ceiling"],
        "status": "DEVELOPMENT_GATE_MET" if all(checks.values()) else "DEVELOPMENT_GATE_NOT_MET",
        "arms": {
            protocol["arms"][0]: baseline,
            protocol["arms"][1]: candidate,
        },
        "comparisons": {
            "improvement_rate_delta": improvement_delta,
            "expected_regret_reduction": regret_reduction,
        },
        "checks": checks,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
