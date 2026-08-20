"""Sky-compatible GC2-B development evaluator; BA metrics remain authoritative."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from scripts.research.goal_copilot_2a.evaluator import evaluate_condition
except ImportError:
    import sys

    BUNDLE_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(BUNDLE_ROOT))
    from scripts.research.goal_copilot_2a.evaluator import evaluate_condition

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "dev_scenarios.json"
if not SCENARIOS.exists():
    SCENARIOS = HERE.parent / "goal_copilot_bridge" / "pilot" / "dev_scenarios.json"

SEARCH_CONDITIONS = (
    "CLEAN",
    "TARGET_DROPOUT_MODERATE",
    "BEARING_JITTER_MODERATE",
    "FALSE_TARGET_MODERATE",
    "NEARNESS_ERROR_MODERATE",
    "TRACKING_COLLAPSE_MODERATE",
    "DELAYED_EVIDENCE_MODERATE",
    "COMBINED_MILD",
    "COMBINED_MODERATE",
)


def evaluate_matrix(program_path: Path) -> dict[str, Any]:
    return {
        condition: evaluate_condition(program_path, SCENARIOS, condition)
        for condition in SEARCH_CONDITIONS
    }


def summarize(matrix: dict[str, Any]) -> dict[str, Any]:
    metrics = {condition: cell["metrics"] for condition, cell in matrix.items()}
    hard_gate = all(
        item["semantic_validity"]
        and item["unsafe_guidance"] == 0
        and item["premature_completion"] == 0
        for item in metrics.values()
    )
    clean = metrics["CLEAN"]
    moderate = metrics["COMBINED_MODERATE"]
    clean_gate = clean["completion_count"] == 12
    family_floor = min(moderate["family_completion_rates"].values())
    condition_floor = min(item["completion_rate"] for item in metrics.values())
    progress = sum(item["normalized_progress_total"] for item in metrics.values())
    timeouts = sum(item["timeouts"] for item in metrics.values())
    wrong_way = sum(item["wrong_way_actions"] for item in metrics.values())
    ast_nodes = clean["candidate_complexity_ast_nodes"]
    score = -1_000_000_000_000.0
    if hard_gate:
        score = (
            int(clean_gate) * 1_000_000_000
            + moderate["completion_count"] * 10_000_000
            + family_floor * 1_000_000
            + moderate["eligible_reacquisition_rate"] * 100_000
            + condition_floor * 10_000
            + progress * 10
            - timeouts
            - wrong_way * 0.001
            - ast_nodes * 0.000001
        )
    return {
        "combined_score": score,
        "validity": int(hard_gate and clean_gate),
        "hard_gate_pass_all_search_conditions": hard_gate,
        "clean_completion_count": clean["completion_count"],
        "combined_mild_completion_count": metrics["COMBINED_MILD"]["completion_count"],
        "combined_moderate_completion_count": moderate["completion_count"],
        "combined_moderate_family_completion_counts": moderate["family_completion_counts"],
        "combined_moderate_eligible_reacquisition_rate": moderate["eligible_reacquisition_rate"],
        "minimum_search_condition_completion_rate": condition_floor,
        "unsafe_guidance_total": sum(item["unsafe_guidance"] for item in metrics.values()),
        "premature_completion_total": sum(item["premature_completion"] for item in metrics.values()),
        "timeouts_total": timeouts,
        "wrong_way_actions_total": wrong_way,
        "candidate_digest": clean["candidate_digest"],
        "candidate_complexity_ast_nodes": ast_nodes,
    }


def evaluate(program_path: str) -> dict[str, Any]:
    try:
        matrix = evaluate_matrix(Path(program_path))
        summary = summarize(matrix)
        failures = {
            condition: [
                outcome["scenario_id"]
                for outcome in cell["outcomes"]
                if not outcome["goal_completion"]
            ]
            for condition, cell in matrix.items()
        }
        return {
            **summary,
            "artifacts": {
                "feedback": json.dumps(failures, sort_keys=True),
                "authority": "DEV_GUIDANCE_ONLY_BLINDASSIST_RETAINS_ACCEPTANCE",
            },
        }
    except Exception as exc:
        return {
            "combined_score": -1_000_000_000_000.0,
            "validity": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "artifacts": {"failure_stage": "candidate_or_evaluator_validation"},
        }
