"""Zero-call saturation audit over the evaluable L10M search traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INITIAL_SCORE = 0.9517241379310345
HALF_BUDGET_GENERATION = 4
EXPECTED_GENERATIONS = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _read_completions(path: Path) -> list[dict[str, Any]]:
    completions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            event = json.loads(line)
            if event.get("kind") != "completion":
                continue
            if event.get("returncode") != 0 or not event.get("semantic_valid"):
                raise RuntimeError(f"inadmissible completion at {path}:{line_number}")
            completions.append(event)
    return completions


def _validate_source(
    *,
    result_path: Path,
    events_path: Path,
    expected_terminal: str,
    expected_completion_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = _read_json(result_path)
    if result.get("terminal") != expected_terminal:
        raise RuntimeError(f"source is not evaluable and terminal: {result_path}")
    if result.get("provider_failure_count", 0) != 0:
        raise RuntimeError(f"source has provider failures: {result_path}")
    completions = _read_completions(events_path)
    if len(completions) != expected_completion_count:
        raise RuntimeError(
            f"expected {expected_completion_count} completions in {events_path}, got {len(completions)}"
        )
    bound_hash = result.get("source_sha256", {}).get("events.jsonl")
    if bound_hash is None:
        bound_hash = result.get("evidence", {}).get("events_sha256")
    if bound_hash != _sha256(events_path):
        raise RuntimeError(f"event ledger hash does not match terminal result: {events_path}")
    return result, completions


def _trajectory_rows(
    completions: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in completions:
        grouped[(str(event["arm"]), int(event["seed"]))].append(event)

    rows: list[dict[str, Any]] = []
    for (arm, seed), events in sorted(grouped.items()):
        ordered = sorted(events, key=lambda item: int(item["generation"]))
        generations = [int(item["generation"]) for item in ordered]
        if generations != list(range(1, EXPECTED_GENERATIONS + 1)):
            raise RuntimeError(f"incomplete trajectory: {source}/{arm}/{seed}: {generations}")
        scores = [float(item["behavioral_score"]) for item in ordered]
        running_best: list[float] = []
        best = INITIAL_SCORE
        first_improvement = None
        improvement_generations: list[int] = []
        for generation, score in zip(generations, scores, strict=True):
            if score > best:
                best = score
                improvement_generations.append(generation)
                if first_improvement is None:
                    first_improvement = generation
            running_best.append(best)
        rows.append(
            {
                "source": source,
                "arm": arm,
                "seed": seed,
                "scores": scores,
                "running_best": running_best,
                "first_improvement_generation": first_improvement,
                "improvement_generations": improvement_generations,
                "best_at_half_budget": running_best[HALF_BUDGET_GENERATION - 1],
                "final_best_score": running_best[-1],
                "late_realized_gain": running_best[-1]
                - running_best[HALF_BUDGET_GENERATION - 1],
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]], ceiling: float) -> dict[str, Any]:
    finals = [float(row["final_best_score"]) for row in rows]
    firsts = [
        int(row["first_improvement_generation"])
        for row in rows
        if row["first_improvement_generation"] is not None
    ]
    ceiling_count = sum(score == ceiling for score in finals)
    reached_by_half = sum(
        float(row["best_at_half_budget"]) == ceiling for row in rows
    )
    late_gain_total = sum(float(row["late_realized_gain"]) for row in rows)
    late_improvement_count = sum(
        generation > HALF_BUDGET_GENERATION
        for row in rows
        for generation in row["improvement_generations"]
    )
    mode_score, mode_count = Counter(finals).most_common(1)[0]
    return {
        "trajectory_count": len(rows),
        "discovery_reach_count": len(firsts),
        "discovery_reach_rate": len(firsts) / len(rows),
        "time_to_first_improvement_generations": firsts,
        "time_to_first_improvement_median": statistics.median(firsts) if firsts else None,
        "ceiling_score": ceiling,
        "ceiling_reach_count": ceiling_count,
        "ceiling_reach_rate": ceiling_count / len(rows),
        "ceiling_by_half_budget_count": reached_by_half,
        "ceiling_by_half_budget_rate": reached_by_half / len(rows),
        "final_score_population_variance": statistics.pvariance(finals),
        "final_score_mode": mode_score,
        "final_score_mode_count": mode_count,
        "final_score_mode_share": mode_count / len(rows),
        "late_generations": [HALF_BUDGET_GENERATION + 1, EXPECTED_GENERATIONS],
        "late_strict_improvement_count": late_improvement_count,
        "late_realized_gain_total": late_gain_total,
        "late_realized_gain_mean": late_gain_total / len(rows),
        "theoretical_headroom_remaining_after_half_budget_total": sum(
            max(0.0, ceiling - float(row["best_at_half_budget"])) for row in rows
        ),
    }


def audit(b1_run: Path, b3a_run: Path) -> dict[str, Any]:
    b1_result_path = b1_run / "result.json"
    b1_events_path = b1_run / "events.jsonl"
    b3a_result_path = b3a_run / "result.json"
    b3a_events_path = b3a_run / "events.jsonl"
    b1_result, b1_events = _validate_source(
        result_path=b1_result_path,
        events_path=b1_events_path,
        expected_terminal="B1_EVALUABLE_COMPLETE",
        expected_completion_count=48,
    )
    b3a_result, b3a_events = _validate_source(
        result_path=b3a_result_path,
        events_path=b3a_events_path,
        expected_terminal="B3A_EVALUABLE_COMPLETE",
        expected_completion_count=48,
    )
    if b1_result.get("initial_score") != INITIAL_SCORE:
        raise RuntimeError("B1 initial score differs from the B4-I0 audit constant")

    trajectories = _trajectory_rows(b1_events, "B1") + _trajectory_rows(
        b3a_events, "B3-A"
    )
    ceiling = max(float(row["final_best_score"]) for row in trajectories)
    baseline = [
        row
        for row in trajectories
        if (row["source"] == "B1" and row["arm"] == "structured")
        or (row["source"] == "B3-A" and row["arm"] == "structured_control")
    ]
    fresh_baseline = [
        row
        for row in baseline
        if row["source"] == "B3-A" and row["arm"] == "structured_control"
    ]
    all_formal = trajectories

    baseline_summary = _summarize(baseline, ceiling)
    fresh_summary = _summarize(fresh_baseline, ceiling)
    all_summary = _summarize(all_formal, ceiling)
    criteria = {
        "fresh_baseline_all_reach_ceiling": fresh_summary["ceiling_reach_rate"] == 1.0,
        "fresh_baseline_all_reach_ceiling_by_half_budget": fresh_summary[
            "ceiling_by_half_budget_rate"
        ]
        == 1.0,
        "pooled_baseline_ceiling_mode_share_at_least_five_sixths": baseline_summary[
            "final_score_mode_share"
        ]
        >= 5 / 6,
        "pooled_baseline_has_no_realized_late_gain": baseline_summary[
            "late_realized_gain_total"
        ]
        == 0.0,
        "all_formal_arms_ceiling_mode_share_at_least_eleven_twelfths": all_summary[
            "final_score_mode_share"
        ]
        >= 11 / 12,
    }
    saturated = all(criteria.values())
    return {
        "schema": "l10m_b4_i0_saturation_audit_v1",
        "audit_id": "L10M-B4-I0-ZERO-CALL-SATURATION-AUDIT-V1",
        "model_call_count": 0,
        "scope": {
            "baseline_population": "all evaluable unmodified Structured proposal trajectories: B1 Structured plus B3-A Structured Control",
            "fresh_baseline_population": "B3-A Structured Control only",
            "context_population": "all evaluable formal B1 and B3-A search arms",
            "excluded": [
                "B1 transport-failed predecessors",
                "transport qualification canaries",
                "B2 seed-89 candidate transplant",
                "B3-I0 diagnostic lineage autopsy",
            ],
            "budget_generations": EXPECTED_GENERATIONS,
            "half_budget_generation": HALF_BUDGET_GENERATION,
            "initial_score": INITIAL_SCORE,
            "observed_ceiling_score": ceiling,
        },
        "baseline": baseline_summary,
        "fresh_baseline": fresh_summary,
        "all_formal_search_arms": all_summary,
        "baseline_trajectories": baseline,
        "decision_rubric": {
            "status": "descriptive governance rubric frozen after headline B3-A outcomes were known; not a blind hypothesis test",
            "criteria": criteria,
        },
        "terminal": "B4_I0_SATURATION_CONFIRMED" if saturated else "B4_I0_SATURATION_NOT_CONFIRMED",
        "benchmark_classification": (
            "MECHANISM_DEBUG_BENCHMARK / NOT_SUITABLE_FOR_SEARCH_VALUE_DISCRIMINATION"
            if saturated
            else "SEARCH_VALUE_DISCRIMINATION_NOT_RULED_OUT"
        ),
        "claim_ceiling": "descriptive saturation evidence for the current finite synthetic L10M instance distribution and eight-generation budget; not general search-value, model, device, user, safety-effect, or production evidence",
        "next_step": (
            "freeze a harder fresh cohort before any model call; preserve the eight-generation budget and evaluate search efficiency separately only under a future preregistered protocol"
            if saturated
            else "retain the current benchmark classification and investigate the unresolved pressure signal without changing the frozen B3-A result"
        ),
        "source_sha256": {
            str(b1_result_path): _sha256(b1_result_path),
            str(b1_events_path): _sha256(b1_events_path),
            str(b3a_result_path): _sha256(b3a_result_path),
            str(b3a_events_path): _sha256(b3a_events_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1-run", type=Path, required=True)
    parser.add_argument("--b3a-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite create-once audit: {args.output}")
    result = audit(args.b1_run, args.b3a_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "terminal": result["terminal"]}))


if __name__ == "__main__":
    main()
