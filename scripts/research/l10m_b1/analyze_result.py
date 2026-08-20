"""Read-only frozen-rule aggregation for a complete L10M-B1 V2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from .evaluator import evaluate_spec
from .policy_space import INITIAL_SPEC
from .protocol import EVALUATIONS_PER_ARM_PER_SEED, FINAL_SCORE_EQUIVALENCE_MARGIN, MIN_DISCOVERY_IMPROVEMENT, PAIRED_SEEDS, PROTOCOL_ID


RESULT_SCHEMA = "l10m_b1_matched_search_result_v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "execution_manifest.json"
    events_path = run_dir / "events.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("status") != "COMPLETE":
        raise RuntimeError("run is not a complete frozen B1 V2 successor")
    completions = [event for event in _events(events_path) if event.get("kind") == "completion"]
    expected = len(PAIRED_SEEDS) * 2 * EVALUATIONS_PER_ARM_PER_SEED
    if len(completions) != expected:
        raise RuntimeError(f"expected {expected} completions, found {len(completions)}")
    if any(event.get("returncode") != 0 or not str(event.get("candidate_output", "")).strip() for event in completions):
        raise RuntimeError("provider failure or empty terminal response makes B1 not evaluable")

    initial_score = float(evaluate_spec(INITIAL_SPEC)["behavioral_score"])
    arms: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    unsafe_counts = {"raw": 0, "structured": 0}
    invalid_counts = {"raw": 0, "structured": 0}
    for seed in PAIRED_SEEDS:
        by_arm: dict[str, dict[str, object]] = {}
        for arm in ("raw", "structured"):
            group = [event for event in completions if event["seed"] == seed and event["arm"] == arm]
            if len(group) != EVALUATIONS_PER_ARM_PER_SEED:
                raise RuntimeError(f"seed {seed} arm {arm} has an incomplete budget")
            invalid = sum(event.get("semantic_valid") is not True for event in group)
            unsafe = sum(event.get("unsafe_candidate") is True for event in group)
            invalid_counts[arm] += invalid
            unsafe_counts[arm] += unsafe
            eligible = [event for event in group if event.get("semantic_valid") is True and event.get("unsafe_candidate") is not True]
            if not eligible:
                raise RuntimeError(f"seed {seed} arm {arm} has no eligible candidate")
            best = max(eligible, key=lambda event: (float(event["behavioral_score"]), -int(event["generation"])))
            hits = [event for event in eligible if float(event["behavioral_score"]) - initial_score >= MIN_DISCOVERY_IMPROVEMENT]
            first_hit = min((int(event["generation"]) for event in hits), default=None)
            row = {
                "seed": seed,
                "arm": arm,
                "best_score": float(best["behavioral_score"]),
                "best_improvement": float(best["behavioral_score"]) - initial_score,
                "best_generation": int(best["generation"]),
                "first_discovery_generation": first_hit,
                "discovery_success": first_hit is not None,
                "invalid_count": invalid,
                "unsafe_count": unsafe,
                "best_changed_components": best.get("changed_components", []),
                "best_behavioral_vector": best.get("behavioral_vector", {}),
                "best_candidate_sha256": best.get("candidate_output_sha256"),
            }
            arms.append(row)
            by_arm[arm] = row
        raw = by_arm["raw"]
        structured = by_arm["structured"]
        delta = float(structured["best_improvement"]) - float(raw["best_improvement"])
        paired.append({
            "seed": seed,
            "structured_minus_raw_best_improvement": delta,
            "final_scores_equivalent": abs(float(structured["best_score"]) - float(raw["best_score"])) <= FINAL_SCORE_EQUIVALENCE_MARGIN,
            "same_best_changed_components": structured["best_changed_components"] == raw["best_changed_components"],
        })

    deltas = [float(row["structured_minus_raw_best_improvement"]) for row in paired]
    paired_median = float(statistics.median(deltas))
    structured_wins = sum(delta > 0 for delta in deltas)
    equivalent_seed_count = sum(bool(row["final_scores_equivalent"]) for row in paired)
    shared_component_seed_count = sum(bool(row["same_best_changed_components"]) for row in paired)
    raw_unsafe_rate = unsafe_counts["raw"] / (len(PAIRED_SEEDS) * EVALUATIONS_PER_ARM_PER_SEED)
    structured_unsafe_rate = unsafe_counts["structured"] / (len(PAIRED_SEEDS) * EVALUATIONS_PER_ARM_PER_SEED)
    no_unsafe_regression = structured_unsafe_rate <= raw_unsafe_rate

    if paired_median > 0 and structured_wins >= 2 and no_unsafe_regression:
        verdict = "STRUCTURED_SEARCHABILITY_VALUE_ESTABLISHED"
    else:
        efficiency_wins = 0
        for seed in PAIRED_SEEDS:
            raw = next(row for row in arms if row["seed"] == seed and row["arm"] == "raw")
            structured = next(row for row in arms if row["seed"] == seed and row["arm"] == "structured")
            s_hit = structured["first_discovery_generation"]
            r_hit = raw["first_discovery_generation"]
            if s_hit is not None and (r_hit is None or int(s_hit) < int(r_hit) or structured["invalid_count"] < raw["invalid_count"]):
                efficiency_wins += 1
        if equivalent_seed_count == len(PAIRED_SEEDS) and efficiency_wins >= 2 and no_unsafe_regression:
            verdict = "STRUCTURED_SEARCH_EFFICIENCY_VALUE_ESTABLISHED"
        elif equivalent_seed_count == len(PAIRED_SEEDS) and shared_component_seed_count >= 2:
            verdict = "REPRESENTATION_NOT_BOTTLENECK_SHARED_CAUSAL_COMPONENT"
        elif sum(row["discovery_success"] for row in arms if row["arm"] == "raw") <= 1 and sum(row["discovery_success"] for row in arms if row["arm"] == "structured") <= 1:
            verdict = "SEARCH_OPERATOR_OR_GENERATION_BOTTLENECK_NOT_RESOLVED"
        else:
            verdict = "B1_INCONCLUSIVE"

    return {
        "schema": RESULT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "run_id": manifest["run_id"],
        "terminal": "B1_EVALUABLE_COMPLETE",
        "scientific_verdict": verdict,
        "claim_ceiling": manifest["protocol_manifest"]["claim_ceiling"],
        "initial_score": initial_score,
        "completion_count": len(completions),
        "provider_failure_count": 0,
        "invalid_count": invalid_counts,
        "unsafe_count": unsafe_counts,
        "arm_results": arms,
        "paired_results": paired,
        "primary": {"paired_deltas": deltas, "paired_median": paired_median, "structured_wins": structured_wins},
        "secondary": {
            "equivalent_seed_count": equivalent_seed_count,
            "shared_best_component_seed_count": shared_component_seed_count,
            "raw_unsafe_rate": raw_unsafe_rate,
            "structured_unsafe_rate": structured_unsafe_rate,
            "equivalence_rule_interpretation": "best scores must be equivalent in all paired seeds; only the shared changed-component clause has an explicit 2-of-3 threshold",
        },
        "evidence": {
            "execution_manifest_sha256": _sha256(manifest_path),
            "events_sha256": _sha256(events_path),
            "transport_qualification": manifest["transport_qualification"],
        },
        "excluded_predecessor": manifest.get("supersedes_non_evaluable_attempt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supersedes-result", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"result receipt already exists: {args.output}")
    result = analyze(args.run_dir)
    if args.supersedes_result is not None:
        result["supersedes_non_authoritative_result"] = {
            "path": str(args.supersedes_result.resolve()),
            "sha256": _sha256(args.supersedes_result),
            "reason": "initial post-run aggregation used a post-hoc median equivalence interpretation; strict preregistered wording requires equivalence in all paired seeds",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
