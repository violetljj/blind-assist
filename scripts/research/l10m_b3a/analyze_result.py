"""Create-once analysis for a complete evaluable B3-A run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.l10m_b1.evaluator import evaluate_spec
from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, canonical_spec, parse_structured
from scripts.research.l10m_b3a.exploration import legal_adjacent_moves, proposal_move_tokens
from scripts.research.l10m_b3a.protocol import (
    ARMS,
    EVALUATIONS_PER_ARM_PER_SEED,
    INITIAL_SCORE,
    MIN_DISCOVERY_IMPROVEMENT,
    PAIRED_SEEDS,
    PROTOCOL_ID,
    build_protocol_manifest,
    canonical_manifest_sha256,
)


TARGET_CANONICAL_SHA256 = "b0110121ef34b54fb82d82be881d01ed07474983a5f897ae005b95a4f0185021"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_create_once(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _trajectory(seed: int, arm: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(
        [event for event in events if event.get("kind") == "completion" and event.get("seed") == seed and event.get("arm") == arm],
        key=lambda event: int(event["generation"]),
    )
    if [event["generation"] for event in rows] != list(range(1, EVALUATIONS_PER_ARM_PER_SEED + 1)):
        raise RuntimeError(f"incomplete trajectory for seed={seed} arm={arm}")
    incumbent = INITIAL_SPEC
    best_score = INITIAL_SCORE
    attempted: set[str] = set()
    unique_specs: set[str] = set()
    direction_tokens: list[str] = []
    first_discovery = None
    exact_target_generations: list[int] = []
    invalid_count = 0
    unsafe_count = 0
    operator_integrity = True
    operator_violations: list[str] = []
    non_improving_count = 0

    for event in rows:
        generation = int(event["generation"])
        if not event.get("semantic_valid"):
            invalid_count += 1
            continue
        candidate = parse_structured(str(event["candidate_output"]))
        candidate_canonical = canonical_spec(candidate)
        recomputed = evaluate_spec(candidate)
        for field in ("semantic_valid", "unsafe_candidate", "behavioral_score", "behavioral_vector", "invariant_counts"):
            if event.get(field) != recomputed.get(field):
                raise RuntimeError(f"seed={seed} arm={arm} generation={generation} evaluator mismatch for {field}")
        unique_specs.add(candidate_canonical)
        tokens = proposal_move_tokens(incumbent, candidate)
        direction_tokens.extend(sorted(tokens))
        score = float(event["behavioral_score"])
        unsafe = bool(event.get("unsafe_candidate"))
        unsafe_count += int(unsafe)
        strict_improvement = not unsafe and score > best_score
        if not strict_improvement:
            non_improving_count += 1
        if score >= INITIAL_SCORE + MIN_DISCOVERY_IMPROVEMENT and not unsafe and first_discovery is None:
            first_discovery = generation
        if _sha256_bytes(candidate_canonical.encode("utf-8")) == TARGET_CANONICAL_SHA256:
            exact_target_generations.append(generation)

        if arm == "structured_balanced":
            legal = legal_adjacent_moves(incumbent)
            untried = set(legal) - attempted
            operator_token = event.get("operator_move_token")
            if untried:
                if operator_token not in untried:
                    operator_integrity = False
                    operator_violations.append(f"generation {generation}: did not admit an untried legal move")
                elif candidate != legal[operator_token]:
                    operator_integrity = False
                    operator_violations.append(f"generation {generation}: candidate differs from admitted move")
            if operator_token is not None:
                attempted.add(str(operator_token))

        if strict_improvement:
            incumbent = candidate
            best_score = score

    return {
        "seed": seed,
        "arm": arm,
        "best_final_score": best_score,
        "discovery_reached": best_score >= INITIAL_SCORE + MIN_DISCOVERY_IMPROVEMENT,
        "first_discovery_generation": first_discovery,
        "exact_target_generations": exact_target_generations,
        "unique_admitted_candidate_count": len(unique_specs),
        "unique_canonical_move_count": len(set(direction_tokens)),
        "canonical_move_attempt_count": len(direction_tokens),
        "non_improving_candidate_count": non_improving_count,
        "semantic_invalid_count": invalid_count,
        "unsafe_count": unsafe_count,
        "operator_integrity": operator_integrity,
        "operator_violations": operator_violations,
        "final_incumbent_canonical_sha256": _sha256_bytes(canonical_spec(incumbent).encode("utf-8")),
    }


def analyze(repo_root: Path, run_dir: Path, protocol_path: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    run_dir = run_dir.resolve()
    protocol_path = protocol_path.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_protocol = build_protocol_manifest(repo_root)
    if protocol != expected_protocol:
        raise RuntimeError("protocol differs from the frozen implementation")
    manifest = json.loads((run_dir / "execution_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "COMPLETE"
        or manifest.get("terminal") != "B3A_EXECUTION_COMPLETE_PENDING_ANALYSIS"
        or manifest.get("protocol_sha256") != _sha256(protocol_path)
        or manifest.get("protocol_manifest_sha256") != canonical_manifest_sha256(protocol)
    ):
        raise RuntimeError("run manifest is not a complete frozen B3-A execution")

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    dispatches = [event for event in events if event.get("kind") == "dispatch"]
    completions = [event for event in events if event.get("kind") == "completion"]
    expected_count = len(PAIRED_SEEDS) * len(ARMS) * EVALUATIONS_PER_ARM_PER_SEED
    if len(dispatches) != expected_count or len(completions) != expected_count:
        raise RuntimeError("event ledger is not the complete paired 48-call matrix")
    if {event["request_id"] for event in dispatches} != {event["request_id"] for event in completions}:
        raise RuntimeError("dispatch/completion request lineage mismatch")
    if any(event.get("transport_runtime_failure") for event in completions):
        raise RuntimeError("complete manifest contains a provider runtime failure")

    trajectories = [_trajectory(seed, arm, completions) for seed in PAIRED_SEEDS for arm in ARMS]
    paired = []
    wins = losses = ties = 0
    control_reach = balanced_reach = 0
    coverage_higher = 0
    for seed in PAIRED_SEEDS:
        control = next(item for item in trajectories if item["seed"] == seed and item["arm"] == "structured_control")
        balanced = next(item for item in trajectories if item["seed"] == seed and item["arm"] == "structured_balanced")
        delta = float(balanced["best_final_score"]) - float(control["best_final_score"])
        if delta > 0:
            wins += 1
            disposition = "balanced_win"
        elif delta < 0:
            losses += 1
            disposition = "balanced_loss"
        else:
            ties += 1
            disposition = "tie"
        control_reach += int(bool(control["discovery_reached"]))
        balanced_reach += int(bool(balanced["discovery_reached"]))
        coverage_higher += int(int(balanced["unique_canonical_move_count"]) > int(control["unique_canonical_move_count"]))
        paired.append(
            {
                "seed": seed,
                "control_best_score": control["best_final_score"],
                "balanced_best_score": balanced["best_final_score"],
                "paired_delta": delta,
                "disposition": disposition,
                "control_discovery": control["discovery_reached"],
                "balanced_discovery": balanced["discovery_reached"],
                "control_first_discovery_generation": control["first_discovery_generation"],
                "balanced_first_discovery_generation": balanced["first_discovery_generation"],
            }
        )

    control_unsafe = sum(int(item["unsafe_count"]) for item in trajectories if item["arm"] == "structured_control")
    balanced_unsafe = sum(int(item["unsafe_count"]) for item in trajectories if item["arm"] == "structured_balanced")
    operator_integrity = all(bool(item["operator_integrity"]) for item in trajectories if item["arm"] == "structured_balanced")
    admitted = (
        balanced_reach > control_reach
        and wins >= 1
        and losses == 0
        and balanced_unsafe <= control_unsafe
        and operator_integrity
    )
    verdict = "B3A_BALANCED_EXPLORATION_ADMITTED" if admitted else "B3A_BALANCED_EXPLORATION_NOT_ADMITTED"
    return {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_dir.name,
        "terminal": "B3A_EVALUABLE_COMPLETE",
        "scientific_verdict": verdict,
        "claim_ceiling": protocol["claim_ceiling"],
        "model_calls": len(completions),
        "fresh_paired_seeds": list(PAIRED_SEEDS),
        "primary": {
            "control_discovery_count": control_reach,
            "balanced_discovery_count": balanced_reach,
            "paired_best_score_wins": wins,
            "paired_best_score_losses": losses,
            "paired_best_score_ties": ties,
            "unsafe_count_control": control_unsafe,
            "unsafe_count_balanced": balanced_unsafe,
            "operator_integrity": operator_integrity,
            "admission_rule_passed": admitted,
        },
        "secondary": {
            "balanced_move_coverage_higher_seed_count": coverage_higher,
            "diversity_without_search_value": coverage_higher > 0 and not admitted,
            "exact_target_is_diagnostic_only": True,
        },
        "paired_results": paired,
        "trajectories": trajectories,
        "source_sha256": {
            "protocol.json": _sha256(protocol_path),
            "execution_manifest.json": _sha256(run_dir / "execution_manifest.json"),
            "events.jsonl": _sha256(run_dir / "events.jsonl"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.repo_root, args.run_dir, args.protocol)
    _write_create_once(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
