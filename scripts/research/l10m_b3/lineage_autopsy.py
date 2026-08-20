"""Canonical, zero-model-call autopsy of the complete L10M-B1 V2 search trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.research.l10m_b1.policy_space import (
    INITIAL_SPEC,
    PolicySpec,
    canonical_spec,
    changed_components,
    parse_raw,
    parse_structured,
)


PROTOCOL_ID = "L10M-B3-I0-CANONICAL-INTERVENTION-LINEAGE-AUTOPSY-V1"
SOURCE_RUN_ID = "b1-20260820T115002-98733875"
SOURCE_HASHES = {
    "events.jsonl": "b6f6e4ff3bccab2e46d0252e017fd7be73d4be533fd1233f0a011e94583e9ed6",
    "execution_manifest.json": "de3b02f0515baa18a9ef226c15bc20f0b98f28e6820dd504b0519d1c3b1b228b",
    "result_adjudication_r1.json": "35b0f816c281c4a3680eacf0aaaf15305864143db790f5aa433405f66ef7f72a",
}
B2_HASHES = {
    "protocol.json": "15d0cbf4d7cdd1d89440a2967eb7fd0b74b5caf2d924cad4e1fbe86052869926",
    "result.json": "7fe39bac5aa31d049dc95628c392b56789c676a2815b06fe6f9bc69df72df9e7",
}
IMPLEMENTATION_HASHES = {
    "scripts/research/l10m_b1/run_search.py": "0aa171481fb3aeb26f832a1cc75eacad95ca306acb78ac140c30d2058cdb49f2",
    "scripts/research/l10m_b1/policy_space.py": "923ea40af80866fb6f98725d2031e2c450972f138e3ce31e8df6de4270779c43",
}
SEEDS = (53, 71, 89)
ARMS = ("raw", "structured")
GENERATIONS = 8
INITIAL_SCORE = 0.9517241379310345
TARGET_SEED = 89
TARGET_CANONICAL_SHA256 = "b0110121ef34b54fb82d82be881d01ed07474983a5f897ae005b95a4f0185021"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha(spec: PolicySpec) -> str:
    return _sha256_bytes(canonical_spec(spec).encode("utf-8"))


def _verify_hashes(root: Path, expected: dict[str, str], *, label: str) -> None:
    for relative, digest in expected.items():
        actual = _sha256(root / relative)
        if actual != digest:
            raise RuntimeError(f"{label} hash mismatch for {relative}: {actual}")


def _parse_candidate(arm: str, source: str) -> PolicySpec:
    if arm == "raw":
        return parse_raw(source)
    if arm == "structured":
        return parse_structured(source)
    raise ValueError(f"unknown arm: {arm}")


def _mechanism_delta(before: PolicySpec, after: PolicySpec) -> dict[str, dict[str, Any]]:
    old = asdict(before)
    new = asdict(after)
    return {
        name: {"from": old[name], "to": new[name]}
        for name in old
        if old[name] != new[name]
    }


def reconstruct_trajectory(completions: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the exact strictly-improving single-incumbent B1 lineage."""
    if not completions:
        raise RuntimeError("trajectory has no completion events")
    ordered = sorted(completions, key=lambda event: int(event["generation"]))
    expected_generations = list(range(1, GENERATIONS + 1))
    if [int(event["generation"]) for event in ordered] != expected_generations:
        raise RuntimeError("trajectory does not contain generations 1..8 exactly once")

    incumbent = INITIAL_SPEC
    incumbent_score = INITIAL_SCORE
    seen: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    target_proposed: list[int] = []
    target_accepted: list[int] = []

    for event in ordered:
        generation = int(event["generation"])
        parent_sha = _canonical_sha(incumbent)
        proposal: PolicySpec | None = None
        parse_error: str | None = None
        try:
            proposal = _parse_candidate(str(event["arm"]), str(event["candidate_output"]))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, SyntaxError) as error:
            parse_error = f"{type(error).__name__}: {error}"

        proposal_sha = None if proposal is None else _canonical_sha(proposal)
        duplicate_of = None if proposal_sha is None else seen.get(proposal_sha)
        if proposal_sha is not None and proposal_sha not in seen:
            seen[proposal_sha] = generation

        valid = bool(event.get("semantic_valid")) and proposal is not None
        safe = not bool(event.get("unsafe_candidate"))
        score_value = event.get("behavioral_score")
        score = float(score_value) if score_value is not None else None
        accepted = bool(valid and safe and score is not None and score > incumbent_score)
        if proposal_sha == TARGET_CANONICAL_SHA256:
            target_proposed.append(generation)
            if accepted:
                target_accepted.append(generation)

        if not valid:
            disposition = "semantic_invalid"
        elif not safe:
            disposition = "unsafe_rejected"
        elif accepted:
            disposition = "accepted_strict_improvement"
        else:
            disposition = "not_retained_not_strictly_better"

        rows.append(
            {
                "generation": generation,
                "request_id": event["request_id"],
                "parent_canonical_sha256": parent_sha,
                "proposal_canonical_sha256": proposal_sha,
                "proposal_spec": None if proposal is None else asdict(proposal),
                "mechanism_delta_from_parent": None if proposal is None else _mechanism_delta(incumbent, proposal),
                "mechanism_delta_from_initial": None if proposal is None else _mechanism_delta(INITIAL_SPEC, proposal),
                "changed_components_from_initial": [] if proposal is None else changed_components(INITIAL_SPEC, proposal),
                "behavioral_score": score,
                "semantic_valid": valid,
                "unsafe_candidate": not safe,
                "target_intervention": proposal_sha == TARGET_CANONICAL_SHA256,
                "duplicate_of_generation": duplicate_of,
                "dedup_applied": False,
                "disposition": disposition,
                "parse_error": parse_error,
            }
        )
        if accepted:
            assert proposal is not None and score is not None
            incumbent = proposal
            incumbent_score = score

    proposal_counts = Counter(row["proposal_canonical_sha256"] for row in rows)
    return {
        "seed": int(ordered[0]["seed"]),
        "arm": str(ordered[0]["arm"]),
        "generation_budget": GENERATIONS,
        "generations_completed": len(rows),
        "budget_completed": len(rows) == GENERATIONS,
        "search_state_model": "single incumbent; replace only on semantic-valid, safe, strictly higher score",
        "dedup_mechanism_present": False,
        "unique_canonical_proposals": len(proposal_counts),
        "canonical_proposal_counts": dict(sorted(proposal_counts.items())),
        "target_proposed_generations": target_proposed,
        "target_accepted_generations": target_accepted,
        "final_incumbent_canonical_sha256": _canonical_sha(incumbent),
        "final_incumbent_score": incumbent_score,
        "lineage": rows,
    }


def _proposal_distribution(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in trajectory["lineage"]:
        signature = json.dumps(row["proposal_spec"], sort_keys=True, separators=(",", ":"))
        entry = grouped.setdefault(
            signature,
            {
                "proposal_spec": row["proposal_spec"],
                "mechanism_delta_from_initial": row["mechanism_delta_from_initial"],
                "generations": [],
                "count": 0,
            },
        )
        entry["generations"].append(row["generation"])
        entry["count"] += 1
    return sorted(grouped.values(), key=lambda entry: entry["generations"][0])


def _load_and_validate_events(run_dir: Path) -> list[dict[str, Any]]:
    _verify_hashes(run_dir, SOURCE_HASHES, label="B1 source")
    adjudication = json.loads((run_dir / "result_adjudication_r1.json").read_text(encoding="utf-8"))
    if (
        adjudication.get("run_id") != SOURCE_RUN_ID
        or adjudication.get("terminal") != "B1_EVALUABLE_COMPLETE"
        or adjudication.get("scientific_verdict") != "B1_INCONCLUSIVE"
        or adjudication.get("completion_count") != 48
    ):
        raise RuntimeError("source B1 receipt is not the frozen 48/48 evaluable closure")

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    dispatches = [event for event in events if event.get("kind") == "dispatch"]
    completions = [event for event in events if event.get("kind") == "completion"]
    if len(events) != 96 or len(dispatches) != 48 or len(completions) != 48:
        raise RuntimeError("B1 event ledger is not exactly 48 dispatch/completion pairs")
    dispatch_ids = Counter(event["request_id"] for event in dispatches)
    completion_ids = Counter(event["request_id"] for event in completions)
    if dispatch_ids != completion_ids or any(count != 1 for count in dispatch_ids.values()):
        raise RuntimeError("B1 request lineage is incomplete or ambiguous")
    expected_cells = {(seed, arm, generation) for seed in SEEDS for arm in ARMS for generation in range(1, 9)}
    actual_cells = {(int(e["seed"]), str(e["arm"]), int(e["generation"])) for e in completions}
    if actual_cells != expected_cells:
        raise RuntimeError("B1 completion ledger does not cover the frozen 3 x 2 x 8 matrix")
    return completions


def analyze(repo_root: Path, run_dir: Path, b2_dir: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    run_dir = run_dir.resolve()
    b2_dir = b2_dir.resolve()
    _verify_hashes(repo_root, IMPLEMENTATION_HASHES, label="B1 implementation")
    _verify_hashes(b2_dir, B2_HASHES, label="B2 source")
    b2 = json.loads((b2_dir / "result.json").read_text(encoding="utf-8"))
    if (
        b2.get("terminal") != "B2_EVALUABLE_COMPLETE"
        or b2.get("scientific_verdict") != "B2_SEARCH_PATH_FAILURE_SIGNAL"
        or b2.get("canonical_spec_sha256", {}).get("source") != TARGET_CANONICAL_SHA256
    ):
        raise RuntimeError("B2 does not identify the frozen seed-89 target intervention")

    completions = _load_and_validate_events(run_dir)
    trajectories = []
    for seed in SEEDS:
        for arm in ARMS:
            subset = [event for event in completions if event["seed"] == seed and event["arm"] == arm]
            trajectories.append(reconstruct_trajectory(subset))

    raw89 = next(item for item in trajectories if item["seed"] == TARGET_SEED and item["arm"] == "raw")
    structured89 = next(item for item in trajectories if item["seed"] == TARGET_SEED and item["arm"] == "structured")
    if raw89["target_proposed_generations"] != [2, 4, 8] or raw89["target_accepted_generations"] != [2]:
        raise RuntimeError("seed-89 Raw target lineage differs from the frozen observation")
    if structured89["target_proposed_generations"]:
        verdict = "B3_I0_TARGET_PROPOSED_REQUIRES_RETENTION_DECOMPOSITION"
        bottleneck = "selection_or_lineage"
        first_breakpoint = None
    else:
        verdict = "B3_I0_PROPOSAL_EXPLORATION_FAILURE_OBSERVED_SEED89"
        bottleneck = "proposal_exploration"
        first_breakpoint = {
            "generation": 2,
            "raw": "target action_selection turn_threshold 0.20 -> 0.10 proposed and strictly retained",
            "structured": "fallback min_quality 0.35 -> 0.50 proposed; target intervention absent",
            "meaning": "first target-reachability divergence, not the first arbitrary proposal-format divergence",
        }

    return {
        "protocol_id": PROTOCOL_ID,
        "terminal": "B3_I0_EVALUABLE_COMPLETE",
        "scientific_verdict": verdict,
        "observed_bottleneck": bottleneck,
        "model_calls": 0,
        "new_search_calls": 0,
        "new_evaluator_calls": 0,
        "claim_ceiling": "diagnostic localization of the consumed B1 V2 traces only; seed 89 cannot establish fix generalization, proposal-distribution causality, or admission of a new search mechanism",
        "source_evidence": {
            "b1_run_id": SOURCE_RUN_ID,
            "b1_sha256": SOURCE_HASHES,
            "b2_sha256": B2_HASHES,
            "b1_implementation_sha256": IMPLEMENTATION_HASHES,
            "target_canonical_sha256": TARGET_CANONICAL_SHA256,
        },
        "matrix_integrity": {
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "generations_per_trajectory": GENERATIONS,
            "completion_events": len(completions),
            "all_budgets_complete": all(item["budget_completed"] for item in trajectories),
        },
        "seed89_first_causal_breakpoint": first_breakpoint,
        "seed89_proposal_distribution": {
            "raw": _proposal_distribution(raw89),
            "structured": _proposal_distribution(structured89),
        },
        "seed89_alternatives": {
            "evaluator_or_ranking_eliminated_target": "ruled_out_for_observed_structured_trace_target_never_proposed",
            "target_failed_to_enter_parent_population": "not_applicable_target_never_proposed_and_runner_has_single_incumbent_not_population",
            "target_overwritten_by_later_mutation": "ruled_out_target_never_retained_or_proposed_in_structured",
            "dedup_or_canonicalization_removed_target": "ruled_out_runner_has_no_dedup_gate_and_every_completion_was_evaluated",
            "budget_ended_early": "ruled_out_all_8_structured_generations_completed",
            "finite_budget_sampling_limit": "still_applicable_absence_is_only_within_the_frozen_8_generation_budget",
        },
        "next_experiment_class_if_authorized": "B3-A proposal/exploration intervention on fresh seeds or fresh instances; do not use seed 89 for admission",
        "algorithm_fix_authorized_by_i0": False,
        "new_search_executed": False,
        "trajectories": trajectories,
    }


def _write_create_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--b1-run-dir", type=Path, required=True)
    parser.add_argument("--b2-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.repo_root, args.b1_run_dir, args.b2_dir)
    _write_create_once(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
