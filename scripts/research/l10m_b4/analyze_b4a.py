"""Create-once analysis for a complete evaluable B4-A run."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, canonical_spec, parse_structured
from scripts.research.l10m_b3a.exploration import legal_adjacent_moves

from .hard_benchmark import evaluate_instance, load_benchmark
from .protocol_b4a import (
    ARMS,
    GENERATIONS_PER_TRAJECTORY,
    PAIRED_IDENTITIES,
    PROTOCOL_ID,
    build_protocol_manifest,
    canonical_manifest_sha256,
)


EPSILON = 1e-12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _trajectory(
    rows: list[dict[str, Any]], instance: dict[str, Any], arm: str, paired_identity: int
) -> dict[str, Any]:
    completions = sorted(
        [row for row in rows if row.get("kind") == "completion"],
        key=lambda row: int(row["generation"]),
    )
    if [row["generation"] for row in completions] != list(range(1, GENERATIONS_PER_TRAJECTORY + 1)):
        raise RuntimeError("trajectory is incomplete or duplicated")
    incumbent = INITIAL_SPEC
    initial_score = float(evaluate_instance(INITIAL_SPEC, instance)["behavioral_score"])
    best_score = initial_score
    first_improvement = None
    attempted: set[str] = set()
    operator_violations: list[str] = []
    unique_candidates: set[str] = set()
    unsafe_count = 0
    invalid_count = 0
    for event in completions:
        if event.get("returncode") != 0 or event.get("transport_runtime_failure"):
            raise RuntimeError("runtime-failed completion in evaluable trajectory")
        if not event.get("semantic_valid"):
            invalid_count += 1
            continue
        candidate = parse_structured(event["candidate_output"])
        result = evaluate_instance(candidate, instance)
        for key in ("behavioral_score", "behavioral_vector", "unsafe_candidate", "semantic_valid"):
            if result[key] != event[key]:
                raise RuntimeError(f"recomputed evaluator field differs: {key}")
        score = float(result["behavioral_score"])
        unsafe = bool(result["unsafe_candidate"])
        unsafe_count += int(unsafe)
        unique_candidates.add(canonical_spec(candidate))
        strict = not unsafe and score > best_score + EPSILON
        if bool(event.get("strict_improvement")) != strict:
            raise RuntimeError("strict-improvement ledger differs from recomputation")
        if arm == "structured_balanced":
            token = event.get("operator_move_token")
            if token is not None:
                if token in attempted:
                    operator_violations.append(f"repeated move token: {token}")
                legal = legal_adjacent_moves(incumbent)
                if token not in legal or legal[token] != candidate:
                    operator_violations.append(f"non-legal admitted move: {token}")
                attempted.add(token)
        if strict:
            incumbent = candidate
            best_score = score
            if first_improvement is None:
                first_improvement = int(event["generation"])
    certificate_score = float(instance["qualified_global_score"])
    normalized = (best_score - initial_score) / (certificate_score - initial_score)
    return {
        "instance_id": instance["instance_id"],
        "paired_identity": paired_identity,
        "arm": arm,
        "initial_score": initial_score,
        "final_best_score": best_score,
        "normalized_progress": normalized,
        "global_optimum_reached": abs(best_score - certificate_score) <= EPSILON,
        "first_strict_improvement_generation": first_improvement,
        "unique_admitted_candidate_count": len(unique_candidates),
        "unique_canonical_move_count": len(attempted),
        "unsafe_count": unsafe_count,
        "semantic_invalid_count": invalid_count,
        "operator_integrity": not operator_violations,
        "operator_violations": operator_violations,
    }


def analyze(repo_root: Path, run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "execution_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    if frozen != build_protocol_manifest(repo_root):
        raise RuntimeError("protocol does not match frozen implementation")
    if manifest.get("terminal") != "B4A_EXECUTION_COMPLETE":
        raise RuntimeError("run is not complete and evaluable")
    if manifest.get("protocol_manifest_sha256") != canonical_manifest_sha256(frozen):
        raise RuntimeError("execution manifest protocol identity mismatch")
    events_path = run_dir / "events.jsonl"
    events = _events(events_path)
    benchmark = load_benchmark()
    qualified = {
        row["instance_id"]: row
        for row in json.loads(
            (repo_root / frozen["harder_cohort"]["certificate_path"]).read_text(encoding="utf-8")
        )["instances"]
    }
    instances = {row["instance_id"]: row for row in benchmark["instances"]}
    expected = len(PAIRED_IDENTITIES) * len(ARMS) * GENERATIONS_PER_TRAJECTORY
    completions = [row for row in events if row.get("kind") == "completion"]
    if len(completions) != expected:
        raise RuntimeError("completion count differs from frozen budget")
    trajectories: list[dict[str, Any]] = []
    for pair in PAIRED_IDENTITIES:
        instance_id = str(pair["instance_id"])
        instance = dict(instances[instance_id])
        instance["qualified_global_score"] = qualified[instance_id]["global_score"]
        identity = int(pair["paired_identity"])
        for arm in ARMS:
            selected = [
                row
                for row in events
                if row.get("instance_id") == instance_id
                and row.get("paired_identity") == identity
                and row.get("arm") == arm
            ]
            trajectories.append(_trajectory(selected, instance, arm, identity))
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in trajectories:
        grouped[(row["instance_id"], row["paired_identity"])][row["arm"]] = row
    pairs = []
    wins = losses = ties = 0
    for (instance_id, identity), arms in sorted(grouped.items()):
        control = arms["structured_control"]
        balanced = arms["structured_balanced"]
        delta = balanced["normalized_progress"] - control["normalized_progress"]
        if delta > EPSILON:
            disposition = "balanced_win"
            wins += 1
        elif delta < -EPSILON:
            disposition = "balanced_loss"
            losses += 1
        else:
            disposition = "tie"
            ties += 1
        pairs.append(
            {
                "instance_id": instance_id,
                "paired_identity": identity,
                "control_normalized_progress": control["normalized_progress"],
                "balanced_normalized_progress": balanced["normalized_progress"],
                "paired_normalized_delta": delta,
                "disposition": disposition,
            }
        )
    deltas = [row["paired_normalized_delta"] for row in pairs]
    control_global = sum(row["global_optimum_reached"] for row in trajectories if row["arm"] == "structured_control")
    balanced_global = sum(row["global_optimum_reached"] for row in trajectories if row["arm"] == "structured_balanced")
    control_unsafe = sum(row["unsafe_count"] for row in trajectories if row["arm"] == "structured_control")
    balanced_unsafe = sum(row["unsafe_count"] for row in trajectories if row["arm"] == "structured_balanced")
    integrity = all(row["operator_integrity"] for row in trajectories if row["arm"] == "structured_balanced")
    admitted = (
        statistics.median(deltas) > EPSILON
        and wins >= 6
        and losses == 0
        and balanced_global >= control_global
        and balanced_unsafe <= control_unsafe
        and integrity
    )
    return {
        "schema": "l10m_b4a_result_v1",
        "protocol_id": PROTOCOL_ID,
        "run_id": run_dir.name,
        "model_calls": expected,
        "terminal": "B4A_EVALUABLE_COMPLETE",
        "scientific_verdict": "B4A_BALANCED_SEARCH_VALUE_ESTABLISHED" if admitted else "B4A_BALANCED_SEARCH_VALUE_NOT_ESTABLISHED",
        "primary": {
            "median_paired_normalized_progress_delta": statistics.median(deltas),
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
            "control_global_optimum_reach": control_global,
            "balanced_global_optimum_reach": balanced_global,
            "operator_integrity": integrity,
            "admission_rule_passed": admitted,
        },
        "safety": {"control_unsafe_count": control_unsafe, "balanced_unsafe_count": balanced_unsafe},
        "paired_results": pairs,
        "trajectories": trajectories,
        "source_sha256": {
            "events.jsonl": _sha256(events_path),
            "execution_manifest.json": _sha256(run_dir / "execution_manifest.json"),
            "protocol.json": _sha256(protocol_path),
        },
        "claim_ceiling": frozen["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite create-once result: {args.output}")
    result = analyze(args.repo_root.resolve(), args.run_dir.resolve(), args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "terminal": result["terminal"], "verdict": result["scientific_verdict"]}))


if __name__ == "__main__":
    main()
