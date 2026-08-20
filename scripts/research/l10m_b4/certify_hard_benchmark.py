"""Exhaustively certify B4 harder-cohort search-pressure properties."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, PolicySpec, all_specs, canonical_spec

from .hard_benchmark import BENCHMARK_PATH, build_instance, legal_neighbors, load_benchmark
from scripts.research.l10m_b1.evaluator import evaluate_spec


EPSILON = 1e-12
MIN_INITIAL_IMPROVING_NEIGHBORS = 3
MIN_INITIAL_NON_IMPROVING_NEIGHBORS = 3
MIN_STRICT_STEPS_TO_GLOBAL_OPTIMUM = 2
BOUND_SOURCE_PATHS = (
    "scripts/research/l10m_b4/hard_benchmark_v1.json",
    "scripts/research/l10m_b4/hard_benchmark.py",
    "scripts/research/l10m_b4/certify_hard_benchmark.py",
    "scripts/research/l10m_b1/evaluator.py",
    "scripts/research/l10m_b1/policy_space.py",
    "scripts/research/l10m_b0/evaluation.py",
    "scripts/research/l10m_b0/b0c_precedence.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _bound_sources(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in BOUND_SOURCE_PATHS:
        path = repo_root / relative
        worktree_bytes = path.read_bytes()
        committed_bytes = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        if worktree_bytes != committed_bytes:
            raise RuntimeError(f"bound source is not identical to HEAD: {relative}")
        hashes[relative] = hashlib.sha256(worktree_bytes).hexdigest()
    return hashes


def _scores(instance: dict[str, Any]) -> dict[PolicySpec, float]:
    cohort = build_instance(instance)
    scores: dict[PolicySpec, float] = {}
    for spec in all_specs():
        result = evaluate_spec(spec, cohort)
        if not result["semantic_valid"] or result["unsafe_candidate"]:
            raise RuntimeError("hard benchmark admitted an invalid or unsafe finite-space candidate")
        scores[spec] = float(result["behavioral_score"])
    return scores


def _shortest_strict_path_to_global(
    scores: dict[PolicySpec, float], global_score: float
) -> list[PolicySpec] | None:
    queue: deque[PolicySpec] = deque([INITIAL_SPEC])
    predecessor: dict[PolicySpec, PolicySpec | None] = {INITIAL_SPEC: None}
    while queue:
        current = queue.popleft()
        if abs(scores[current] - global_score) <= EPSILON:
            path: list[PolicySpec] = []
            cursor: PolicySpec | None = current
            while cursor is not None:
                path.append(cursor)
                cursor = predecessor[cursor]
            return list(reversed(path))
        for neighbor in legal_neighbors(current):
            if neighbor in predecessor or scores[neighbor] <= scores[current] + EPSILON:
                continue
            predecessor[neighbor] = current
            queue.append(neighbor)
    return None


def certify_instance(instance: dict[str, Any]) -> dict[str, Any]:
    scores = _scores(instance)
    initial_score = scores[INITIAL_SPEC]
    initial_neighbors = legal_neighbors(INITIAL_SPEC)
    improving = [spec for spec in initial_neighbors if scores[spec] > initial_score + EPSILON]
    non_improving = [spec for spec in initial_neighbors if scores[spec] <= initial_score + EPSILON]
    global_score = max(scores.values())
    global_specs = [spec for spec, score in scores.items() if abs(score - global_score) <= EPSILON]
    path = _shortest_strict_path_to_global(scores, global_score)
    if path is None:
        strict_steps = None
    else:
        strict_steps = len(path) - 1
    local_maxima = [
        spec
        for spec, score in scores.items()
        if all(scores[neighbor] <= score + EPSILON for neighbor in legal_neighbors(spec))
    ]
    criteria = {
        "at_least_three_initial_improving_neighbors": len(improving)
        >= MIN_INITIAL_IMPROVING_NEIGHBORS,
        "at_least_three_initial_non_improving_neighbors": len(non_improving)
        >= MIN_INITIAL_NON_IMPROVING_NEIGHBORS,
        "no_initial_neighbor_reaches_global_optimum": all(
            scores[spec] < global_score - EPSILON for spec in initial_neighbors
        ),
        "global_optimum_requires_at_least_two_strict_steps": strict_steps is not None
        and strict_steps >= MIN_STRICT_STEPS_TO_GLOBAL_OPTIMUM,
        "quality_interaction_is_present": all(
            spec.fallback_min_quality == INITIAL_SPEC.fallback_min_quality
            for spec in improving
        ),
    }
    return {
        "instance_id": instance["instance_id"],
        "finite_spec_count": len(scores),
        "episode_count": len(build_instance(instance)),
        "initial_score": initial_score,
        "global_score": global_score,
        "global_optimum_count": len(global_specs),
        "initial_neighbor_count": len(initial_neighbors),
        "initial_improving_neighbor_count": len(improving),
        "initial_non_improving_neighbor_count": len(non_improving),
        "initial_improving_neighbors": [canonical_spec(spec) for spec in improving],
        "shortest_strict_steps_to_global_optimum": strict_steps,
        "shortest_strict_path": [] if path is None else [canonical_spec(spec) for spec in path],
        "local_maximum_count": len(local_maxima),
        "criteria": criteria,
        "qualified": all(criteria.values()),
    }


def certify(path: Path = BENCHMARK_PATH, repo_root: Path | None = None) -> dict[str, Any]:
    benchmark = load_benchmark(path)
    instances = [certify_instance(instance) for instance in benchmark["instances"]]
    qualified = all(instance["qualified"] for instance in instances)
    return {
        "schema": "l10m_b4_hard_benchmark_certificate_v1",
        "benchmark_id": benchmark["benchmark_id"],
        "model_call_count": 0,
        "construction_role": "benchmark development and qualification only; no search-arm outcome observed",
        "instances": instances,
        "all_instances_qualified": qualified,
        "terminal": "B4_HARD_BENCHMARK_QUALIFIED" if qualified else "B4_HARD_BENCHMARK_NOT_QUALIFIED",
        "claim_ceiling": "exhaustive finite-landscape search-pressure qualification only; not evidence that any search mechanism is better",
        "source_commit": None if repo_root is None else _git(repo_root, "rev-parse", "HEAD"),
        "source_sha256": (
            {str(path): _sha256(path)}
            if repo_root is None
            else _bound_sources(repo_root)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.repo_root is None:
        parser.error("--repo-root is required for a create-once frozen certificate")
    result = certify(args.benchmark, args.repo_root)
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite create-once certificate: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "terminal": result["terminal"]}))


if __name__ == "__main__":
    main()
