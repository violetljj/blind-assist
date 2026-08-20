"""Exhaustively certify the fresh B5-A cohort before any search-arm call."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .certify_hard_benchmark import certify_instance
from .fresh_benchmark import BENCHMARK_PATH, load_fresh_benchmark


MIN_STRICT_STEPS_TO_GLOBAL_OPTIMUM = 5
BOUND_SOURCE_PATHS = (
    "scripts/research/l10m_b4/fresh_benchmark_v1.json",
    "scripts/research/l10m_b4/fresh_benchmark.py",
    "scripts/research/l10m_b4/certify_fresh_benchmark.py",
    "scripts/research/l10m_b4/hard_benchmark.py",
    "scripts/research/l10m_b4/certify_hard_benchmark.py",
    "scripts/research/l10m_b1/evaluator.py",
    "scripts/research/l10m_b1/policy_space.py",
    "scripts/research/l10m_b0/evaluation.py",
    "scripts/research/l10m_b0/b0c_precedence.py",
)


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


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


def certify(path: Path = BENCHMARK_PATH, repo_root: Path | None = None) -> dict[str, Any]:
    benchmark = load_fresh_benchmark(path)
    instances = []
    for source in benchmark["instances"]:
        row = certify_instance(source)
        row["criteria"]["global_optimum_requires_at_least_five_strict_steps"] = (
            row["shortest_strict_steps_to_global_optimum"] is not None
            and row["shortest_strict_steps_to_global_optimum"]
            >= MIN_STRICT_STEPS_TO_GLOBAL_OPTIMUM
        )
        row["qualified"] = all(row["criteria"].values())
        instances.append(row)
    qualified = all(row["qualified"] for row in instances)
    return {
        "schema": "l10m_b5_fresh_harder_benchmark_certificate_v1",
        "benchmark_id": benchmark["benchmark_id"],
        "model_call_count": 0,
        "construction_role": "outcome-blind benchmark construction and exhaustive landscape qualification only; no B5-A search-arm call or outcome observed",
        "instances": instances,
        "all_instances_qualified": qualified,
        "terminal": "B5_FRESH_HARDER_COHORT_QUALIFIED" if qualified else "B5_FRESH_HARDER_COHORT_NOT_QUALIFIED",
        "claim_ceiling": "finite-landscape freshness and search-pressure qualification only; not search-operator evidence",
        "source_commit": None if repo_root is None else _git(repo_root, "rev-parse", "HEAD"),
        "source_sha256": {} if repo_root is None else _bound_sources(repo_root),
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
