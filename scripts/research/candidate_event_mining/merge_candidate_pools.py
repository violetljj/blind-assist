#!/usr/bin/env python3
"""Merge disjoint finalized review batches into one discovery pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.research.candidate_event_mining.pipeline import (
    CANDIDATE_REPORT_SCHEMA,
    ContractError,
    POOL_SCHEMA,
    read_json,
    refuse_overwrite,
    sha256_file,
    write_json,
)


def _candidate_ids(value: dict[str, Any], path: Path) -> set[str]:
    ids: list[str] = []
    for key in ("pool", "quarantine"):
        items = value.get(key)
        if not isinstance(items, list):
            raise ContractError(f"{path}: {key} must be a list")
        for item in items:
            if not isinstance(item, dict) or not item.get("candidate_id"):
                raise ContractError(f"{path}: invalid {key} candidate")
            ids.append(str(item["candidate_id"]))
    if len(ids) != len(set(ids)):
        raise ContractError(f"{path}: duplicate candidate IDs")
    return set(ids)


def run(args: argparse.Namespace) -> dict[str, Any]:
    full_report_path = args.candidate_report.resolve()
    full_report = read_json(full_report_path)
    if full_report.get("schema") != CANDIDATE_REPORT_SCHEMA:
        raise ContractError("unexpected full candidate report schema")
    full_candidates = full_report.get("candidates")
    if not isinstance(full_candidates, list):
        raise ContractError("full candidate report has no candidates list")
    full_ids = {str(item["candidate_id"]) for item in full_candidates}
    if len(full_ids) != len(full_candidates):
        raise ContractError("full candidate report contains duplicate IDs")
    if len(args.candidate_pool) != len(args.review_queue):
        raise ContractError("each candidate pool must have one matching review queue")

    reviewed_ids: set[str] = set()
    pool_items: list[dict[str, Any]] = []
    quarantine_items: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for pool_arg, queue_arg in zip(args.candidate_pool, args.review_queue):
        pool_path = pool_arg.resolve()
        queue_path = queue_arg.resolve()
        pool = read_json(pool_path)
        queue = read_json(queue_path)
        if pool.get("schema") != POOL_SCHEMA:
            raise ContractError(f"unexpected candidate pool schema: {pool_path}")
        if queue.get("schema") != CANDIDATE_REPORT_SCHEMA:
            raise ContractError(f"unexpected review queue schema: {queue_path}")
        pool_ids = _candidate_ids(pool, pool_path)
        queue_candidates = queue.get("candidates")
        if not isinstance(queue_candidates, list):
            raise ContractError(f"review queue has no candidates: {queue_path}")
        queue_ids = {str(item["candidate_id"]) for item in queue_candidates}
        if len(queue_ids) != len(queue_candidates):
            raise ContractError(f"review queue contains duplicate IDs: {queue_path}")
        if pool_ids != queue_ids:
            raise ContractError(f"pool/queue candidate IDs differ: {pool_path}")
        if not queue_ids <= full_ids:
            raise ContractError(f"pool contains IDs outside full report: {pool_path}")
        overlap = reviewed_ids & queue_ids
        if overlap:
            raise ContractError(f"review batches overlap: {sorted(overlap)[:5]}")
        reviewed_ids.update(queue_ids)
        pool_items.extend(dict(item) for item in pool["pool"])
        quarantine_items.extend(dict(item) for item in pool["quarantine"])
        batches.append(
            {
                "candidate_pool": {"path": str(pool_path), "sha256": sha256_file(pool_path)},
                "review_queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
                "candidate_count": len(queue_ids),
                "pool_count": int(pool.get("summary", {}).get("pool_count", len(pool["pool"]))),
                "quarantine_count": int(pool.get("summary", {}).get("quarantine_count", len(pool["quarantine"]))),
            }
        )
    if args.require_complete and reviewed_ids != full_ids:
        missing = sorted(full_ids - reviewed_ids)
        raise ContractError(f"merged review coverage is incomplete: {len(missing)} candidates remain")
    if not reviewed_ids:
        raise ContractError("no reviewed candidates supplied")

    output = args.output.resolve()
    refuse_overwrite(output)
    refuse_overwrite(Path(str(output) + ".sha256"))
    result = {
        "schema": POOL_SCHEMA,
        "pool_version": "r0-merged",
        "candidate_report_path": str(full_report_path),
        "candidate_report_sha256": sha256_file(full_report_path),
        "candidate_output_visibility": False,
        "review_batches": batches,
        "review_coverage": {
            "full_candidate_count": len(full_ids),
            "reviewed_candidate_count": len(reviewed_ids),
            "unreviewed_candidate_count": len(full_ids - reviewed_ids),
            "complete": reviewed_ids == full_ids,
        },
        "pool": sorted(pool_items, key=lambda item: str(item["candidate_id"])),
        "quarantine": sorted(quarantine_items, key=lambda item: str(item["candidate_id"])),
        "summary": {
            "candidate_count": len(reviewed_ids),
            "pool_count": len(pool_items),
            "quarantine_count": len(quarantine_items),
            "unreviewed_candidate_count": len(full_ids - reviewed_ids),
        },
        "authority": {
            "research_lane": "THESIS_DEVELOPMENT",
            "discovery_only": True,
            "event_truth": False,
            "training": False,
            "confirmation": False,
            "production": False,
            "safety": False,
            "default_app": False,
        },
    }
    write_json(output, result)
    Path(str(output) + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, action="append", required=True)
    parser.add_argument("--review-queue", type=Path, action="append", required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        result = run(parse_args())
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **result["summary"], "complete": result["review_coverage"]["complete"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
