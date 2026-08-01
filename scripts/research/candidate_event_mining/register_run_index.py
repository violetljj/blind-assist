#!/usr/bin/env python3
"""Append a hash-bound candidate-mining run to the F:\\ba-data run index."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.research.candidate_event_mining.pipeline import (
    BUNDLE_SCHEMA,
    CANDIDATE_REPORT_SCHEMA,
    ContractError,
    POOL_SCHEMA,
    PROJECT_INDEX_SCHEMA,
    read_json,
    refuse_overwrite,
    sha256_file,
    validate_project_index,
    write_json,
)


RUN_INDEX_SCHEMA = "blindassist_candidate_event_mining_run_index_v1"


def _artifact(path: Path, label: str) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"{label} is missing: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _optional_artifact(path: Path | None, label: str) -> dict[str, str] | None:
    return None if path is None else _artifact(path, label)


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_index_path = args.project_index.resolve()
    project_index = validate_project_index(read_json(project_index_path))
    if project_index.get("schema") != PROJECT_INDEX_SCHEMA:
        raise ContractError("unexpected project index schema")
    output = args.output.resolve()
    if output.exists():
        index = read_json(output)
        if index.get("schema") != RUN_INDEX_SCHEMA:
            raise ContractError("unexpected run index schema")
        if index.get("project_index", {}).get("path") != str(project_index_path):
            raise ContractError("run index is bound to a different project index path")
        runs = index.get("runs")
        if not isinstance(runs, list):
            raise ContractError("run index runs must be a list")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        index = {
            "schema": RUN_INDEX_SCHEMA,
            "index_version": "r0",
            "project_id": project_index["project_id"],
            "project_index": {"path": str(project_index_path), "sha256": sha256_file(project_index_path)},
            "runs": [],
            "authority": {
                "research_lane": "THESIS_DEVELOPMENT",
                "event_truth": False,
                "training": False,
                "confirmation": False,
                "production": False,
                "safety": False,
                "default_app": False,
            },
        }
        runs = index["runs"]
    if any(item.get("run_id") == args.run_id for item in runs):
        raise ContractError(f"run_id already indexed: {args.run_id}")

    full_report = read_json(args.candidate_report.resolve())
    if full_report.get("schema") != CANDIDATE_REPORT_SCHEMA:
        raise ContractError("full candidate report schema mismatch")
    queue_report = read_json(args.review_queue_report.resolve())
    if queue_report.get("schema") != CANDIDATE_REPORT_SCHEMA:
        raise ContractError("review queue report schema mismatch")
    bundle_manifest = read_json(args.review_bundle.resolve() / "review_bundle_manifest.json")
    if bundle_manifest.get("schema") != BUNDLE_SCHEMA:
        raise ContractError("review bundle schema mismatch")
    if args.candidate_pool is not None:
        pool = read_json(args.candidate_pool.resolve())
        if pool.get("schema") != POOL_SCHEMA:
            raise ContractError("candidate pool schema mismatch")

    queue_meta = queue_report.get("review_queue", {})
    run_record = {
        "run_id": args.run_id,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": args.status,
        "data_role": "THESIS_DEVELOPMENT_CONSUMED_DISCOVERY",
        "source_count": len({item["source_id"] for item in full_report.get("candidates", [])}),
        "project_index": {"path": str(project_index_path), "sha256": sha256_file(project_index_path)},
        "adapter_manifest": _artifact(args.adapter_manifest, "adapter manifest"),
        "adapter_trace": _artifact(args.adapter_trace, "adapter trace"),
        "full_candidate_report": _artifact(args.candidate_report, "full candidate report"),
        "review_queue_report": _artifact(args.review_queue_report, "review queue report"),
        "review_queue": {
            "full_candidate_count": int(queue_meta.get("full_candidate_count", len(full_report.get("candidates", [])))),
            "selected_candidate_count": int(queue_meta.get("selected_candidate_count", len(queue_report.get("candidates", [])))),
            "unreviewed_candidate_count": int(queue_meta.get("unreviewed_candidate_count", 0)),
            "selection_policy": queue_meta.get("selection_policy"),
        },
        "review_bundle_manifest": _artifact(args.review_bundle.resolve() / "review_bundle_manifest.json", "review bundle manifest"),
        "luna_reviews": _optional_artifact(args.luna_reviews, "Luna reviews"),
        "candidate_pool": _optional_artifact(args.candidate_pool, "candidate pool"),
        "authorization": {
            "event_truth": False,
            "training": False,
            "confirmation": False,
            "production": False,
            "safety": False,
            "default_app": False,
        },
    }
    runs.append(run_record)
    runs.sort(key=lambda item: item["run_id"])
    index["project_index"]["sha256"] = sha256_file(project_index_path)
    index["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output, index)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(r"F:\ba-data\blindassist-candidate-event-mining\run_index.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", choices=("batch_complete_review_pending", "review_complete", "finalized"), required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--adapter-trace", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--review-queue-report", type=Path, required=True)
    parser.add_argument("--review-bundle", type=Path, required=True)
    parser.add_argument("--luna-reviews", type=Path)
    parser.add_argument("--candidate-pool", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        index = run(args)
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "run_count": len(index["runs"]), "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
