#!/usr/bin/env python3
"""Select a deterministic, coverage-aware review-budget report.

The full candidate report remains the discovery inventory. This command makes
an explicitly bounded review queue so an independent reviewer can inspect a
human-sized set without silently treating all unreviewed candidates as
negative. The output is a candidate-report-shaped subset with parent-report
lineage; only that subset may be finalized into a reviewed candidate pool.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.research.candidate_event_mining.pipeline import (
    CANDIDATE_REPORT_SCHEMA,
    ContractError,
    load_contract,
    read_json,
    refuse_overwrite,
    sha256_file,
    write_json,
)


def _rank(candidate: dict[str, Any]) -> tuple[float, int, int, str]:
    return (
        -float(candidate.get("trigger_score_peak", 0.0)),
        -int(candidate.get("active_frame_count", 0)),
        int(candidate.get("start_timestamp_ms", 0)),
        str(candidate.get("candidate_id", "")),
    )


def _select(
    candidates: list[dict[str, Any]],
    maximum: int,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if maximum <= 0:
        raise ContractError("max_candidates must be positive")
    excluded_ids = excluded_ids or set()
    eligible = [
        candidate
        for candidate in candidates
        if str(candidate.get("candidate_id", "")) not in excluded_ids
    ]
    if not eligible:
        raise ContractError("no eligible candidates remain after exclusions")
    ordered = sorted(eligible, key=_rank)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in ordered:
        key = (str(candidate["source_id"]), str(candidate["trigger_type"]))
        groups.setdefault(key, []).append(candidate)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    group_heads = sorted((members[0] for members in groups.values()), key=_rank)
    for candidate in group_heads:
        if len(selected) >= maximum:
            break
        selected.append(candidate)
        selected_ids.add(str(candidate["candidate_id"]))

    clusters: dict[str, list[dict[str, Any]]] = {}
    for candidate in ordered:
        clusters.setdefault(str(candidate.get("cluster_id", "")), []).append(candidate)
    cluster_heads = sorted((members[0] for members in clusters.values()), key=_rank)
    for candidate in cluster_heads:
        candidate_id = str(candidate["candidate_id"])
        if len(selected) >= maximum:
            break
        if candidate_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate_id)

    for candidate in ordered:
        if len(selected) >= maximum:
            break
        candidate_id = str(candidate["candidate_id"])
        if candidate_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate_id)
    return sorted(
        selected,
        key=lambda item: (
            str(item["source_id"]),
            int(item["start_timestamp_ms"]),
            str(item["trigger_type"]),
            str(item["candidate_id"]),
        ),
    )


def _candidate_ids_from_exclusion(path: Path) -> set[str]:
    value = read_json(path)
    candidate_ids: set[str] = set()
    raw_ids = value.get("candidate_ids")
    if isinstance(raw_ids, list):
        candidate_ids.update(str(item) for item in raw_ids if str(item))
    for key in ("candidates", "pool", "quarantine"):
        raw_items = value.get(key)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if isinstance(item, dict) and item.get("candidate_id"):
                candidate_ids.add(str(item["candidate_id"]))
    if not candidate_ids:
        raise ContractError(f"exclusion report has no candidate IDs: {path}")
    return candidate_ids


def run(args: argparse.Namespace) -> dict[str, Any]:
    _contract, contract_meta = load_contract(args.contract.resolve())
    input_path = args.candidate_report.resolve()
    report = read_json(input_path)
    if report.get("schema") != CANDIDATE_REPORT_SCHEMA:
        raise ContractError("unexpected candidate report schema")
    if report.get("contract", {}).get("sha256") != contract_meta["sha256"]:
        raise ContractError("candidate report was not produced under the supplied contract")
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ContractError("full candidate report has no candidates")
    all_candidates = [dict(candidate) for candidate in candidates]
    excluded_reports: list[dict[str, Any]] = []
    excluded_ids: set[str] = set()
    for exclusion_path in args.exclude_report:
        path = exclusion_path.resolve()
        ids = _candidate_ids_from_exclusion(path)
        excluded_ids.update(ids)
        excluded_reports.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "candidate_count": len(ids),
            }
        )
    eligible_candidates = [
        candidate
        for candidate in all_candidates
        if str(candidate["candidate_id"]) not in excluded_ids
    ]
    selected = _select(all_candidates, args.max_candidates, excluded_ids)
    selected_counts = Counter(str(candidate["trigger_type"]) for candidate in selected)
    selected_clusters = {str(candidate.get("cluster_id", "")) for candidate in selected}
    output = args.output.resolve()
    refuse_overwrite(output)
    refuse_overwrite(Path(str(output) + ".sha256"))
    queue = {
        "schema": CANDIDATE_REPORT_SCHEMA,
        "run_id": report.get("run_id"),
        "contract": report["contract"],
        "project_index": report.get("project_index"),
        "input_trace": report.get("input_trace"),
        "summary": {
            "input_frame_count": report.get("summary", {}).get("input_frame_count"),
            "raw_candidate_count": len(selected),
            "deduplicated_candidate_count": len(selected),
            "cluster_count": len(selected_clusters),
            "candidate_type_counts": {key: selected_counts.get(key, 0) for key in sorted(selected_counts)},
            "review_queue_full_candidate_count": len(candidates),
            "review_queue_excluded_candidate_count": len(excluded_ids),
            "review_queue_eligible_candidate_count": len(eligible_candidates),
            "review_queue_selected_candidate_count": len(selected),
            "review_queue_unreviewed_candidate_count": len(eligible_candidates) - len(selected),
        },
        "review_queue": {
            "selection_policy": (
                "score_rank_then_source_x_trigger_coverage_then_cluster_coverage"
                "+exclude_prior_reviewed_ids"
            ),
            "max_candidates": args.max_candidates,
            "parent_candidate_report_path": str(input_path),
            "parent_candidate_report_sha256": sha256_file(input_path),
            "full_candidate_count": len(candidates),
            "excluded_reports": excluded_reports,
            "excluded_candidate_count": len(excluded_ids),
            "eligible_candidate_count": len(eligible_candidates),
            "selected_candidate_count": len(selected),
            "unreviewed_candidate_count": len(eligible_candidates) - len(selected),
            "unreviewed_disposition": "not_reviewed_and_excluded_from_candidate_pool",
        },
        "candidates": selected,
        "authority": {
            "event_truth": False,
            "training": False,
            "confirmation": False,
            "production": False,
            "safety": False,
            "default_app": False,
        },
    }
    write_json(output, queue)
    Path(str(output) + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    return queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[],
        help="Candidate report, queue, bundle manifest, or pool whose IDs are already reviewed",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        queue = run(args)
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                **queue["summary"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
