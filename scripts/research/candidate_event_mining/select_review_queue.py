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


def _select(candidates: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if maximum <= 0:
        raise ContractError("max_candidates must be positive")
    ordered = sorted(candidates, key=_rank)
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
    selected = _select([dict(candidate) for candidate in candidates], args.max_candidates)
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
            "review_queue_selected_candidate_count": len(selected),
            "review_queue_unreviewed_candidate_count": len(candidates) - len(selected),
        },
        "review_queue": {
            "selection_policy": "score_rank_then_source_x_trigger_coverage_then_cluster_coverage",
            "max_candidates": args.max_candidates,
            "parent_candidate_report_path": str(input_path),
            "parent_candidate_report_sha256": sha256_file(input_path),
            "full_candidate_count": len(candidates),
            "selected_candidate_count": len(selected),
            "unreviewed_candidate_count": len(candidates) - len(selected),
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
