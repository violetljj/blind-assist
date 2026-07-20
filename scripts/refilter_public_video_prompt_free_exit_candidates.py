#!/usr/bin/env python3
"""Re-filter a hash-bound prompt-free exit scan with absence persistence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import scan_public_video_prompt_free_exit_candidates as discovery


SCHEMA = "blindassist_public_video_prompt_free_exit_persistence_refilter_v1"


def verify_report(path: Path) -> dict[str, Any]:
    mil.reject_independent_direction(path)
    sidecar = Path(str(path) + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"report or sidecar is missing: {path}")
    expected = sidecar.read_text(encoding="ascii").strip().lower()
    actual = common.sha256_file(path)
    if expected != actual:
        raise ValueError(f"report sidecar mismatch: {path}")
    report = common.load_json(path)
    if report.get("schema") != discovery.SCHEMA:
        raise ValueError("unexpected discovery report schema")
    return report


def refilter(report: dict[str, Any], *, min_absent_run_samples: int) -> dict[str, Any]:
    if min_absent_run_samples <= 0:
        raise ValueError("minimum absent run must be positive")
    interval_ms = int(report["sampling"]["sample_interval_ms"])
    sources: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for source in report["sources"]:
        source_candidates = discovery.discover_adjacent_exits(
            source["source_id"],
            source["samples"],
            sample_interval_ms=interval_ms,
            min_absent_run_samples=min_absent_run_samples,
        )
        sources.append({**source, "exit_candidates": source_candidates})
        candidates.extend(source_candidates)
    return {
        "sources": sources,
        "exit_candidates": candidates,
        "summary": {
            "source_count": len(sources),
            "sample_count": sum(int(source["sample_count"]) for source in sources),
            "exit_candidate_count": len(candidates),
            "exit_candidate_count_by_group": dict(sorted(Counter(
                candidate["semantic_group"] for candidate in candidates
            ).items())),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.input_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    source_report = verify_report(args.input_report)
    filtered = refilter(
        source_report,
        min_absent_run_samples=args.min_absent_run_samples,
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_report": str(args.input_report.resolve()),
        "input_report_sha256": common.sha256_file(args.input_report),
        "source_discovery_schema": source_report["schema"],
        "model": source_report["model"],
        "sampling": {
            **source_report["sampling"],
            "minimum_consecutive_absent_samples": args.min_absent_run_samples,
            "minimum_absent_duration_ms": (
                args.min_absent_run_samples
                * int(source_report["sampling"]["sample_interval_ms"])
            ),
        },
        **filtered,
        "evidence_limit": "Persistence re-filter of discovery proposals only; no candidate is training truth, calibration evidence, blind evidence, or production evidence.",
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(str(args.output) + ".sha256").write_text(
        common.sha256_file(args.output) + "\n", encoding="ascii"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-absent-run-samples", type=int, required=True)
    args = parser.parse_args()
    if args.min_absent_run_samples <= 0:
        parser.error("minimum absent run must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
