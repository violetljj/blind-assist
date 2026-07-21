#!/usr/bin/env python3
"""Evaluate the frozen six-source R1 geometry gate without pooling sources."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from .contract import load_json, require_false_flags, sha256_file, write_json
except ImportError:  # Direct execution through scripts/run_research_tool.py.
    from contract import load_json, require_false_flags, sha256_file, write_json


PREREGISTRATION_SCHEMA = "blindassist_ustrf_crosscam_geometry_multisource_preregistration_v1"
ANDROID_SCHEMA = "blindassist_ustrf_crosscam_multisource_android_output_v1"
REPORT_SCHEMA = "blindassist_ustrf_crosscam_multisource_geometry_gate_report_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    prereg = load_json(args.preregistration)
    android = load_json(args.android_output)
    _require(prereg.get("schema") == PREREGISTRATION_SCHEMA, "preregistration schema mismatch")
    _require(android.get("schema") == ANDROID_SCHEMA, "Android multisource schema mismatch")
    require_false_flags(prereg["authority"], "preregistration.authority")
    _require(android.get("preregistration_sha256") == sha256_file(args.preregistration),
             "Android output is not bound to the frozen preregistration")
    _require(android.get("uncertainty_frame_ratios") == [0.01, 0.02, 0.03],
             "Android output changed the frozen uncertainty profiles")
    for key in ("threshold_fit", "parameter_search", "thresholds_changed", "training_performed",
                "pexels_used_for_gate"):
        _require(android.get(key) is False, f"Android output must declare {key}=false")

    expected = {row["source_id"]: row for row in prereg["held_out_events"]}
    sources = android.get("sources")
    _require(isinstance(sources, list), "Android output sources must be an array")
    actual_ids = [row.get("source_id") for row in sources]
    _require(len(actual_ids) == len(set(actual_ids)), "Android output repeats a source")
    _require(set(actual_ids) == set(expected), "Android output must contain exactly the six held-out sources")

    source_rows: list[dict[str, Any]] = []
    for source in sources:
        source_id = source["source_id"]
        event = expected[source_id]
        expected_class = event["expected_class"]
        _require(source.get("expected_class") == expected_class, f"{source_id}: expected class drift")
        _require(source.get("event_id") == event["event_id"], f"{source_id}: event id drift")
        _require(source.get("window_ms") == event["window_ms"], f"{source_id}: window drift")
        status = source.get("status")
        _require(status in ("resolved", "unresolved"), f"{source_id}: invalid status")
        if status == "unresolved":
            _require(bool(source.get("unresolved_reason")), f"{source_id}: unresolved needs a reason")
        profile_summaries = source.get("profile_summaries")
        _require(isinstance(profile_summaries, list) and
                 [row.get("uncertainty_frame_ratio") for row in profile_summaries] == [0.01, 0.02, 0.03],
                 f"{source_id}: missing frozen per-profile summaries")
        for profile in profile_summaries:
            for key in ("inside_count", "outside_count", "uncertain_count"):
                _require(isinstance(profile.get(key), int) and profile[key] >= 0,
                         f"{source_id}: invalid {key}")
        robust_inside = source.get("robust_inside")
        _require(isinstance(robust_inside, bool), f"{source_id}: robust_inside must be boolean")
        expected_recall = int(robust_inside) if expected_class == "positive" and status == "resolved" else None
        expected_false_alarm = robust_inside if expected_class == "negative" and status == "resolved" else None
        _require(source.get("event_recall") == expected_recall, f"{source_id}: event recall mismatch")
        _require(source.get("false_alarm") == expected_false_alarm, f"{source_id}: false alarm mismatch")
        source_rows.append({
            "event_id": event["event_id"],
            "source_id": source_id,
            "expected_class": expected_class,
            "status": status,
            "unresolved_reason": source.get("unresolved_reason"),
            "profile_summaries": profile_summaries,
            "robust_counts": source.get("detection_relation_counts"),
            "robust_inside": robust_inside,
            "event_recall": expected_recall,
            "false_alarm": expected_false_alarm,
            "projection_receipt_sha256": source.get("projection_receipt_sha256"),
            "video_sha256": source.get("video_sha256"),
        })

    resolved = [row for row in source_rows if row["status"] == "resolved"]
    positive_robust = sum(row["expected_class"] == "positive" and row["robust_inside"] for row in resolved)
    negative_robust = sum(row["expected_class"] == "negative" and row["robust_inside"] for row in resolved)
    unresolved_count = len(source_rows) - len(resolved)
    stopping = prereg["stopping_rules"]
    checks = {
        "at_least_two_positive_sources_with_robust_inside":
            positive_robust >= stopping["minimum_positive_sources_with_robust_inside"],
        "all_negative_sources_without_robust_inside":
            negative_robust <= stopping["maximum_negative_sources_with_robust_inside"],
        "at_most_one_unresolved_source":
            unresolved_count <= stopping["maximum_unresolved_sources"],
        "pexels_did_not_rescue_gate": android["pexels_used_for_gate"] is False,
    }
    passed = all(checks.values())
    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256_file(args.preregistration),
        "android_output_sha256": sha256_file(args.android_output),
        "uncertainty_frame_ratios": [0.01, 0.02, 0.03],
        "source_results": source_rows,
        "gate_counts": {
            "positive_sources_with_robust_inside": positive_robust,
            "negative_sources_with_robust_inside": negative_robust,
            "unresolved_sources": unresolved_count,
        },
        "gate_checks": checks,
        "passed": passed,
        "next_stage": "dynamic_route_projection" if passed else "diagnose_polygon_projection_or_detector",
        "pooled_frame_average_reported": False,
        "pexels_used_for_gate": False,
        "training_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    write_json(args.output, report)
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--android-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(parse_args(argv))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "passed": report["passed"], **report["gate_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
