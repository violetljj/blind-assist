#!/usr/bin/env python3
"""Attribute R1.1 failures in geometry -> detector -> semantics order."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from .contract import load_json, sha256_file, write_json
    from .diagnostic_contract import ANDROID_SCHEMA, ATTRIBUTION_SCHEMA, DIAGNOSTIC_ROLE, ORACLE_SCHEMA, require
except ImportError:
    from contract import load_json, sha256_file, write_json
    from diagnostic_contract import ANDROID_SCHEMA, ATTRIBUTION_SCHEMA, DIAGNOSTIC_ROLE, ORACLE_SCHEMA, require


def _attribute(oracle: dict[str, Any], android: dict[str, Any]) -> tuple[str, str]:
    if not oracle.get("projection_contract_valid") or not oracle.get("oracle_geometry_passed"):
        return "polygon_or_projection_contract", "oracle geometry did not validate the frozen target contact"
    coverage = android.get("detector_coverage")
    require(isinstance(coverage, dict), "Android source is missing detector coverage")
    if coverage.get("status") == "unsupported_taxonomy":
        return "detector_class_unsupported", "model label inventory does not cover the frozen target category"
    require(coverage.get("status") == "supported", "invalid detector coverage status")
    visible = android.get("visible_target_frame_count")
    matched = android.get("target_match_frame_count")
    require(isinstance(visible, int) and visible >= 0 and isinstance(matched, int) and 0 <= matched <= visible,
            "invalid target match counts")
    if visible > 0 and matched == 0:
        failure = "detector_zero_detection" if android.get("zero_detection_frame_count") == visible else "detector_target_miss"
        return failure, "oracle passed but Android did not match the frozen target"
    if android.get("android_oracle_geometry_parity") is not True:
        return "android_geometry_parity_failure", "matched target relation disagrees with the oracle geometry arm"
    if android.get("cooccurrence_runtime_alert_count", 0) > 0:
        return "risk_semantics_or_target_association", "co-occurring objects still generated runtime alerts"
    return "no_diagnostic_failure", "geometry, target matching, and target-only alert attribution agree"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    oracle = load_json(args.oracle_output)
    android = load_json(args.android_output)
    require(oracle.get("schema") == ORACLE_SCHEMA, "oracle schema mismatch")
    require(android.get("schema") == ANDROID_SCHEMA, "Android schema mismatch")
    require(oracle.get("diagnostic_set_role") == DIAGNOSTIC_ROLE and android.get("diagnostic_set_role") == DIAGNOSTIC_ROLE,
            "R1.1 inputs cannot claim held-out status")
    require(android.get("target_ledger_sha256") == oracle.get("target_ledger_sha256"), "target ledger binding mismatch")
    oracle_sources = {source["event_id"]: source for source in oracle.get("sources", [])}
    android_sources = {source["event_id"]: source for source in android.get("sources", [])}
    require(len(oracle_sources) == len(oracle.get("sources", [])), "oracle repeats event")
    require(len(android_sources) == len(android.get("sources", [])), "Android repeats event")
    require(set(oracle_sources) == set(android_sources), "source inventory mismatch")
    rows = []
    for event_id, oracle_source in oracle_sources.items():
        android_source = android_sources[event_id]
        require(android_source.get("source_id") == oracle_source.get("source_id"), f"{event_id}: source drift")
        require(android_source.get("target_instance_id") == oracle_source.get("target_instance_id"), f"{event_id}: target drift")
        category, reason = _attribute(oracle_source, android_source)
        rows.append({"event_id": event_id, "source_id": oracle_source["source_id"],
                     "target_instance_id": oracle_source["target_instance_id"],
                     "attribution": category, "reason": reason,
                     "oracle_geometry_passed": oracle_source["oracle_geometry_passed"],
                     "detector_coverage": android_source["detector_coverage"],
                     "target_match_frame_count": android_source.get("target_match_frame_count"),
                     "cooccurrence_runtime_alert_count": android_source.get("cooccurrence_runtime_alert_count", 0)})
    report = {
        "schema": ATTRIBUTION_SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_set_role": DIAGNOSTIC_ROLE, "oracle_output_sha256": sha256_file(args.oracle_output),
        "android_output_sha256": sha256_file(args.android_output), "sources": rows,
        "training_authorized": False, "held_out_gate_authorized": False,
        "next_round_requires_new_sources": True,
    }
    write_json(args.output, report)
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--android-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(parse_args(argv))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False)); return 2
    print(json.dumps({"ok": True, "attributions": {row["event_id"]: row["attribution"] for row in report["sources"]}}, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
