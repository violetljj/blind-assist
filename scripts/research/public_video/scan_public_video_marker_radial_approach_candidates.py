#!/usr/bin/env python3
"""Freeze r7.25 radial-approach candidates before visual review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_marker_radial_approach_contract as approach_contract
import public_video_tristate_contract as tristate_contract
import run_public_silver_frozen_feature_probe as common
import run_public_video_marker_radial_approach_probe as probe


SCHEMA = "blindassist_public_video_marker_radial_approach_candidates_v1"


def candidate_rows(features: dict[str, Any], base_contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = probe.diagnose(features, base_contract)
    return [
        {"source_id": source["source_id"], "events": [event for event in source["events"] if event["radial_approach_passed"]]}
        for source in rows
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract, contract_meta = approach_contract.load_contract(args.contract)
    base_contract, base_meta = tristate_contract.load_contract(args.base_contract)
    features = lifecycle.verify_json_sidecar(args.features)
    if base_meta["sha256"] != contract["bound_inputs"]["chromatic_marker_contract_sha256"]:
        raise ValueError("base chromatic contract hash mismatch")
    if features.get("prospective_contract", {}).get("sha256") != base_meta["sha256"]:
        raise ValueError("feature report was not extracted under the bound base contract")
    forbidden = {contract["bound_inputs"]["japan_video_sha256"], contract["bound_inputs"]["matoaka_video_sha256"]}
    if any(source.get("video_sha256") in forbidden for source in features.get("sources", [])):
        raise ValueError("derivation video cannot become prospective")
    rows = candidate_rows(features, base_contract)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approach_contract": contract_meta,
        "base_contract": base_meta,
        "feature_report_sha256": common.sha256_file(args.features),
        "review_received": False,
        "sources": rows,
        "summary": {
            "source_count": len(rows),
            "candidate_event_count": sum(len(source["events"]) for source in rows),
        },
        "authorizations": {"training": False, "calibration": False, "blind": False, "android_runtime_change": False, "production_model_replacement": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, **value["summary"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
