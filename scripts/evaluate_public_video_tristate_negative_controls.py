#!/usr/bin/env python3
"""Evaluate frozen tri-state lifecycle behavior on no-exit nuisance controls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_tristate_contract as prospective
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_tristate_negative_controls_v1"


def evaluate_source(source: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    samples = source.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("negative-control source has no samples")
    settings = contract["lifecycle"]
    result = lifecycle.tristate_exit_intervals(
        samples,
        settings["selected_groups"],
        entry_window_samples=settings["entry_window_samples"],
        entry_min_active_samples=settings["entry_min_active_samples"],
        clear_absent_samples=settings["clear_absent_samples"],
    )
    active_samples = [
        sample for sample in samples
        if lifecycle.sample_is_active(sample, set(settings["selected_groups"]))
    ]
    active_classes = Counter(
        class_name
        for sample in active_samples
        for class_name, count in sample.get("semantic_class_counts", {}).items()
        for _ in range(int(count))
    )
    passed = not result["intervals"] and result["terminal_state"] == "clear"
    return {
        "source_id": source["source_id"],
        "control_role": source.get("control_role"),
        "sample_count": len(samples),
        "active_sample_count": len(active_samples),
        "active_fraction": len(active_samples) / len(samples),
        "active_class_counts": dict(sorted(active_classes.items())),
        "lifecycle": result,
        "passed": passed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    scan = lifecycle.verify_json_sidecar(args.scan)
    contract, contract_attestation = prospective.load_contract(args.contract)
    if scan.get("schema") != "blindassist_public_video_prompt_free_exit_discovery_v1":
        raise ValueError("unexpected scan schema")
    if scan.get("prospective_contract") != contract_attestation:
        raise ValueError("negative-control scan is not bound to the supplied contract")
    sources = scan.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("negative-control scan has no sources")
    results = [evaluate_source(source, contract) for source in sources]
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "scan": {"path": str(args.scan.resolve()), "sha256": common.sha256_file(args.scan)},
            "contract": contract_attestation,
        },
        "contract": {
            "selected_groups": contract["lifecycle"]["selected_groups"],
            "entry_window_samples": contract["lifecycle"]["entry_window_samples"],
            "entry_min_active_samples": contract["lifecycle"]["entry_min_active_samples"],
            "clear_absent_samples": contract["lifecycle"]["clear_absent_samples"],
            "expected_control_behavior": "zero completed events and terminal clear",
        },
        "sources": results,
        "summary": {
            "source_count": len(results),
            "passed_source_count": sum(result["passed"] for result in results),
            "failed_source_count": sum(not result["passed"] for result in results),
            "passed": all(result["passed"] for result in results),
        },
        "evidence_limit": "GPT-reviewed nuisance controls only; not human truth, training truth, calibration, blind evaluation, Android runtime authorization, or production evidence.",
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(
        common.sha256_file(args.output) + "\n", encoding="ascii"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
