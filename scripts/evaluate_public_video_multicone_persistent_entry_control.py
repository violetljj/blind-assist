#!/usr/bin/env python3
"""Evaluate the frozen multi-cone expert on a persistent-risk entry control."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_multicone_policy as multicone
import public_video_tristate_contract as prospective
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_multicone_persistent_entry_control_v1"


def score_persistent_entry(
    lifecycle_result: dict[str, Any], transition_window_ms: list[int]
) -> dict[str, Any]:
    if len(transition_window_ms) != 2:
        raise ValueError("entry transition window must contain two timestamps")
    start_ms, end_ms = map(int, transition_window_ms)
    open_event = lifecycle_result.get("open_event")
    entry_ms = (
        int(open_event["event_entry_timestamp_ms"])
        if isinstance(open_event, dict)
        else None
    )
    return {
        "completed_exit_interval_count": len(lifecycle_result["intervals"]),
        "terminal_state": lifecycle_result["terminal_state"],
        "open_event_present": isinstance(open_event, dict),
        "event_entry_timestamp_ms": entry_ms,
        "entry_inside_visual_transition": (
            entry_ms is not None and start_ms <= entry_ms <= end_ms
        ),
        "passed": bool(
            not lifecycle_result["intervals"]
            and lifecycle_result["terminal_state"] in {"present", "uncertain"}
            and entry_ms is not None
            and start_ms <= entry_ms <= end_ms
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    scan = lifecycle.verify_json_sidecar(args.scan)
    review = lifecycle.verify_json_sidecar(args.review)
    contract, contract_attestation = prospective.load_contract(args.contract)
    policy = multicone.validate_policy(contract)
    if scan.get("prospective_contract") != contract_attestation:
        raise ValueError("scan is not bound to the supplied multi-cone contract")
    if review.get("schema") != "blindassist_public_video_persistent_entry_gpt_review_v1":
        raise ValueError("unexpected persistent-entry review schema")
    review_body = review.get("review")
    if not isinstance(review_body, dict):
        raise ValueError("persistent-entry review body is missing")
    sources = scan.get("sources")
    if not isinstance(sources, list):
        raise ValueError("scan sources are missing")
    matches = [
        source for source in sources
        if source.get("source_id") == review_body.get("source_id")
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("samples"), list):
        raise ValueError("review source does not bind exactly one scan sequence")
    samples = multicone.apply_policy(matches[0]["samples"], policy)
    settings = contract["lifecycle"]
    result = lifecycle.tristate_exit_intervals(
        samples,
        ["barrier_structure"],
        entry_window_samples=settings["entry_window_samples"],
        entry_min_active_samples=settings["entry_min_active_samples"],
        clear_absent_samples=settings["clear_absent_samples"],
    )
    score = score_persistent_entry(
        result, review_body["visual_entry_transition_window_ms"]
    )
    clear = lifecycle.activity_window_diagnostics(
        samples, ["barrier_structure"], review_body["stable_clear_window_ms"]
    )
    risk = lifecycle.activity_window_diagnostics(
        samples, ["barrier_structure"], review_body["risk_present_window_ms"]
    )
    acceptance = contract["acceptance"]
    coverage_passed = bool(
        clear["active_fraction"] is not None
        and clear["active_fraction"] <= acceptance["maximum_stable_clear_active_fraction"]
        and risk["active_fraction"] is not None
        and risk["active_fraction"] >= acceptance["minimum_risk_present_active_fraction"]
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "scan": {"path": str(args.scan.resolve()), "sha256": common.sha256_file(args.scan)},
            "review": {"path": str(args.review.resolve()), "sha256": common.sha256_file(args.review)},
            "contract": contract_attestation,
        },
        "risk_evidence_policy": policy,
        "lifecycle": result,
        "diagnostics": {
            "stable_clear_window": clear,
            "risk_present_window": risk,
        },
        "score": score,
        "acceptance": {
            "lifecycle_entry_and_persistence_passed": score["passed"],
            "coverage_passed": coverage_passed,
            "passed": score["passed"] and coverage_passed,
        },
        "evidence_limit": "Licensed same-city vehicle-view persistent-risk mechanism control only; not an independent pedestrian source, human truth, training truth, calibration, blind evaluation, Android runtime authorization, or production evidence.",
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
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["acceptance"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
