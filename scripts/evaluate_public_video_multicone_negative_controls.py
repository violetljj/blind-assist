#!/usr/bin/env python3
"""Prospectively evaluate the frozen multi-cone expert on nuisance controls."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import evaluate_public_video_tristate_negative_controls as negative
import public_video_multicone_policy as multicone
import public_video_tristate_contract as prospective
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_multicone_negative_controls_v1"


def evaluate_source(
    source: dict[str, Any], contract: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    filtered = dict(source)
    filtered["samples"] = multicone.apply_policy(source["samples"], policy)
    return negative.evaluate_source(filtered, contract)


def run(args: argparse.Namespace) -> dict[str, Any]:
    scan = lifecycle.verify_json_sidecar(args.scan)
    contract, contract_attestation = prospective.load_contract(args.contract)
    policy = multicone.validate_policy(contract)
    if scan.get("schema") != "blindassist_public_video_prompt_free_exit_discovery_v1":
        raise ValueError("unexpected scan schema")
    if scan.get("prospective_contract") != contract_attestation:
        raise ValueError("scan is not bound to the supplied multi-cone contract")
    sources = scan.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("multi-cone negative-control scan has no sources")
    results = [evaluate_source(source, contract, policy) for source in sources]
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "scan": {"path": str(args.scan.resolve()), "sha256": common.sha256_file(args.scan)},
            "contract": contract_attestation,
        },
        "risk_evidence_policy": policy,
        "sources": results,
        "summary": {
            "source_count": len(results),
            "passed_source_count": sum(result["passed"] for result in results),
            "failed_source_count": sum(not result["passed"] for result in results),
            "passed": all(result["passed"] for result in results),
        },
        "evidence_limit": "Prospective GPT-reviewed nuisance controls only; not human truth, training truth, calibration, blind evaluation, Android runtime authorization, or production evidence.",
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
