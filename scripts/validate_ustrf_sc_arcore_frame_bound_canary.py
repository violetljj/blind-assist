#!/usr/bin/env python3
"""Stable adapter for the USTRF-SC ARCore frame-bound canary validator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


IMPLEMENTATION = Path(__file__).resolve().parent / "research" / "ustrf_sc" / "validate_ustrf_sc_arcore_frame_bound_canary.py"


def load_implementation():
    spec = importlib.util.spec_from_file_location("ustrf_sc_arcore_frame_bound_validator", IMPLEMENTATION)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw ARCore single-Frame binding evidence fail closed.")
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--device-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    inputs = {args.raw.resolve(), args.summary.resolve(), args.device_receipt.resolve()}
    if output in inputs:
        print(json.dumps({"ok": False, "error": "output must not overwrite an input"}, ensure_ascii=False))
        return 2

    implementation = load_implementation()
    try:
        report = implementation.validate(args.raw, args.summary, args.device_receipt)
    except (implementation.ContractError, OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        failure = {
            "schema": implementation.AUDIT_SCHEMA,
            "ok": False,
            "gate_open": False,
            "verdict": "FREEZE_FRAME_BOUND_METRIC_GEOMETRY",
            "error": str(error),
            "evidence_boundary": {
                "benchmark_only": True,
                "app_runtime_involved": False,
                "navigation_output_issued": False,
                "training_authority": False,
                "production_authorized": False,
                "human_truth": False,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False))
        return 2

    report["ok"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "gate_open": report["gate_open"],
        "verdict": report["verdict"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0 if report["gate_open"] else 2


if __name__ == "__main__":
    sys.exit(main())
