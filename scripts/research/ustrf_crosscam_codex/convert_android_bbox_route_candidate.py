#!/usr/bin/env python3
"""Convert shipped Android USTRF bbox-route output into the proxy candidate schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from .contract import BUNDLE_SCHEMA, CANDIDATE_SCHEMA, CONTRACT_ID, load_json, sha256_file, write_json
except ImportError:  # Direct execution through scripts/run_research_tool.py.
    from contract import BUNDLE_SCHEMA, CANDIDATE_SCHEMA, CONTRACT_ID, load_json, sha256_file, write_json


RISK_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
ACTION = {"LOW": "slow", "MEDIUM": "slow", "HIGH": "detour", "CRITICAL": "stop"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    bundle = load_json(args.bundle_manifest)
    adapter = load_json(args.adapter_output)
    if bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("contract_id") != CONTRACT_ID:
        raise ValueError("bundle schema/contract mismatch")
    if adapter.get("schema") != "blindassist_ustrf_sc_u0_candidate_adapter_output_v1":
        raise ValueError("Android adapter output schema mismatch")
    if adapter.get("arm_id") != "detector_bbox_explicit_route":
        raise ValueError("Android output is not the route-gated detector arm")
    start_ms = int(bundle["window"]["start_ms"])
    active: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    def flush() -> None:
        if not active:
            return
        peak = max(active, key=lambda row: RISK_ORDER[row["level"]])
        events.append({
            "category": "other",
            "route_relation": "inside",
            "required_action": ACTION[peak["level"]],
            "ttc_band": "unknown",
            "start_ms": active[0]["time_ms"],
            "end_ms": active[-1]["time_ms"] + 500,
            "peak_ms": peak["time_ms"],
        })
        active.clear()

    for row in adapter.get("frames", []):
        level = row.get("decision", {}).get("stable_risk_level")
        if level not in RISK_ORDER:
            raise ValueError("Android output has invalid stable risk level")
        time_ms = int(row["video_pts_ms"]) - start_ms
        if level == "NONE":
            flush()
        else:
            active.append({"level": level, "time_ms": time_ms})
    flush()
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "contract_id": CONTRACT_ID,
        "candidate_id": "ustrf_android_detector_bbox_explicit_route_v1",
        "candidate_kind": "shipped_android_yolo_bbox_explicit_route_proxy",
        "bundle_manifest_sha256": sha256_file(args.bundle_manifest),
        "adapter_output_path": str(args.adapter_output.resolve()),
        "adapter_output_sha256": sha256_file(args.adapter_output),
        "events": events,
        "human_event_truth_present": False,
        "metric_geometry_present": False,
        "training_authorized": False,
        "u0_authority_granted": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    write_json(args.output, candidate)
    return candidate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        candidate = run(parse_args(argv))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "event_count": len(candidate["events"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
