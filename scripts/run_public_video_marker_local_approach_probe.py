#!/usr/bin/env python3
"""Retrospectively test local radial-approach windows inside chromatic events."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_chromatic_marker_policy as chromatic
import public_video_tristate_contract as tristate_contract
import run_public_silver_frozen_feature_probe as common
import run_public_video_marker_radial_approach_probe as radial


SCHEMA = "blindassist_public_video_marker_local_approach_probe_v1"


def selected_rows(samples: list[dict[str, Any]], event: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    start = int(event["event_entry_timestamp_ms"])
    end = int(event["last_active_timestamp_ms"])
    rows: list[dict[str, Any]] = []
    for sample in samples:
        timestamp = int(sample["timestamp_ms"])
        if not start <= timestamp <= end:
            continue
        accepted = radial.accepted_detections(sample, policy)
        if accepted:
            rows.append({
                "timestamp_ms": timestamp,
                "detection": max(accepted, key=lambda row: float(row["features"]["bottom_y_norm"])),
            })
    return rows


def local_windows(
    samples: list[dict[str, Any]],
    event: dict[str, Any],
    policy: dict[str, Any],
    width: int,
) -> list[dict[str, Any]]:
    rows = selected_rows(samples, event, policy)
    passed: list[dict[str, Any]] = []
    for index in range(max(0, len(rows) - width + 1)):
        window = rows[index:index + width]
        subevent = {
            "event_entry_timestamp_ms": window[0]["timestamp_ms"],
            "last_active_timestamp_ms": window[-1]["timestamp_ms"],
        }
        diagnostic = radial.event_diagnostics(samples, subevent, policy)
        if diagnostic["radial_approach_passed"]:
            passed.append(diagnostic)
    return passed


def diagnose(features: dict[str, Any], contract: dict[str, Any], widths: list[int]) -> list[dict[str, Any]]:
    policy = chromatic.validate_policy(contract)
    lc = contract["lifecycle"]
    results: list[dict[str, Any]] = []
    for source in features["sources"]:
        filtered = chromatic.apply_policy(source["samples"], policy)
        state = lifecycle.tristate_exit_intervals(
            filtered,
            lc["selected_groups"],
            entry_window_samples=int(lc["entry_window_samples"]),
            entry_min_active_samples=int(lc["entry_min_active_samples"]),
            clear_absent_samples=int(lc["clear_absent_samples"]),
        )
        events = list(state["intervals"])
        if state["open_event"]:
            events.append(state["open_event"])
        event_rows = []
        for event in events:
            whole = radial.event_diagnostics(source["samples"], event, policy)
            by_width = {}
            for width in widths:
                passes = local_windows(source["samples"], event, policy, width)
                by_width[str(width)] = {
                    "passing_window_count": len(passes),
                    "first_passing_window": passes[0] if passes else None,
                }
            event_rows.append({"whole_event": whole, "local_windows": by_width})
        results.append({"source_id": source["source_id"], "events": event_rows})
    return results


def parse_group(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("group must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("group label cannot be empty")
    return label, Path(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract, contract_meta = tristate_contract.load_contract(args.contract)
    widths = sorted(set(args.window_width))
    if not widths or any(width < radial.MIN_ACCEPTED_SAMPLES for width in widths):
        raise ValueError(f"window widths must be >= {radial.MIN_ACCEPTED_SAMPLES}")
    groups = {}
    inputs = {}
    for label, path in args.group:
        features = lifecycle.verify_json_sidecar(path)
        groups[label] = diagnose(features, contract, widths)
        inputs[label] = {"path": str(path.resolve()), "sha256": common.sha256_file(path)}
    summary = {}
    for label, sources in groups.items():
        summary[label] = {
            str(width): sum(
                1
                for source in sources
                for event in source["events"]
                if event["local_windows"][str(width)]["passing_window_count"] > 0
            )
            for width in widths
        }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "hypothesis": "Whole-event endpoints can hide a genuine local approach phase inside a long chromatic event.",
        "contract": contract_meta,
        "inputs": inputs,
        "fixed_search": {
            "window_widths_in_accepted_samples": widths,
            "within_window_gate": "exact r7.25 radial gate",
            "selection": "every sliding window; report counts without choosing a winner",
        },
        "groups": groups,
        "summary_event_counts_with_any_passing_window": summary,
        "authorizations": {
            "future_contract_freeze": False,
            "training": False,
            "calibration": False,
            "blind": False,
            "android_runtime_change": False,
            "production_model_replacement": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--group", action="append", type=parse_group, required=True)
    parser.add_argument("--window-width", action="append", type=int, default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "summary": value["summary_event_counts_with_any_passing_window"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
