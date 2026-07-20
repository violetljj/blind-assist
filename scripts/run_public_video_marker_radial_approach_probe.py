#!/usr/bin/env python3
"""Retrospectively diagnose radial approach for frozen chromatic marker events."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_chromatic_marker_policy as chromatic
import public_video_tristate_contract as prospective
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_marker_radial_approach_probe_v1"
MIN_ACCEPTED_SAMPLES = 5
ENDPOINT_SAMPLES = 3
MIN_BOTTOM_PROGRESS = 0.05


def accepted_detections(sample: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    targets = set(policy["target_classes"])
    return [
        row for row in sample.get("detections", [])
        if row.get("class_name") in targets
        and float(row["features"]["high_saturation_fraction"])
        > float(row["features"]["dark_fraction"])
    ]


def event_diagnostics(
    samples: Sequence[dict[str, Any]],
    event: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    start = int(event["event_entry_timestamp_ms"])
    end = int(event["last_active_timestamp_ms"])
    rows = []
    for sample in samples:
        timestamp = int(sample["timestamp_ms"])
        if not start <= timestamp <= end:
            continue
        detections = accepted_detections(sample, policy)
        if not detections:
            continue
        # Closest accepted marker per timestamp; no class, threshold or track search.
        selected = max(detections, key=lambda row: float(row["features"]["bottom_y_norm"]))
        features = selected["features"]
        rows.append({
            "timestamp_ms": timestamp,
            "bottom_y_norm": float(features["bottom_y_norm"]),
            "center_x_norm": float(features["center_x_norm"]),
            "area_ratio": float(features["area_ratio"]),
        })
    enough = len(rows) >= MIN_ACCEPTED_SAMPLES
    if not rows:
        return {"accepted_sample_count": 0, "radial_approach_passed": False}
    endpoint = min(ENDPOINT_SAMPLES, max(1, len(rows) // 2))
    early, late = rows[:endpoint], rows[-endpoint:]
    early_bottom = median(row["bottom_y_norm"] for row in early)
    late_bottom = median(row["bottom_y_norm"] for row in late)
    early_center = median(row["center_x_norm"] for row in early)
    late_center = median(row["center_x_norm"] for row in late)
    early_area = median(row["area_ratio"] for row in early)
    late_area = median(row["area_ratio"] for row in late)
    bottom_progress = late_bottom - early_bottom
    horizontal_shift = abs(late_center - early_center)
    log_area_growth = math.log(max(late_area, 1e-12) / max(early_area, 1e-12))
    checks = {
        "minimum_accepted_samples": enough,
        "bottom_progress_at_least_five_percent": bottom_progress >= MIN_BOTTOM_PROGRESS,
        "vertical_progress_exceeds_horizontal_sweep": bottom_progress > horizontal_shift,
        "positive_area_growth": log_area_growth > 0.0,
    }
    return {
        "event_entry_timestamp_ms": start,
        "last_active_timestamp_ms": end,
        "accepted_sample_count": len(rows),
        "endpoint_sample_count": endpoint,
        "early_bottom_y_median": early_bottom,
        "late_bottom_y_median": late_bottom,
        "bottom_progress": bottom_progress,
        "early_center_x_median": early_center,
        "late_center_x_median": late_center,
        "absolute_horizontal_shift": horizontal_shift,
        "early_area_median": early_area,
        "late_area_median": late_area,
        "log_area_growth": log_area_growth,
        "checks": checks,
        "radial_approach_passed": all(checks.values()),
    }


def diagnose(features: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    policy = chromatic.validate_policy(contract)
    lc = contract["lifecycle"]
    results = []
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
        results.append({
            "source_id": source["source_id"],
            "events": [event_diagnostics(source["samples"], event, policy) for event in events],
        })
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract, contract_meta = prospective.load_contract(args.contract)
    positive = lifecycle.verify_json_sidecar(args.positive_features)
    negative = lifecycle.verify_json_sidecar(args.negative_features)
    derivation = lifecycle.verify_json_sidecar(args.derivation_features)
    groups = {
        "prospective_positive_japan": diagnose(positive, contract),
        "prospective_negative_matoaka": diagnose(negative, contract),
        "historical_vehicle_derivation": diagnose(derivation, contract),
    }
    japan_pass = any(event["radial_approach_passed"] for row in groups["prospective_positive_japan"] for event in row["events"])
    matoaka_pass = any(event["radial_approach_passed"] for row in groups["prospective_negative_matoaka"] for event in row["events"])
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "derivation_disclosure": "Rule selected after inspecting the Japan positive and Matoaka negative. Results are retrospective and cannot repair r7.23.",
        "contract": contract_meta,
        "inputs": {
            "positive_features_sha256": common.sha256_file(args.positive_features),
            "negative_features_sha256": common.sha256_file(args.negative_features),
            "derivation_features_sha256": common.sha256_file(args.derivation_features),
        },
        "fixed_rule": {
            "closest_detection_per_timestamp": "maximum bottom_y_norm",
            "minimum_accepted_samples": MIN_ACCEPTED_SAMPLES,
            "endpoint_samples": ENDPOINT_SAMPLES,
            "minimum_bottom_progress": MIN_BOTTOM_PROGRESS,
            "vertical_progress_must_exceed_horizontal_sweep": True,
            "positive_area_growth_required": True,
        },
        "groups": groups,
        "diagnostic_gate": {
            "japan_positive_retained": japan_pass,
            "matoaka_negative_rejected": not matoaka_pass,
            "passed": japan_pass and not matoaka_pass,
        },
        "authorizations": {
            "future_prospective_contract_freeze": japan_pass and not matoaka_pass,
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
    parser.add_argument("--positive-features", type=Path, required=True)
    parser.add_argument("--negative-features", type=Path, required=True)
    parser.add_argument("--derivation-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, **value["diagnostic_gate"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
