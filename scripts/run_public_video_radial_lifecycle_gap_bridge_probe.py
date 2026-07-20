#!/usr/bin/env python3
"""Retrospectively sweep post-entry evidence-gap persistence after the r7.29 false clear."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import evaluate_public_video_marker_radial_lifecycle_positive as positive
import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_chromatic_marker_policy as chromatic
import public_video_tristate_contract as tristate_contract
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_radial_lifecycle_gap_bridge_probe_v1"


def radial_entry_lifecycle(
    samples: Sequence[dict[str, Any]],
    policy: dict[str, Any],
    candidate_events: Sequence[dict[str, Any]],
    *,
    clear_absent_samples: int,
) -> dict[str, Any]:
    if clear_absent_samples <= 0:
        raise ValueError("clear_absent_samples must be positive")
    active_samples = chromatic.apply_policy(samples, policy)
    candidate_entries = {
        int(event["event_entry_timestamp_ms"])
        for event in candidate_events
        if event.get("radial_approach_passed") is True
    }
    state = "clear"
    entry_ms: int | None = None
    last_active_ms: int | None = None
    first_absent_ms: int | None = None
    absent_run = 0
    intervals: list[dict[str, Any]] = []
    reminder_timestamps: list[int] = []

    for sample in sorted(active_samples, key=lambda row: int(row["timestamp_ms"])):
        timestamp_ms = int(sample["timestamp_ms"])
        active = bool(sample.get("semantic_group_counts", {}).get("barrier_structure", 0))
        radial_entry = timestamp_ms in candidate_entries
        if state == "clear":
            if radial_entry:
                state = "present"
                entry_ms = timestamp_ms
                last_active_ms = timestamp_ms if active else None
                first_absent_ms = None
                absent_run = 0
                reminder_timestamps.append(timestamp_ms)
            continue
        if active:
            state = "present"
            last_active_ms = timestamp_ms
            first_absent_ms = None
            absent_run = 0
        else:
            if state == "present":
                state = "uncertain"
                first_absent_ms = timestamp_ms
                absent_run = 1
            else:
                absent_run += 1
            if absent_run >= clear_absent_samples:
                assert entry_ms is not None and last_active_ms is not None and first_absent_ms is not None
                intervals.append({
                    "event_entry_timestamp_ms": entry_ms,
                    "last_active_timestamp_ms": last_active_ms,
                    "first_absent_timestamp_ms": first_absent_ms,
                    "confirmed_clear_timestamp_ms": timestamp_ms,
                    "clear_absent_sample_count": absent_run,
                })
                state = "clear"
                entry_ms = None
                last_active_ms = None
                first_absent_ms = None
                absent_run = 0
    open_event = None
    if state != "clear":
        open_event = {
            "event_entry_timestamp_ms": entry_ms,
            "last_active_timestamp_ms": last_active_ms,
            "first_absent_timestamp_ms": first_absent_ms,
            "terminal_state": state,
        }
    return {
        "intervals": intervals,
        "open_event": open_event,
        "terminal_state": state,
        "reminder_timestamps_ms": reminder_timestamps,
    }


def score_sweep_row(state: dict[str, Any], review_item: dict[str, Any]) -> dict[str, Any]:
    risk = list(map(int, review_item["visual_risk_present_window_ms"]))
    stable_clear = list(map(int, review_item["stable_route_clear_window_ms"]))
    intervals = [
        event for event in state["intervals"]
        if int(event["event_entry_timestamp_ms"]) <= risk[1]
        and int(event["confirmed_clear_timestamp_ms"]) >= risk[0]
    ]
    if len(intervals) != 1:
        return {
            "passed": False,
            "matching_interval_count": len(intervals),
            "reminder_count": len(state["reminder_timestamps_ms"]),
        }
    event = intervals[0]
    clear_ms = int(event["confirmed_clear_timestamp_ms"])
    entry_ms = int(event["event_entry_timestamp_ms"])
    checks = {
        "opens_in_time": risk[0] <= entry_ms <= int(review_item["latest_acceptable_open_timestamp_ms"]),
        "persists_through_visual_risk": clear_ms >= risk[1],
        "clears_inside_stable_route_clear_window": stable_clear[0] <= clear_ms <= stable_clear[1],
        "same_episode_reminder_once": len(state["reminder_timestamps_ms"]) == 1,
    }
    return {
        "event": event,
        "reminder_timestamps_ms": state["reminder_timestamps_ms"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    base_contract, base_meta = tristate_contract.load_contract(args.base_contract)
    features = lifecycle.verify_json_sidecar(args.features)
    candidates = lifecycle.verify_json_sidecar(args.candidates)
    review = lifecycle.verify_json_sidecar(args.review)
    if features.get("prospective_contract", {}).get("sha256") != base_meta["sha256"]:
        raise ValueError("feature report base contract mismatch")
    if candidates.get("feature_report_sha256") != common.sha256_file(args.features):
        raise ValueError("candidate feature report mismatch")
    positive_items = [
        item for item in review.get("sources", [])
        if item.get("evaluation_role") == "prospective_positive_lifecycle"
    ]
    if len(positive_items) != 1:
        raise ValueError("probe requires exactly one positive lifecycle review item")
    item = positive_items[0]
    source_id = item["source_id"]
    source_rows = [row for row in features["sources"] if row["source_id"] == source_id]
    candidate_rows = [row for row in candidates["sources"] if row["source_id"] == source_id]
    if len(source_rows) != 1 or len(candidate_rows) != 1:
        raise ValueError("positive source does not bind exactly one feature and candidate row")
    policy = chromatic.validate_policy(base_contract)
    sweep = []
    for threshold in range(args.minimum_clear_absent_samples, args.maximum_clear_absent_samples + 1):
        state = radial_entry_lifecycle(
            source_rows[0]["samples"],
            policy,
            candidate_rows[0]["events"],
            clear_absent_samples=threshold,
        )
        sweep.append({
            "clear_absent_samples": threshold,
            "state": state,
            "score": score_sweep_row(state, item),
        })
    passing = [row["clear_absent_samples"] for row in sweep if row["score"]["passed"]]
    selected = min(passing) if passing else None

    negative_reports = []
    negative_candidate_count = 0
    for path in args.prior_negative_candidates:
        value = lifecycle.verify_json_sidecar(path)
        count = int(value.get("summary", {}).get("candidate_event_count", -1))
        if count < 0:
            raise ValueError(f"negative candidate summary missing: {path}")
        negative_candidate_count += count
        negative_reports.append({"path": str(path.resolve()), "sha256": common.sha256_file(path), "candidate_event_count": count})

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "derivation_disclosure": (
            "The persistence sweep was opened after inspecting the frozen r7.29 Edmonton false-clear result. "
            "It is retrospective repair evidence only and cannot convert r7.29 into a prospective pass."
        ),
        "inputs": {
            "base_contract": base_meta,
            "feature_report_sha256": common.sha256_file(args.features),
            "candidate_report_sha256": common.sha256_file(args.candidates),
            "review_sha256": common.sha256_file(args.review),
            "prior_negative_candidate_reports": negative_reports,
        },
        "fixed_architecture": {
            "entry": "frozen r7.25 radial candidate only",
            "after_entry_active_evidence": "frozen r7.11 chromatic marker evidence resets absence run without a second reminder",
            "gap_state": "uncertain",
            "exit": "clear only after N consecutive absent one-second samples",
            "learned_parameters": 0,
        },
        "sweep": sweep,
        "selection": {
            "minimum_passing_clear_absent_samples": selected,
            "passing_values": passing,
            "selected_for_future_freeze": selected is not None and negative_candidate_count == 0,
        },
        "negative_stress": {
            "prior_report_count": len(negative_reports),
            "frozen_radial_candidate_event_count": negative_candidate_count,
            "no_new_negative_event_can_open": negative_candidate_count == 0,
            "limitation": "This only checks entry isolation; it does not measure overlong persistence after a true radial open.",
        },
        "authorizations": {
            "future_prospective_contract_freeze": selected is not None and negative_candidate_count == 0,
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
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--prior-negative-candidates", type=Path, nargs="*", default=[])
    parser.add_argument("--minimum-clear-absent-samples", type=int, default=5)
    parser.add_argument("--maximum-clear-absent-samples", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({"ok": True, **value["selection"], "output_sha256": common.sha256_file(parsed.output)}, ensure_ascii=False))
