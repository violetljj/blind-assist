#!/usr/bin/env python3
"""Causally replay Japan through radial+route entry and r7.30 lifecycle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_chromatic_marker_policy as chromatic
import public_video_radial_lifecycle_gap_bridge_contract as gap_contract
import public_video_tristate_contract as tristate_contract
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_explicit_ego_route_relation_probe as route_relation
import run_public_video_marker_radial_approach_probe as radial
import run_public_video_obstacle_aware_route_width_probe as route_width
import run_public_video_radial_lifecycle_gap_bridge_probe as gap


SCHEMA = "blindassist_public_video_japan_causal_lifecycle_replay_v1"
SOURCE_ID = "wikimedia_commons_japan_rural_riverside_walk_2025"


def first_supported_entry(rows: Sequence[dict[str, Any]], baseline: float) -> int | None:
    for row in sorted(rows, key=lambda value: int(value["timestamp_ms"])):
        if row["radial_prefix_passed"] and float(row["route_intrusion_score"]) > baseline:
            return int(row["timestamp_ms"])
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.gap_contract, args.chromatic_contract, args.features, args.r725_probe, args.review, args.model_dir, args.output]
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    frozen_gap, gap_meta = gap_contract.load_contract(args.gap_contract)
    base_contract, base_meta = tristate_contract.load_contract(args.chromatic_contract)
    policy = chromatic.validate_policy(base_contract)
    features = lifecycle.verify_json_sidecar(args.features)
    r725 = lifecycle.verify_json_sidecar(args.r725_probe)
    review_report = lifecycle.verify_json_sidecar(args.review)
    review = review_report.get("review", {})
    if review.get("source_id") != SOURCE_ID or review.get("accepted_for_prospective_evaluation") is not True:
        raise ValueError("Japan visual review is missing or not accepted")
    source_rows = [row for row in features.get("sources", []) if row.get("source_id") == SOURCE_ID]
    if len(source_rows) != 1:
        raise ValueError("Japan feature source is missing")
    source = source_rows[0]
    groups = r725.get("groups", {}).get("prospective_positive_japan", [])
    if len(groups) != 1 or groups[0].get("source_id") != SOURCE_ID:
        raise ValueError("Japan r7.25 group is missing")
    events = [event for event in groups[0].get("events", []) if event.get("radial_approach_passed") is True]
    if len(events) != 1:
        raise ValueError("Japan must have exactly one frozen radial event")
    frozen_event = events[0]
    event_start = int(frozen_event["event_entry_timestamp_ms"])
    event_end = int(frozen_event["last_active_timestamp_ms"])

    pre_clear = list(map(int, review["pre_risk_clear_window_ms"]))
    post_clear = list(map(int, review["stable_post_clear_window_ms"]))
    timeline_start = pre_clear[0]
    timeline_end = post_clear[1]
    samples = [row for row in source["samples"] if timeline_start <= int(row["timestamp_ms"]) <= timeline_end]
    timestamps = [int(row["timestamp_ms"]) for row in samples]
    frames = route_width.decode_at(Path(source["local_video_path"]), timestamps)
    teacher = route_width.FrozenWalkableTeacher(args.model_dir)
    maps = teacher.probability_maps(frames, batch_size=args.batch_size)
    timeline = []
    for walkable, sample in zip(maps, samples):
        timestamp = int(sample["timestamp_ms"])
        obstacle = route_width.obstacle_mask_from_detections(
            sample.get("detections", []),
            walkable.shape,
            safety_margin_object_heights=route_relation.SAFETY_MARGIN_OBJECT_HEIGHTS,
        )
        relation = route_relation.explicit_route_relation(walkable, obstacle)
        prefix = radial.event_diagnostics(
            source["samples"],
            {"event_entry_timestamp_ms": event_start, "last_active_timestamp_ms": min(timestamp, event_end)},
            policy,
        ) if event_start <= timestamp <= event_end else {"radial_approach_passed": False, "accepted_sample_count": 0}
        timeline.append({
            "timestamp_ms": timestamp,
            "route_intrusion_score": float(relation["route_intrusion_score"]),
            "radial_prefix_passed": bool(prefix["radial_approach_passed"]),
            "radial_prefix_accepted_sample_count": int(prefix["accepted_sample_count"]),
        })
    baseline_values = [row["route_intrusion_score"] for row in timeline if pre_clear[0] <= row["timestamp_ms"] < pre_clear[1]]
    if not baseline_values:
        raise ValueError("Japan pre-risk route baseline is empty")
    baseline = float(median(baseline_values))
    supported_entry = first_supported_entry(timeline, baseline)
    if supported_entry is None:
        raise ValueError("Japan never obtains causal radial and route support")
    state = gap.radial_entry_lifecycle(
        source["samples"],
        policy,
        [{"event_entry_timestamp_ms": supported_entry, "radial_approach_passed": True}],
        clear_absent_samples=int(frozen_gap["lifecycle"]["clear_absent_samples"]),
    )
    intervals = [row for row in state["intervals"] if int(row["event_entry_timestamp_ms"]) == supported_entry]
    if len(intervals) != 1:
        raise ValueError("Japan replay did not produce exactly one closed event")
    interval = intervals[0]
    risk = list(map(int, review["risk_present_window_ms"]))
    checks = {
        "opens_inside_reviewed_risk_window": risk[0] <= supported_entry <= risk[1],
        "persists_through_reviewed_risk_end": int(interval["confirmed_clear_timestamp_ms"]) >= risk[1],
        "clears_inside_stable_post_clear_window": post_clear[0] <= int(interval["confirmed_clear_timestamp_ms"]) <= post_clear[1],
        "same_episode_reminder_once": state["reminder_timestamps_ms"] == [supported_entry],
    }
    passed = all(checks.values())
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_causal_prefix_replay_under_frozen_r725_route_relation_and_r730_lifecycle",
        "inputs": {
            "gap_contract": gap_meta,
            "chromatic_contract": base_meta,
            "features_sha256": common.sha256_file(args.features),
            "r725_probe_sha256": common.sha256_file(args.r725_probe),
            "review_sha256": common.sha256_file(args.review),
            "segformer_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin"),
        },
        "causal_contract": {
            "trusted_clear_baseline_window_ms": pre_clear,
            "route_baseline": "median explicit relation over half-open pre-risk clear window",
            "radial_support": "r7.25 rule recomputed on every prefix; future samples are unavailable",
            "entry": "first prefix with radial pass and route score above trusted clear baseline",
            "lifecycle": "frozen r7.30 nine-absent bridge and one reminder",
            "learned_parameters": 0,
            "threshold_fitted": False,
        },
        "route_baseline_intrusion_score": baseline,
        "timeline": timeline,
        "causal_supported_entry_timestamp_ms": supported_entry,
        "reviewed_risk_window_ms": risk,
        "entry_lead_before_reviewed_risk_ms": max(0, risk[0] - supported_entry),
        "lifecycle_state": state,
        "matching_interval": interval,
        "checks": checks,
        "diagnostic_gate": {"passed": passed},
        "failure": None if passed else "Causal radial+route support opens before the frozen reviewed risk window.",
        "authorizations": {
            "five_prototype_bootstrap_short_runs": False,
            "future_prospective_contract_freeze": False,
            "training": False,
            "calibration": False,
            "blind": False,
            "android_runtime_change": False,
            "production_model_replacement": False,
        },
        "evidence_limit": "Retrospective GPT/VLM silver timing replay. An early alert is retained as a contract failure, not relabeled as correct after inspection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-contract", type=Path, required=True)
    parser.add_argument("--chromatic-contract", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--r725-probe", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "gate_passed": value["diagnostic_gate"]["passed"],
        "entry_timestamp_ms": value["causal_supported_entry_timestamp_ms"],
        "clear_timestamp_ms": value["matching_interval"]["confirmed_clear_timestamp_ms"],
        "output_sha256": common.sha256_file(parsed.output),
    }, ensure_ascii=False))
