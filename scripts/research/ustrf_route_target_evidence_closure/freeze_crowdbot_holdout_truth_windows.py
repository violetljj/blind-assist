#!/usr/bin/env python3
"""Freeze evaluable person-route events and matched negative windows before candidate execution."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from contract import load_json, sha256_file, validate_prereg


ACTIVE_ROLES = {"route_intersecting", "approaching_route"}


def consecutive_true_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replacement", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite frozen holdout truth windows")
    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    truth_contract = config["sealed_holdout"]["holdout_truth_freeze_contract"]
    presence_contract = truth_contract["all_person_presence"]
    replacement = load_json(args.replacement) if args.replacement else None
    planned_output = (
        replacement["planned_outputs"]["truth_windows"]
        if replacement
        else presence_contract["planned_truth_windows_path"]
    )
    if args.output.resolve() != (repo / planned_output).resolve():
        raise RuntimeError("truth-window output differs from preregistration")
    fusion = load_json(args.fusion)
    if fusion.get("candidate_outputs_executed") is not False or fusion.get("app_detector_or_event_outputs_exposed") is not False:
        raise RuntimeError("truth-window input is not candidate/App blind")
    if replacement and fusion.get("replacement_preregistration_sha256") != sha256_file(args.replacement):
        raise RuntimeError("truth-window replacement input binding mismatch")
    frames_by_sequence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for frame in fusion["frames"]:
        frames_by_sequence[(frame["source_id"], frame["sequence_id"])].append(frame)
    for frames in frames_by_sequence.values():
        frames.sort(key=lambda row: int(row["frame_id"]))
        if [int(row["frame_id"]) for row in frames] != list(range(len(frames))):
            raise RuntimeError("holdout sequence frame numbering is not contiguous")
    pre = int(truth_contract["window_freeze"]["positive_pre_context_frames"])
    post = int(truth_contract["window_freeze"]["positive_post_clear_context_frames"])
    accepted_events = []
    quarantined_events = []
    positive_ranges_by_sequence: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for event in fusion["event_proposals"]:
        key = (event["source_id"], event["sequence_id"])
        frames = frames_by_sequence[key]
        onset = int(event["onset_frame"])
        alertable = int(event["alertable_start_frame"])
        clear = event.get("clear_frame")
        reasons = []
        if clear is None:
            reasons.append("censored_without_terminal_clear")
            clear_index = len(frames) - 1
        else:
            clear_index = int(clear)
        start = onset - pre
        end = clear_index + post
        if start < 0 or end >= len(frames):
            reasons.append("insufficient_fixed_context")
        bounded_start = max(0, start)
        bounded_end = min(len(frames) - 1, end)
        if event.get("truth_origin") != "visible_metric_person_role_episode":
            reasons.append("event_not_derived_from_visible_metric_person")
        if event.get("identity_continuous") is not True:
            reasons.append("event_person_visibility_not_continuous")
        event_frames = frames[onset : clear_index + 1]
        if any(
            not any(
                (
                    event.get("published_track_id") is not None
                    and event["published_track_id"] in person.get("published_track_alias_ids", [person.get("published_track_id")])
                    or event.get("published_track_id") is None
                    and event.get("person_id") == person.get("person_id")
                )
                and person.get("event_id") == event["event_id"]
                for person in frame["persons"]
            )
            for frame in event_frames
        ):
            reasons.append("event_person_identity_or_role_not_continuously_confirmed")
        frozen = {
            **event,
            "window_start_frame": bounded_start,
            "window_end_frame": bounded_end,
            "window_start_timestamp_ns": frames[bounded_start]["source_capture_timestamp_ns"],
            "window_end_timestamp_ns": frames[bounded_end]["source_capture_timestamp_ns"],
        }
        if reasons:
            quarantined_events.append({**frozen, "quarantine_reasons": sorted(set(reasons))})
        else:
            accepted_events.append(frozen)
            positive_ranges_by_sequence[key].append((bounded_start, bounded_end))
    accepted_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in accepted_events:
        accepted_by_source[event["source_id"]].append(event)
    negative_windows = []
    negative_exposure_seconds: dict[str, float] = defaultdict(float)
    negative_flags_by_sequence: dict[tuple[str, str], list[bool]] = {}
    for key, frames in frames_by_sequence.items():
        source_id, sequence_id = key
        flags = []
        for frame in frames:
            active = any(person.get("role") in ACTIVE_ROLES for person in frame["persons"])
            flags.append(frame["route_relevant_person_truth_complete"] and not active)
        negative_flags_by_sequence[key] = flags
        for index in range(len(frames) - 1):
            if flags[index] and flags[index + 1]:
                delta = (frames[index + 1]["source_capture_timestamp_ns"] - frames[index]["source_capture_timestamp_ns"]) / 1e9
                if 0.0 < delta <= 1.0:
                    negative_exposure_seconds[source_id] += delta
    used_negative: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for event in sorted(accepted_events, key=lambda row: (row["source_id"], row["sequence_id"], row["onset_frame"])):
        key = (event["source_id"], event["sequence_id"])
        frames = frames_by_sequence[key]
        length = event["window_end_frame"] - event["window_start_frame"] + 1
        event_midpoint = (event["window_start_frame"] + event["window_end_frame"]) / 2.0
        candidates = []
        for run_start, run_end in consecutive_true_runs(negative_flags_by_sequence[key]):
            if run_end - run_start + 1 < length:
                continue
            for start in range(run_start, run_end - length + 2):
                interval = (start, start + length - 1)
                if any(overlap(interval, positive) for positive in positive_ranges_by_sequence[key]):
                    continue
                if any(overlap(interval, used) for used in used_negative[key]):
                    continue
                midpoint = (interval[0] + interval[1]) / 2.0
                candidates.append((abs(midpoint - event_midpoint), interval[0], interval))
        if not candidates:
            continue
        _, _, interval = min(candidates)
        used_negative[key].append(interval)
        negative_windows.append({
            "window_id": f"{event['event_id']}:matched-negative",
            "matched_event_id": event["event_id"],
            "source_id": event["source_id"],
            "sequence_id": event["sequence_id"],
            "start_frame": interval[0],
            "end_frame": interval[1],
            "start_timestamp_ns": frames[interval[0]]["source_capture_timestamp_ns"],
            "end_timestamp_ns": frames[interval[1]]["source_capture_timestamp_ns"],
        })
    source_rows = []
    selected_source_ids = (
        [row["source_id"] for row in replacement["replacement_sources"]]
        if replacement
        else load_json(repo / config["sealed_holdout"]["content_qualification_receipt"]["path"])[
            "selected_source_ids"
        ]
    )
    for source_id in selected_source_ids:
        events = accepted_by_source[source_id]
        matched = [row for row in negative_windows if row["source_id"] == source_id]
        positive_count = len(events)
        critical_count = sum(row["critical"] for row in events)
        exposure_minutes = negative_exposure_seconds[source_id] / 60.0
        gates = {
            "positive_events": positive_count >= config["sealed_holdout"]["minimum_positive_events_each_source"],
            "critical_events": critical_count >= config["sealed_holdout"]["minimum_critical_events_each_source"],
            "matched_negative_windows": len(matched) >= config["sealed_holdout"]["minimum_matched_negative_windows_each_source"],
            "negative_exposure_minutes": exposure_minutes >= config["sealed_holdout"]["minimum_scorable_negative_exposure_minutes_each_source"],
        }
        source_rows.append({
            "source_id": source_id,
            "accepted_positive_event_count": positive_count,
            "accepted_critical_event_count": critical_count,
            "matched_negative_window_count": len(matched),
            "scorable_negative_exposure_minutes": exposure_minutes,
            "admission_gates": gates,
            "admission_pass": all(gates.values()),
        })
    admitted_count = sum(row["admission_pass"] for row in source_rows)
    payload = {
        "schema": "blindassist_crowdbot_holdout_route_role_truth_windows_r1",
        "authority": "sealed_model_proxy_benchmark_truth_not_human_or_production_truth",
        "candidate_outputs_executed": False,
        "app_detector_or_event_outputs_exposed": False,
        "config_sha256": sha256_file(args.config),
        "replacement_preregistration_sha256": sha256_file(args.replacement) if args.replacement else None,
        "fusion_sha256": sha256_file(args.fusion),
        "truth_scoring_unit": "visible_metric_event_and_route_relevant_person_complete_negative_frame",
        "unrelated_unknown_person_invalidates_whole_positive_window": False,
        "candidate_alert_overlapping_unresolved_person": "hard_fail_unknown_person_active_alert_for_source",
        "candidate_run_unit": "each_candidate_runs_each_full_sequence_once_without_window_reset",
        "accepted_events": accepted_events,
        "quarantined_events": quarantined_events,
        "matched_negative_windows": negative_windows,
        "sources": source_rows,
        "admitted_source_count": admitted_count,
        "selection_authority": admitted_count == 2,
        "production_authority": False,
        "candidate_h2_authority": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "HOLDOUT_ADMITTED_FOR_ONE_SHOT_SELECTION" if admitted_count == 2 else "FAIL_CLOSED_HOLDOUT_NOT_ADMITTED",
        "admitted_source_count": admitted_count,
        "sources": source_rows,
        "output_sha256": sha256_file(args.output),
    }))
    return 0 if admitted_count == 2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
