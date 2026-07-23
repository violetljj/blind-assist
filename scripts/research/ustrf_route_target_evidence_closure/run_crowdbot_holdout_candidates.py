#!/usr/bin/env python3
"""One-shot full-sequence C1-C3 replay and source-wise sealed-holdout scoring."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

TRACKER_DIR = Path(__file__).resolve().parents[1] / "ustrf_tracker_ttc_ablation"
sys.path.insert(0, str(TRACKER_DIR))
from run_ablation import ArmState, associate, iou, route_hit  # noqa: E402

from candidates import (  # noqa: E402
    C1PerPersonRelationFSM,
    C2RouteOccupancyEpisodeFSM,
    C3DualKeyClearanceFSM,
    relation_observation,
)
from contract import load_json, sha256_file, validate_prereg  # noqa: E402


ACTIVE_ROLES = {"route_intersecting", "approaching_route"}
CANDIDATE_CLASSES = {
    "C1_CAUSAL_ROUTE_RELATION_FSM": C1PerPersonRelationFSM,
    "C2_ROUTE_OCCUPANCY_EPISODE_FSM": C2RouteOccupancyEpisodeFSM,
    "C3_DUAL_KEY_CLEARANCE_FSM": C3DualKeyClearanceFSM,
}


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q, method="higher")) if values else None


def candidate_active(state: Any) -> bool:
    active = getattr(state, "active", False)
    return bool(active)


def truth_matches(
    observed: dict[int, list[float]],
    persons: list[dict[str, Any]],
    minimum_iou: float,
) -> dict[int, dict[str, Any]]:
    pairs = sorted(
        (
            iou(box, person["bbox_xyxy"]),
            track_id,
            person_index,
        )
        for track_id, box in observed.items()
        for person_index, person in enumerate(persons)
    )
    used_tracks: set[int] = set()
    used_persons: set[int] = set()
    result = {}
    for overlap, track_id, person_index in reversed(pairs):
        if overlap < minimum_iou:
            break
        if track_id in used_tracks or person_index in used_persons:
            continue
        used_tracks.add(track_id)
        used_persons.add(person_index)
        result[track_id] = persons[person_index]
    return result


def intervals_by_sequence(truth: dict[str, Any]) -> dict[tuple[str, str], list[tuple[int, int, str]]]:
    rows: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    for event in truth["accepted_events"]:
        rows[(event["source_id"], event["sequence_id"])].append(
            (int(event["window_start_frame"]), int(event["window_end_frame"]), "positive")
        )
    for window in truth["matched_negative_windows"]:
        rows[(window["source_id"], window["sequence_id"])].append(
            (int(window["start_frame"]), int(window["end_frame"]), "negative")
        )
    return rows


def window_types(intervals: list[tuple[int, int, str]], frame_number: int) -> set[str]:
    return {kind for start, end, kind in intervals if start <= frame_number <= end}


def run_sequence(
    candidate_id: str,
    detector_frames: list[dict[str, Any]],
    route_frames: list[dict[str, Any]],
    truth_frames: list[dict[str, Any]],
    tracker_config: dict[str, Any],
    config: dict[str, Any],
    intervals: list[tuple[int, int, str]],
) -> dict[str, Any]:
    if not (len(detector_frames) == len(route_frames) == len(truth_frames)):
        raise RuntimeError("candidate sequence input coverage mismatch")
    fsm = CANDIDATE_CLASSES[candidate_id](
        int(config["frozen_axes"]["min_alert_frames"]),
        int(config["frozen_axes"]["min_clear_frames"]),
    )
    tracker = ArmState()
    histories: dict[int, deque[tuple[int, float]]] = defaultdict(lambda: deque(maxlen=3))
    deliveries = []
    closures = []
    evidence_ages = []
    unknown_active_frames = 0
    trace = []
    for detector_frame, route, truth_frame in zip(detector_frames, route_frames, truth_frames, strict=True):
        identity = (detector_frame["frame_id"], detector_frame["source_capture_timestamp_ns"])
        if identity != (route["frame_id"], route["source_capture_timestamp_ns"]) or identity != (
            truth_frame["frame_id"], truth_frame["source_capture_timestamp_ns"]
        ):
            raise RuntimeError(f"candidate frame identity mismatch: {identity}")
        frame_number = int(detector_frame["frame_id"])
        detections = detector_frame["person_detections"]
        observed_pairs = associate(detections, frame_number, "T0", tracker, tracker_config)
        observed_boxes = {track.track_id: track.box for track, _ in observed_pairs}
        width, height = detector_frame["source_size"]
        route_known = route.get("status") == "known" and route.get("uv") is not None
        relations = {}
        for track, _ in observed_pairs:
            relations[track.track_id] = relation_observation(
                track_id=track.track_id,
                frame_number=frame_number,
                box=track.box,
                route=route,
                width=int(width),
                height=int(height),
                route_intersects=route_hit(
                    track.box,
                    route,
                    int(width),
                    int(height),
                    float(config["frozen_axes"]["route_point_margin_fraction"]),
                ),
                histories=histories,
            )
        output = fsm.update(frame_number, route_known, relations)
        if route_known and isinstance(route.get("pose_age_ms"), (int, float)):
            evidence_ages.append(float(route["pose_age_ms"]))
        if not route_known and candidate_active(fsm):
            unknown_active_frames += 1
        matches = truth_matches(
            observed_boxes,
            truth_frame["persons"],
            float(config["frozen_axes"]["target_match_iou"]),
        )
        unresolved_people = [
            *truth_frame.get("presence_only_role_unknown", []),
            *truth_frame.get("quarantined_visual_candidates", []),
        ]
        active_track_ids = {track_id for track_id, role in relations.items() if role in ACTIVE_ROLES}
        for delivery_key in output["deliveries"]:
            attributed_tracks = {int(delivery_key)} if candidate_id.startswith("C1_") else active_track_ids
            matched_people = [matches[track_id] for track_id in attributed_tracks if track_id in matches]
            unknown_person_track_ids = sorted(
                track_id
                for track_id in attributed_tracks
                if track_id in observed_boxes
                and any(
                    iou(observed_boxes[track_id], person["bbox_xyxy"])
                    >= float(config["frozen_axes"]["target_match_iou"])
                    for person in unresolved_people
                )
            )
            event_ids = sorted({
                person["event_id"]
                for person in matched_people
                if person.get("event_id") is not None and person.get("role") in ACTIVE_ROLES
            })
            deliveries.append({
                "frame": frame_number,
                "delivery_key": delivery_key,
                "truth_event_ids": event_ids,
                "matched_person_ids": sorted(person["person_id"] for person in matched_people),
                "unknown_person_overlap": bool(unknown_person_track_ids),
                "unknown_person_track_ids": unknown_person_track_ids,
                "route_known": route_known,
                "window_types": sorted(window_types(intervals, frame_number)),
            })
        for closure_key in output["closures"]:
            closures.append({"frame": frame_number, "closure_key": closure_key})
        trace.append({
            "frame": frame_number,
            "route_known": route_known,
            "observed_track_ids": sorted(observed_boxes),
            "active_relation_track_ids": sorted(active_track_ids),
            "deliveries": output["deliveries"],
            "closures": output["closures"],
            "candidate_active": candidate_active(fsm),
        })
    return {
        "frame_count": len(trace),
        "trace_sha256": __import__("hashlib").sha256(
            json.dumps(trace, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "deliveries": deliveries,
        "closures": closures,
        "evidence_ages_ms": evidence_ages,
        "active_alert_on_unknown_route_frame_count": unknown_active_frames,
    }


def score_source(
    source_id: str,
    truth: dict[str, Any],
    sequence_runs: dict[str, dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    events = [row for row in truth["accepted_events"] if row["source_id"] == source_id]
    source_truth = next(row for row in truth["sources"] if row["source_id"] == source_id)
    deliveries = [
        {"sequence_id": sequence_id, **delivery}
        for sequence_id, run in sequence_runs.items()
        for delivery in run["deliveries"]
    ]
    closures = [
        {"sequence_id": sequence_id, **closure}
        for sequence_id, run in sequence_runs.items()
        for closure in run["closures"]
    ]
    recalled = 0
    critical_miss = 0
    repeat = 0
    regeneration = 0
    clearance_successes = 0
    clearance_delays = []
    event_rows = []
    for event in events:
        correct = [
            delivery for delivery in deliveries
            if delivery["sequence_id"] == event["sequence_id"]
            and event["event_id"] in delivery["truth_event_ids"]
            and int(event["alertable_start_frame"]) <= delivery["frame"] <= int(event["clear_frame"])
        ]
        recalled += int(bool(correct))
        critical_miss += int(bool(event["critical"]) and not correct)
        repeat += max(0, len(correct) - 1)
        regeneration += max(0, len(correct) - 1)
        clearance_delay = None
        if correct:
            delivery = correct[0]
            matching_closures = [
                closure for closure in closures
                if closure["sequence_id"] == event["sequence_id"]
                and closure["closure_key"] == delivery["delivery_key"]
                and closure["frame"] >= delivery["frame"]
            ]
            if matching_closures:
                closure = matching_closures[0]
                if int(event["clear_frame"]) <= closure["frame"] <= int(event["window_end_frame"]):
                    clearance_delay = (
                        closure["frame"] - int(event["clear_frame"])
                    ) * (1000.0 / 15.0)
                    clearance_successes += 1
                    clearance_delays.append(clearance_delay)
        event_rows.append({
            "event_id": event["event_id"],
            "recalled": bool(correct),
            "correct_delivery_count": len(correct),
            "clearance_success": clearance_delay is not None,
            "clearance_delay_ms": clearance_delay,
        })
    false_alerts = sum(
        not delivery["truth_event_ids"]
        and delivery["route_known"]
        and not delivery["unknown_person_overlap"]
        for delivery in deliveries
    )
    unknown_person_active_alert_count = sum(delivery["unknown_person_overlap"] for delivery in deliveries)
    negative_minutes = float(source_truth["scorable_negative_exposure_minutes"])
    evidence = [age for run in sequence_runs.values() for age in run["evidence_ages_ms"]]
    metrics = {
        "event_recall": recalled / len(events) if events else None,
        "critical_miss": critical_miss,
        "false_alerts_per_minute": false_alerts / negative_minutes if negative_minutes > 0 else None,
        "clearance_rate": clearance_successes / len(events) if events else None,
        "clearance_p95_ms": percentile(clearance_delays, 95),
        "repeat_alert_count": repeat,
        "event_regeneration_count": regeneration,
        "evidence_age_p95_ms": percentile(evidence, 95),
        "active_alert_on_unknown_route_frames": sum(
            run["active_alert_on_unknown_route_frame_count"] for run in sequence_runs.values()
        ),
        "unknown_person_active_alert_count": unknown_person_active_alert_count,
    }
    checks = {
        "event_recall": metrics["event_recall"] is not None and metrics["event_recall"] >= gate["event_recall_min"],
        "critical_miss": metrics["critical_miss"] <= gate["critical_miss_max"],
        "false_alerts_per_minute": metrics["false_alerts_per_minute"] is not None and metrics["false_alerts_per_minute"] <= gate["false_alerts_per_minute_max"],
        "clearance_rate": metrics["clearance_rate"] is not None and metrics["clearance_rate"] >= gate["clearance_rate_min"],
        "clearance_p95_ms": metrics["clearance_p95_ms"] is not None and metrics["clearance_p95_ms"] <= gate["clearance_p95_ms_max"],
        "repeat_alert_count": metrics["repeat_alert_count"] <= gate["repeat_alert_count_max"],
        "event_regeneration_count": metrics["event_regeneration_count"] <= gate["event_regeneration_count_max"],
        "evidence_age_p95_ms": metrics["evidence_age_p95_ms"] is not None and metrics["evidence_age_p95_ms"] <= gate["evidence_age_p95_ms_max"],
        "active_alert_on_unknown_route_frames": metrics["active_alert_on_unknown_route_frames"] <= gate["active_alert_on_unknown_route_frames_max"],
        "unknown_person_active_alert_count": metrics["unknown_person_active_alert_count"] == 0,
    }
    return {
        "source_id": source_id,
        "event_count": len(events),
        "critical_event_count": sum(bool(row["critical"]) for row in events),
        "false_alert_count": false_alerts,
        "negative_exposure_minutes": negative_minutes,
        "metrics": metrics,
        "gate_checks": checks,
        "gate_pass": all(checks.values()),
        "events": event_rows,
    }


def winner_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    metrics = [source["metrics"] for source in candidate["sources"]]
    return (
        sum(row["critical_miss"] for row in metrics),
        -min(row["event_recall"] for row in metrics),
        max(row["false_alerts_per_minute"] for row in metrics),
        -min(row["clearance_rate"] for row in metrics),
        max(row["clearance_p95_ms"] for row in metrics),
        sum(row["repeat_alert_count"] for row in metrics),
        max(row["evidence_age_p95_ms"] for row in metrics),
        candidate["candidate_id"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--route-ledger", required=True, type=Path)
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--truth-windows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replacement", type=Path)
    parser.add_argument("--scoring-amendment", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite holdout candidate report")
    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    execution = config["candidate_execution_contract"]
    replacement = load_json(args.replacement) if args.replacement else None
    scoring_amendment = load_json(args.scoring_amendment) if args.scoring_amendment else None
    if replacement:
        if scoring_amendment is None:
            raise RuntimeError("replacement candidate run requires frozen scoring amendment")
        if scoring_amendment["replacement_preregistration_sha256"] != sha256_file(args.replacement):
            raise RuntimeError("scoring amendment replacement binding mismatch")
        if scoring_amendment["candidate_implementation_sha256"] != sha256_file(
            Path(__file__).with_name("candidates.py")
        ):
            raise RuntimeError("scoring amendment candidate binding mismatch")
        if scoring_amendment["runner_implementation_sha256"] != sha256_file(Path(__file__)):
            raise RuntimeError("scoring amendment runner binding mismatch")
        if scoring_amendment["candidate_outputs_executed_before_freeze"] is not False:
            raise RuntimeError("scoring amendment was frozen after candidate execution")
    planned_candidate_output = (
        replacement["planned_outputs"]["candidate_result"]
        if replacement
        else execution["planned_candidate_report_path"]
    )
    if args.output.resolve() != (repo / planned_candidate_output).resolve():
        raise RuntimeError("candidate report path differs from preregistration")
    truth_contract = config["sealed_holdout"]["holdout_truth_freeze_contract"]["all_person_presence"]
    expected_inputs = (
        {
            args.detector.resolve(): (repo / replacement["planned_outputs"]["app_detector_ledger"]).resolve(),
            args.route_ledger.resolve(): (repo / replacement["planned_outputs"]["causal_route_ledger"]).resolve(),
            args.fusion.resolve(): (repo / replacement["planned_outputs"]["fusion"]).resolve(),
            args.truth_windows.resolve(): (repo / replacement["planned_outputs"]["truth_windows"]).resolve(),
        }
        if replacement
        else {
            args.detector.resolve(): (repo / execution["app_detector"]["planned_output_path"]).resolve(),
            args.route_ledger.resolve(): (repo / config["sealed_holdout"]["causal_route_input_contract"]["planned_output_path"]).resolve(),
            args.fusion.resolve(): (repo / truth_contract["planned_fusion_output_path"]).resolve(),
            args.truth_windows.resolve(): (repo / truth_contract["planned_truth_windows_path"]).resolve(),
        }
    )
    if any(actual != expected for actual, expected in expected_inputs.items()):
        raise RuntimeError("candidate input path differs from preregistration")
    detector = load_json(args.detector)
    route = load_json(args.route_ledger)
    fusion = load_json(args.fusion)
    truth = load_json(args.truth_windows)
    if truth.get("selection_authority") is not True or truth.get("admitted_source_count") != 2:
        raise RuntimeError("truth windows lack two-source selection authority")
    if detector.get("candidate_outputs_executed") is not False or route.get("candidate_outputs_executed") is not False:
        raise RuntimeError("candidate input contamination")
    if fusion.get("candidate_outputs_executed") is not False or truth.get("candidate_outputs_executed") is not False:
        raise RuntimeError("truth input contamination")
    if replacement:
        replacement_sha = sha256_file(args.replacement)
        for payload in (detector, route, fusion, truth):
            if payload.get("replacement_preregistration_sha256") != replacement_sha:
                raise RuntimeError("candidate replacement input binding mismatch")
    if detector.get("truth_windows_sha256") != sha256_file(args.truth_windows):
        raise RuntimeError("App detector ledger is not bound to the frozen truth windows")
    tracker_config = load_json(repo / execution["association"]["config_path"])
    detector_sequences = {
        (source["source_id"], sequence["sequence_id"]): sequence["frames"]
        for source in detector["sources"] for sequence in source["sequences"]
    }
    route_sequences = {
        (source["source_id"], sequence["sequence_id"]): sequence["route_predictions"]
        for source in route["sources"] for sequence in source["sequences"]
    }
    truth_sequences: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for frame in fusion["frames"]:
        truth_sequences[(frame["source_id"], frame["sequence_id"])].append(frame)
    for frames in truth_sequences.values():
        frames.sort(key=lambda row: int(row["frame_id"]))
    if set(detector_sequences) != set(route_sequences) or set(detector_sequences) != set(truth_sequences):
        raise RuntimeError("candidate detector/route/truth sequence inventories differ")
    intervals = intervals_by_sequence(truth)
    candidates = []
    for candidate in config["candidate_roster"]:
        candidate_id = candidate["id"]
        runs_by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for key in sorted(detector_sequences):
            source_id, sequence_id = key
            runs_by_source[source_id][sequence_id] = run_sequence(
                candidate_id,
                detector_sequences[key],
                route_sequences[key],
                truth_sequences[key],
                tracker_config,
                config,
                intervals.get(key, []),
            )
        sources = [
            score_source(source_id, truth, sequences, config["selection_gate_each_source"])
            for source_id, sequences in sorted(runs_by_source.items())
        ]
        candidates.append({
            "candidate_id": candidate_id,
            "source_gate_pass_count": sum(row["gate_pass"] for row in sources),
            "eligible": len(sources) == 2 and all(row["gate_pass"] for row in sources),
            "sources": sources,
            "sequence_receipts": {
                source_id: {
                    sequence_id: {
                        "frame_count": run["frame_count"],
                        "trace_sha256": run["trace_sha256"],
                        "deliveries": run["deliveries"],
                        "closures": run["closures"],
                        "active_alert_on_unknown_route_frame_count": run["active_alert_on_unknown_route_frame_count"],
                    }
                    for sequence_id, run in sequences.items()
                }
                for source_id, sequences in runs_by_source.items()
            },
        })
    eligible = [row for row in candidates if row["eligible"]]
    winner = min(eligible, key=winner_sort_key)["candidate_id"] if eligible else None
    payload = {
        "schema": "blindassist_crowdbot_holdout_candidate_selection_r1",
        "authority": "sealed_holdout_model_proxy_selection_not_android_or_production_authority",
        "candidate_outputs_executed": True,
        "candidate_outputs_executed_only_after_truth_windows_hash_frozen": True,
        "config_sha256": sha256_file(args.config),
        "replacement_preregistration_sha256": sha256_file(args.replacement) if args.replacement else None,
        "scoring_amendment_sha256": sha256_file(args.scoring_amendment) if args.scoring_amendment else None,
        "false_alert_numerator": "all_full_sequence_route_known_deliveries_without_truth_event_excluding_unknown_person_hard_fail",
        "detector_sha256": sha256_file(args.detector),
        "route_ledger_sha256": sha256_file(args.route_ledger),
        "fusion_sha256": sha256_file(args.fusion),
        "truth_windows_sha256": sha256_file(args.truth_windows),
        "candidate_implementation_sha256": sha256_file(Path(__file__).with_name("candidates.py")),
        "candidate_run_unit": "each_candidate_each_full_sequence_once_without_window_reset",
        "candidates": candidates,
        "winner_candidate_id": winner,
        "android_shadow_authorized": winner is not None,
        "candidate_h2_authority": False,
        "production_authority": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "HOLDOUT_WINNER_ELIGIBLE_FOR_ANDROID_SHADOW" if winner else "STOP_NO_ANDROID_SHADOW_KEEP_H2_CLOSED",
        "winner_candidate_id": winner,
        "output_sha256": sha256_file(args.output),
    }))
    return 0 if winner else 2


if __name__ == "__main__":
    raise SystemExit(main())
