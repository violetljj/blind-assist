from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0]); top = max(first[1], second[1])
    right = min(first[2], second[2]); bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


@dataclass
class Track:
    track_id: int
    box: list[float]
    confidence: float
    last_frame: int
    previous_box: list[float] | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    misses: int = 0
    route_hits: int = 0


@dataclass
class ArmState:
    next_track_id: int = 1
    tracks: list[Track] = field(default_factory=list)
    track_births: int = 0
    fragmented_track_count: int = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self.track_births = 0
        self.fragmented_track_count = 0


def predicted_box(track: Track, arm: str) -> list[float]:
    if arm == "T0":
        return track.box
    dx, dy = track.velocity
    if arm == "T3" and track.previous_box is not None:
        current = center(track.box)
        previous = center(track.previous_box)
        dx, dy = current[0] - previous[0], current[1] - previous[1]
    return [track.box[0] + dx, track.box[1] + dy, track.box[2] + dx, track.box[3] + dy]


def match_score(track: Track, detection: dict[str, Any], arm: str) -> tuple[float, float] | None:
    box = detection["box"]
    predicted = predicted_box(track, arm)
    overlap = iou(predicted, box)
    first_center = center(predicted); second_center = center(box)
    delta = math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
    normalized_delta = delta / max(1.0, math.hypot(predicted[2] - predicted[0], predicted[3] - predicted[1]))
    if track.last_frame < 0 or track.misses > 1:
        return None
    threshold = 0.16 if arm == "T3" else 0.12
    if overlap >= 0.25 or normalized_delta <= threshold:
        return (overlap, normalized_delta)
    return None


def associate(detections: list[dict[str, Any]], frame_index: int, arm: str, state: ArmState, config: dict) -> list[tuple[Track, dict[str, Any]]]:
    if frame_index <= 0 or (state.tracks and frame_index - max(t.last_frame for t in state.tracks) > 1):
        state.reset()
    indexed = list(enumerate(detections))
    if arm == "T2":
        high = [(index, detection) for index, detection in indexed if detection["confidence"] >= float(config["arms"][arm]["high_confidence_min"])]
        low = [(index, detection) for index, detection in indexed if detection["confidence"] < float(config["arms"][arm]["high_confidence_min"])]
        groups = (high, low)
    else:
        groups = (indexed,)
    matched: list[tuple[Track, dict[str, Any]]] = []
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    for group in groups:
        candidates = []
        for detection_index, detection in group:
            for track in state.tracks:
                if track.track_id in used_tracks:
                    continue
                score = match_score(track, detection, arm)
                if score is not None:
                    candidates.append((score[0], -score[1], track.track_id, detection_index, track, detection))
        for _, _, _, detection_index, track, detection in sorted(candidates, reverse=True):
            if track.track_id in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track.track_id)
            used_detections.add(detection_index)
            matched.append((track, detection))
    for track in state.tracks:
        if track.track_id not in used_tracks:
            track.misses += 1
    for track, detection in matched:
        old_center = center(track.box)
        new_box = detection["box"]
        new_center = center(new_box)
        vx = new_center[0] - old_center[0]
        vy = new_center[1] - old_center[1]
        if arm == "T1":
            alpha = float(config["arms"][arm]["alpha"])
            beta = float(config["arms"][arm]["beta"])
            track.velocity = (alpha * vx + (1.0 - alpha) * track.velocity[0] + beta * vx, alpha * vy + (1.0 - alpha) * track.velocity[1] + beta * vy)
        else:
            track.velocity = (vx, vy)
        track.previous_box = track.box
        track.box = new_box
        track.confidence = detection["confidence"]
        track.last_frame = frame_index
        track.misses = 0
    unmatched = [
        detection
        for index, detection in indexed
        if index not in used_detections
        and (arm != "T2" or detection["confidence"] >= float(config["arms"][arm]["high_confidence_min"]))
    ]
    for detection in unmatched:
        track = Track(state.next_track_id, detection["box"], detection["confidence"], frame_index)
        state.next_track_id += 1
        state.track_births += 1
        state.tracks.append(track)
        matched.append((track, detection))
    before = len(state.tracks)
    state.tracks = [track for track in state.tracks if track.misses <= 1]
    state.fragmented_track_count += max(0, before - len(state.tracks))
    return matched


def route_hit(box: list[float], route: dict[str, Any], width: int, height: int, margin_fraction: float) -> bool:
    if route.get("status") != "known" or not route.get("uv"):
        return False
    u, v = route["uv"]
    left, top, right, bottom = box
    margin = min(width, height) * margin_fraction
    return left - margin <= u <= right + margin and top - margin <= v <= bottom + margin


def evaluate_window(window: dict, source: dict, ledger: dict, candidate: dict, arm: str, config: dict) -> dict:
    frames = {row["frame_id"]: row for row in ledger["frames"]}
    route = {row["frame_id"]: row for row in candidate["route_predictions"]}
    state = ArmState()
    alerts: list[dict[str, Any]] = []
    active_start: int | None = None
    clear_run = 0
    risk_run = 0
    route_hits = 0
    frame_ids = [f"{index:06d}" for index in range(window["start_frame"], window["end_frame"] + 1)]
    width, height = 640, 480
    for frame_index, frame_id in enumerate(frame_ids):
        frame = frames.get(frame_id)
        if frame is None:
            raise ValueError(f"ledger missing frame {frame_id}")
        matched = associate(frame["detections"], int(frame_id), arm, state, config)
        hit = any(route_hit(track.box, route[frame_id], width, height, float(config["route_event"]["route_point_margin_fraction"])) for track, _ in matched)
        if hit:
            route_hits += 1
            clear_run = 0
            risk_run += 1
            if active_start is None and risk_run >= int(config["route_event"]["min_alert_frames"]):
                active_start = int(frame_id) - int(config["route_event"]["min_alert_frames"]) + 1
        else:
            risk_run = 0
            clear_run += 1
            if active_start is not None and clear_run >= int(config["route_event"]["min_clear_frames"]):
                end = int(frame_id) - int(config["route_event"]["min_clear_frames"])
                alerts.append({"start_frame": active_start, "end_frame": end})
                active_start = None
                clear_run = 0
    if active_start is not None:
        alerts.append({"start_frame": active_start, "end_frame": int(frame_ids[-1])})
    truth = window.get("truth_anchors")
    matched_event = False
    if window["window_type"] == "positive" and truth:
        matched_event = any(alert["start_frame"] <= truth["passed_or_cleared_frame"] and alert["end_frame"] >= truth["alertable_frame"] for alert in alerts)
    duration_s = max(1e-9, (int(frame_ids[-1]) - int(frame_ids[0]) + 1) / 15.0)
    return {
        "window_id": window["window_id"],
        "source_id": window["source_id"],
        "window_type": window["window_type"],
        "critical": bool(window.get("critical", False)),
        "alerts": alerts,
        "route_hit_frames": route_hits,
        "matched_event": matched_event,
        "track_births": state.track_births,
        "track_fragmentations": state.fragmented_track_count,
        "duration_s": duration_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--dynamics-ledger", type=Path, required=True)
    parser.add_argument("--lt-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    config = read_json(args.config)
    windows_payload = read_json(args.windows)
    if sha256(args.config) != windows_payload["config_sha256"]:
        raise ValueError("window/config hash mismatch")
    rows_by_source: dict[str, dict] = {}
    source_names = {"lilocbench_dynamics_0_front": "dynamics_0", "lilocbench_lt_changes_dynamics_0_front": "lt_changes_dynamics_0"}
    for source_id, source_name in source_names.items():
        ledger_path = args.dynamics_ledger if source_name == "dynamics_0" else args.lt_ledger
        candidate_path = Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1") / ("dynamics_0-candidate-evaluation-frozen-v1.json" if source_name == "dynamics_0" else "lt_changes_dynamics_0-candidate-v1.json")
        source = config["inputs"][source_name]
        ledger = read_json(ledger_path)
        if (
            ledger.get("schema") != "blindassist_ustrf_tracker_ttc_detector_ledger_v1"
            or ledger.get("config_sha256") != sha256(args.config)
            or ledger.get("windows_sha256") != sha256(args.windows)
            or ledger.get("source_id") != source_id
            or ledger.get("event_truth_visible") is not False
            or ledger.get("candidate_alerts_visible") is not False
            or sha256(candidate_path) != source["candidate_sha256"]
        ):
            raise ValueError(f"input hash mismatch: {source_name}")
        rows_by_source[source_id] = {"ledger": ledger, "candidate": next(row for row in read_json(candidate_path)["sources"] if row["source_id"] == source_id), "ledger_sha256": sha256(ledger_path)}
    detector_person_box_count = sum(
        len(row["detections"])
        for source in rows_by_source.values()
        for row in source["ledger"]["frames"]
    )
    arms_to_run = ("T0",) if detector_person_box_count == 0 else ("T0", "T1", "T2", "T3")
    results = {}
    for arm in arms_to_run:
        windows = []
        for window in windows_payload["windows"]:
            source = rows_by_source[window["source_id"]]
            windows.append(evaluate_window(window, source, source["ledger"], source["candidate"], arm, config))
        positives = [row for row in windows if row["window_type"] == "positive"]
        negatives = [row for row in windows if row["window_type"] == "negative"]
        matched = sum(row["matched_event"] for row in positives)
        critical = sum(row["critical"] for row in positives)
        false_alerts = sum(len(row["alerts"]) for row in negatives)
        negative_minutes = sum(row["duration_s"] for row in negatives) / 60.0
        source_metrics = {}
        for source_id in source_names:
            source_windows = [row for row in windows if row["source_id"] == source_id]
            source_positive = [row for row in source_windows if row["window_type"] == "positive"]
            source_negative = [row for row in source_windows if row["window_type"] == "negative"]
            source_critical = [row for row in source_positive if row["critical"]]
            source_negative_minutes = sum(row["duration_s"] for row in source_negative) / 60.0
            source_metrics[source_id] = {
                "event_recall": sum(row["matched_event"] for row in source_positive) / len(source_positive) if source_positive else None,
                "critical_miss_rate": 1.0 - sum(row["matched_event"] for row in source_critical) / len(source_critical) if source_critical else None,
                "false_alerts_per_minute": sum(len(row["alerts"]) for row in source_negative) / source_negative_minutes if source_negative_minutes else None,
                "clearance_rate": None,
                "clearance_p95_ms": None,
                "positive_window_count": len(source_positive),
                "negative_window_count": len(source_negative),
            }
        results[arm] = {
            "windows": windows,
            "event_recall": matched / len(positives) if positives else None,
            "critical_miss_rate": (critical - sum(row["matched_event"] for row in positives if row["critical"])) / critical if critical else None,
            "false_alerts_per_minute": false_alerts / negative_minutes if negative_minutes else None,
            "clearance_rate": None,
            "clearance_p95_ms": None,
            "identity_switch_rate": None,
            "id_truth_status": "not_evaluable_no_object_track_truth",
            "track_fragmentations": sum(row["track_fragmentations"] for row in windows),
            "detector_person_box_count": detector_person_box_count,
            "source_metrics": source_metrics,
            "worst_sources": {
                "event_recall": min(source_metrics.items(), key=lambda item: item[1]["event_recall"])[0],
                "critical_miss_rate": max(source_metrics.items(), key=lambda item: item[1]["critical_miss_rate"])[0],
                "false_alerts_per_minute": max(source_metrics.items(), key=lambda item: item[1]["false_alerts_per_minute"])[0],
                "clearance_rate": None,
                "clearance_p95_ms": None,
            },
        }
    payload = {
        "schema": "blindassist_ustrf_tracker_ttc_ablation_result_v1",
        "authority": "benchmark_only_research_ablation",
        "config_sha256": sha256(args.config),
        "windows_sha256": sha256(args.windows),
        "source_ledgers": {source_name: rows_by_source[source_id]["ledger_sha256"] for source_id, source_name in source_names.items()},
        "r3_evaluator_ran": False,
        "production_authority": False,
        "results": results,
        "not_run_arms": {
            arm: "frozen detector produced zero person boxes; association arm cannot alter detector recall"
            for arm in ("T1", "T2", "T3")
            if arm not in arms_to_run
        },
        "stop_decision": "STOP_T1_T3_NO_PERSON_DETECTIONS_AND_ID_TRUTH_UNAVAILABLE" if detector_person_box_count == 0 else "REVIEW_REQUIRED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arms": list(results), "stop_decision": payload["stop_decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
