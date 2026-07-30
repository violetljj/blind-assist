from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any


BASELINE_BRANCH = (
    "CURRENT_FULL_PRODUCTION_TEMPORAL_GEOMETRY"
)
EXPECTED_FRAMES = 4422
MAXIMUM_GAP_NS = 500_000_000
HISTORY_FRAMES = 7
TRACK_SLOPE_THRESHOLD_PER_S = 0.2
SCENE_SCALE_THRESHOLD_PER_S = -0.05


@dataclass
class Track:
    epoch: int
    last: dict[str, Any]
    timestamp_ns: int
    history: list[tuple[int, float]] = field(default_factory=list)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    temporary.replace(path)


def detection_identity(detection: dict[str, Any]) -> tuple[Any, ...]:
    fields = (
        "class_id",
        "label",
        "confidence",
        "left",
        "top",
        "right",
        "bottom",
        "frame_width",
        "frame_height",
        "source",
        "temporal_promotion_eligible",
    )
    return tuple(detection[field] for field in fields)


def association_metrics(
    previous: dict[str, Any], current: dict[str, Any]
) -> tuple[float, float, float]:
    intersection = max(
        0.0, min(previous["right"], current["right"]) - max(previous["left"], current["left"])
    ) * max(
        0.0, min(previous["bottom"], current["bottom"]) - max(previous["top"], current["top"])
    )
    previous_area = (previous["right"] - previous["left"]) * (
        previous["bottom"] - previous["top"]
    )
    current_area = (current["right"] - current["left"]) * (
        current["bottom"] - current["top"]
    )
    union = previous_area + current_area - intersection
    iou = intersection / union if union > 0 else 0.0
    dx = (
        (current["left"] + current["right"] - previous["left"] - previous["right"])
        / 2
        / max(1, previous["frame_width"])
    )
    dy = (
        (current["top"] + current["bottom"] - previous["top"] - previous["bottom"])
        / 2
        / max(1, previous["frame_height"])
    )
    distance = math.hypot(dx, dy)
    return iou, distance, 2 * iou + max(0.0, 1.0 - distance)


def compatible(previous: Track, current: dict[str, Any], timestamp_ns: int) -> float | None:
    gap = timestamp_ns - previous.timestamp_ns
    if gap <= 0 or gap > MAXIMUM_GAP_NS:
        return None
    for field in ("class_id", "label", "source", "frame_width", "frame_height"):
        if previous.last[field] != current[field]:
            return None
    iou, distance, score = association_metrics(previous.last, current)
    return score if iou >= 0.25 or distance <= 0.12 else None


def associate(
    tracks: list[Track],
    detections: list[dict[str, Any]],
    timestamp_ns: int,
    next_epoch: int,
) -> tuple[list[Track], dict[int, Track], list[float], int]:
    live = [
        track
        for track in tracks
        if 0 < timestamp_ns - track.timestamp_ns <= MAXIMUM_GAP_NS
    ]
    pairs: list[tuple[float, int, int, int]] = []
    for track_index, track in enumerate(live):
        for detection_index, detection in enumerate(detections):
            score = compatible(track, detection, timestamp_ns)
            if score is not None:
                pairs.append((-score, track.epoch, detection_index, track_index))
    pairs.sort()
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: dict[int, Track] = {}
    scene_rates: list[float] = []
    for _, _, detection_index, track_index in pairs:
        if track_index in used_tracks or detection_index in used_detections:
            continue
        track = live[track_index]
        detection = detections[detection_index]
        previous_height = track.last["bottom"] - track.last["top"]
        current_height = detection["bottom"] - detection["top"]
        gap_s = (timestamp_ns - track.timestamp_ns) / 1_000_000_000
        if previous_height > 0 and current_height > 0:
            scene_rates.append(math.log(current_height / previous_height) / gap_s)
        used_tracks.add(track_index)
        used_detections.add(detection_index)
        matches[detection_index] = track
    for index, detection in enumerate(detections):
        track = matches.get(index)
        if track is None:
            track = Track(next_epoch, detection, timestamp_ns)
            next_epoch += 1
            live.append(track)
            matches[index] = track
        height = detection["bottom"] - detection["top"]
        if math.isfinite(height) and height > 0:
            track.history.append((timestamp_ns, math.log(height)))
            track.history = track.history[-HISTORY_FRAMES:]
        else:
            track.history.clear()
        track.last = detection
        track.timestamp_ns = timestamp_ns
    return live, matches, scene_rates, next_epoch


def track_decision(history: list[tuple[int, float]]) -> str:
    if len(history) != HISTORY_FRAMES:
        return "ABSTAIN"
    origin = history[0][0]
    times = [(timestamp - origin) / 1_000_000_000 for timestamp, _ in history]
    values = [value for _, value in history]
    mean_time = sum(times) / len(times)
    mean_value = sum(values) / len(values)
    denominator = sum((value - mean_time) ** 2 for value in times)
    if denominator <= 0:
        return "ABSTAIN"
    slope = sum(
        (time - mean_time) * (value - mean_value)
        for time, value in zip(times, values)
    ) / denominator
    deltas = [right - left for left, right in zip(values, values[1:])]
    if slope >= TRACK_SLOPE_THRESHOLD_PER_S and all(value > 0 for value in deltas):
        return "CONFIRM_APPROACH"
    if slope <= -TRACK_SLOPE_THRESHOLD_PER_S and all(value < 0 for value in deltas):
        return "CONTRADICT_APPROACH"
    return "ABSTAIN"


def box_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    return association_metrics(first, second)[0]


def simulate_feedback(
    ordered_keys: list[tuple[str, str]],
    baseline: dict[tuple[str, str], dict[str, Any]],
    timestamps: dict[tuple[str, str], int],
    vetoes: dict[tuple[str, str], bool] | None = None,
) -> dict[tuple[str, str], bool]:
    output: dict[tuple[str, str], bool] = {}
    session: str | None = None
    origin_ns = 0
    last_alert: dict[tuple[str, str], int] = {}
    fatigue_last_ms = 0
    fatigue_count = 0
    previous_low_confidence: dict[str, Any] | None = None
    for key in ordered_keys:
        row = baseline[key]
        timestamp_ns = timestamps[key]
        if key[0] != session:
            session = key[0]
            origin_ns = timestamp_ns
            last_alert.clear()
            fatigue_last_ms = 0
            fatigue_count = 0
            previous_low_confidence = None
        now_ms = (timestamp_ns - origin_ns) // 1_000_000
        risk = row["stable_risk"]
        detection = risk["source_detection"]
        requires_confirmation = (
            detection is not None
            and detection["source"] == "OBJECT_DETECTOR"
            and detection["label"] == "person"
            and detection["confidence"] < 0.5
            and risk["direction"] != "CENTER"
        )
        confirmed = True
        if requires_confirmation:
            confirmed = (
                previous_low_confidence is not None
                and previous_low_confidence["label"] == detection["label"]
                and previous_low_confidence["source"] == detection["source"]
                and (
                    box_iou(previous_low_confidence, detection) >= 0.25
                    or abs(
                        (
                            previous_low_confidence["left"]
                            + previous_low_confidence["right"]
                        )
                        / 2
                        / previous_low_confidence["frame_width"]
                        - (detection["left"] + detection["right"])
                        / 2
                        / detection["frame_width"]
                    )
                    <= 0.12
                )
            )
            previous_low_confidence = detection
        else:
            previous_low_confidence = None

        base_cooldown: int | None = None
        if risk["proximity"] == "CRITICAL" and risk["level"] == "HIGH":
            base_cooldown = 850
        elif risk["proximity"] == "NEAR" and risk["level"] in ("HIGH", "MEDIUM"):
            base_cooldown = 1500
        triggered = False
        if confirmed and base_cooldown is not None and not (vetoes or {}).get(key, False):
            alert_key = (risk["direction"], risk["proximity"])
            critical = base_cooldown == 850
            fatigue_level = (
                0
                if critical
                else fatigue_count if now_ms - fatigue_last_ms <= 12_000 else 0
            )
            multiplier = 2.0 if fatigue_level >= 4 else 1.5 if fatigue_level >= 2 else 1.0
            cooldown = base_cooldown if critical else int(base_cooldown * multiplier)
            previous_alert = last_alert.get(alert_key)
            if previous_alert is None or now_ms - previous_alert >= cooldown:
                triggered = True
                last_alert[alert_key] = now_ms
                if critical:
                    fatigue_count = 0
                    fatigue_last_ms = 0
                else:
                    fatigue_count = (
                        fatigue_count + 1
                        if now_ms - fatigue_last_ms <= 12_000
                        else 1
                    )
                    fatigue_last_ms = now_ms
        output[key] = triggered
    return output


def score_items(
    truth_items: list[dict[str, Any]],
    ordered_keys: list[tuple[str, str]],
    timestamps: dict[tuple[str, str], int],
    triggers: dict[tuple[str, str], bool],
) -> dict[str, Any]:
    origins: dict[str, int] = {}
    frame_index: dict[tuple[str, str], int] = {}
    for index, key in enumerate(ordered_keys):
        origins.setdefault(key[0], timestamps[key])
        frame_index[key] = index
    result: dict[str, Any] = {}
    for item in truth_items:
        if item["scoring_status"] != "SCORED":
            continue
        interval = (
            item["interval_ns"]
            if item["item_kind"] == "negative_window"
            else item["valid_interval_ns"]
        )
        keys = [
            key
            for key in ordered_keys
            if key[0] == item["session_id"]
            and interval[0] <= timestamps[key] - origins[key[0]] <= interval[1]
            and triggers[key]
        ]
        result[item["item_id"]] = {
            "kind": item["item_kind"],
            "trigger_count": len(keys),
            "first_trigger_timestamp_ns": timestamps[keys[0]] if keys else None,
            "first_trigger_frame_index": frame_index[keys[0]] if keys else None,
        }
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dump_rows = read_jsonl(args.dump)
    baseline_rows = [
        row for row in read_jsonl(args.baseline) if row["branch_id"] == BASELINE_BRANCH
    ]
    baseline_evaluation = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
    if len(dump_rows) != EXPECTED_FRAMES or len(baseline_rows) != EXPECTED_FRAMES:
        raise ValueError("expected exactly 4422 dump and baseline frames")
    dump = {(row["session_id"], row["frame_id"]): row for row in dump_rows}
    baseline = {(row["session_id"], row["frame_id"]): row for row in baseline_rows}
    if len(dump) != EXPECTED_FRAMES or set(dump) != set(baseline):
        raise ValueError("dump/baseline frame identity mismatch")
    ordered_keys = [(row["session_id"], row["frame_id"]) for row in dump_rows]
    timestamps = {key: dump[key]["source_capture_timestamp_ns"] for key in ordered_keys}
    hash_mismatches = [
        key
        for key in ordered_keys
        if dump[key]["detector_output_sha256"]
        != baseline[key]["detector_output_sha256"]
    ]
    if hash_mismatches:
        raise ValueError(f"detector output hash mismatch at {hash_mismatches[0]}")

    tracks: list[Track] = []
    next_epoch = 0
    session: str | None = None
    track_decisions: dict[tuple[str, str], str] = {}
    scene_decisions: dict[tuple[str, str], str] = {}
    for key in ordered_keys:
        row = dump[key]
        if key[0] != session:
            session = key[0]
            tracks = []
        tracks, matches, scene_rates, next_epoch = associate(
            tracks, row["detections"], timestamps[key], next_epoch
        )
        selected = baseline[key]["raw_risk"]["source_detection"]
        decision = "NO_TARGET"
        if selected is not None:
            selected_indices = [
                index
                for index, detection in enumerate(row["detections"])
                if detection_identity(detection) == detection_identity(selected)
            ]
            if len(selected_indices) != 1:
                raise ValueError(f"selected target binding mismatch at {key}")
            decision = track_decision(matches[selected_indices[0]].history)
        track_decisions[key] = decision
        scene_decisions[key] = (
            "CONTRADICT_APPROACH"
            if len(scene_rates) >= 2 and median(scene_rates) <= SCENE_SCALE_THRESHOLD_PER_S
            else "ABSTAIN"
        )

    replay = simulate_feedback(ordered_keys, baseline, timestamps)
    replay_mismatches = [
        key
        for key in ordered_keys
        if replay[key] != baseline[key]["feedback_triggered"]
    ]
    if replay_mismatches:
        raise ValueError(f"baseline feedback replay mismatch at {replay_mismatches[0]}")
    candidate = simulate_feedback(
        ordered_keys,
        baseline,
        timestamps,
        vetoes={
            key: scene_decisions[key] == "CONTRADICT_APPROACH" for key in ordered_keys
        },
    )
    active_replay_path = getattr(args, "active_replay", None)
    implementation_parity: dict[str, Any] | None = None
    if active_replay_path is not None:
        active_rows = read_jsonl(active_replay_path)
        active = {(row["session_id"], row["frame_id"]): row for row in active_rows}
        if len(active_rows) != EXPECTED_FRAMES or set(active) != set(ordered_keys):
            raise ValueError("active replay frame identity mismatch")
        detector_mismatches = [
            key
            for key in ordered_keys
            if active[key]["detector_output_sha256"]
            != dump[key]["detector_output_sha256"]
        ]
        baseline_mismatches = [
            key
            for key in ordered_keys
            if active[key]["baseline_feedback_triggered"] != replay[key]
        ]
        candidate_mismatches = [
            key
            for key in ordered_keys
            if active[key]["candidate_feedback_triggered"] != candidate[key]
        ]
        decision_mismatches = [
            key
            for key in ordered_keys
            if (active[key]["dual_loop"]["correction_decision"] or "ABSTAIN")
            != scene_decisions[key]
        ]
        event_mutation_rows = [
            key
            for key in ordered_keys
            if active[key]["dual_loop"]["event_mutation_allowed"]
        ]
        mismatch_groups = {
            "detector": detector_mismatches,
            "baseline_feedback": baseline_mismatches,
            "candidate_feedback": candidate_mismatches,
            "scene_decision": decision_mismatches,
            "event_mutation": event_mutation_rows,
        }
        first_failure = next(
            (
                (name, values[0])
                for name, values in mismatch_groups.items()
                if values
            ),
            None,
        )
        if first_failure is not None:
            raise ValueError(
                f"active implementation parity mismatch: "
                f"{first_failure[0]} at {first_failure[1]}"
            )
        implementation_parity = {
            "status": "VALID",
            "active_replay_sha256":
                hashlib.sha256(active_replay_path.read_bytes()).hexdigest(),
            "frame_count": len(active_rows),
            "detector_hash_mismatch_count": 0,
            "baseline_feedback_mismatch_count": 0,
            "candidate_feedback_mismatch_count": 0,
            "scene_decision_mismatch_count": 0,
            "event_mutation_row_count": 0,
            "baseline_feedback_trigger_count": sum(
                active[key]["baseline_feedback_triggered"] for key in ordered_keys
            ),
            "candidate_feedback_trigger_count": sum(
                active[key]["candidate_feedback_triggered"] for key in ordered_keys
            ),
            "vetoed_feedback_opportunity_count": sum(
                active[key]["candidate_feedback_reason"] == "DUAL_LOOP_CONTRADICTED"
                for key in ordered_keys
            ),
        }
    truth_items = baseline_evaluation["truth_item_table"]
    baseline_items = score_items(truth_items, ordered_keys, timestamps, replay)
    candidate_items = score_items(truth_items, ordered_keys, timestamps, candidate)
    negative_ids = [
        item_id
        for item_id, item in baseline_items.items()
        if item["kind"] == "negative_window"
    ]
    positive_ids = [
        item_id
        for item_id, item in baseline_items.items()
        if item["kind"] == "positive_event"
    ]
    negative_trigger_keys = {
        key
        for key in ordered_keys
        if replay[key]
        and any(
            item["session_id"] == key[0]
            and item["item_kind"] == "negative_window"
            and item["scoring_status"] == "SCORED"
            and item["interval_ns"][0]
            <= timestamps[key]
            - next(
                timestamps[candidate_key]
                for candidate_key in ordered_keys
                if candidate_key[0] == key[0]
            )
            <= item["interval_ns"][1]
            for item in truth_items
        )
    }
    positive_delays = {
        item_id: (
            None
            if candidate_items[item_id]["first_trigger_frame_index"] is None
            else candidate_items[item_id]["first_trigger_frame_index"]
            - baseline_items[item_id]["first_trigger_frame_index"]
        )
        for item_id in positive_ids
    }
    result = {
        "schema_version": "blindassist.dual_loop_multitrack_counterfactual_evaluation.v1",
        "status": "VALID",
        "authority": "DEVELOPMENT_ONLY_BURNED_RGB",
        "inputs": {
            "dump_sha256": hashlib.sha256(args.dump.read_bytes()).hexdigest(),
            "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            "baseline_evaluation_sha256": hashlib.sha256(
                args.baseline_evaluation.read_bytes()
            ).hexdigest(),
            "frame_count": len(ordered_keys),
            "detector_hash_mismatch_count": 0,
        },
        "baseline_replay": {
            "feedback_trigger_count": sum(replay.values()),
            "frame_mismatch_count": 0,
        },
        "r0_multitrack": {
            "decision_counts": {
                decision: list(track_decisions.values()).count(decision)
                for decision in (
                    "CONFIRM_APPROACH",
                    "CONTRADICT_APPROACH",
                    "ABSTAIN",
                    "NO_TARGET",
                )
            },
            "feedback_trigger_decision_counts": {
                decision: sum(
                    replay[key] and track_decisions[key] == decision
                    for key in ordered_keys
                )
                for decision in (
                    "CONFIRM_APPROACH",
                    "CONTRADICT_APPROACH",
                    "ABSTAIN",
                    "NO_TARGET",
                )
            },
            "scored_negative_trigger_contradict_count": sum(
                track_decisions[key] == "CONTRADICT_APPROACH"
                for key in negative_trigger_keys
            ),
            "scientific_terminal": "REJECT_MULTITRACK_ACTIVE_ROUTE",
            "reason": "zero scored negative trigger rows carry a contradict decision",
        },
        "r1_scene_scale_discovery": {
            "rule": {
                "minimum_matched_detections": 2,
                "median_log_height_rate_threshold_per_s":
                    SCENE_SCALE_THRESHOLD_PER_S,
                "hold_ms": 0,
                "effect": "suppress current feedback opportunity only",
            },
            "scene_contradict_frame_count": sum(
                value == "CONTRADICT_APPROACH"
                for value in scene_decisions.values()
            ),
            "candidate_feedback_trigger_count": sum(candidate.values()),
            "scored_negative_trigger_count_baseline": sum(
                baseline_items[item_id]["trigger_count"] for item_id in negative_ids
            ),
            "scored_negative_trigger_count_candidate": sum(
                candidate_items[item_id]["trigger_count"] for item_id in negative_ids
            ),
            "negative_windows_eliminated": sum(
                baseline_items[item_id]["trigger_count"] > 0
                and candidate_items[item_id]["trigger_count"] == 0
                for item_id in negative_ids
            ),
            "positive_recall_baseline": sum(
                baseline_items[item_id]["trigger_count"] > 0 for item_id in positive_ids
            ),
            "positive_recall_candidate": sum(
                candidate_items[item_id]["trigger_count"] > 0 for item_id in positive_ids
            ),
            "maximum_positive_delay_frames": max(
                delay for delay in positive_delays.values() if delay is not None
            ),
            "positive_delay_frames": positive_delays,
            "per_item": {
                item_id: {
                    "baseline": baseline_items[item_id],
                    "candidate": candidate_items[item_id],
                }
                for item_id in baseline_items
            },
            "implementation_parity": implementation_parity or "NOT_PROVIDED",
            "scientific_terminal": "ROW_BURDEN_SIGNAL_ONLY_NOT_WINDOW_EFFECT",
        },
        "claim_ceiling": "ADAPTIVE_TWO_SESSION_DEVELOPMENT_ONLY",
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--active-replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_exclusive(args.output, evaluate(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
