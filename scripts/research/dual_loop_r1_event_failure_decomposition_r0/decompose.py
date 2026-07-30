from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any, Iterable


TASK_ID = "DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0"
SCENE_SCALE_THRESHOLD_PER_S = -0.05
MAXIMUM_GAP_NS = 500_000_000
NANOS_PER_MILLISECOND = 1_000_000
NANOS_PER_SECOND = 1_000_000_000
ALLOWED_TERMINALS = {
    "POLICY_GRANULARITY_MISMATCH_SUPPORTED",
    "SIGNAL_ABSENT_OR_IRRELEVANT",
    "TARGET_ASSOCIATION_LIMITATION_SUPPORTED",
    "MIXED_NO_CLEAR_SUCCESSOR",
}
RETAINED_FALSE_CLASSES = {
    "A_SIGNAL_ABSENT",
    "B_SIGNAL_LATE",
    "C_FRAME_VETO_THEN_RETRY",
    "D_TARGET_OR_ASSOCIATION_MISMATCH",
    "E_SCALE_SIGNAL_TASK_MISMATCH",
    "MIXED_OR_UNRESOLVED",
}


@dataclass(frozen=True)
class TruthItem:
    source_name: str
    source_id: str
    item_id: str
    item_kind: str
    should_alert: bool
    trace_session: str
    score_start_rel_ns: int
    score_end_rel_ns: int
    category: str
    role: str
    outcome_access_state: str
    scoring_status: str = "SCORED"


@dataclass(frozen=True)
class FrameRow:
    source_name: str
    session: str
    frame_id: str
    source_time_ns: int
    relative_time_ns: int
    baseline_feedback: bool
    baseline_reason: str | None
    candidate_feedback: bool
    candidate_reason: str | None
    contradiction: bool
    scene_rate_per_s: float | None
    order_index: int


@dataclass
class TargetDiagnostics:
    selected_rates: dict[tuple[str, str], float]
    scene_medians: dict[tuple[str, str], float]
    reset_rows: dict[tuple[str, str], int]
    target_observability: str
    scene_observability: str


@dataclass
class SourceBundle:
    name: str
    source_id: str
    protocol_id: str
    rows: list[FrameRow]
    truth: list[TruthItem]
    diagnostics: TargetDiagnostics
    input_hashes: dict[str, str]
    pre_frozen_delay_limit_ns: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    require_equal(actual, expected, label)
    return actual


def require_complete_receipt(
    trace_path: Path,
    receipt_path: Path,
    expected_frame_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = read_json(receipt_path)
    require_equal(receipt.get("status"), "COMPLETE", f"{receipt_path} status")
    if "truth_read" in receipt:
        require_equal(receipt.get("truth_read"), False, f"{receipt_path} truth access")
    require_hash(trace_path, str(receipt["trace_sha256"]), f"{receipt_path} trace hash")
    rows = read_jsonl(trace_path)
    if expected_frame_count is not None:
        require_equal(len(rows), expected_frame_count, f"{trace_path} frame count")
    if "frame_count" in receipt:
        require_equal(len(rows), int(receipt["frame_count"]), f"{receipt_path} frame count")
    return receipt, rows


def seconds_to_ns(value: Any) -> int:
    return int(Decimal(str(value)) * Decimal(NANOS_PER_SECOND))


def frame_key(row: dict[str, Any], session: str) -> tuple[str, str]:
    return session, str(row["frame_id"])


def detection_identity(detection: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        detection.get(field)
        for field in (
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
    )


def association_score(
    previous: dict[str, Any], current: dict[str, Any]
) -> float | None:
    for field in ("class_id", "label", "source", "frame_width", "frame_height"):
        if previous.get(field) != current.get(field):
            return None
    first = previous
    second = current
    intersection = max(
        0.0,
        min(float(first["right"]), float(second["right"]))
        - max(float(first["left"]), float(second["left"])),
    ) * max(
        0.0,
        min(float(first["bottom"]), float(second["bottom"]))
        - max(float(first["top"]), float(second["top"])),
    )
    previous_area = (float(first["right"]) - float(first["left"])) * (
        float(first["bottom"]) - float(first["top"])
    )
    current_area = (float(second["right"]) - float(second["left"])) * (
        float(second["bottom"]) - float(second["top"])
    )
    union = previous_area + current_area - intersection
    iou = intersection / union if union > 0.0 else 0.0
    dx = (
        (float(second["left"]) + float(second["right"]))
        - (float(first["left"]) + float(first["right"]))
    ) / 2.0 / max(1.0, float(first["frame_width"]))
    dy = (
        (float(second["top"]) + float(second["bottom"]))
        - (float(first["top"]) + float(first["bottom"]))
    ) / 2.0 / max(1.0, float(first["frame_height"]))
    distance = math.hypot(dx, dy)
    if iou < 0.25 and distance > 0.12:
        return None
    return 2.0 * iou + max(0.0, 1.0 - distance)


def source_detection(risk: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(risk, dict):
        return None
    value = risk.get("source_detection")
    return value if isinstance(value, dict) else None


def active_trace_session(source_name: str, row: dict[str, Any], source_id: str) -> str:
    if source_name == "CrowdBot":
        return str(row["session_id"])
    return source_id


def normalize_active_rows(
    source_name: str,
    source_id: str,
    protocol_id: str,
    raw_rows: list[dict[str, Any]],
) -> list[FrameRow]:
    sessions: dict[str, int] = {}
    for raw in raw_rows:
        session = active_trace_session(source_name, raw, source_id)
        timestamp = int(raw["source_capture_timestamp_ns"])
        sessions.setdefault(session, timestamp)
    normalized: list[FrameRow] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_rows):
        require_equal(raw.get("protocol_id"), protocol_id, f"{source_name} protocol {index}")
        session = active_trace_session(source_name, raw, source_id)
        key = frame_key(raw, session)
        if key in seen:
            raise ValueError(f"{source_name}: duplicate active frame {key}")
        seen.add(key)
        scene_rate = raw.get("dual_loop", {}).get("signed_approach_rate_per_s")
        if scene_rate is not None:
            scene_rate = float(scene_rate)
            if not math.isfinite(scene_rate):
                raise ValueError(f"{source_name}: non-finite scene rate at {key}")
        normalized.append(
            FrameRow(
                source_name=source_name,
                session=session,
                frame_id=str(raw["frame_id"]),
                source_time_ns=int(raw["source_capture_timestamp_ns"]),
                relative_time_ns=int(raw["source_capture_timestamp_ns"]) - sessions[session],
                baseline_feedback=bool(
                    raw.get("baseline_feedback_triggered", raw.get("feedback_triggered", False))
                ),
                baseline_reason=raw.get(
                    "baseline_feedback_reason", raw.get("feedback_reason")
                ),
                candidate_feedback=bool(
                    raw.get("candidate_feedback_triggered", raw.get("feedback_triggered", False))
                ),
                candidate_reason=raw.get(
                    "candidate_feedback_reason", raw.get("feedback_reason")
                ),
                contradiction=(
                    raw.get("dual_loop", {}).get("correction_decision")
                    == "CONTRADICT_APPROACH"
                ),
                scene_rate_per_s=scene_rate,
                order_index=index,
            )
        )
    normalized.sort(key=lambda row: (row.session, row.relative_time_ns, row.order_index))
    for previous, current in zip(normalized, normalized[1:]):
        if previous.session == current.session and current.relative_time_ns <= previous.relative_time_ns:
            raise ValueError(f"{source_name}: non-increasing frame time in {current.session}")
    return normalized


def parse_combined_truth(
    source_name: str,
    source_id: str,
    ledger_path: Path,
    expected_role: str,
    expected_ids: set[str],
    score_overrides: dict[str, tuple[int, int, str]] | None = None,
) -> tuple[list[TruthItem], str]:
    ledger = read_jsonl(ledger_path)
    selected = [row for row in ledger if row.get("source_id") == source_id]
    if not selected:
        raise ValueError(f"{source_name}: no truth rows for {source_id}")
    if any(row.get("role") != expected_role for row in selected):
        raise ValueError(f"{source_name}: truth role drift")
    ids = {str(row.get("event_id") or row.get("negative_window_id")) for row in selected}
    require_equal(ids, expected_ids, f"{source_name} truth item ids")
    items: list[TruthItem] = []
    for row in selected:
        item_id = str(row.get("event_id") or row.get("negative_window_id"))
        positive = row.get("item_kind") == "positive_event"
        if row.get("item_kind") not in {"positive_event", "negative_window"}:
            raise ValueError(f"{source_name}: unknown truth item kind {item_id}")
        if bool(row.get("should_alert")) != positive:
            raise ValueError(f"{source_name}: truth polarity drift {item_id}")
        override = score_overrides.get(item_id) if score_overrides is not None else None
        if override is not None:
            start, end, scoring_status = override
        else:
            if positive:
                interval = row.get("alertable_start_interval_seconds")
            else:
                interval = row.get("window_interval_seconds")
            if not isinstance(interval, dict):
                raise ValueError(f"{source_name}: missing score interval {item_id}")
            start = seconds_to_ns(interval["start"])
            end = seconds_to_ns(interval["end"])
            scoring_status = "SCORED"
        if end < start:
            raise ValueError(f"{source_name}: inverted score interval {item_id}")
        trace_session = (
            str(row["session_id"]) if source_name == "CrowdBot" else source_id
        )
        items.append(
            TruthItem(
                source_name=source_name,
                source_id=source_id,
                item_id=item_id,
                item_kind=str(row["item_kind"]),
                should_alert=positive,
                trace_session=trace_session,
                score_start_rel_ns=start,
                score_end_rel_ns=end,
                category=str(row.get("negative_type") or row.get("event_type") or ""),
                role=str(row["role"]),
                outcome_access_state=str(row.get("outcome_access_state_after_repair", "")),
                scoring_status=scoring_status,
            )
        )
    items.sort(key=lambda item: item.item_id)
    return items, sha256_file(ledger_path)


def parse_shiraz_truth(
    ledger_path: Path,
    receipt_path: Path,
    expected_ids: set[str],
) -> tuple[list[TruthItem], str]:
    receipt = read_json(receipt_path)
    require_equal(receipt.get("status"), "TRUTH_FROZEN_ADEQUATE", "Shiraz truth status")
    require_equal(receipt.get("truth_adequacy"), True, "Shiraz truth adequacy")
    require_equal(receipt.get("baseline_output_opened"), False, "Shiraz truth baseline state")
    require_equal(receipt.get("candidate_output_opened"), False, "Shiraz truth candidate state")
    ledger_hash = sha256_file(ledger_path)
    require_equal(receipt.get("truth_ledger_sha256"), ledger_hash, "Shiraz truth ledger hash")
    selected = read_jsonl(ledger_path)
    ids = {str(row["item_id"]) for row in selected}
    require_equal(ids, expected_ids, "Shiraz truth item ids")
    items: list[TruthItem] = []
    for row in selected:
        positive = bool(row["should_alert"])
        start = int(row["alertable_start_ns"] if positive else row["start_ns"])
        end = int(row["end_ns"])
        if end < start or start < 0:
            raise ValueError(f"Shiraz: inverted truth interval {row['item_id']}")
        items.append(
            TruthItem(
                source_name="Shiraz",
                source_id=str(row["source_id"]),
                item_id=str(row["item_id"]),
                item_kind="positive_event" if positive else "negative_window",
                should_alert=positive,
                trace_session=str(row["source_id"]),
                score_start_rel_ns=start,
                score_end_rel_ns=end,
                category=str(row["category"]),
                role="DEVELOPMENT",
                outcome_access_state=str(row["outcome_access_state"]),
                scoring_status="SCORED",
            )
        )
    items.sort(key=lambda item: item.item_id)
    return items, ledger_hash


def selected_index(
    detections: list[dict[str, Any]], selected: dict[str, Any] | None
) -> int | None:
    if selected is None:
        return None
    exact = [
        index
        for index, detection in enumerate(detections)
        if detection_identity(detection) == detection_identity(selected)
    ]
    if len(exact) == 1:
        return exact[0]
    candidates: list[tuple[float, int]] = []
    for index, detection in enumerate(detections):
        score = association_score(selected, detection)
        if score is not None:
            candidates.append((score, index))
    if len(candidates) != 1:
        return None
    return candidates[0][1]


def recompute_full_target_diagnostics(
    active_rows: list[FrameRow],
    baseline_raw_rows: list[dict[str, Any]],
    detection_raw_rows: list[dict[str, Any]],
    source_name: str,
    source_id: str,
) -> TargetDiagnostics:
    baseline_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    detection_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    origins: dict[str, int] = {}
    for raw in baseline_raw_rows:
        session = active_trace_session(source_name, raw, source_id)
        baseline_by_key[frame_key(raw, session)] = raw
    for raw in detection_raw_rows:
        session = active_trace_session(source_name, raw, source_id)
        detection_by_key[frame_key(raw, session)] = raw
        origins.setdefault(session, int(raw["source_capture_timestamp_ns"]))
    if set(baseline_by_key) != set(detection_by_key):
        raise ValueError(f"{source_name}: baseline/detection frame identity mismatch")
    if set((row.session, row.frame_id) for row in active_rows) != set(detection_by_key):
        raise ValueError(f"{source_name}: active/detection frame identity mismatch")

    @dataclass
    class Track:
        epoch: int
        detection: dict[str, Any]
        timestamp_ns: int

    selected_rates: dict[tuple[str, str], float] = {}
    scene_medians: dict[tuple[str, str], float] = {}
    reset_rows: dict[tuple[str, str], int] = {}
    tracks_by_session: dict[str, list[Track]] = {}
    next_epoch_by_session: dict[str, int] = {}
    previous_selected_epoch: dict[str, int | None] = {}
    previous_selected_detection: dict[str, dict[str, Any] | None] = {}
    previous_selected_time: dict[str, int | None] = {}

    for active in active_rows:
        key = (active.session, active.frame_id)
        raw_dump = detection_by_key[key]
        detections = raw_dump.get("detections")
        if not isinstance(detections, list):
            raise ValueError(f"{source_name}: missing detection list at {key}")
        timestamp_ns = int(raw_dump["source_capture_timestamp_ns"])
        tracks = tracks_by_session.setdefault(active.session, [])
        tracks[:] = [
            track
            for track in tracks
            if 0 < timestamp_ns - track.timestamp_ns <= MAXIMUM_GAP_NS
        ]
        pairs: list[tuple[float, int, int, int]] = []
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                score = association_score(track.detection, detection)
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
            track = tracks[track_index]
            detection = detections[detection_index]
            previous_height = float(track.detection["bottom"]) - float(track.detection["top"])
            current_height = float(detection["bottom"]) - float(detection["top"])
            gap_s = (timestamp_ns - track.timestamp_ns) / NANOS_PER_SECOND
            if previous_height > 0.0 and current_height > 0.0 and gap_s > 0.0:
                scene_rates.append(math.log(current_height / previous_height) / gap_s)
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches[detection_index] = track
        for index, detection in enumerate(detections):
            track = matches.get(index)
            if track is None:
                epoch = next_epoch_by_session.get(active.session, 0)
                next_epoch_by_session[active.session] = epoch + 1
                track = Track(epoch, detection, timestamp_ns)
                tracks.append(track)
                matches[index] = track
            track.detection = detection
            track.timestamp_ns = timestamp_ns
        selected = source_detection(baseline_by_key[key].get("raw_risk"))
        index = selected_index(detections, selected)
        # The Kotlin producer updates tracks before checking selectedTarget, but it
        # emits no evidence when selectedTarget is null. Keep the scene summary
        # observable only on those emitted-evidence opportunities.
        if index is not None and len(scene_rates) >= 2:
            scene_medians[key] = float(median(scene_rates))
        current_epoch: int | None = None
        current_detection: dict[str, Any] | None = None
        if index is not None:
            current_epoch = matches[index].epoch
            current_detection = detections[index]
            previous_epoch = previous_selected_epoch.get(active.session)
            if previous_epoch is not None and current_epoch != previous_epoch:
                reset_rows[key] = 1
            previous_detection = previous_selected_detection.get(active.session)
            previous_time = previous_selected_time.get(active.session)
            if (
                previous_epoch is not None
                and previous_epoch == current_epoch
                and previous_detection is not None
                and previous_time is not None
            ):
                previous_height = float(previous_detection["bottom"]) - float(previous_detection["top"])
                current_height = float(current_detection["bottom"]) - float(current_detection["top"])
                gap_s = (timestamp_ns - previous_time) / NANOS_PER_SECOND
                if previous_height > 0.0 and current_height > 0.0 and 0.0 < gap_s <= 0.5:
                    selected_rates[key] = math.log(current_height / previous_height) / gap_s
            previous_selected_epoch[active.session] = current_epoch
            previous_selected_detection[active.session] = current_detection
            previous_selected_time[active.session] = timestamp_ns
        else:
            previous_selected_epoch.setdefault(active.session, None)
            previous_selected_detection.setdefault(active.session, None)
            previous_selected_time.setdefault(active.session, None)
    return TargetDiagnostics(
        selected_rates=selected_rates,
        scene_medians=scene_medians,
        reset_rows=reset_rows,
        target_observability="RECOMPUTED_FROM_FULL_DETECTION_TRACE",
        scene_observability="RECOMPUTED_FROM_FULL_DETECTION_TRACE",
    )


def selected_only_diagnostics(
    active_rows: list[FrameRow],
    raw_rows: list[dict[str, Any]],
    source_id: str,
) -> TargetDiagnostics:
    selected_by_key: dict[tuple[str, str], dict[str, Any] | None] = {}
    for raw in raw_rows:
        selected_by_key[(source_id, str(raw["frame_id"]))] = source_detection(
            raw.get("raw_risk")
        )
    selected_rates: dict[tuple[str, str], float] = {}
    previous: dict[str, tuple[dict[str, Any], int] | None] = {source_id: None}
    for row in active_rows:
        selected = selected_by_key.get((row.session, row.frame_id))
        if selected is None:
            continue
        old = previous[row.session]
        if old is not None:
            old_detection, old_time = old
            score = association_score(old_detection, selected)
            gap_s = (row.source_time_ns - old_time) / NANOS_PER_SECOND
            old_height = float(old_detection["bottom"]) - float(old_detection["top"])
            current_height = float(selected["bottom"]) - float(selected["top"])
            if score is not None and old_height > 0.0 and current_height > 0.0 and 0.0 < gap_s <= 0.5:
                selected_rates[(row.session, row.frame_id)] = math.log(
                    current_height / old_height
                ) / gap_s
        previous[row.session] = (selected, row.source_time_ns)
    scene_medians = {
        (row.session, row.frame_id): row.scene_rate_per_s
        for row in active_rows
        if row.scene_rate_per_s is not None
    }
    return TargetDiagnostics(
        selected_rates=selected_rates,
        scene_medians=scene_medians,
        reset_rows={},
        target_observability="NOT_OBSERVABLE_NO_FULL_DETECTION_TRACE",
        scene_observability="CONTRADICTION_OR_ADMITTED_RATE_ROWS_ONLY",
    )


def compact_summary(
    values: Iterable[float],
    source: str,
    coverage: str,
) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "source": source,
            "coverage": coverage,
            "count": 0,
            "min_per_s": None,
            "median_per_s": None,
            "max_per_s": None,
        }
    return {
        "source": source,
        "coverage": coverage,
        "count": len(finite),
        "min_per_s": round(min(finite), 6),
        "median_per_s": round(float(median(finite)), 6),
        "max_per_s": round(max(finite), 6),
    }


def longest_contradiction_run_ms(rows: list[FrameRow]) -> float:
    contradiction_times = [row.source_time_ns for row in rows if row.contradiction]
    if not contradiction_times:
        return 0.0
    longest = 0
    start = contradiction_times[0]
    previous = start
    for timestamp in contradiction_times[1:]:
        if timestamp - previous > MAXIMUM_GAP_NS:
            longest = max(longest, previous - start)
            start = timestamp
        previous = timestamp
    longest = max(longest, previous - start)
    return round(longest / NANOS_PER_MILLISECOND, 3)


def first_or_none(rows: list[FrameRow], predicate: Any) -> int | None:
    values = [row.source_time_ns for row in rows if predicate(row)]
    return min(values) if values else None


def classify_retained_false(result: dict[str, Any]) -> str:
    reset_count = result.get("target_association_reset_count")
    first_alert = result.get("baseline_first_feedback_ns")
    first_contradiction = result.get("first_contradiction_ns")
    first_veto = result.get("first_actual_veto_ns")
    retry = result.get("next_candidate_feedback_after_veto_ns")
    if isinstance(reset_count, int) and reset_count > 0 and (
        first_contradiction is None
        or first_alert is None
        or first_contradiction >= first_alert
    ):
        return "D_TARGET_OR_ASSOCIATION_MISMATCH"
    if first_veto is not None and retry is not None:
        return "C_FRAME_VETO_THEN_RETRY"
    if first_contradiction is None:
        return "A_SIGNAL_ABSENT"
    if first_alert is not None and first_contradiction > first_alert:
        return "B_SIGNAL_LATE"
    if first_veto is not None and first_alert is not None and first_veto > first_alert:
        return "B_SIGNAL_LATE"
    if result.get("truth_category") in {
        "STATIC_SCENE",
        "STATIC_OBSTACLE_OFF_CORRIDOR",
        "TURN_OR_NEAR_IN_PLACE_ROTATION",
        "LATERAL_PASS_OR_RECEDING",
        "NORMAL_WALKING_SHAKE",
    }:
        return "E_SCALE_SIGNAL_TASK_MISMATCH"
    return "MIXED_OR_UNRESOLVED"


def summarize_item(
    item: TruthItem,
    frames: list[FrameRow],
    diagnostics: TargetDiagnostics,
) -> dict[str, Any]:
    selected = [
        row
        for row in frames
        if row.session == item.trace_session
        and item.score_start_rel_ns <= row.relative_time_ns <= item.score_end_rel_ns
    ]
    selected.sort(key=lambda row: (row.relative_time_ns, row.order_index))
    baseline_first = first_or_none(selected, lambda row: row.baseline_feedback)
    candidate_first = first_or_none(selected, lambda row: row.candidate_feedback)
    contradiction_first = first_or_none(selected, lambda row: row.contradiction)
    veto_first = first_or_none(
        selected,
        lambda row: row.baseline_feedback
        and not row.candidate_feedback
        and row.candidate_reason == "DUAL_LOOP_CONTRADICTED",
    )
    retry = first_or_none(
        selected,
        lambda row: veto_first is not None
        and row.candidate_feedback
        and row.source_time_ns > veto_first,
    )
    contradiction_lead = (
        None
        if baseline_first is None or contradiction_first is None
        else round(
            (baseline_first - contradiction_first) / NANOS_PER_MILLISECOND,
            3,
        )
    )
    retry_after_veto = (
        None
        if retry is None or veto_first is None
        else round((retry - veto_first) / NANOS_PER_MILLISECOND, 3)
    )
    keyset = {(row.session, row.frame_id) for row in selected}
    selected_rate_values = [
        diagnostics.selected_rates[key]
        for key in keyset
        if key in diagnostics.selected_rates
    ]
    scene_rate_values = [
        diagnostics.scene_medians[key]
        for key in keyset
        if key in diagnostics.scene_medians
    ]
    reset_count: int | None
    if diagnostics.target_observability == "NOT_OBSERVABLE_NO_FULL_DETECTION_TRACE":
        reset_count = None
    else:
        reset_count = sum(
            diagnostics.reset_rows.get((row.session, row.frame_id), 0)
            for row in selected
        )
    if item.should_alert:
        if baseline_first is not None and candidate_first is not None:
            outcome = "POSITIVE_RETAINED_BASELINE_HIT"
        elif baseline_first is not None:
            outcome = "POSITIVE_LOST_BY_CANDIDATE"
        elif candidate_first is not None:
            outcome = "POSITIVE_CANDIDATE_ONLY"
        else:
            outcome = "POSITIVE_BOTH_MISSED"
    else:
        if baseline_first is not None and candidate_first is None:
            outcome = "NEGATIVE_WINDOW_CORRECTED"
        elif baseline_first is not None and candidate_first is not None:
            outcome = "RETAINED_FALSE"
        elif baseline_first is None and candidate_first is not None:
            outcome = "INDUCED_FALSE"
        else:
            outcome = "BOTH_CLEAR"
    result: dict[str, Any] = {
        "source": item.source_name,
        "source_id": item.source_id,
        "truth_item_id": item.item_id,
        "item_kind": item.item_kind,
        "truth_category": item.category,
        "truth_session_id": item.trace_session,
        "truth_role": item.role,
        "truth_outcome_access_state": item.outcome_access_state,
        "scoring_status": item.scoring_status,
        "closed_effect_scored": item.scoring_status == "SCORED",
        "score_interval_start_rel_ns": item.score_start_rel_ns,
        "score_interval_end_rel_ns": item.score_end_rel_ns,
        "baseline_first_feedback_ns": baseline_first,
        "candidate_first_feedback_ns": candidate_first,
        "first_contradiction_ns": contradiction_first,
        "first_actual_veto_ns": veto_first,
        "next_candidate_feedback_after_veto_ns": retry,
        "contradiction_before_first_alert": (
            None
            if baseline_first is None or contradiction_first is None
            else contradiction_first < baseline_first
        ),
        "contradiction_lead_ms": contradiction_lead,
        "retry_after_veto_ms": retry_after_veto,
        "contradiction_row_count": sum(row.contradiction for row in selected),
        "longest_contradiction_run_ms": longest_contradiction_run_ms(selected),
        "selected_target_scale_rate_summary": compact_summary(
            selected_rate_values,
            diagnostics.target_observability,
            "VALID_CONSECUTIVE_SELECTED_TARGET_RATES",
        ),
        "scene_median_scale_rate_summary": compact_summary(
            scene_rate_values,
            diagnostics.scene_observability,
            "RECORDED_OR_RECOMPUTED_SCENE_MEDIANS",
        ),
        "target_association_reset_count": reset_count,
        "target_association_observability": diagnostics.target_observability,
        "baseline_feedback_row_count": sum(row.baseline_feedback for row in selected),
        "candidate_feedback_row_count": sum(row.candidate_feedback for row in selected),
        "feedback_row_delta": sum(row.candidate_feedback for row in selected)
        - sum(row.baseline_feedback for row in selected),
        "final_event_outcome": outcome,
        "retained_false_class": None,
    }
    if outcome == "RETAINED_FALSE":
        result["retained_false_class"] = classify_retained_false(result)
    if baseline_first is not None and candidate_first is not None:
        result["candidate_first_feedback_delay_ms"] = round(
            (candidate_first - baseline_first) / NANOS_PER_MILLISECOND,
            3,
        )
    else:
        result["candidate_first_feedback_delay_ms"] = None
    return result


def truth_for_session(bundle: SourceBundle, item: TruthItem) -> list[FrameRow]:
    return [
        row
        for row in bundle.rows
        if row.session == item.trace_session
        and item.score_start_rel_ns <= row.relative_time_ns <= item.score_end_rel_ns
    ]


def audit_upper_bound(
    bundle: SourceBundle,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = bundle.rows
    breakpoints = {0}
    for row in rows:
        if not row.candidate_feedback:
            continue
        prior = [
            previous.source_time_ns
            for previous in rows
            if previous.session == row.session
            and previous.contradiction
            and previous.source_time_ns <= row.source_time_ns
        ]
        if prior:
            breakpoints.add(row.source_time_ns - max(prior))

    row_keys = [(row.session, row.frame_id) for row in rows]

    def simulate(hold_ns: int) -> tuple[list[bool], dict[str, Any]]:
        suppression_until: dict[str, int] = {}
        candidate_by_key: dict[tuple[str, str], bool] = {}
        for row in rows:
            if row.contradiction:
                suppression_until[row.session] = max(
                    suppression_until.get(row.session, -1),
                    row.source_time_ns + hold_ns,
                )
            candidate_by_key[(row.session, row.frame_id)] = (
                row.candidate_feedback
                and row.source_time_ns > suppression_until.get(row.session, -1)
            )
        candidate = [candidate_by_key[key] for key in row_keys]
        by_item: dict[str, tuple[list[int], list[int]]] = {}
        for item in bundle.truth:
            window = truth_for_session(bundle, item)
            baseline = [row.source_time_ns for row in window if row.baseline_feedback]
            candidate_times = [
                row.source_time_ns
                for row in window
                if candidate_by_key[(row.session, row.frame_id)]
            ]
            by_item[item.item_id] = (baseline, candidate_times)
        corrected = [
            item.item_id
            for item in bundle.truth
            if not item.should_alert
            and by_item[item.item_id][0]
            and not by_item[item.item_id][1]
        ]
        induced = [
            item.item_id
            for item in bundle.truth
            if not item.should_alert
            and not by_item[item.item_id][0]
            and by_item[item.item_id][1]
        ]
        positive_bad: list[dict[str, Any]] = []
        positive_delays: dict[str, float | None] = {}
        for item in bundle.truth:
            if not item.should_alert:
                continue
            baseline, candidate_times = by_item[item.item_id]
            if not baseline:
                continue
            if not candidate_times:
                positive_bad.append({"item_id": item.item_id, "reason": "MISS"})
                positive_delays[item.item_id] = None
                continue
            delay_ms = (candidate_times[0] - baseline[0]) / NANOS_PER_MILLISECOND
            positive_delays[item.item_id] = round(delay_ms, 3)
            if delay_ms * NANOS_PER_MILLISECOND > bundle.pre_frozen_delay_limit_ns:
                positive_bad.append(
                    {
                        "item_id": item.item_id,
                        "reason": "DELAY_OVER_PREFROZEN_LIMIT",
                        "delay_ms": round(delay_ms, 3),
                    }
                )
        return candidate, {
            "hold_ms": round(hold_ns / NANOS_PER_MILLISECOND, 3),
            "corrected_negative_windows": corrected,
            "induced_negative_windows": induced,
            "positive_guardrail_failures": positive_bad,
            "positive_first_feedback_delay_ms": positive_delays,
            "safe": bool(corrected) and not induced and not positive_bad,
            "by_item": by_item,
        }

    safe_candidates: list[tuple[int, dict[str, Any]]] = []
    for hold_ns in sorted(breakpoints):
        _, audit = simulate(hold_ns)
        if audit["safe"]:
            safe_candidates.append((hold_ns, audit))
    witness: dict[str, Any] | None = None
    if safe_candidates:
        hold_ns, audit = safe_candidates[0]
        corrected_id = audit["corrected_negative_windows"][0]
        item = next(item for item in bundle.truth if item.item_id == corrected_id)
        window = truth_for_session(bundle, item)
        baseline_times = [row.source_time_ns for row in window if row.baseline_feedback]
        opportunity_times = [row.source_time_ns for row in window if row.candidate_feedback]
        latest_opportunity = max(baseline_times + opportunity_times)
        trigger_rows = [
            row
            for row in window
            if row.contradiction and row.source_time_ns <= latest_opportunity
        ]
        witness = {
            "source": bundle.name,
            "hold_ms": round(hold_ns / NANOS_PER_MILLISECOND, 3),
            "eliminated_negative_window": corrected_id,
            "baseline_feedback_times_ns": baseline_times,
            "last_suppressing_contradiction_ns": (
                trigger_rows[-1].source_time_ns if trigger_rows else None
            ),
            "all_baseline_hit_positives_retained": not audit[
                "positive_guardrail_failures"
            ],
            "maximum_positive_added_delay_ms": (
                max(audit["positive_first_feedback_delay_ms"].values())
                if audit["positive_first_feedback_delay_ms"]
                else 0.0
            ),
            "induced_negative_window_count": len(audit["induced_negative_windows"]),
        }
    return {
        "source": bundle.name,
        "audit_scope": "DEVELOPMENT_ONLY_IN_MEMORY_COUNTERFACTUAL",
        "strategy_family": "CONTRADICTION_TRIGGERED_FINITE_HOLD",
        "uses_existing_r1_evidence": True,
        "changes_r1_threshold": False,
        "reads_future_frames": False,
        "requires_new_runtime_state_if_implemented": True,
        "pre_frozen_positive_delay_limit_ms": round(
            bundle.pre_frozen_delay_limit_ns / NANOS_PER_MILLISECOND,
            3,
        ),
        "candidate_trace_written": False,
        "safe_witness_exists": bool(safe_candidates),
        "witness": witness,
        "safe_candidate_count": len(safe_candidates),
        "tested_breakpoint_count": len(breakpoints),
        "source_audit_status": (
            "POLICY_GRANULARITY_MISMATCH_SUPPORTED"
            if safe_candidates
            else "NO_SAFE_FINITE_HOLD_WITNESS"
        ),
    }


CSV_COLUMNS = [
    "source",
    "source_id",
    "truth_item_id",
    "item_kind",
    "truth_category",
    "truth_session_id",
    "scoring_status",
    "closed_effect_scored",
    "score_interval_start_rel_ns",
    "score_interval_end_rel_ns",
    "baseline_first_feedback_ns",
    "first_contradiction_ns",
    "first_actual_veto_ns",
    "next_candidate_feedback_after_veto_ns",
    "contradiction_before_first_alert",
    "contradiction_lead_ms",
    "retry_after_veto_ms",
    "contradiction_row_count",
    "longest_contradiction_run_ms",
    "selected_target_scale_rate_summary",
    "scene_median_scale_rate_summary",
    "target_association_reset_count",
    "baseline_feedback_row_count",
    "candidate_feedback_row_count",
    "feedback_row_delta",
    "candidate_first_feedback_ns",
    "candidate_first_feedback_delay_ms",
    "final_event_outcome",
    "retained_false_class",
]


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def write_lf_exclusive(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    temporary.replace(path)


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    audit = result["upper_bound_audit"]
    lines = [
        "# DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0",
        "",
        f"Terminal：`{result['terminal']}`",
        "",
        "## 结论",
        "",
        "active R1 的作用边界是当前 feedback opportunity 的 frame-level veto；它不写入",
        "RiskEventTracker 的 event identity/lifecycle。因此 contradiction 可以减少一行，",
        "但只要后续 candidate feedback opportunity 重新可用，负窗仍会保持 false。这个结论",
        "只适用于已关闭的 Development evidence，不形成新的效果主张。",
        "",
        "## 三来源汇总",
        "",
            "| source | ledger positives | closed scored positives | baseline/candidate positive hit | baseline/candidate false windows | baseline/candidate feedback rows |",
            "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for source in summary["sources"]:
        lines.append(
            "| {source} | {positive_count} | {scored_positive_count} | "
            "{baseline_positive_hits}/{candidate_positive_hits} | "
            "{baseline_false_windows}/{candidate_false_windows} | "
            "{baseline_feedback_rows}/{candidate_feedback_rows} |".format(
                **source
            )
        )
    lines.extend(
        [
            "",
            "## Retained-false 分类",
            "",
            "| class | count | 解释 |",
            "| --- | ---: | --- |",
            "| A_SIGNAL_ABSENT | {A_SIGNAL_ABSENT} | 评分窗没有 contradiction evidence。 |".format(
                **summary["retained_false_class_counts"]
            ),
            "| B_SIGNAL_LATE | {B_SIGNAL_LATE} | 首个 contradiction/veto 晚于 baseline 首次反馈。 |".format(
                **summary["retained_false_class_counts"]
            ),
            "| C_FRAME_VETO_THEN_RETRY | {C_FRAME_VETO_THEN_RETRY} | 同帧 veto 后同一评分窗重新出现 candidate feedback。 |".format(
                **summary["retained_false_class_counts"]
            ),
            "| D_TARGET_OR_ASSOCIATION_MISMATCH | {D_TARGET_OR_ASSOCIATION_MISMATCH} | 仅在 full detection trace 可观测到 target association reset 时使用。 |".format(
                **summary["retained_false_class_counts"]
            ),
            "| E_SCALE_SIGNAL_TASK_MISMATCH | {E_SCALE_SIGNAL_TASK_MISMATCH} | signal 存在但未落在可 veto 的 feedback opportunity，或语义与 scene-scale task 不相称。 |".format(
                **summary["retained_false_class_counts"]
            ),
            "| MIXED_OR_UNRESOLVED | {MIXED_OR_UNRESOLVED} | 现有字段不足以唯一归因。 |".format(
                **summary["retained_false_class_counts"]
            ),
            "",
            "## Upper-bound audit（Development-only）",
            "",
        ]
    )
    if audit["safe_witness_exists"]:
        witness = audit["witness_sources"][0]
        lines.extend(
            [
                "存在一个只用于审计的 causal witness：在已有 R1 contradiction 后维持有限抑制，",
                f"{witness['source']} 以约 `{witness['hold_ms']} ms` 的 duration 可消除 `{witness['eliminated_negative_window']}`，",
                f"保留所有 baseline-hit 正例、最大新增首反馈时延 `{witness['maximum_positive_added_delay_ms']} ms`、",
                "induced negative window 为 0。它需要新的 runtime state，且只在单一 Development",
                "source 的已见 trace 上成立；因此支持 policy-granularity mismatch，但不是 R2 授权。",
            ]
        )
    else:
        lines.append("在各来源预冻结正例时延上限内，没有找到可消除完整负窗的有限 causal witness。")
    lines.extend(
        [
            "",
            "## R2 决策",
            "",
            f"是否值得设计单变量 R2：`{result['recommendation']['worth_designing_single_variable_r2']}`。",
            "推荐关闭 scene-scale active 路线；不自动实现 hold/latch、事件状态或任何 R2。",
            "当前 R1 只保留 mechanism、row-density diagnostic、回归夹具和 failure decomposition 价值。",
            "",
            "## 证据限制",
            "",
            "- 所有指标都来自已关闭 Development trace/truth/receipt；不读取未来帧、不读取运行时 truth。",
            "- Matoaka 没有完整 detection dump，selected target rate 只来自 trace 中已记录的 selected risk box；scene median 与 target reset 不可观测。",
            "- `signed_approach_rate_per_s` 不是用户收到提醒的行数；实际 veto 只按同帧 baseline/candidate/reason 重新计算。",
            "- 本报告不是 Confirmation、产品、安全或真人助行证据。",
            "",
        ]
    )
    lines.extend(["## 逐窗口分解", "", "以下逐项保留 truth ledger 中的每个正例事件与负例窗口；`scoring_status` 沿用 closed evaluation。", ""])
    window_fields = [
        "baseline_first_feedback_ns",
        "first_contradiction_ns",
        "first_actual_veto_ns",
        "next_candidate_feedback_after_veto_ns",
        "contradiction_before_first_alert",
        "contradiction_lead_ms",
        "retry_after_veto_ms",
        "contradiction_row_count",
        "longest_contradiction_run_ms",
        "selected_target_scale_rate_summary",
        "scene_median_scale_rate_summary",
        "target_association_reset_count",
        "baseline_feedback_row_count",
        "candidate_feedback_row_count",
        "final_event_outcome",
        "retained_false_class",
    ]
    for window in result["windows"]:
        lines.extend(
            [
                f"### {window['source']} / {window['truth_item_id']}",
                "",
                f"kind=`{window['item_kind']}`; scoring_status=`{window['scoring_status']}`; "
                f"closed_effect_scored=`{window['closed_effect_scored']}`",
                "",
                "| field | value |",
                "| --- | --- |",
            ]
        )
        for field in window_fields:
            lines.append(f"| {field} | `{json_cell(window.get(field))}` |")
        lines.append("")
    return "\n".join(lines)


def publish_outputs(
    output_dir: Path,
    result: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    audit_payload = json.dumps(
        result["upper_bound_audit"], ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    csv_lines: list[str] = []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in result["windows"]:
        writer.writerow({column: json_cell(row.get(column)) for column in CSV_COLUMNS})
    csv_payload = buffer.getvalue()
    markdown_payload = render_markdown(result)
    if not markdown_payload.endswith("\n"):
        markdown_payload += "\n"
    files = {
        "event_failure_decomposition.json": json_payload.encode("utf-8"),
        "upper_bound_audit.json": audit_payload.encode("utf-8"),
        "event_failure_decomposition.csv": csv_payload.encode("utf-8"),
        "event_failure_decomposition.md": markdown_payload.encode("utf-8"),
    }
    for name, payload in files.items():
        if b"\r\n" in payload:
            raise ValueError(f"{name}: output is not LF")
        write_lf_exclusive(output_dir / name, payload)


def default_paths(repo_root: Path) -> dict[str, Path]:
    evidence = repo_root / "artifacts.local/evidence"
    dual = evidence / "dual-loop"
    unseen = evidence / "dual-loop-r1-unseen-natural-event-r0/rank2-shiraz"
    return {
        "crowd_active_trace": dual / "scene-scale-veto-r1/device-active-replay/trace.jsonl",
        "crowd_active_receipt": dual / "scene-scale-veto-r1/device-active-replay/producer_receipt.json",
        "crowd_evaluation": dual / "scene-scale-veto-r1/evaluation.json",
        "crowd_dump_trace": dual / "multitrack-counterfactual-r0/device-dump/trace.jsonl",
        "crowd_dump_receipt": dual / "multitrack-counterfactual-r0/device-dump/producer_receipt.json",
        "crowd_baseline_trace": dual / "production-temporal-geometry-factorial-ab-r0/device-producer/trace.jsonl",
        "crowd_baseline_receipt": dual / "production-temporal-geometry-factorial-ab-r0/device-producer/producer_receipt.json",
        "crowd_scoring_evaluation": dual / "production-temporal-geometry-factorial-ab-r0/evaluation/result.json",
        "crowd_truth": dual / "f1a-negative-category-supplement-r1/combined_event_window_ledger.jsonl",
        "crowd_truth_validation": dual / "f1a-negative-category-supplement-r1/validation.json",
        "matoaka_active_trace": dual / "scene-scale-veto-r1/matoaka-device-output/trace.jsonl",
        "matoaka_active_receipt": dual / "scene-scale-veto-r1/matoaka-device-output/producer_receipt.json",
        "matoaka_evaluation": dual / "scene-scale-veto-r1/matoaka-evaluation.json",
        "matoaka_truth": dual / "f1a-negative-category-supplement-r1/combined_event_window_ledger.jsonl",
        "matoaka_truth_validation": dual / "f1a-negative-category-supplement-r1/validation.json",
        "shiraz_baseline_trace": unseen / "device-r1/baseline-output/trace.jsonl",
        "shiraz_baseline_receipt": unseen / "device-r1/baseline-output/producer_receipt.json",
        "shiraz_candidate_trace": unseen / "device-r1/candidate-output/trace.jsonl",
        "shiraz_candidate_receipt": unseen / "device-r1/candidate-output/producer_receipt.json",
        "shiraz_truth": unseen / "truth-freeze-r2/truth_ledger.jsonl",
        "shiraz_truth_receipt": unseen / "truth-freeze-r2/truth_freeze_receipt.json",
        "shiraz_effect": unseen / "effect-evaluation-r1/effect_result.json",
        "shiraz_terminal_receipt": unseen / "effect-evaluation-r1/terminal.json",
    }


def validate_combined_truth_receipt(path: Path, ledger_hash: str) -> None:
    receipt = read_json(path)
    require_equal(receipt.get("combined_ledger_sha256"), ledger_hash, f"{path} ledger hash")
    require_equal(receipt.get("data_protocol_status"), "VALID", f"{path} protocol status")
    require_equal(receipt.get("candidate_output_visibility"), False, f"{path} candidate visibility")


def closed_score_overrides(evaluation: dict[str, Any]) -> dict[str, tuple[int, int, str]]:
    table = evaluation.get("truth_item_table")
    if not isinstance(table, list):
        raise ValueError("closed evaluation: missing truth_item_table")
    overrides: dict[str, tuple[int, int, str]] = {}
    for entry in table:
        if not isinstance(entry, dict):
            raise ValueError("closed evaluation: malformed truth item table entry")
        item_id = str(entry["item_id"])
        interval_key = "valid_interval_ns" if entry.get("item_kind") == "positive_event" else "interval_ns"
        interval = entry.get(interval_key)
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"closed evaluation: missing {interval_key} for {item_id}")
        start, end = int(interval[0]), int(interval[1])
        if start < 0 or end < start:
            raise ValueError(f"closed evaluation: inverted interval for {item_id}")
        overrides[item_id] = (start, end, str(entry.get("scoring_status", "SCORED")))
    return overrides


def load_crowdbot(paths: dict[str, Path]) -> SourceBundle:
    active_receipt, active_raw = require_complete_receipt(
        paths["crowd_active_trace"], paths["crowd_active_receipt"], 4422
    )
    require_equal(active_receipt.get("protocol_id"), "DUAL_LOOP_SCENE_SCALE_VETO_R1", "CrowdBot active protocol")
    require_equal(active_receipt.get("risk_mutation_count"), 0, "CrowdBot risk mutation")
    dump_receipt, dump_raw = require_complete_receipt(
        paths["crowd_dump_trace"], paths["crowd_dump_receipt"], 4422
    )
    require_equal(dump_receipt.get("truth_read"), False, "CrowdBot dump truth access")
    require_equal(active_receipt.get("input_dump_sha256"), sha256_file(paths["crowd_dump_trace"]), "CrowdBot active input dump hash")
    baseline_receipt = read_json(paths["crowd_baseline_receipt"])
    require_equal(baseline_receipt.get("status"), "COMPLETE", "CrowdBot baseline status")
    require_equal(baseline_receipt.get("truth_joined"), False, "CrowdBot baseline truth join")
    require_hash(paths["crowd_baseline_trace"], str(baseline_receipt["trace_sha256"]), "CrowdBot baseline trace hash")
    baseline_all = read_jsonl(paths["crowd_baseline_trace"])
    baseline_raw = [
        row
        for row in baseline_all
        if row.get("branch_id") == "CURRENT_FULL_PRODUCTION_TEMPORAL_GEOMETRY"
    ]
    require_equal(len(baseline_raw), 4422, "CrowdBot baseline branch frame count")
    active_hash_by_key = {
        (str(row["session_id"]), str(row["frame_id"])): row["detector_output_sha256"]
        for row in active_raw
    }
    baseline_by_key = {
        (str(row["session_id"]), str(row["frame_id"])): row for row in baseline_raw
    }
    dump_by_key = {
        (str(row["session_id"]), str(row["frame_id"])): row for row in dump_raw
    }
    if set(active_hash_by_key) != set(baseline_by_key) or set(active_hash_by_key) != set(dump_by_key):
        raise ValueError("CrowdBot active/baseline/dump identity mismatch")
    for active in active_raw:
        key = (str(active["session_id"]), str(active["frame_id"]))
        baseline = baseline_by_key[key]
        dump = dump_by_key[key]
        require_equal(active["detector_output_sha256"], baseline["detector_output_sha256"], f"CrowdBot detector hash {key}")
        require_equal(active["detector_output_sha256"], dump["detector_output_sha256"], f"CrowdBot dump hash {key}")
        require_equal(active["baseline_feedback_triggered"], baseline["feedback_triggered"], f"CrowdBot baseline feedback parity {key}")
    evaluation = read_json(paths["crowd_evaluation"])
    require_equal(evaluation.get("status"), "VALID", "CrowdBot evaluation status")
    require_equal(evaluation["inputs"]["dump_sha256"], sha256_file(paths["crowd_dump_trace"]), "CrowdBot evaluation dump hash")
    require_equal(evaluation["r1_scene_scale_discovery"]["implementation_parity"]["active_replay_sha256"], sha256_file(paths["crowd_active_trace"]), "CrowdBot evaluation active hash")
    scoring_evaluation = read_json(paths["crowd_scoring_evaluation"])
    require_equal(scoring_evaluation.get("status"), "VALID", "CrowdBot scoring evaluation status")
    require_equal(scoring_evaluation.get("trace_sha256"), sha256_file(paths["crowd_baseline_trace"]), "CrowdBot scoring evaluation trace hash")
    score_overrides = closed_score_overrides(scoring_evaluation)
    ledger_hash = sha256_file(paths["crowd_truth"])
    validate_combined_truth_receipt(paths["crowd_truth_validation"], ledger_hash)
    truth, _ = parse_combined_truth(
        "CrowdBot",
        "crowdbot_0327_shared_control",
        paths["crowd_truth"],
        "DECISION",
        {f"F1A-N-{index:03d}" for index in range(1, 8)}
        | {f"F1A-P-{index:03d}" for index in range(1, 11)},
        score_overrides,
    )
    rows = normalize_active_rows(
        "CrowdBot",
        "crowdbot_0327_shared_control",
        "DUAL_LOOP_SCENE_SCALE_VETO_R1",
        active_raw,
    )
    diagnostics = recompute_full_target_diagnostics(
        rows,
        baseline_raw,
        dump_raw,
        "CrowdBot",
        "crowdbot_0327_shared_control",
    )
    for row in rows:
        key = (row.session, row.frame_id)
        recomputed = diagnostics.scene_medians.get(key)
        active_decision = next(
            raw["dual_loop"]["correction_decision"]
            for raw in active_raw
            if (str(raw["session_id"]), str(raw["frame_id"])) == key
        )
        expected = (
            "CONTRADICT_APPROACH"
            if recomputed is not None and recomputed <= SCENE_SCALE_THRESHOLD_PER_S
            else "ABSTAIN"
        )
        require_equal(active_decision or "ABSTAIN", expected, f"CrowdBot scene parity {key}")
    return SourceBundle(
        name="CrowdBot",
        source_id="crowdbot_0327_shared_control",
        protocol_id="DUAL_LOOP_SCENE_SCALE_VETO_R1",
        rows=rows,
        truth=truth,
        diagnostics=diagnostics,
        input_hashes={
            "active_trace_sha256": sha256_file(paths["crowd_active_trace"]),
            "active_receipt_sha256": sha256_file(paths["crowd_active_receipt"]),
            "detection_trace_sha256": sha256_file(paths["crowd_dump_trace"]),
            "detection_receipt_sha256": sha256_file(paths["crowd_dump_receipt"]),
            "baseline_trace_sha256": sha256_file(paths["crowd_baseline_trace"]),
            "baseline_receipt_sha256": sha256_file(paths["crowd_baseline_receipt"]),
            "scoring_evaluation_sha256": sha256_file(paths["crowd_scoring_evaluation"]),
            "truth_ledger_sha256": ledger_hash,
            "truth_validation_sha256": sha256_file(paths["crowd_truth_validation"]),
        },
        pre_frozen_delay_limit_ns=0,
    )


def load_matoaka(paths: dict[str, Path]) -> SourceBundle:
    receipt, active_raw = require_complete_receipt(
        paths["matoaka_active_trace"], paths["matoaka_active_receipt"], 10_724
    )
    require_equal(receipt.get("protocol_id"), "DUAL_LOOP_SCENE_SCALE_VETO_R1_CROSS_SOURCE_MATOAKA", "Matoaka protocol")
    require_equal(receipt.get("risk_mutation_count"), 0, "Matoaka risk mutation")
    evaluation = read_json(paths["matoaka_evaluation"])
    require_equal(evaluation.get("status"), "VALID", "Matoaka evaluation status")
    require_equal(evaluation.get("scientific_terminal"), "CROSS_SOURCE_DEVELOPMENT_SIGNAL_REPLICATED", "Matoaka evaluation terminal")
    require_equal(evaluation["inputs"]["trace_sha256"], sha256_file(paths["matoaka_active_trace"]), "Matoaka evaluation trace hash")
    ledger_hash = sha256_file(paths["matoaka_truth"])
    require_equal(evaluation["inputs"]["truth_ledger_sha256"], ledger_hash, "Matoaka evaluation truth hash")
    validate_combined_truth_receipt(paths["matoaka_truth_validation"], ledger_hash)
    truth, _ = parse_combined_truth(
        "Matoaka",
        "wikimedia_commons_matoaka_west_virginia_walk_2019",
        paths["matoaka_truth"],
        "DEVELOPMENT",
        {f"F1A-N-{index:03d}" for index in range(8, 20)}
        | {f"F1A-P-{index:03d}" for index in range(11, 18)},
    )
    rows = normalize_active_rows(
        "Matoaka",
        "wikimedia_commons_matoaka_west_virginia_walk_2019",
        "DUAL_LOOP_SCENE_SCALE_VETO_R1_CROSS_SOURCE_MATOAKA",
        active_raw,
    )
    diagnostics = selected_only_diagnostics(rows, active_raw, rows[0].session)
    return SourceBundle(
        name="Matoaka",
        source_id="wikimedia_commons_matoaka_west_virginia_walk_2019",
        protocol_id="DUAL_LOOP_SCENE_SCALE_VETO_R1_CROSS_SOURCE_MATOAKA",
        rows=rows,
        truth=truth,
        diagnostics=diagnostics,
        input_hashes={
            "active_trace_sha256": sha256_file(paths["matoaka_active_trace"]),
            "active_receipt_sha256": sha256_file(paths["matoaka_active_receipt"]),
            "evaluation_sha256": sha256_file(paths["matoaka_evaluation"]),
            "truth_ledger_sha256": ledger_hash,
            "truth_validation_sha256": sha256_file(paths["matoaka_truth_validation"]),
        },
        pre_frozen_delay_limit_ns=0,
    )


def load_shiraz(paths: dict[str, Path]) -> SourceBundle:
    baseline_receipt, baseline_raw = require_complete_receipt(
        paths["shiraz_baseline_trace"], paths["shiraz_baseline_receipt"], 4891
    )
    candidate_receipt, candidate_raw = require_complete_receipt(
        paths["shiraz_candidate_trace"], paths["shiraz_candidate_receipt"], 4891
    )
    require_equal(baseline_receipt.get("protocol_id"), "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_SHIRAZ", "Shiraz baseline protocol")
    require_equal(candidate_receipt.get("protocol_id"), "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_SHIRAZ", "Shiraz candidate protocol")
    require_equal(candidate_receipt.get("risk_mutation_count"), 0, "Shiraz candidate risk mutation")
    require_equal(candidate_receipt.get("event_mutation_allowed_count"), 0, "Shiraz candidate event mutation")
    if len(baseline_raw) != len(candidate_raw):
        raise ValueError("Shiraz baseline/candidate row count mismatch")
    for index, (base, candidate) in enumerate(zip(baseline_raw, candidate_raw)):
        for field in ("frame_id", "source_capture_timestamp_ns", "detector_output_sha256"):
            require_equal(candidate.get(field), base.get(field), f"Shiraz identity {field} {index}")
        require_equal(candidate.get("baseline_feedback_triggered"), base.get("feedback_triggered"), f"Shiraz baseline feedback parity {index}")
        require_equal(candidate.get("baseline_raw_risk_sha256"), base.get("raw_risk_sha256"), f"Shiraz raw risk parity {index}")
        require_equal(candidate.get("baseline_stable_risk_sha256"), base.get("stable_risk_sha256"), f"Shiraz stable risk parity {index}")
    effect = read_json(paths["shiraz_effect"])
    require_equal(effect.get("status"), "COMPLETE", "Shiraz effect status")
    require_equal(effect.get("terminal"), "FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT", "Shiraz effect terminal")
    require_equal(effect.get("baseline_trace_sha256"), sha256_file(paths["shiraz_baseline_trace"]), "Shiraz effect baseline hash")
    require_equal(effect.get("candidate_trace_sha256"), sha256_file(paths["shiraz_candidate_trace"]), "Shiraz effect candidate hash")
    require_equal(effect.get("baseline_receipt_sha256"), sha256_file(paths["shiraz_baseline_receipt"]), "Shiraz effect baseline receipt hash")
    require_equal(effect.get("candidate_receipt_sha256"), sha256_file(paths["shiraz_candidate_receipt"]), "Shiraz effect candidate receipt hash")
    terminal_receipt = read_json(paths["shiraz_terminal_receipt"])
    require_equal(terminal_receipt.get("status"), effect.get("terminal"), "Shiraz terminal receipt")
    truth, ledger_hash = parse_shiraz_truth(
        paths["shiraz_truth"],
        paths["shiraz_truth_receipt"],
        {f"R2_P00{index}" for index in range(1, 8)} | {f"R2_N00{index}" for index in range(1, 7)},
    )
    require_equal(effect.get("truth_ledger_sha256"), ledger_hash, "Shiraz effect truth hash")
    rows = normalize_active_rows(
        "Shiraz",
        "commons_iran_shiraz_city_tour_2021_5",
        "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_SHIRAZ",
        candidate_raw,
    )
    diagnostics = recompute_full_target_diagnostics(
        rows,
        baseline_raw,
        baseline_raw,
        "Shiraz",
        "commons_iran_shiraz_city_tour_2021_5",
    )
    for row in rows:
        key = (row.session, row.frame_id)
        recomputed = diagnostics.scene_medians.get(key)
        expected = (
            "CONTRADICT_APPROACH"
            if recomputed is not None and recomputed <= SCENE_SCALE_THRESHOLD_PER_S
            else "ABSTAIN"
        )
        raw = next(raw for raw in candidate_raw if str(raw["frame_id"]) == row.frame_id)
        require_equal(raw["dual_loop"].get("correction_decision") or "ABSTAIN", expected, f"Shiraz scene parity {key}")
    return SourceBundle(
        name="Shiraz",
        source_id="commons_iran_shiraz_city_tour_2021_5",
        protocol_id="DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_SHIRAZ",
        rows=rows,
        truth=truth,
        diagnostics=diagnostics,
        input_hashes={
            "baseline_trace_sha256": sha256_file(paths["shiraz_baseline_trace"]),
            "baseline_receipt_sha256": sha256_file(paths["shiraz_baseline_receipt"]),
            "candidate_trace_sha256": sha256_file(paths["shiraz_candidate_trace"]),
            "candidate_receipt_sha256": sha256_file(paths["shiraz_candidate_receipt"]),
            "truth_ledger_sha256": ledger_hash,
            "truth_receipt_sha256": sha256_file(paths["shiraz_truth_receipt"]),
            "effect_result_sha256": sha256_file(paths["shiraz_effect"]),
            "terminal_receipt_sha256": sha256_file(paths["shiraz_terminal_receipt"]),
        },
        pre_frozen_delay_limit_ns=250_000_000,
    )


def aggregate_summary(bundles: list[SourceBundle], windows: list[dict[str, Any]]) -> dict[str, Any]:
    source_summaries: list[dict[str, Any]] = []
    for bundle in bundles:
        source_rows = [row for row in windows if row["source"] == bundle.name]
        positives = [row for row in source_rows if row["item_kind"] == "positive_event"]
        negatives = [row for row in source_rows if row["item_kind"] == "negative_window"]
        source_summaries.append(
            {
                "source": bundle.name,
                "positive_count": len(positives),
                "scored_positive_count": sum(row["closed_effect_scored"] for row in positives),
                "baseline_positive_hits": sum(row["baseline_first_feedback_ns"] is not None for row in positives),
                "candidate_positive_hits": sum(row["candidate_first_feedback_ns"] is not None for row in positives),
                "baseline_false_windows": sum(row["baseline_first_feedback_ns"] is not None for row in negatives),
                "candidate_false_windows": sum(row["candidate_first_feedback_ns"] is not None for row in negatives),
                "scored_negative_window_count": sum(row["closed_effect_scored"] for row in negatives),
                "baseline_feedback_rows": sum(row["baseline_feedback_row_count"] for row in source_rows),
                "candidate_feedback_rows": sum(row["candidate_feedback_row_count"] for row in source_rows),
                "retained_false_windows": sum(row["final_event_outcome"] == "RETAINED_FALSE" for row in negatives),
                "corrected_negative_windows": sum(row["final_event_outcome"] == "NEGATIVE_WINDOW_CORRECTED" for row in negatives),
            }
        )
    class_counts = {name: 0 for name in sorted(RETAINED_FALSE_CLASSES)}
    for row in windows:
        if row["retained_false_class"] in class_counts:
            class_counts[row["retained_false_class"]] += 1
    return {
        "sources": source_summaries,
        "window_count": len(windows),
        "positive_event_count": sum(row["item_kind"] == "positive_event" for row in windows),
        "negative_window_count": sum(row["item_kind"] == "negative_window" for row in windows),
        "closed_effect_scored_count": sum(row["closed_effect_scored"] for row in windows),
        "closed_effect_not_evaluable_count": sum(not row["closed_effect_scored"] for row in windows),
        "retained_false_count": sum(row["final_event_outcome"] == "RETAINED_FALSE" for row in windows),
        "retained_false_class_counts": class_counts,
        "baseline_feedback_rows": sum(row["baseline_feedback_row_count"] for row in windows),
        "candidate_feedback_rows": sum(row["candidate_feedback_row_count"] for row in windows),
    }


def analyze(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    paths = default_paths(repo_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing closed evidence: " + ", ".join(missing))
    bundles = [load_crowdbot(paths), load_matoaka(paths), load_shiraz(paths)]
    windows: list[dict[str, Any]] = []
    upper_bounds: list[dict[str, Any]] = []
    for bundle in bundles:
        for item in bundle.truth:
            windows.append(summarize_item(item, bundle.rows, bundle.diagnostics))
        upper_bounds.append(audit_upper_bound(bundle, windows[-len(bundle.truth) :]))
    upper_witnesses = [audit for audit in upper_bounds if audit["safe_witness_exists"]]
    if upper_witnesses:
        terminal = "POLICY_GRANULARITY_MISMATCH_SUPPORTED"
    elif all(
        row["first_contradiction_ns"] is None
        for row in windows
        if row["final_event_outcome"] == "RETAINED_FALSE"
    ):
        terminal = "SIGNAL_ABSENT_OR_IRRELEVANT"
    elif any(
        row["retained_false_class"] == "D_TARGET_OR_ASSOCIATION_MISMATCH"
        for row in windows
    ):
        terminal = "TARGET_ASSOCIATION_LIMITATION_SUPPORTED"
    else:
        terminal = "MIXED_NO_CLEAR_SUCCESSOR"
    require_equal(terminal in ALLOWED_TERMINALS, True, "decomposition terminal allowlist")
    result = {
        "schema_version": "blindassist.dual_loop_r1_event_failure_decomposition_r0.v1",
        "task_id": TASK_ID,
        "status": "COMPLETE",
        "stage": "DEVELOPMENT_POST_TERMINAL_ANALYSIS",
        "terminal": terminal,
        "frozen_r1_terminal": "FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY",
        "scope": {
            "source_rule": "consume closed Development trace/truth/receipt only",
            "r1_implementation_modified": False,
            "threshold_modified": False,
            "new_hold_latch_or_event_state": False,
            "new_sensor_or_model": False,
            "candidate_rerun": False,
            "future_frames_read": False,
            "new_effect_claim": False,
        },
        "inputs": {
            bundle.name: {
                "source_id": bundle.source_id,
                "protocol_id": bundle.protocol_id,
                "pre_frozen_delay_limit_ms": round(
                    bundle.pre_frozen_delay_limit_ns / NANOS_PER_MILLISECOND,
                    3,
                ),
                "hashes": bundle.input_hashes,
            }
            for bundle in bundles
        },
        "summary": aggregate_summary(bundles, windows),
        "windows": windows,
        "upper_bound_audit": {
            "schema_version": "blindassist.dual_loop_r1_upper_bound_audit_r0.v1",
            "source_results": upper_bounds,
            "safe_witness_exists": bool(upper_witnesses),
            "witness_sources": [audit["witness"] for audit in upper_witnesses],
            "interpretation": (
                "A finite causal policy witness exists in one Development source, but it requires "
                "new runtime state and is not a new R1 effect result."
                if upper_witnesses
                else "No safe finite causal witness was found in the audited strategy family."
            ),
        },
        "recommendation": {
            "worth_designing_single_variable_r2": False,
            "decision": "CLOSE_SCENE_SCALE_ACTIVE_ROUTE",
            "reason": (
                "The only witness is a single-source, approximately 19.1 s finite hold in a "
                "Development replay; it requires new state, is not cross-source, and does not "
                "justify a new effect claim or R2 implementation."
            ),
            "r2_implemented": False,
        },
    }
    publish_outputs(output_dir, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TASK_ID)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else args.repo_root / args.output_dir
    if "artifacts.local" not in output_dir.parts:
        raise ValueError("output must be under artifacts.local")
    result = analyze(args.repo_root.resolve(), output_dir.resolve())
    print(json.dumps({"status": result["status"], "terminal": result["terminal"], "output_dir": str(output_dir.resolve())}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
