"""Run the public-JRDB robot-relative diagnostic ceiling for DTR-R0.

Both arms receive only the ordered current-and-past source-native
robot-relative 3-D person annotations. Future JRDB annotations are read only by
this evaluator to decide whether the person enters the time-aligned relative
tube inside the three-second horizon. Because JRDB boxes may be interpolated by
the dataset, this is privileged annotation access rather than runtime-causal
sensor evidence.

This is intentionally an information ceiling.  It does not run RGB detection,
claim human-authored risk truth, or establish product/safety performance.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence
import zipfile

from dtr_r0 import (
    Arm,
    CausalFrame,
    DTRConfig,
    EgoPose,
    Observation,
    Prediction,
    Signal,
    run_arm,
)
from dtr_r1 import FROZEN_R1_CONFIG, run_r1_arm
from dtr_r2 import FROZEN_R2_CONFIG, run_r2_arm
from dtr_r3 import FROZEN_R3_CONFIG, R3Arm, run_r3_arm


SCHEMA = "dtr-r0-jrdb-native-ceiling-v1"
R1_SCHEMA = "dtr-r1-jrdb-native-ceiling-v1"
R2_SCHEMA = "dtr-r2-jrdb-native-ceiling-v1"
R3_SCHEMA = "dtr-r3-jrdb-native-ceiling-v1"
CLAIM_CEILING = "PUBLIC_REAL_PRIVILEGED_NATIVE_TRACK_CEILING_ONLY"
LABEL_PREFIXES = ("labels_3d/", "labels/labels_3d/")
HORIZON_S = 3.0
ROUTE_HALF_WIDTH_M = 0.65
MINIMUM_FALSE_ALERT_REDUCTION = 0.40
TIMESTAMP_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class NativeSample:
    frame_index: int
    time_s: float
    forward_m: float
    left_m: float
    radius_m: float
    interpolated: bool

    @property
    def distance_m(self) -> float:
        return math.hypot(self.forward_m, self.left_m)

    @property
    def tube_threshold_m(self) -> float:
        return ROUTE_HALF_WIDTH_M + self.radius_m


@dataclass(frozen=True)
class TruthEvent:
    start_index: int
    end_index: int
    contact_index: int
    category: str


@dataclass(frozen=True)
class AlertSegment:
    start_index: int
    end_index: int


@dataclass
class ArmAccumulator:
    critical_events: int = 0
    recalled_events: int = 0
    alert_segments: int = 0
    false_alert_segments: int = 0
    event_fragments: int = 0
    lead_times_s: list[float] = field(default_factory=list)
    escalated_events: int = 0
    escalation_lead_times_s: list[float] = field(default_factory=list)
    clear_delays_s: list[float] = field(default_factory=list)
    clear_eligible_events: int = 0
    cleared_events: int = 0
    unknown_prediction_frames: int = 0
    evaluated_prediction_frames: int = 0
    evaluated_exposure_s: float = 0.0
    known_negative_exposure_s: float = 0.0
    matched_event_alerts: int = 0
    evaluable_event_alert_segments: int = 0
    fragmented_events: int = 0
    extra_alert_fragments: int = 0
    prematurely_cleared_recalled_events: int = 0
    clear_followup_right_censored_events: int = 0
    ranking_scores: array = field(default_factory=lambda: array("d"))
    ranking_labels: bytearray = field(default_factory=bytearray)
    ranking_evaluable_frames: int = 0
    missing_ranking_score_frames: int = 0
    category_total: dict[str, int] = field(default_factory=dict)
    category_recalled: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "ArmAccumulator") -> None:
        for name in (
            "critical_events",
            "recalled_events",
            "alert_segments",
            "false_alert_segments",
            "event_fragments",
            "escalated_events",
            "clear_eligible_events",
            "cleared_events",
            "unknown_prediction_frames",
            "evaluated_prediction_frames",
            "matched_event_alerts",
            "evaluable_event_alert_segments",
            "fragmented_events",
            "extra_alert_fragments",
            "prematurely_cleared_recalled_events",
            "clear_followup_right_censored_events",
            "ranking_evaluable_frames",
            "missing_ranking_score_frames",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.evaluated_exposure_s += other.evaluated_exposure_s
        self.known_negative_exposure_s += other.known_negative_exposure_s
        self.lead_times_s.extend(other.lead_times_s)
        self.escalation_lead_times_s.extend(other.escalation_lead_times_s)
        self.clear_delays_s.extend(other.clear_delays_s)
        self.ranking_scores.extend(other.ranking_scores)
        self.ranking_labels.extend(other.ranking_labels)
        for category, count in other.category_total.items():
            self.category_total[category] = self.category_total.get(category, 0) + count
        for category, count in other.category_recalled.items():
            self.category_recalled[category] = (
                self.category_recalled.get(category, 0) + count
            )

    def to_dict(self, *, include_escalation: bool = False) -> dict[str, Any]:
        category_recall = {
            category: {
                "events": total,
                "recalled": self.category_recalled.get(category, 0),
                "recall": ratio(self.category_recalled.get(category, 0), total),
            }
            for category, total in sorted(self.category_total.items())
        }
        event_precision = ratio(
            self.matched_event_alerts,
            self.evaluable_event_alert_segments,
        )
        matched_event_recall = ratio(self.matched_event_alerts, self.critical_events)
        result = {
            "critical_events": self.critical_events,
            "critical_events_recalled": self.recalled_events,
            "critical_event_recall": ratio(
                self.recalled_events, self.critical_events
            ),
            "by_event_geometry": category_recall,
            "alert_segments": self.alert_segments,
            "false_alert_segments": self.false_alert_segments,
            "alert_segment_precision": ratio(
                self.alert_segments - self.false_alert_segments,
                self.alert_segments,
            ),
            "event_detection_true_positives": self.matched_event_alerts,
            "event_detection_evaluable_alert_segments": self.evaluable_event_alert_segments,
            "event_detection_ignored_alert_segments": (
                self.alert_segments - self.evaluable_event_alert_segments
            ),
            "event_detection_precision": event_precision,
            "event_detection_recall": matched_event_recall,
            "event_detection_f1": harmonic_f1(
                event_precision,
                matched_event_recall,
            ),
            "evaluated_target_track_minutes": self.evaluated_exposure_s / 60.0,
            "false_alert_segments_per_target_track_minute": ratio(
                self.false_alert_segments,
                self.evaluated_exposure_s / 60.0,
            ),
            "known_negative_target_track_minutes": self.known_negative_exposure_s / 60.0,
            "false_alert_segments_per_known_negative_target_track_minute": ratio(
                self.false_alert_segments,
                self.known_negative_exposure_s / 60.0,
            ),
            "mean_alert_segments_per_critical_event": ratio(
                self.event_fragments, self.critical_events
            ),
            "onset_fragments_in_event_windows": self.event_fragments,
            "mean_onset_fragments_per_matched_event": ratio(
                self.event_fragments,
                self.matched_event_alerts,
            ),
            "fragmented_events": self.fragmented_events,
            "fragmented_event_rate": ratio(
                self.fragmented_events,
                self.matched_event_alerts,
            ),
            "extra_alert_fragments": self.extra_alert_fragments,
            "extra_onsets_per_matched_event": ratio(
                self.extra_alert_fragments,
                self.matched_event_alerts,
            ),
            "median_first_alert_lead_s": median_or_none(self.lead_times_s),
            "clear_eligible_events": self.clear_eligible_events,
            "cleared_events": self.cleared_events,
            "uncleared_clear_eligible_events": (
                self.clear_eligible_events - self.cleared_events
            ),
            "clear_rate": ratio(self.cleared_events, self.clear_eligible_events),
            "median_clear_delay_s": median_or_none(self.clear_delays_s),
            "prematurely_cleared_recalled_events": self.prematurely_cleared_recalled_events,
            "clear_followup_right_censored_events": self.clear_followup_right_censored_events,
            "unknown_prediction_frames": self.unknown_prediction_frames,
            "evaluated_prediction_frames": self.evaluated_prediction_frames,
            "known_prediction_coverage": ratio(
                self.evaluated_prediction_frames
                - self.unknown_prediction_frames,
                self.evaluated_prediction_frames,
            ),
            "frame_auprc": (
                average_precision(self.ranking_scores, self.ranking_labels)
                if self.ranking_scores
                else None
            ),
            "frame_auprc_evaluable_frames": len(self.ranking_scores),
            "frame_auprc_missing_score_frames": self.missing_ranking_score_frames,
            "frame_auprc_score_coverage": ratio(
                len(self.ranking_scores),
                self.ranking_evaluable_frames,
            ),
            "frame_auprc_scope": (
                "score-known frames among countable-event and known-negative frames"
            ),
            "frame_auprc_rankable_frames": self.ranking_evaluable_frames,
            "frame_auprc_ignored_evaluator_known_frames": (
                self.evaluated_prediction_frames - self.ranking_evaluable_frames
            ),
        }
        if include_escalation:
            result.update(
                {
                    "critical_events_escalated": self.escalated_events,
                    "critical_event_escalation_rate": ratio(
                        self.escalated_events, self.critical_events
                    ),
                    "median_escalation_lead_s": median_or_none(
                        self.escalation_lead_times_s
                    ),
                }
            )
        return result


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def harmonic_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Tie-aware frame-level average precision for a continuous decision score."""

    if len(scores) != len(labels):
        raise ValueError("ranking score/label lengths differ")
    positives = sum(labels)
    if not scores or positives == 0:
        return None
    ordered = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    result = 0.0
    index = 0
    while index < len(ordered):
        score = scores[ordered[index]]
        group_true = 0
        group_false = 0
        while index < len(ordered) and scores[ordered[index]] == score:
            label = bool(labels[ordered[index]])
            group_true += int(label)
            group_false += int(not label)
            index += 1
        true_positives += group_true
        false_positives += group_false
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


def median_or_none(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_prefix(label_zip: zipfile.ZipFile) -> str:
    names = label_zip.namelist()
    for prefix in LABEL_PREFIXES:
        if any(name.startswith(prefix) and name.endswith(".json") for name in names):
            return prefix
    raise ValueError("JRDB archive contains no supported labels_3d directory")


def label_sequences(label_zip: zipfile.ZipFile, prefix: str) -> list[str]:
    sequences = []
    for name in label_zip.namelist():
        if name.startswith(prefix) and name.endswith(".json"):
            sequences.append(Path(name).stem)
    return sorted(sequences)


def read_timestamps(
    timestamp_zip: zipfile.ZipFile, sequence: str
) -> dict[int, float]:
    member = f"timestamps/{sequence}/frames_pc.json"
    with timestamp_zip.open(member) as handle:
        payload = json.load(handle)
    timestamps: dict[int, float] = {}
    for row in payload["data"]:
        candidates = [
            item
            for item in row.get("pointclouds", [])
            if item.get("name") == "lower_velodyne"
        ] or list(row.get("pointclouds", []))
        if not candidates:
            continue
        stem = Path(str(candidates[0]["url"])).stem
        timestamps[int(stem)] = float(row["timestamp"])
    return timestamps


def read_tracks(
    label_zip: zipfile.ZipFile,
    prefix: str,
    sequence: str,
    timestamps: dict[int, float],
) -> tuple[dict[str, list[NativeSample]], dict[str, int]]:
    with label_zip.open(f"{prefix}{sequence}.json") as handle:
        payload = json.load(handle)
    tracks: dict[str, list[NativeSample]] = {}
    counts = {
        "label_frames": len(payload["labels"]),
        "objects_seen": 0,
        "objects_used": 0,
        "objects_no_eval": 0,
        "objects_missing_timestamp": 0,
        "interpolated_objects_used": 0,
    }
    for frame_name, people in payload["labels"].items():
        frame_index = int(Path(frame_name).stem)
        time_s = timestamps.get(frame_index)
        for person in people:
            counts["objects_seen"] += 1
            attributes = person.get("attributes", {})
            if bool(attributes.get("no_eval", False)):
                counts["objects_no_eval"] += 1
                continue
            if time_s is None:
                counts["objects_missing_timestamp"] += 1
                continue
            box = person["box"]
            forward_m = float(box["cx"])
            left_m = float(box["cy"])
            width_m = float(box.get("w", 0.60))
            length_m = float(box.get("l", 0.60))
            if not all(
                math.isfinite(value)
                for value in (time_s, forward_m, left_m, width_m, length_m)
            ):
                continue
            radius_m = max(0.15, 0.5 * max(width_m, length_m))
            interpolated = bool(attributes.get("interpolated", False))
            tracks.setdefault(str(person["label_id"]), []).append(
                NativeSample(
                    frame_index=frame_index,
                    time_s=time_s,
                    forward_m=forward_m,
                    left_m=left_m,
                    radius_m=radius_m,
                    interpolated=interpolated,
                )
            )
            counts["objects_used"] += 1
            counts["interpolated_objects_used"] += int(interpolated)
    for samples in tracks.values():
        samples.sort(key=lambda item: (item.frame_index, item.time_s))
    return tracks, counts


def contiguous_segments(samples: Sequence[NativeSample]) -> Iterable[list[NativeSample]]:
    current: list[NativeSample] = []
    for sample in samples:
        if current and (
            sample.frame_index != current[-1].frame_index + 1
            or sample.time_s <= current[-1].time_s
        ):
            if current:
                yield current
            current = []
        current.append(sample)
    if current:
        yield current


def causal_frames(track_id: str, samples: Sequence[NativeSample]) -> list[CausalFrame]:
    origin = samples[0].time_s
    pose = EgoPose(0.0, 0.0, 0.0, 0.0)
    return [
        CausalFrame(
            time_s=sample.time_s - origin,
            ego_pose=pose,
            observations=(
                Observation(
                    track_id=track_id,
                    forward_m=sample.forward_m,
                    left_m=sample.left_m,
                    radius_m=sample.radius_m,
                ),
            ),
            person_detection_count=1,
        )
        for sample in samples
    ]


def future_hits(samples: Sequence[NativeSample]) -> tuple[list[bool | None], list[int | None]]:
    truth: list[bool | None] = []
    contact_indices: list[int | None] = []
    final_time = samples[-1].time_s
    for index, sample in enumerate(samples):
        hit: int | None = None
        future_index = index
        while (
            future_index < len(samples)
            and samples[future_index].time_s - sample.time_s <= HORIZON_S + 1e-9
        ):
            future = samples[future_index]
            if future.distance_m <= future.tube_threshold_m:
                hit = future_index
                break
            future_index += 1
        has_full_future = (
            final_time - sample.time_s
            >= HORIZON_S - TIMESTAMP_TOLERANCE_S
        )
        if hit is not None:
            truth.append(True)
            contact_indices.append(hit)
        elif has_full_future:
            truth.append(False)
            contact_indices.append(None)
        else:
            truth.append(None)
            contact_indices.append(None)
    return truth, contact_indices


def classify_event(
    samples: Sequence[NativeSample], start_index: int, contact_index: int
) -> str:
    start = samples[start_index]
    contact = samples[contact_index]
    if (
        abs(start.left_m) > start.tube_threshold_m
        and abs(contact.left_m) <= contact.tube_threshold_m
    ):
        return "lateral_crossing"
    if (
        start.forward_m > start.tube_threshold_m
        and abs(start.left_m) <= start.tube_threshold_m
        and contact.forward_m < start.forward_m
    ):
        return "oncoming_corridor"
    return "other_close_approach"


def truth_events(
    samples: Sequence[NativeSample],
    truth: Sequence[bool | None],
    contacts: Sequence[int | None],
    minimum_history_s: float,
) -> tuple[list[TruthEvent], list[bool]]:
    known = [
        value is not None
        and sample.time_s - samples[0].time_s + 1e-9 >= minimum_history_s
        for sample, value in zip(samples, truth)
    ]
    events: list[TruthEvent] = []
    index = 0
    while index < len(samples):
        if not known[index] or truth[index] is not True:
            index += 1
            continue
        # A track that first appears inside an event lacks pre-event context.
        if index == 0 or not known[index - 1]:
            while index < len(samples) and known[index] and truth[index] is True:
                index += 1
            continue
        if truth[index - 1] is True:
            index += 1
            continue
        start = index
        while (
            index + 1 < len(samples)
            and known[index + 1]
            and truth[index + 1] is True
        ):
            index += 1
        end = index
        contact = contacts[start]
        if contact is not None:
            events.append(
                TruthEvent(
                    start_index=start,
                    end_index=end,
                    contact_index=contact,
                    category=classify_event(samples, start, contact),
                )
            )
        index += 1
    return events, known


def alert_segments(predictions: Sequence[Prediction]) -> list[AlertSegment]:
    segments: list[AlertSegment] = []
    start: int | None = None
    for index, prediction in enumerate(predictions):
        if prediction.signal is Signal.ONSET:
            if start is not None:
                segments.append(AlertSegment(start, index - 1))
            start = index
        elif prediction.signal is Signal.CLEAR and start is not None:
            segments.append(AlertSegment(start, max(start, index - 1)))
            start = None
    if start is not None:
        segments.append(AlertSegment(start, len(predictions) - 1))
    return segments


def overlaps(segment: AlertSegment, start: int, end: int) -> bool:
    return segment.start_index <= end and segment.end_index >= start


def maximum_event_alert_assignment(
    segments: Sequence[AlertSegment],
    events: Sequence[TruthEvent],
) -> dict[int, int]:
    """Map event index to alert index using ONSET-in-window one-to-one matching."""

    segment_to_event: dict[int, int] = {}

    def assign(event_index: int, visited: set[int]) -> bool:
        event = events[event_index]
        for segment_index, segment in enumerate(segments):
            if (
                segment_index in visited
                or not event.start_index
                <= segment.start_index
                <= event.contact_index
            ):
                continue
            visited.add(segment_index)
            previous = segment_to_event.get(segment_index)
            if previous is None or assign(previous, visited):
                segment_to_event[segment_index] = event_index
                return True
        return False

    for event_index in range(len(events)):
        assign(event_index, set())
    return {
        event_index: segment_index
        for segment_index, event_index in segment_to_event.items()
    }


def maximum_event_alert_matching(
    segments: Sequence[AlertSegment],
    events: Sequence[TruthEvent],
) -> int:
    return len(maximum_event_alert_assignment(segments, events))


def continuous_decision_score(prediction: Prediction) -> float | None:
    value = prediction.diagnostic.get("decision_score")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def score_arm(
    samples: Sequence[NativeSample],
    predictions: Sequence[Prediction],
    events: Sequence[TruthEvent],
    known: Sequence[bool],
    truth: Sequence[bool | None],
    clear_grace_s: float = 0.50,
) -> ArmAccumulator:
    result = ArmAccumulator()
    segments = [
        segment
        for segment in alert_segments(predictions)
        if any(known[index] for index in range(segment.start_index, segment.end_index + 1))
    ]
    result.alert_segments = len(segments)
    result.false_alert_segments = sum(
        not any(
            known[index] and truth[index] is True
            for index in range(segment.start_index, segment.end_index + 1)
        )
        for segment in segments
    )
    countable_positive = {
        index
        for event in events
        for index in range(event.start_index, event.end_index + 1)
    }
    ignored_positive = {
        index
        for index, (is_known, truth_value) in enumerate(zip(known, truth))
        if is_known and truth_value is True and index not in countable_positive
    }
    precision_segments = [
        segment
        for segment in segments
        if known[segment.start_index]
        and segment.start_index not in ignored_positive
    ]
    result.evaluable_event_alert_segments = len(precision_segments)
    event_assignment = maximum_event_alert_assignment(
        precision_segments,
        events,
    )
    result.matched_event_alerts = len(event_assignment)
    result.evaluated_prediction_frames = sum(known)
    result.unknown_prediction_frames = sum(
        is_known and prediction.raw_alert is None
        for is_known, prediction in zip(known, predictions)
    )
    known_indices = [index for index, is_known in enumerate(known) if is_known]
    if len(known_indices) > 1:
        result.evaluated_exposure_s = (
            samples[known_indices[-1]].time_s - samples[known_indices[0]].time_s
        )
    result.known_negative_exposure_s = sum(
        samples[index + 1].time_s - samples[index].time_s
        for index in range(len(samples) - 1)
        if known[index]
        and known[index + 1]
        and truth[index] is False
        and truth[index + 1] is False
    )
    for index, (is_known, truth_value, prediction) in enumerate(
        zip(known, truth, predictions)
    ):
        if not is_known or index in ignored_positive:
            continue
        result.ranking_evaluable_frames += 1
        if prediction.raw_alert is None:
            result.missing_ranking_score_frames += 1
            continue
        score = continuous_decision_score(prediction)
        if score is None:
            result.missing_ranking_score_frames += 1
            continue
        result.ranking_scores.append(score)
        result.ranking_labels.append(truth_value is True)

    for event_index, event in enumerate(events):
        result.critical_events += 1
        result.category_total[event.category] = (
            result.category_total.get(event.category, 0) + 1
        )
        warning_segments = [
            segment
            for segment in segments
            if overlaps(segment, event.start_index, event.contact_index)
        ]
        event_segments = [
            segment
            for segment in segments
            if event.start_index <= segment.start_index <= event.contact_index
        ]
        result.event_fragments += len(event_segments)
        if len(event_segments) > 1:
            result.fragmented_events += 1
            result.extra_alert_fragments += len(event_segments) - 1
        if warning_segments:
            result.recalled_events += 1
            result.category_recalled[event.category] = (
                result.category_recalled.get(event.category, 0) + 1
            )
            first_alert = min(item.start_index for item in warning_segments)
            result.lead_times_s.append(
                samples[event.contact_index].time_s - samples[first_alert].time_s
            )
        escalations = [
            index
            for index in range(event.start_index, event.contact_index + 1)
            if known[index] and predictions[index].signal is Signal.ESCALATE
        ]
        if escalations:
            result.escalated_events += 1
            result.escalation_lead_times_s.append(
                samples[event.contact_index].time_s - samples[escalations[0]].time_s
            )

        next_start = (
            events[event_index + 1].start_index
            if event_index + 1 < len(events)
            else len(samples)
        )
        matched_segment_index = event_assignment.get(event_index)
        if matched_segment_index is None:
            continue
        matched_segment = precision_segments[matched_segment_index]
        active_through_event = any(
            segment.end_index >= event.end_index for segment in (matched_segment,)
        )
        if not active_through_event:
            result.prematurely_cleared_recalled_events += 1
            continue
        negative_followup: list[int] = []
        followup_index = event.end_index + 1
        while (
            followup_index < next_start
            and known[followup_index]
            and truth[followup_index] is False
        ):
            negative_followup.append(followup_index)
            followup_index += 1
        if not negative_followup:
            result.clear_followup_right_censored_events += 1
            continue
        clear_candidates = [
            index
            for index in negative_followup
            if predictions[index].signal is Signal.CLEAR
        ]
        followup_span_s = (
            samples[negative_followup[-1]].time_s
            - samples[negative_followup[0]].time_s
        )
        if followup_span_s + 1e-9 >= clear_grace_s:
            result.clear_eligible_events += 1
            if clear_candidates:
                result.cleared_events += 1
                result.clear_delays_s.append(
                    samples[clear_candidates[0]].time_s
                    - samples[negative_followup[0]].time_s
                )
        else:
            result.clear_followup_right_censored_events += 1
    return result


def evaluate_track_segment(
    track_id: str,
    samples: Sequence[NativeSample],
    config: DTRConfig,
    arms: Sequence[Arm | R3Arm] = (Arm.B2_RADIAL_TTC, Arm.C_ROUTE_INTERSECTION),
) -> tuple[dict[Arm | R3Arm, ArmAccumulator], int, float]:
    if (
        len(samples) < 2
        or samples[-1].time_s - samples[0].time_s
        < config.minimum_track_span_s + HORIZON_S
    ):
        return {}, 0, 0.0
    frames = causal_frames(track_id, samples)
    predictions = {
        arm: (
                run_r3_arm(
                    frames,
                    arm,
                    r0_config=config,
                    guard_frames=(
                        frames
                        if arm is R3Arm.C_CURVED_DISTRIBUTIONAL_GUARDED
                        else None
                    ),
                )
                if isinstance(arm, R3Arm)
                else run_r1_arm(frames)
                if arm is Arm.D_R1_OCCUPANCY_CONSENSUS
                else run_r2_arm(frames, config)
                if arm is Arm.E_R2_GUARDED_CONSENSUS
                else run_arm(frames, arm, config)
            )
        for arm in arms
    }
    truth, contacts = future_hits(samples)
    events, known = truth_events(
        samples, truth, contacts, config.minimum_track_span_s
    )
    if not any(known):
        return {}, 0, 0.0
    scored = {
        arm: score_arm(
            samples,
            predictions[arm],
            events,
            known,
            truth,
            clear_grace_s=config.clear_grace_s,
        )
        for arm in arms
    }
    known_indices = [index for index, value in enumerate(known) if value]
    exposure_s = (
        samples[known_indices[-1]].time_s - samples[known_indices[0]].time_s
        if len(known_indices) > 1
        else 0.0
    )
    return scored, len(events), exposure_s


def arm_recall(metrics: dict[str, Any], category: str | None = None) -> float | None:
    if category is None:
        return metrics["critical_event_recall"]
    return metrics["by_event_geometry"].get(category, {}).get("recall")


def nondecreasing(left: float | None, right: float | None) -> bool | None:
    if left is None or right is None:
        return None
    return left + 1e-12 >= right


def decision(pooled: dict[str, dict[str, Any]]) -> dict[str, Any]:
    b2 = pooled[Arm.B2_RADIAL_TTC.value]
    challenger = pooled[Arm.C_ROUTE_INTERSECTION.value]
    if b2["false_alert_segments"]:
        reduction = (
            b2["false_alert_segments"] - challenger["false_alert_segments"]
        ) / b2["false_alert_segments"]
    else:
        reduction = None
    overall = nondecreasing(
        arm_recall(challenger), arm_recall(b2)
    )
    category_checks = {
        category: nondecreasing(
            arm_recall(challenger, category), arm_recall(b2, category)
        )
        for category in ("lateral_crossing", "oncoming_corridor")
    }
    evaluable = (
        reduction is not None
        and overall is not None
        and all(value is not None for value in category_checks.values())
    )
    passed = bool(
        evaluable
        and overall
        and all(category_checks.values())
        and reduction >= MINIMUM_FALSE_ALERT_REDUCTION
    )
    if not evaluable:
        status = "PRIVILEGED_CEILING_NOT_EVALUABLE"
    elif passed:
        status = "PRIVILEGED_CEILING_GO_TO_RGB_DETECTOR_TRACKER"
    else:
        status = "PRIVILEGED_CEILING_STOP_BEFORE_DETECTOR"
    return {
        "status": status,
        "passed": passed,
        "primary_comparator": Arm.B2_RADIAL_TTC.value,
        "primary_challenger": Arm.C_ROUTE_INTERSECTION.value,
        "critical_recall_non_decrease": overall,
        "critical_recall_delta": (
            None
            if arm_recall(challenger) is None or arm_recall(b2) is None
            else arm_recall(challenger) - arm_recall(b2)
        ),
        "category_recall_non_decrease": category_checks,
        "false_alert_segment_reduction_fraction": reduction,
        "required_false_alert_segment_reduction_fraction": (
            MINIMUM_FALSE_ALERT_REDUCTION
        ),
    }


def r1_dominance_decision(pooled: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Report direct R1-vs-R0 dominance without inventing a new tuned gate."""

    if Arm.D_R1_OCCUPANCY_CONSENSUS.value not in pooled:
        return None
    comparator = pooled[Arm.C_ROUTE_INTERSECTION.value]
    challenger = pooled[Arm.D_R1_OCCUPANCY_CONSENSUS.value]
    recall_ok = nondecreasing(arm_recall(challenger), arm_recall(comparator))
    false_delta = (
        comparator["false_alert_segments"] - challenger["false_alert_segments"]
    )
    false_reduction = (
        false_delta / comparator["false_alert_segments"]
        if comparator["false_alert_segments"]
        else None
    )
    return {
        "status": (
            "R1_DOMINATES_R0"
            if recall_ok is True and false_delta > 0
            else "R1_DOES_NOT_DOMINATE_R0"
        ),
        "primary_comparator": Arm.C_ROUTE_INTERSECTION.value,
        "primary_challenger": Arm.D_R1_OCCUPANCY_CONSENSUS.value,
        "critical_recall_non_decrease": recall_ok,
        "critical_recall_delta": (
            None
            if arm_recall(challenger) is None or arm_recall(comparator) is None
            else arm_recall(challenger) - arm_recall(comparator)
        ),
        "false_alert_segment_delta": false_delta,
        "false_alert_segment_reduction_fraction": false_reduction,
    }


def successor_comparison(
    pooled: dict[str, dict[str, Any]],
    challenger: Arm | R3Arm,
    comparator: Arm | R3Arm,
) -> dict[str, Any] | None:
    if challenger.value not in pooled or comparator.value not in pooled:
        return None
    left = pooled[challenger.value]
    right = pooled[comparator.value]
    left_recall = left["critical_event_recall"]
    right_recall = right["critical_event_recall"]
    recall_delta = (
        None
        if left_recall is None or right_recall is None
        else left_recall - right_recall
    )
    false_delta = right["false_alert_segments"] - left["false_alert_segments"]
    return {
        "status": (
            "DOMINATES"
            if recall_delta is not None and recall_delta >= -1e-12 and false_delta > 0
            else "DOES_NOT_DOMINATE"
        ),
        "challenger": challenger.value,
        "comparator": comparator.value,
        "critical_recall_delta": recall_delta,
        "false_alert_segment_delta": false_delta,
        "false_alert_segment_reduction_fraction": ratio(
            false_delta, right["false_alert_segments"]
        ),
        "clear_rate_delta": (
            None
            if left["clear_rate"] is None or right["clear_rate"] is None
            else left["clear_rate"] - right["clear_rate"]
        ),
    }


def evaluate(
    label_zip_path: Path,
    timestamp_zip_path: Path,
    selected_sequences: set[str] | None,
    include_r1: bool = False,
    include_r2: bool = False,
    include_r3: bool = False,
) -> dict[str, Any]:
    include_r2_effective = include_r2 or include_r3
    include_successor = include_r1 or include_r2_effective
    config = DTRConfig(
        route_horizon_s=HORIZON_S,
        route_half_width_m=ROUTE_HALF_WIDTH_M,
        nominal_wearer_speed_mps=0.0,
    )
    arms = (
        Arm.B2_RADIAL_TTC,
        Arm.C_ROUTE_INTERSECTION,
        *((Arm.D_R1_OCCUPANCY_CONSENSUS,) if include_successor else ()),
        *((Arm.E_R2_GUARDED_CONSENSUS,) if include_r2_effective else ()),
        *(tuple(R3Arm) if include_r3 else ()),
    )
    pooled = {arm: ArmAccumulator() for arm in arms}
    sequence_results = []
    totals = {
        "sequences": 0,
        "label_frames": 0,
        "objects_seen": 0,
        "objects_used": 0,
        "objects_no_eval": 0,
        "objects_missing_timestamp": 0,
        "interpolated_objects_used": 0,
        "native_identities": 0,
        "contiguous_track_segments": 0,
        "evaluable_track_segments": 0,
        "critical_events": 0,
        "track_segment_exposure_s": 0.0,
    }

    with zipfile.ZipFile(label_zip_path) as label_zip, zipfile.ZipFile(
        timestamp_zip_path
    ) as timestamp_zip:
        prefix = label_prefix(label_zip)
        sequences = label_sequences(label_zip, prefix)
        if selected_sequences is not None:
            missing = selected_sequences.difference(sequences)
            if missing:
                raise ValueError(f"unknown sequence(s): {sorted(missing)}")
            sequences = [item for item in sequences if item in selected_sequences]
        for sequence_index, sequence in enumerate(sequences, start=1):
            timestamps = read_timestamps(timestamp_zip, sequence)
            tracks, source_counts = read_tracks(
                label_zip, prefix, sequence, timestamps
            )
            sequence_arms = {arm: ArmAccumulator() for arm in arms}
            track_segments = 0
            evaluable_segments = 0
            critical_events = 0
            exposure_s = 0.0
            for track_id, track_samples in tracks.items():
                for segment_index, segment in enumerate(
                    contiguous_segments(track_samples)
                ):
                    track_segments += 1
                    scored, event_count, segment_exposure_s = evaluate_track_segment(
                        f"{sequence}/{track_id}/{segment_index}",
                        segment,
                        config,
                        arms,
                    )
                    if not scored:
                        continue
                    evaluable_segments += 1
                    critical_events += event_count
                    exposure_s += segment_exposure_s
                    for arm, metrics in scored.items():
                        sequence_arms[arm].merge(metrics)
                        pooled[arm].merge(metrics)
            sequence_result = {
                "sequence": sequence,
                "source": source_counts,
                "native_identities": len(tracks),
                "contiguous_track_segments": track_segments,
                "evaluable_track_segments": evaluable_segments,
                "critical_events": critical_events,
                "track_segment_exposure_s": exposure_s,
                "arms": {
                    arm.value: metrics.to_dict(include_escalation=include_successor)
                    for arm, metrics in sequence_arms.items()
                },
            }
            sequence_results.append(sequence_result)
            totals["sequences"] += 1
            for name, value in source_counts.items():
                totals[name] += value
            totals["native_identities"] += len(tracks)
            totals["contiguous_track_segments"] += track_segments
            totals["evaluable_track_segments"] += evaluable_segments
            totals["critical_events"] += critical_events
            totals["track_segment_exposure_s"] += exposure_s
            print(
                f"[{sequence_index}/{len(sequences)}] {sequence}: "
                f"events={critical_events}, tracks={evaluable_segments}",
                flush=True,
            )

    pooled_dict = {
        arm.value: metrics.to_dict(include_escalation=include_successor)
        for arm, metrics in pooled.items()
    }
    result = {
        "schema_version": (
            R3_SCHEMA
            if include_r3
            else R2_SCHEMA
            if include_r2
            else R1_SCHEMA
            if include_r1
            else SCHEMA
        ),
        "claim_ceiling": CLAIM_CEILING,
        "source": {
            "dataset": "JRDB public caller-provided labels/timestamps split",
            "labels": str(label_zip_path.resolve()),
            "timestamps": str(timestamp_zip_path.resolve()),
            "labels_sha256": sha256_file(label_zip_path),
            "timestamps_sha256": sha256_file(timestamp_zip_path),
            "coordinate_contract": (
                "JRDB lidar coordinates relative to the robot; "
                "box.cx=forward_m, box.cy=left_m"
            ),
            "input_access": (
                "ordered native identity and 3-D center through current frame only; "
                "source interpolation means this is not runtime-causal evidence"
            ),
            "truth_access": (
                "future native annotation used only by evaluator"
            ),
            "odometry_note": (
                "robot-relative tracks already contain combined ego/person relative "
                "motion; nominal wearer velocity is zero to avoid double compensation"
            ),
        },
        "configuration": {
            "route_horizon_s": HORIZON_S,
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "person_radius": "0.5 * max(source box width, source box length)",
            "track_window_s": config.track_window_s,
            "minimum_track_span_s": config.minimum_track_span_s,
            "clear_grace_s": config.clear_grace_s,
            "minimum_false_alert_reduction_fraction": (
                MINIMUM_FALSE_ALERT_REDUCTION
            ),
            "r1": FROZEN_R1_CONFIG.to_dict() if include_successor else None,
            "r2": FROZEN_R2_CONFIG.to_dict() if include_r2_effective else None,
            "r3": FROZEN_R3_CONFIG.to_dict() if include_r3 else None,
            "r3_route_authority": (
                "robot-relative identity-pose degeneration only; no synchronized ego yaw-rate"
                if include_r3
                else None
            ),
            "r3_source_evaluability": (
                {
                    R3Arm.A_CURVED_ROBUST_CV.value: "NOT_EVALUABLE_CURVED_ROUTE_AUTHORITY_ABSENT",
                    R3Arm.B_STRAIGHT_DISTRIBUTIONAL.value: "EVALUABLE_IDENTITY_RELATIVE_DIAGNOSTIC",
                    R3Arm.C_CURVED_DISTRIBUTIONAL_GUARDED.value: "EVALUABLE_STRAIGHT_DEGENERATION_WITH_GUARD_ONLY",
                }
                if include_r3
                else None
            ),
            "r3_ablation_interpretation": (
                "coupled-arm performance only; no single-component causal attribution"
                if include_r3
                else None
            ),
        },
        "coverage": totals,
        "pooled": pooled_dict,
        "decision": decision(pooled_dict),
        "r1_decision": r1_dominance_decision(pooled_dict),
        "r2_vs_r0": successor_comparison(
            pooled_dict,
            Arm.E_R2_GUARDED_CONSENSUS,
            Arm.C_ROUTE_INTERSECTION,
        ),
        "r2_vs_r1": successor_comparison(
            pooled_dict,
            Arm.E_R2_GUARDED_CONSENSUS,
            Arm.D_R1_OCCUPANCY_CONSENSUS,
        ),
        "r3_vs_r2": {
            arm.value: successor_comparison(
                pooled_dict,
                arm,
                Arm.E_R2_GUARDED_CONSENSUS,
            )
            for arm in R3Arm
        }
        if include_r3
        else None,
        "by_sequence": sequence_results,
        "limitations": [
            "JRDB 3-D person boxes are privileged source annotations and may be interpolated.",
            "The processed labels/timestamps release has no synchronized ego pose, so this is robot-relative closure rather than exact wearer-route evaluation.",
            "R3-A/C therefore exercise only the identity-pose straight degeneration here; JRDB does not contribute curved-route evidence.",
            "Geometry-derived route entry is not human-authored alertability or safety truth.",
            "No RGB detector, tracker, phone metric depth, Android runtime, or user study is evaluated.",
            "False-alert rates are per target-track exposure, not merged user wall-clock alerts per minute.",
            "This is a retrospective ceiling on a previously inspected public cohort, not a fresh holdout confirmation.",
        ],
    }
    if not include_successor:
        result["configuration"].pop("r1")
        result.pop("r1_decision")
    if not include_r2_effective:
        result["configuration"].pop("r2")
        result.pop("r2_vs_r0")
        result.pop("r2_vs_r1")
    if not include_r3:
        result["configuration"].pop("r3")
        result["configuration"].pop("r3_route_authority")
        result["configuration"].pop("r3_source_evaluability")
        result["configuration"].pop("r3_ablation_interpretation")
        result.pop("r3_vs_r2")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--timestamps-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-r1",
        action="store_true",
        help="Add the fixed robust occupancy-consensus challenger.",
    )
    parser.add_argument(
        "--include-r2",
        action="store_true",
        help="Add R1 plus the fixed half-horizon guarded R2 successor.",
    )
    parser.add_argument(
        "--include-r3",
        action="store_true",
        help="Add R1/R2 plus the fixed R3 A/B/C ablation.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        help="Evaluate only the named sequence; repeat for more than one.",
    )
    args = parser.parse_args()
    if not args.labels_zip.is_file() or not args.timestamps_zip.is_file():
        raise FileNotFoundError("JRDB labels/timestamps zip is missing")
    result = evaluate(
        args.labels_zip,
        args.timestamps_zip,
        set(args.sequence) if args.sequence else None,
        include_r1=args.include_r1,
        include_r2=args.include_r2,
        include_r3=args.include_r3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "coverage": result["coverage"],
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
