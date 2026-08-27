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


SCHEMA = "dtr-r0-jrdb-native-ceiling-v1"
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
    clear_delays_s: list[float] = field(default_factory=list)
    clear_eligible_events: int = 0
    cleared_events: int = 0
    unknown_prediction_frames: int = 0
    evaluated_prediction_frames: int = 0
    category_total: dict[str, int] = field(default_factory=dict)
    category_recalled: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "ArmAccumulator") -> None:
        for name in (
            "critical_events",
            "recalled_events",
            "alert_segments",
            "false_alert_segments",
            "event_fragments",
            "clear_eligible_events",
            "cleared_events",
            "unknown_prediction_frames",
            "evaluated_prediction_frames",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.lead_times_s.extend(other.lead_times_s)
        self.clear_delays_s.extend(other.clear_delays_s)
        for category, count in other.category_total.items():
            self.category_total[category] = self.category_total.get(category, 0) + count
        for category, count in other.category_recalled.items():
            self.category_recalled[category] = (
                self.category_recalled.get(category, 0) + count
            )

    def to_dict(self) -> dict[str, Any]:
        category_recall = {
            category: {
                "events": total,
                "recalled": self.category_recalled.get(category, 0),
                "recall": ratio(self.category_recalled.get(category, 0), total),
            }
            for category, total in sorted(self.category_total.items())
        }
        return {
            "critical_events": self.critical_events,
            "critical_events_recalled": self.recalled_events,
            "critical_event_recall": ratio(
                self.recalled_events, self.critical_events
            ),
            "by_event_geometry": category_recall,
            "alert_segments": self.alert_segments,
            "false_alert_segments": self.false_alert_segments,
            "mean_alert_segments_per_critical_event": ratio(
                self.event_fragments, self.critical_events
            ),
            "median_first_alert_lead_s": median_or_none(self.lead_times_s),
            "clear_eligible_events": self.clear_eligible_events,
            "cleared_events": self.cleared_events,
            "clear_rate": ratio(self.cleared_events, self.clear_eligible_events),
            "median_clear_delay_s": median_or_none(self.clear_delays_s),
            "unknown_prediction_frames": self.unknown_prediction_frames,
            "evaluated_prediction_frames": self.evaluated_prediction_frames,
            "known_prediction_coverage": ratio(
                self.evaluated_prediction_frames
                - self.unknown_prediction_frames,
                self.evaluated_prediction_frames,
            ),
        }


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


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


def score_arm(
    samples: Sequence[NativeSample],
    predictions: Sequence[Prediction],
    events: Sequence[TruthEvent],
    known: Sequence[bool],
    truth: Sequence[bool | None],
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
    result.evaluated_prediction_frames = sum(known)
    result.unknown_prediction_frames = sum(
        is_known and prediction.raw_alert is None
        for is_known, prediction in zip(known, predictions)
    )

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
            if overlaps(segment, event.start_index, event.end_index)
        ]
        result.event_fragments += len(event_segments)
        if warning_segments:
            result.recalled_events += 1
            result.category_recalled[event.category] = (
                result.category_recalled.get(event.category, 0) + 1
            )
            first_alert = min(item.start_index for item in warning_segments)
            result.lead_times_s.append(
                samples[event.contact_index].time_s - samples[first_alert].time_s
            )

        next_start = (
            events[event_index + 1].start_index
            if event_index + 1 < len(events)
            else len(samples)
        )
        clear_candidates = [
            index
            for index in range(event.end_index + 1, next_start)
            if known[index] and predictions[index].signal is Signal.CLEAR
        ]
        if any(known[index] for index in range(event.end_index + 1, next_start)):
            result.clear_eligible_events += 1
            if clear_candidates:
                result.cleared_events += 1
                result.clear_delays_s.append(
                    samples[clear_candidates[0]].time_s - samples[event.end_index].time_s
                )
    return result


def evaluate_track_segment(
    track_id: str,
    samples: Sequence[NativeSample],
    config: DTRConfig,
) -> tuple[dict[Arm, ArmAccumulator], int, float]:
    if (
        len(samples) < 2
        or samples[-1].time_s - samples[0].time_s
        < config.minimum_track_span_s + HORIZON_S
    ):
        return {}, 0, 0.0
    frames = causal_frames(track_id, samples)
    truth, contacts = future_hits(samples)
    events, known = truth_events(
        samples, truth, contacts, config.minimum_track_span_s
    )
    if not any(known):
        return {}, 0, 0.0
    scored = {
        arm: score_arm(
            samples,
            run_arm(frames, arm, config),
            events,
            known,
            truth,
        )
        for arm in (Arm.B2_RADIAL_TTC, Arm.C_ROUTE_INTERSECTION)
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


def evaluate(
    label_zip_path: Path,
    timestamp_zip_path: Path,
    selected_sequences: set[str] | None,
) -> dict[str, Any]:
    config = DTRConfig(
        route_horizon_s=HORIZON_S,
        route_half_width_m=ROUTE_HALF_WIDTH_M,
        nominal_wearer_speed_mps=0.0,
    )
    pooled = {
        arm: ArmAccumulator()
        for arm in (Arm.B2_RADIAL_TTC, Arm.C_ROUTE_INTERSECTION)
    }
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
            sequence_arms = {
                arm: ArmAccumulator()
                for arm in (Arm.B2_RADIAL_TTC, Arm.C_ROUTE_INTERSECTION)
            }
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
                        f"{sequence}/{track_id}/{segment_index}", segment, config
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
                    arm.value: metrics.to_dict()
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

    pooled_dict = {arm.value: metrics.to_dict() for arm, metrics in pooled.items()}
    return {
        "schema_version": SCHEMA,
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
        },
        "coverage": totals,
        "pooled": pooled_dict,
        "decision": decision(pooled_dict),
        "by_sequence": sequence_results,
        "limitations": [
            "JRDB 3-D person boxes are privileged source annotations and may be interpolated.",
            "The processed labels/timestamps release has no synchronized ego pose, so this is robot-relative closure rather than exact wearer-route evaluation.",
            "Geometry-derived route entry is not human-authored alertability or safety truth.",
            "No RGB detector, tracker, phone metric depth, Android runtime, or user study is evaluated.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--timestamps-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
