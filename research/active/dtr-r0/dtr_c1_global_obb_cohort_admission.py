"""Admit a fresh JRDB cohort under the DTR-C0 global OBB contract.

C1 is metadata-only.  It reads native JRDB 3-D labels and image timestamps,
never RGB, LiDAR, detector output, or an algorithm prediction.  Every sequence
is converted into a wearer-global realized-future truth timeline:

    CONTACT   any native future OBB intersects the 0.65 m route body;
    PROXIMITY no OBB contact, but a circularized native envelope intersects;
    CLEAR     a complete 3 s future contains neither;
    UNKNOWN   neither is observed and the future is right-censored.

The consumed C0 sequence is excluded in full.  Remaining sequences are ordered
lexicographically, and the shortest prefix reaching the frozen preferred
denominators is selected.  If the preferred target is unavailable, the
shortest prefix reaching the frozen minimum is selected.  No truth metric is
used to reorder sequences.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from coda_static_ceiling import point_to_box_clearance
from jrdb_rgb_bridge import (
    BASE_LINK_FROM_LOGICAL_RGB360_X_M,
    BASE_LINK_FROM_LOGICAL_RGB360_Y_M,
    HORIZON_S,
    ROUTE_HALF_WIDTH_M,
)


SCHEMA = "blindassist-dtr-c1-global-obb-cohort-admission-v1"
ROSTER_SCHEMA = "blindassist-dtr-c1-fresh-global-obb-roster-v1"
CONTACT = "CONTACT"
PROXIMITY = "PROXIMITY"
CLEAR = "CLEAR"
UNKNOWN = "UNKNOWN"
CONSUMED_SEQUENCES = ("packard-poster-session-2019-03-20_1",)

MINIMUM_BOUNDED_CONTACT_EVENTS = 12
MINIMUM_UNIQUE_RESPONSIBLE_EVENTS = 6
MINIMUM_KNOWN_NON_CONTACT_S = 60.0
PREFERRED_BOUNDED_CONTACT_EVENTS = 20
PREFERRED_UNIQUE_RESPONSIBLE_EVENTS = 10
PREFERRED_KNOWN_NON_CONTACT_S = 120.0

STATUS_ADMITTED_PREFERRED = "DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_METADATA_ONLY"
STATUS_ADMITTED_MINIMUM = "DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_MINIMUM_ONLY"
STATUS_INSUFFICIENT = "JRDB_GLOBAL_OBB_CONTACT_COHORT_INSUFFICIENT"


@dataclass(frozen=True)
class NativeBox:
    label_id: str
    center_forward_m: float
    center_left_m: float
    length_m: float
    width_m: float
    yaw_ego_rad: float


@dataclass(frozen=True)
class FrameInterval:
    first_index: int
    last_index: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(partial, path)


def _archive_sequences(labels: zipfile.ZipFile, timestamps: zipfile.ZipFile) -> list[str]:
    label_sequences = {
        PurePosixPath(name).stem
        for name in labels.namelist()
        if name.startswith("labels/labels_3d/")
        and name.endswith(".json")
        and not PurePosixPath(name).name.startswith(".")
    }
    timestamp_sequences = {
        PurePosixPath(name).parts[1]
        for name in timestamps.namelist()
        if name.startswith("timestamps/") and name.endswith("/frames_img.json")
    }
    require(label_sequences == timestamp_sequences, "label_timestamp_sequence_mismatch")
    return sorted(label_sequences)


def _load_timestamps(bundle: zipfile.ZipFile, sequence: str) -> dict[int, float]:
    payload = json.loads(bundle.read(f"timestamps/{sequence}/frames_img.json"))
    output: dict[int, float] = {}
    for row in payload["data"]:
        cameras = [item for item in row["cameras"] if item["name"] == "stitched_image0"]
        require(len(cameras) == 1, f"stitched_timestamp_not_unique:{sequence}")
        frame = int(PurePosixPath(cameras[0]["url"]).stem)
        output[frame] = float(cameras[0]["timestamp"])
    frames = sorted(output)
    require(bool(frames), f"empty_timestamps:{sequence}")
    require(frames == list(range(frames[0], frames[-1] + 1)), f"noncontiguous_frames:{sequence}")
    require(
        all(output[right] > output[left] for left, right in zip(frames, frames[1:])),
        f"nonmonotonic_timestamps:{sequence}",
    )
    return output


def _load_boxes(bundle: zipfile.ZipFile, sequence: str) -> dict[int, list[NativeBox]]:
    values = json.loads(bundle.read(f"labels/labels_3d/{sequence}.json"))["labels"]
    output: dict[int, list[NativeBox]] = {}
    for filename, items in values.items():
        frame = int(PurePosixPath(filename).stem)
        boxes = []
        for item in items:
            if bool(item.get("attributes", {}).get("no_eval", False)):
                continue
            box = item["box"]
            fields = (
                float(box["cx"]),
                float(box["cy"]),
                float(box["l"]),
                float(box["w"]),
                float(box["rot_z"]),
            )
            if not all(math.isfinite(value) for value in fields):
                continue
            require(fields[2] > 0.0 and fields[3] > 0.0, f"invalid_box_extent:{sequence}:{frame}")
            boxes.append(
                NativeBox(
                    label_id=str(item["label_id"]),
                    center_forward_m=fields[0] + BASE_LINK_FROM_LOGICAL_RGB360_X_M,
                    center_left_m=fields[1] + BASE_LINK_FROM_LOGICAL_RGB360_Y_M,
                    length_m=fields[2],
                    width_m=fields[3],
                    yaw_ego_rad=fields[4],
                )
            )
        output[frame] = boxes
    return output


def _obb_clearance(box: NativeBox, route_radius_m: float) -> float:
    return point_to_box_clearance(
        0.0,
        0.0,
        box.center_forward_m,
        box.center_left_m,
        box.yaw_ego_rad,
        box.length_m,
        box.width_m,
    ) - route_radius_m


def _circle_clearance(box: NativeBox, route_radius_m: float) -> float:
    radius_m = max(0.15, 0.5 * max(box.width_m, box.length_m))
    return math.hypot(box.center_forward_m, box.center_left_m) - (
        route_radius_m + radius_m
    )


def global_truth_timeline(
    *,
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    boxes_by_frame: Mapping[int, Sequence[NativeBox]],
    horizon_s: float = HORIZON_S,
    route_radius_m: float = ROUTE_HALF_WIDTH_M,
) -> list[dict[str, Any]]:
    """Return C0-compatible truth for one complete sequence."""

    ordered_frames = list(frames)
    times = [float(timestamps[frame]) for frame in ordered_frames]
    require(bool(ordered_frames), "empty_sequence")
    require(len(ordered_frames) == len(times), "frame_timestamp_length_mismatch")
    output = []
    for origin_index, (origin_frame, origin_time) in enumerate(zip(ordered_frames, times)):
        final_index = bisect.bisect_right(times, origin_time + horizon_s + 1e-9) - 1
        obb_hits: list[dict[str, Any]] = []
        circle_hits: list[dict[str, Any]] = []
        minimum_obb_clearance = None
        minimum_circle_clearance = None
        for future_index in range(origin_index, final_index + 1):
            future_frame = ordered_frames[future_index]
            delta_s = times[future_index] - origin_time
            for box in boxes_by_frame.get(future_frame, ()):
                obb_clearance = _obb_clearance(box, route_radius_m)
                circle_clearance = _circle_clearance(box, route_radius_m)
                minimum_obb_clearance = (
                    obb_clearance
                    if minimum_obb_clearance is None
                    else min(minimum_obb_clearance, obb_clearance)
                )
                minimum_circle_clearance = (
                    circle_clearance
                    if minimum_circle_clearance is None
                    else min(minimum_circle_clearance, circle_clearance)
                )
                row = {
                    "label_id": box.label_id,
                    "frame": future_frame,
                    "delta_s": delta_s,
                }
                if obb_clearance <= 1e-9:
                    obb_hits.append({**row, "clearance_m": obb_clearance})
                if circle_clearance <= 1e-9:
                    circle_hits.append({**row, "clearance_m": circle_clearance})

        full_future = times[-1] - origin_time >= horizon_s - 0.05
        label = (
            CONTACT
            if obb_hits
            else PROXIMITY
            if circle_hits
            else CLEAR
            if full_future
            else UNKNOWN
        )
        relevant = obb_hits if label == CONTACT else circle_hits if label == PROXIMITY else []
        first_delta = min((float(row["delta_s"]) for row in relevant), default=None)
        first_rows = (
            []
            if first_delta is None
            else [row for row in relevant if abs(float(row["delta_s"]) - first_delta) <= 1e-9]
        )
        obb_components = {str(row["label_id"]) for row in obb_hits}
        circle_components = {str(row["label_id"]) for row in circle_hits}
        output.append(
            {
                "frame": origin_frame,
                "time_s": origin_time,
                "label": label,
                "full_future": full_future,
                "first_hit_delta_s": first_delta,
                "responsible_components": sorted({str(row["label_id"]) for row in first_rows}),
                "contact_components_in_horizon": sorted(obb_components),
                "proximity_components_in_horizon": sorted(circle_components),
                "circle_only_components_in_horizon": sorted(circle_components - obb_components),
                "secondary_circle_only_proximity": bool(circle_components - obb_components),
                "minimum_obb_clearance_m": minimum_obb_clearance,
                "minimum_circle_clearance_m": minimum_circle_clearance,
            }
        )
    return output


def _intervals(rows: Sequence[Mapping[str, Any]], label: str) -> list[FrameInterval]:
    indices = [index for index, row in enumerate(rows) if row["label"] == label]
    if not indices:
        return []
    output = []
    first = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            output.append(FrameInterval(first, previous))
            first = index
        previous = index
    output.append(FrameInterval(first, previous))
    return output


def bounded_contact_events(timeline: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return CONTACT intervals bounded by known non-CONTACT on both sides."""

    output = []
    for interval in _intervals(timeline, CONTACT):
        if interval.first_index == 0 or interval.last_index == len(timeline) - 1:
            continue
        before = timeline[interval.first_index - 1]["label"]
        after = timeline[interval.last_index + 1]["label"]
        if before not in {PROXIMITY, CLEAR} or after not in {PROXIMITY, CLEAR}:
            continue
        onset = timeline[interval.first_index]
        responsible = list(onset["responsible_components"])
        output.append(
            {
                "first_index": interval.first_index,
                "last_index": interval.last_index,
                "first_frame": int(onset["frame"]),
                "last_frame": int(timeline[interval.last_index]["frame"]),
                "first_time_s": float(onset["time_s"]),
                "last_time_s": float(timeline[interval.last_index]["time_s"]),
                "onset_first_hit_delta_s": onset["first_hit_delta_s"],
                "responsible_components": responsible,
                "responsible_component_count": len(responsible),
                "unique_responsible_component": len(responsible) == 1,
            }
        )
    return output


def _duration_by_label(timeline: Sequence[Mapping[str, Any]]) -> Counter[str]:
    durations: Counter[str] = Counter()
    for left, right in zip(timeline, timeline[1:]):
        delta_s = float(right["time_s"]) - float(left["time_s"])
        require(delta_s > 0.0, "nonpositive_timeline_delta")
        durations[str(left["label"])] += delta_s
    return durations


def summarize_sequence(sequence: str, timeline: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events = bounded_contact_events(timeline)
    events = [
        {"event_id": f"{sequence}:contact:{index:03d}", "sequence": sequence, **event}
        for index, event in enumerate(events, start=1)
    ]
    durations = _duration_by_label(timeline)
    counts = Counter(str(row["label"]) for row in timeline)
    known_s = durations[CONTACT] + durations[PROXIMITY] + durations[CLEAR]
    non_contact_s = durations[PROXIMITY] + durations[CLEAR]
    contact_intervals = _intervals(timeline, CONTACT)
    proximity_intervals = _intervals(timeline, PROXIMITY)
    return {
        "sequence": sequence,
        "first_frame": int(timeline[0]["frame"]),
        "last_frame": int(timeline[-1]["frame"]),
        "frames": len(timeline),
        "timeline_duration_s": float(timeline[-1]["time_s"]) - float(timeline[0]["time_s"]),
        "truth_frame_counts": dict(sorted(counts.items())),
        "truth_duration_s": {key: durations[key] for key in (CONTACT, PROXIMITY, CLEAR, UNKNOWN)},
        "known_timeline_s": known_s,
        "known_non_contact_s": non_contact_s,
        "contact_duty_cycle": None if known_s == 0.0 else durations[CONTACT] / known_s,
        "contact_intervals_descriptive": len(contact_intervals),
        "proximity_intervals_descriptive": len(proximity_intervals),
        "bounded_contact_events": len(events),
        "unique_responsible_events": sum(bool(event["unique_responsible_component"]) for event in events),
        "first_contact_responsible_component_counts": dict(
            sorted(Counter(str(event["responsible_component_count"]) for event in events).items())
        ),
        "events": events,
    }


def _totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth_duration_s = {
        label: sum(float(row["truth_duration_s"][label]) for row in rows)
        for label in (CONTACT, PROXIMITY, CLEAR, UNKNOWN)
    }
    known_s = truth_duration_s[CONTACT] + truth_duration_s[PROXIMITY] + truth_duration_s[CLEAR]
    return {
        "sequences": len(rows),
        "frames": sum(int(row["frames"]) for row in rows),
        "timeline_duration_s": sum(float(row["timeline_duration_s"]) for row in rows),
        "bounded_contact_events": sum(int(row["bounded_contact_events"]) for row in rows),
        "unique_responsible_events": sum(int(row["unique_responsible_events"]) for row in rows),
        "known_non_contact_s": sum(float(row["known_non_contact_s"]) for row in rows),
        "truth_duration_s": truth_duration_s,
        "contact_duty_cycle": None if known_s == 0.0 else truth_duration_s[CONTACT] / known_s,
    }


def _meets(totals: Mapping[str, Any], *, preferred: bool) -> bool:
    if preferred:
        return (
            totals["bounded_contact_events"] >= PREFERRED_BOUNDED_CONTACT_EVENTS
            and totals["unique_responsible_events"] >= PREFERRED_UNIQUE_RESPONSIBLE_EVENTS
            and totals["known_non_contact_s"] >= PREFERRED_KNOWN_NON_CONTACT_S
        )
    return (
        totals["bounded_contact_events"] >= MINIMUM_BOUNDED_CONTACT_EVENTS
        and totals["unique_responsible_events"] >= MINIMUM_UNIQUE_RESPONSIBLE_EVENTS
        and totals["known_non_contact_s"] >= MINIMUM_KNOWN_NON_CONTACT_S
    )


def select_roster(sequence_rows: Sequence[Mapping[str, Any]]) -> tuple[str, list[Mapping[str, Any]]]:
    ordered = sorted(sequence_rows, key=lambda row: str(row["sequence"]))
    minimum_prefix = None
    preferred_prefix = None
    for length in range(1, len(ordered) + 1):
        totals = _totals(ordered[:length])
        if minimum_prefix is None and _meets(totals, preferred=False):
            minimum_prefix = length
        if _meets(totals, preferred=True):
            preferred_prefix = length
            break
    if preferred_prefix is not None:
        return STATUS_ADMITTED_PREFERRED, ordered[:preferred_prefix]
    if minimum_prefix is not None:
        return STATUS_ADMITTED_MINIMUM, ordered[:minimum_prefix]
    return STATUS_INSUFFICIENT, []


def _write_sequence_csv(path: Path, rows: Sequence[Mapping[str, Any]], selected: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "sequence",
        "selected",
        "frames",
        "timeline_duration_s",
        "bounded_contact_events",
        "unique_responsible_events",
        "known_non_contact_s",
        "contact_duty_cycle",
        "contact_intervals_descriptive",
        "proximity_intervals_descriptive",
    )
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (str(row["sequence"]) in selected if key == "selected" else row[key])
                    for key in fields
                }
            )
    os.replace(partial, path)


def _write_event_csv(path: Path, rows: Sequence[Mapping[str, Any]], selected: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "event_id",
        "sequence",
        "selected",
        "first_frame",
        "last_frame",
        "first_time_s",
        "last_time_s",
        "onset_first_hit_delta_s",
        "responsible_component_count",
        "responsible_components",
        "unique_responsible_component",
    )
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for sequence_row in rows:
            for event in sequence_row["events"]:
                writer.writerow(
                    {
                        **{key: event[key] for key in fields if key not in {"selected", "responsible_components"}},
                        "selected": str(sequence_row["sequence"]) in selected,
                        "responsible_components": "|".join(event["responsible_components"]),
                    }
                )
    os.replace(partial, path)


def run(*, labels_path: Path, timestamps_path: Path, output: Path, roster_path: Path) -> dict[str, Any]:
    labels_path = labels_path.resolve(strict=True)
    timestamps_path = timestamps_path.resolve(strict=True)
    labels_hash = sha256_file(labels_path)
    timestamps_hash = sha256_file(timestamps_path)
    timelines = []
    sequence_rows = []
    with zipfile.ZipFile(labels_path) as labels, zipfile.ZipFile(timestamps_path) as timestamps:
        available_sequences = _archive_sequences(labels, timestamps)
        candidate_sequences = [
            sequence for sequence in available_sequences if sequence not in CONSUMED_SEQUENCES
        ]
        for sequence in candidate_sequences:
            frame_timestamps = _load_timestamps(timestamps, sequence)
            boxes = _load_boxes(labels, sequence)
            require(
                set(boxes) == set(frame_timestamps),
                f"label_timestamp_frame_mismatch:{sequence}",
            )
            timeline = global_truth_timeline(
                frames=sorted(frame_timestamps),
                timestamps=frame_timestamps,
                boxes_by_frame=boxes,
            )
            sequence_rows.append(summarize_sequence(sequence, timeline))
            timelines.extend({"sequence": sequence, **row} for row in timeline)

    status, selected_rows = select_roster(sequence_rows)
    selected_sequences = {str(row["sequence"]) for row in selected_rows}
    selected_totals = _totals(selected_rows)
    all_totals = _totals(sequence_rows)
    timeline_path = output.with_name(output.stem + ".timeline.jsonl")
    csv_path = output.with_name(output.stem + ".sequences.csv")
    event_csv_path = output.with_name(output.stem + ".events.csv")
    write_jsonl(timeline_path, timelines)
    _write_sequence_csv(csv_path, sequence_rows, selected_sequences)
    _write_event_csv(event_csv_path, sequence_rows, selected_sequences)

    roster = {
        "schema": ROSTER_SCHEMA,
        "status": status,
        "claim_ceiling": "METADATA_ONLY_TRUTH_COHORT_ADMISSION_NO_ALGORITHM_RESULT",
        "dataset": "JRDB public train split",
        "contract": {
            "primary": "wearer-global realized future native OBB union within 0-3 s",
            "secondary": "circle-only PROXIMITY",
            "route_body_radius_m": ROUTE_HALF_WIDTH_M,
            "future_horizon_s": HORIZON_S,
            "bounded_event": "CONTACT interval preceded and followed by known non-CONTACT",
            "responsibility": "component identities at the earliest realized OBB hit from event onset; diagnostic only",
        },
        "source_authority": {
            "labels_archive_name": labels_path.name,
            "labels_sha256": labels_hash,
            "timestamps_archive_name": timestamps_path.name,
            "timestamps_sha256": timestamps_hash,
        },
        "ordering": "lexicographic sequence name; shortest preferred prefix, else shortest minimum prefix",
        "excluded_consumed_sequences": list(CONSUMED_SEQUENCES),
        "selected_sequences": [
            {
                "sequence": row["sequence"],
                "first_frame": row["first_frame"],
                "last_frame": row["last_frame"],
                "frames": row["frames"],
                "bounded_contact_events": row["bounded_contact_events"],
                "unique_responsible_events": row["unique_responsible_events"],
                "known_non_contact_s": row["known_non_contact_s"],
                "contact_duty_cycle": row["contact_duty_cycle"],
                "bounded_contact_event_details": row["events"],
            }
            for row in selected_rows
        ],
        "selected_totals": selected_totals,
        "stage2_authorization": (
            "RAW_SENSOR_ACQUISITION_AND_FROZEN_REPLAY_MAY_BE_PROTOCOLIZED_SEPARATELY"
            if status != STATUS_INSUFFICIENT
            else "JRDB_STAGE2_CLOSED_USE_ANOTHER_ROUTE_AUTHORITATIVE_SOURCE"
        ),
        "forbidden": [
            "training or tuning on C1 truth",
            "reordering sequences by observed event yield",
            "adding the consumed C0 sequence",
            "treating multi-responsible events as single-target dropout stress",
            "forecasting, R8, TeFlow, or DeltaFlow work before frozen replay",
        ],
    }
    write_json(roster_path, roster)
    result = {
        "schema": SCHEMA,
        "status": status,
        "question": "Can JRDB admit a fresh global-OBB cohort with valid CONTACT and non-CONTACT denominators before any algorithm replay?",
        "contract": roster["contract"],
        "admission_policy": {
            "minimum": {
                "bounded_contact_events": MINIMUM_BOUNDED_CONTACT_EVENTS,
                "unique_responsible_events": MINIMUM_UNIQUE_RESPONSIBLE_EVENTS,
                "known_non_contact_s": MINIMUM_KNOWN_NON_CONTACT_S,
            },
            "preferred": {
                "bounded_contact_events": PREFERRED_BOUNDED_CONTACT_EVENTS,
                "unique_responsible_events": PREFERRED_UNIQUE_RESPONSIBLE_EVENTS,
                "known_non_contact_s": PREFERRED_KNOWN_NON_CONTACT_S,
            },
            "ordering": roster["ordering"],
        },
        "source": {
            "labels": str(labels_path),
            "labels_sha256": labels_hash,
            "timestamps": str(timestamps_path),
            "timestamps_sha256": timestamps_hash,
            "available_sequences": len(sequence_rows) + len(CONSUMED_SEQUENCES),
            "candidate_sequences": len(sequence_rows),
            "excluded_consumed_sequences": list(CONSUMED_SEQUENCES),
        },
        "all_candidate_totals": all_totals,
        "selected_totals": selected_totals,
        "selected_sequence_names": sorted(selected_sequences),
        "sequence_scan": sequence_rows,
        "artifacts": {
            "timeline_jsonl": str(timeline_path.resolve()),
            "timeline_jsonl_sha256": sha256_file(timeline_path),
            "sequences_csv": str(csv_path.resolve()),
            "sequences_csv_sha256": sha256_file(csv_path),
            "events_csv": str(event_csv_path.resolve()),
            "events_csv_sha256": sha256_file(event_csv_path),
            "frozen_roster": str(roster_path.resolve()),
            "frozen_roster_sha256": sha256_file(roster_path),
        },
        "decision": (
            "Freeze the admitted sequence roster before acquiring raw sensors or replaying R2/R3-C/R7-P/M1-O."
            if status != STATUS_INSUFFICIENT
            else "Close JRDB C1 and seek a different route-authoritative source such as CODa."
        ),
        "claim_limits": [
            "C1 is truth-only cohort admission, not an algorithm result.",
            "Unique responsibility admits later single-target dropout stress but does not restore per-target correctness.",
            "No RGB, LiDAR, detector, tracker, model, or prediction was read or executed.",
        ],
    }
    write_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    dataset = repo / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo
        / "artifacts.local"
        / "evidence"
        / "dtr-c1"
        / "global-obb-cohort-admission"
        / "result.json",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path(__file__).resolve().with_name("dtr_c1_fresh_global_obb_roster.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        labels_path=args.labels,
        timestamps_path=args.timestamps,
        output=args.output,
        roster_path=args.roster,
    )
    print(json.dumps({"status": result["status"], "selected_totals": result["selected_totals"]}))


if __name__ == "__main__":
    main()
