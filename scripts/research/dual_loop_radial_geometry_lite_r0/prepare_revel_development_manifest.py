#!/usr/bin/env python3
"""Freeze continuous REveL Development inputs and truth-only natural events.

This script does not run a radial-geometry candidate.  It separates the exact
RGB/ROI allowlist visible to a future producer from the Vicon truth visible
only to a later evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from align_revel_detector_failures_with_vicon import (  # noqa: E402
    RADIAL_DEADBAND_MPS,
    SYNC_MAX_DELTA_MS,
    _bracketing_pair_indices,
    _source_radial_motion,
)
from audit_revel_dynamic_vicon_trajectories import (  # noqa: E402
    PERSON_TOPICS,
    SENSOR_TOPIC,
    _extract_topic,
)


FORMAT = "blindassist_dual_loop_revel_development_input_freeze_v1"
TARGETS = (
    {"class_id": 0, "oracle_label": "green-helmet", "target_id": "track-000"},
    {"class_id": 1, "oracle_label": "yellow-helmet", "target_id": "track-001"},
)
MAX_FRAME_GAP_NS = 100_000_000
PRIMARY_EVENT_MIN_FRAMES = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def region_for_center_x(center_x: float) -> str:
    if center_x < 1.0 / 3.0:
        return "LEFT"
    if center_x > 2.0 / 3.0:
        return "RIGHT"
    return "CENTER"


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        class_id = int(fields[0])
        values = tuple(float(value) for value in fields[1:])
        if class_id not in (0, 1) or not all(np.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: invalid class or coordinate")
        center_x, center_y, width, height = values
        epsilon = 1e-9
        if not (
            -epsilon <= center_x <= 1.0 + epsilon
            and -epsilon <= center_y <= 1.0 + epsilon
            and 0.0 < width <= 1.0 + epsilon
            and 0.0 < height <= 1.0 + epsilon
        ):
            raise ValueError(f"{path}:{line_number}: box outside normalized domain")
        rows.append((
            class_id,
            min(1.0, max(0.0, center_x)),
            min(1.0, max(0.0, center_y)),
            min(1.0, width),
            min(1.0, height),
        ))
    return rows


def _close_event(
    events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    target_id: str,
    ordinal: int,
) -> None:
    if not rows:
        return
    event_id = f"{target_id}:event-{ordinal:04d}"
    region_counts = Counter(str(row["region"]) for row in rows)
    start_ns = int(rows[0]["bag_image_timestamp_ns"])
    end_ns = int(rows[-1]["bag_image_timestamp_ns"])
    primary = len(rows) >= PRIMARY_EVENT_MIN_FRAMES
    event = {
        "event_id": event_id,
        "capture_id": "REVEL_DYNAMIC_V1",
        "target_id": target_id,
        "truth_state": rows[0]["truth_state"],
        "start_source_frame_index": rows[0]["source_frame_index"],
        "end_source_frame_index": rows[-1]["source_frame_index"],
        "start_timestamp_ns": start_ns,
        "end_timestamp_ns": end_ns,
        "duration_s": (end_ns - start_ns) / 1e9,
        "eligible_frame_count": len(rows),
        "anchor_region": rows[0]["region"],
        "region_frame_counts": {name: region_counts.get(name, 0) for name in ("LEFT", "CENTER", "RIGHT")},
        "primary_event_eligible": primary,
        "primary_event_min_frames": PRIMARY_EVENT_MIN_FRAMES,
    }
    events.append(event)
    for row in rows:
        row["event_id"] = event_id
        row["event_anchor_region"] = event["anchor_region"]
        row["primary_event_eligible"] = primary


def segment_natural_events(truth_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = str(target["target_id"])
        eligible = [
            row
            for row in truth_rows
            if row["target_id"] == target_id
            and row["unique_roi_available"]
            and row["truth_available"]
        ]
        current: list[dict[str, Any]] = []
        ordinal = 0
        for row in eligible:
            continues = bool(
                current
                and row["source_frame_index"] == current[-1]["source_frame_index"] + 1
                and row["bag_image_timestamp_ns"] > current[-1]["bag_image_timestamp_ns"]
                and row["bag_image_timestamp_ns"] - current[-1]["bag_image_timestamp_ns"] <= MAX_FRAME_GAP_NS
                and row["truth_state"] == current[-1]["truth_state"]
            )
            if current and not continues:
                _close_event(events, current, target_id, ordinal)
                ordinal += 1
                current = []
            current.append(row)
        if current:
            _close_event(events, current, target_id, ordinal)
    return events


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            stream.write("\n")


def _event_coverage(events: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [event for event in events if event["primary_event_eligible"]]
    by_target_region_state: dict[str, int] = {}
    for target in TARGETS:
        for region in ("LEFT", "CENTER", "RIGHT"):
            for state in ("approaching", "quasi_static", "receding"):
                key = f"{target['target_id']}|{region}|{state}"
                by_target_region_state[key] = sum(
                    event["target_id"] == target["target_id"]
                    and event["anchor_region"] == region
                    and event["truth_state"] == state
                    for event in primary
                )
    return {
        "raw_event_count": len(events),
        "primary_event_count": len(primary),
        "primary_event_min_frames": PRIMARY_EVENT_MIN_FRAMES,
        "by_target_anchor_region_truth_state": by_target_region_state,
    }


def prepare(bag_root: Path, image_label_root: Path, output_root: Path) -> dict[str, Any]:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    bag = bag_root / "dynamic.bag"
    calibration = bag_root / "calibration.yaml"
    classes = bag_root / "classes.txt"
    vicon_audit_path = bag_root / "qa" / "revel_dynamic_vicon_trajectory_audit.json"
    images_zip = image_label_root / "images.zip"
    labels_zip = image_label_root / "labels.zip"
    images_root = image_label_root / "extracted" / "images" / "images"
    labels_root = image_label_root / "extracted" / "labels" / "labels"
    required = (bag, calibration, classes, vicon_audit_path, images_zip, labels_zip, images_root, labels_root)
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(f"missing REveL inputs: {missing}")

    class_names = [line.strip() for line in classes.read_text(encoding="utf-8").splitlines() if line.strip()]
    if class_names != ["green-helmet", "yellow-helmet"]:
        raise ValueError(f"unexpected class map: {class_names}")
    image_paths = sorted(images_root.glob("*.jpg"), key=lambda path: int(path.stem))
    label_paths = sorted(labels_root.glob("*.txt"), key=lambda path: int(path.stem))
    if len(image_paths) != 8580 or [path.stem for path in image_paths] != [path.stem for path in label_paths]:
        raise ValueError("REveL full RGB/label sequence is not an exact 8,580-frame pair")
    stems = [path.stem for path in image_paths]

    vicon_audit = json.loads(vicon_audit_path.read_text(encoding="utf-8"))
    expected_bag_hash = vicon_audit.get("source", {}).get("sha256")
    if expected_bag_hash != "6b10752b0d4cb401751e57f3ac55ebe45fcbb785f89d8a43fe1cbfd30dc0b08a":
        raise ValueError("unexpected REveL bag identity in Vicon audit")
    if bag.stat().st_size != vicon_audit.get("source", {}).get("bytes"):
        raise ValueError("REveL bag size drifted from Vicon audit")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag) as reader:
        image_connection = reader.topics["/dvs/image_raw"].connections[0]
        bag_image_timestamps = np.asarray(
            [timestamp for _, timestamp, _ in reader.messages(connections=[image_connection])],
            dtype=np.int64,
        )
        sensor = _extract_topic(reader, typestore, SENSOR_TOPIC)
        people = [_extract_topic(reader, typestore, topic) for topic in PERSON_TOPICS]
    if len(bag_image_timestamps) != len(stems) or np.any(np.diff(bag_image_timestamps) <= 0):
        raise ValueError("bag RGB timestamps are not a strictly increasing 8,580-frame sequence")

    radial_by_target: dict[str, dict[str, Any]] = {}
    for target, person in zip(TARGETS, people):
        radial = _source_radial_motion(person, sensor)
        pair_indices = _bracketing_pair_indices(
            bag_image_timestamps,
            np.asarray(person["timestamps_ns"], dtype=np.int64),
        )
        radial_by_target[str(target["target_id"])] = {
            "radial": radial,
            "pair_indices": pair_indices,
        }

    replay_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    epoch_by_target = {str(target["target_id"]): 0 for target in TARGETS}
    active_by_target = {str(target["target_id"]): False for target in TARGETS}
    for frame_index, (stem, label_path, bag_timestamp_ns) in enumerate(zip(stems, label_paths, bag_image_timestamps)):
        labels = parse_label(label_path)
        by_class = {class_id: [row for row in labels if row[0] == class_id] for class_id in (0, 1)}
        for target in TARGETS:
            target_id = str(target["target_id"])
            matching = by_class[int(target["class_id"])]
            unique_roi = len(matching) == 1
            history_reset = unique_roi and not active_by_target[target_id]
            if history_reset:
                epoch_by_target[target_id] += 1
            active_by_target[target_id] = unique_roi
            track_epoch = f"{target_id}:epoch-{epoch_by_target[target_id]:04d}" if unique_roi else None
            bbox = list(matching[0][1:]) if unique_roi else None
            region = region_for_center_x(float(bbox[0])) if bbox is not None else None

            radial_entry = radial_by_target[target_id]
            pair_index = int(radial_entry["pair_indices"][frame_index])
            radial = radial_entry["radial"]
            truth_available = bool(pair_index >= 0 and radial["valid"][pair_index])
            truth_state = str(radial["state"][pair_index]) if truth_available else None
            truth_rate = float(-radial["range_rate_mps"][pair_index]) if truth_available else None
            truth_reason = None
            if pair_index < 0:
                truth_reason = "NO_STRICT_VICON_BRACKET"
            elif not truth_available:
                truth_reason = "VICON_CONTINUITY_FILTER_REJECTED"

            source_frame_id = f"revel-dynamic:{frame_index:05d}"
            truth_row = {
                "capture_id": "REVEL_DYNAMIC_V1",
                "source_frame_id": source_frame_id,
                "source_frame_index": frame_index,
                "archive_timestamp_ns": int(stem),
                "bag_image_timestamp_ns": int(bag_timestamp_ns),
                "target_id": target_id,
                "oracle_target_label": target["oracle_label"],
                "source_box_count": len(matching),
                "unique_roi_available": unique_roi,
                "region": region,
                "truth_available": truth_available,
                "truth_unavailable_reason": truth_reason,
                "truth_signed_approach_mps": truth_rate,
                "truth_state": truth_state,
                "truth_deadband_mps": RADIAL_DEADBAND_MPS,
                "truth_offline_noncausal": True,
                "event_id": None,
                "event_anchor_region": None,
                "primary_event_eligible": False,
            }
            truth_rows.append(truth_row)
            if unique_roi:
                replay_rows.append({
                    "capture_id": "REVEL_DYNAMIC_V1",
                    "source_frame_id": source_frame_id,
                    "source_frame_index": frame_index,
                    "archive_timestamp_ns": int(stem),
                    "captured_at_ns": int(bag_timestamp_ns),
                    "image_relative_path": f"{stem}.jpg",
                    "target_id": target_id,
                    "track_epoch": track_epoch,
                    "history_reset": history_reset,
                    "roi_xywh_normalized": bbox,
                    "region": region,
                })

    events = segment_natural_events(truth_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    replay_path = output_root / "replay_input.jsonl"
    truth_path = output_root / "truth.jsonl"
    events_path = output_root / "natural_events.jsonl"
    _write_jsonl(replay_path, replay_rows)
    _write_jsonl(truth_path, truth_rows)
    _write_jsonl(events_path, events)

    manifest = {
        "format": FORMAT,
        "protocol_id": "DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0",
        "stage": "DEVELOPMENT",
        "preparation": {
            "script_path": Path(__file__).resolve().as_posix(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rosbags": importlib.metadata.version("rosbags"),
        },
        "capture_id": "REVEL_DYNAMIC_V1",
        "independence_group": "REVEL_DYNAMIC_SINGLE_CAPTURE",
        "nested_units": ["target", "parent_natural_event", "frame_pair"],
        "source_identity": {
            "bag": {"bytes": bag.stat().st_size, "sha256_from_verified_vicon_audit": expected_bag_hash},
            "vicon_audit": {"path": vicon_audit_path.as_posix(), "sha256": sha256_file(vicon_audit_path)},
            "images_zip": {"bytes": images_zip.stat().st_size, "sha256": sha256_file(images_zip)},
            "labels_zip": {"bytes": labels_zip.stat().st_size, "sha256": sha256_file(labels_zip)},
            "calibration": {"sha256": sha256_file(calibration)},
            "classes": {"sha256": sha256_file(classes)},
            "extracted_image_tree_sha256": tree_sha256(image_paths, image_label_root),
            "extracted_label_tree_sha256": tree_sha256(label_paths, image_label_root),
        },
        "producer_allowlist": {
            "replay_input": {"path": replay_path.as_posix(), "rows": len(replay_rows), "sha256": sha256_file(replay_path)},
            "image_root": images_root.as_posix(),
            "vicon_or_truth_access": "FORBIDDEN",
        },
        "evaluator_truth": {
            "truth": {"path": truth_path.as_posix(), "rows": len(truth_rows), "sha256": sha256_file(truth_path)},
            "natural_events": {"path": events_path.as_posix(), "rows": len(events), "sha256": sha256_file(events_path)},
            "truth_sign": "positive=sensor-target approach; equals negative source radial range-rate",
            "truth_state_rule": "approaching if signed approach >=0.10 m/s; receding if <=-0.10 m/s; otherwise quasi_static",
            "offline_noncausal": True,
        },
        "fixed_denominators": {
            "source_frame_count": len(stems),
            "target_frame_rows": len(truth_rows),
            "unique_roi_replay_opportunities": len(replay_rows),
            "target_ids": [target["target_id"] for target in TARGETS],
            "regions": ["LEFT", "CENTER", "RIGHT"],
            "truth_states": ["approaching", "quasi_static", "receding"],
            "event_coverage": _event_coverage(events),
        },
        "event_definition": {
            "parent_unit": "maximal same-target, same-truth-state run over consecutive RGB frame indices",
            "max_interframe_gap_ns": MAX_FRAME_GAP_NS,
            "missing_or_ambiguous_roi_breaks_event": True,
            "truth_unavailable_breaks_event": True,
            "region_change_does_not_split_parent_event": True,
            "anchor_region": "region at first truth-eligible frame",
            "primary_event_min_frames": PRIMARY_EVENT_MIN_FRAMES,
            "short_runs_preserved": True,
        },
        "access_and_claims": {
            "result_access_state": "OUTPUT_INSPECTED_DEVELOPMENT_ONLY",
            "research_track": "DEVELOPMENT_DIAGNOSTIC",
            "confirmation_reuse": "FORBIDDEN_FOR_SAME_CLAIM",
            "runtime_input_role": "ORACLE_TRACK_CONDITIONED_GEOMETRY_DEVELOPMENT",
            "claim_ceiling": "SINGLE_CAPTURE_ORACLE_ROI_CONDITIONED_DEVELOPMENT_ONLY",
            "old_f1b_decision_output_access": "FORBIDDEN",
        },
        "sync_contract": {
            "timestamp_basis": "rosbag record time",
            "vicon_sync_max_delta_ms": SYNC_MAX_DELTA_MS,
            "vicon_pair_selection": "strict source person-pose bracket around each bag image timestamp",
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--image-label-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.bag_root, args.image_label_root, args.output_root)
    print(json.dumps({
        "status": "INPUT_FREEZE_COMPLETE",
        "replay_opportunities": manifest["fixed_denominators"]["unique_roi_replay_opportunities"],
        "primary_events": manifest["fixed_denominators"]["event_coverage"]["primary_event_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
