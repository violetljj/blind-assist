#!/usr/bin/env python3
"""Align bounded REveL detector outcomes with source Vicon person range.

This joins already-audited archive-frame order, label classes, bag timestamps,
helmet markers, and the event/LiDAR sensor marker.  Range remains source-native
helmet-to-sensor evidence, not user-body distance or an assistive TTC label.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from audit_revel_dynamic_vicon_trajectories import (
    MAX_CONTINUOUS_INTERVAL_S,
    MAX_CONTINUOUS_WORLD_SPEED_MPS,
    MIN_CONTINUOUS_INTERVAL_S,
    ORIGIN_EPSILON_M,
    PERSON_TOPICS,
    SENSOR_TOPIC,
    _extract_topic,
    _nearest_indices,
)


SYNC_MAX_DELTA_MS = 20.0
RANGE_BINS = ((0.0, 2.0, "0-2m"), (2.0, 3.0, "2-3m"), (3.0, 4.0, "3-4m"), (4.0, 5.0, "4-5m"), (5.0, float("inf"), "5m+"))
RADIAL_DEADBAND_MPS = 0.10
RADIAL_STATES = ("approaching", "quasi_static", "receding")
TTC_PROXY_BINS = ((0.0, 1.0, "0-1s"), (1.0, 2.0, "1-2s"), (2.0, 3.0, "2-3s"), (3.0, float("inf"), "3s+"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    q = quaternion / np.maximum(norm, 1e-12)
    x, y, z, w = q.T
    return np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), axis=1).reshape(-1, 3, 3)


def _valid_and_continuous(track: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(track["positions"], dtype=np.float64)
    quaternion = np.asarray(track["quaternions"], dtype=np.float64)
    finite = np.isfinite(position).all(axis=1) & np.isfinite(quaternion).all(axis=1)
    valid = finite & (np.linalg.norm(position, axis=1) > ORIGIN_EPSILON_M) & (np.linalg.norm(quaternion, axis=1) > 1e-6)
    timestamps_s = np.asarray(track["timestamps_ns"], dtype=np.float64) / 1e9
    dt = np.diff(timestamps_s)
    interval_eligible = valid[1:] & valid[:-1] & (dt >= MIN_CONTINUOUS_INTERVAL_S) & (dt <= MAX_CONTINUOUS_INTERVAL_S)
    speed = np.full(dt.shape, np.inf, dtype=np.float64)
    speed[interval_eligible] = np.linalg.norm(np.diff(position, axis=0)[interval_eligible], axis=1) / dt[interval_eligible]
    return valid, interval_eligible & (speed <= MAX_CONTINUOUS_WORLD_SPEED_MPS)


def _radial_state(range_rate_mps: float) -> str:
    if range_rate_mps <= -RADIAL_DEADBAND_MPS:
        return "approaching"
    if range_rate_mps >= RADIAL_DEADBAND_MPS:
        return "receding"
    return "quasi_static"


def _source_radial_motion(person: dict[str, Any], sensor: dict[str, Any]) -> dict[str, Any]:
    """Compute source-native radial motion on complete continuity-filtered Vicon pairs."""
    person_valid, person_continuous = _valid_and_continuous(person)
    sensor_valid, sensor_continuous = _valid_and_continuous(sensor)
    person_times = np.asarray(person["timestamps_ns"], dtype=np.int64)
    sensor_times = np.asarray(sensor["timestamps_ns"], dtype=np.int64)
    nearest = _nearest_indices(person_times, sensor_times)
    sync_delta_ms = np.abs(sensor_times[nearest] - person_times).astype(np.float64) / 1e6
    aligned = person_valid & sensor_valid[nearest] & (sync_delta_ms <= SYNC_MAX_DELTA_MS)
    relative_world = np.asarray(person["positions"], dtype=np.float64) - np.asarray(sensor["positions"], dtype=np.float64)[nearest]
    range_m = np.linalg.norm(relative_world, axis=1)
    dt_s = np.diff(person_times).astype(np.float64) / 1e9
    sensor_index_delta = np.diff(nearest)
    sensor_pair_ok = sensor_index_delta == 0
    advances = sensor_index_delta == 1
    if np.any(advances):
        sensor_pair_ok[advances] = sensor_continuous[nearest[:-1][advances]]
    pair_valid = aligned[1:] & aligned[:-1] & person_continuous & sensor_pair_ok & (dt_s > 0.0)

    range_start_m = range_m[:-1]
    range_end_m = range_m[1:]
    range_midpoint_m = (range_start_m + range_end_m) / 2.0
    range_rate_mps = np.full(len(dt_s), np.nan, dtype=np.float64)
    range_rate_mps[pair_valid] = np.diff(range_m)[pair_valid] / dt_s[pair_valid]
    state = np.full(len(dt_s), None, dtype=object)
    ttc_proxy_s = np.full(len(dt_s), np.nan, dtype=np.float64)
    for slot in np.flatnonzero(pair_valid):
        state[slot] = _radial_state(float(range_rate_mps[slot]))
        if state[slot] == "approaching":
            ttc_proxy_s[slot] = range_midpoint_m[slot] / -range_rate_mps[slot]
    return {
        "valid": pair_valid,
        "start_timestamp_ns": person_times[:-1],
        "end_timestamp_ns": person_times[1:],
        "midpoint_timestamp_ns": person_times[:-1] + np.floor_divide(person_times[1:] - person_times[:-1], 2),
        "interval_s": dt_s,
        "range_start_m": range_start_m,
        "range_end_m": range_end_m,
        "range_midpoint_m": range_midpoint_m,
        "range_rate_mps": range_rate_mps,
        "state": state,
        "ttc_proxy_s": ttc_proxy_s,
        "continuity_filtered_pair_count": int(pair_valid.sum()),
    }


def _bracketing_pair_indices(query_timestamps_ns: np.ndarray, person_timestamps_ns: np.ndarray) -> np.ndarray:
    right = np.searchsorted(person_timestamps_ns, query_timestamps_ns, side="left")
    pair = right - 1
    invalid = (right <= 0) | (right >= len(person_timestamps_ns))
    pair[invalid] = -1
    return pair


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(array.min()), "median": float(np.median(array)), "p95": float(np.quantile(array, .95)), "max": float(array.max())}


def _summarize(boxes: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    def wilson95(matched: int, total: int) -> list[float] | None:
        if total == 0:
            return None
        z = 1.959963984540054
        estimate = matched / total
        denominator = 1 + z * z / total
        centre = (estimate + z * z / (2 * total)) / denominator
        radius = z * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total)) / denominator
        return [centre - radius, centre + radius]

    def metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
        ground_truth = len(selected)
        matched = sum(bool(item["matched_at_fixed_score"]) for item in selected)
        return {"ground_truth": ground_truth, "matched": matched, "missed": ground_truth - matched, "recall": matched / ground_truth if ground_truth else None, "recall_wilson95": wilson95(matched, ground_truth)}

    aligned = [item for item in boxes if item["vicon_available"]]
    range_bins = {}
    for lower, upper, name in RANGE_BINS:
        selected = [item for item in aligned if lower <= item["sensor_local_range_m"] < upper]
        range_bins[name] = metrics(selected)
    by_class = {name: metrics([item for item in aligned if item["class_name"] == name]) for name in class_names}
    by_stratum = {name: metrics([item for item in aligned if item["stratum"] == name]) for name in ("small", "medium", "large")}
    motion = [item for item in boxes if item["source_motion_available"]]
    by_motion = {name: metrics([item for item in motion if item["source_radial_motion"] == name]) for name in RADIAL_STATES}
    by_ttc_proxy = {}
    for lower, upper, name in TTC_PROXY_BINS:
        selected = [item for item in motion if item["source_ttc_proxy_s"] is not None and lower <= item["source_ttc_proxy_s"] < upper]
        by_ttc_proxy[name] = metrics(selected)
    range_by_motion = {}
    for lower, upper, range_name in RANGE_BINS:
        range_by_motion[range_name] = {
            state: metrics([item for item in motion if lower <= item["sensor_local_range_m"] < upper and item["source_radial_motion"] == state])
            for state in RADIAL_STATES
        }
    return {
        "box_count": len(boxes),
        "vicon_aligned_box_count": len(aligned),
        "vicon_unavailable_box_count": len(boxes) - len(aligned),
        "recall_by_sensor_local_range": range_bins,
        "document_range_summary": {
            "within_0_5m": metrics([item for item in aligned if item["sensor_local_range_m"] < 5.0]),
            "beyond_5m": metrics([item for item in aligned if item["sensor_local_range_m"] >= 5.0]),
        },
        "recall_by_class": by_class,
        "recall_by_area_stratum": by_stratum,
        "source_motion_aligned_box_count": len(motion),
        "source_motion_unavailable_box_count": len(boxes) - len(motion),
        "source_motion_coverage_fraction": len(motion) / len(boxes) if boxes else None,
        "source_motion_unavailable_reasons": dict(sorted(Counter(item["source_motion_unavailable_reason"] for item in boxes if not item["source_motion_available"]).items())),
        "recall_by_source_radial_motion": by_motion,
        "recall_by_source_ttc_proxy": by_ttc_proxy,
        "recall_by_sensor_local_range_and_radial_motion": range_by_motion,
        "document_motion_summary": {
            **by_motion,
            "ttc_proxy_within_3s": metrics([item for item in motion if item["source_ttc_proxy_s"] is not None and item["source_ttc_proxy_s"] < 3.0]),
            "ttc_proxy_3s_or_more": metrics([item for item in motion if item["source_ttc_proxy_s"] is not None and item["source_ttc_proxy_s"] >= 3.0]),
        },
        "sensor_local_range_by_outcome_m": {
            "matched": _stats([item["sensor_local_range_m"] for item in aligned if item["matched_at_fixed_score"]]),
            "missed": _stats([item["sensor_local_range_m"] for item in aligned if not item["matched_at_fixed_score"]]),
            "small_matched": _stats([item["sensor_local_range_m"] for item in aligned if item["stratum"] == "small" and item["matched_at_fixed_score"]]),
            "small_missed": _stats([item["sensor_local_range_m"] for item in aligned if item["stratum"] == "small" and not item["matched_at_fixed_score"]]),
        },
        "source_radial_range_rate_by_outcome_mps": {
            "matched": _stats([item["source_radial_range_rate_mps"] for item in motion if item["matched_at_fixed_score"]]),
            "missed": _stats([item["source_radial_range_rate_mps"] for item in motion if not item["matched_at_fixed_score"]]),
        },
        "source_ttc_proxy_by_outcome_s": {
            "matched": _stats([item["source_ttc_proxy_s"] for item in motion if item["source_ttc_proxy_s"] is not None and item["matched_at_fixed_score"]]),
            "missed": _stats([item["source_ttc_proxy_s"] for item in motion if item["source_ttc_proxy_s"] is not None and not item["matched_at_fixed_score"]]),
        },
    }


def align(
    bag_root: Path,
    image_label_root: Path,
    benchmark_path: Path,
    details_path: Path,
    vicon_audit_path: Path,
) -> dict[str, Any]:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if benchmark.get("format") != "blindassist_revel_yolo11n_person_benchmark_v2":
        raise ValueError("unexpected detector benchmark format")
    details_receipt = benchmark.get("details_receipt") or {}
    if details_receipt.get("sha256") != _sha256(details_path):
        raise ValueError("detector details SHA256 mismatch")
    frame_details = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(frame_details) != benchmark.get("dataset", {}).get("evaluated_frames"):
        raise ValueError("detector frame detail count mismatch")
    selected_indices = np.asarray([record["selected_index"] for record in frame_details], dtype=np.int64)
    if len(np.unique(selected_indices)) != len(selected_indices) or np.any(np.diff(selected_indices) <= 0):
        raise ValueError("detector selected indices must be unique and strictly increasing")

    bag = bag_root / "dynamic.bag"
    classes_path = bag_root / "classes.txt"
    labels_root = image_label_root / "extracted" / "labels" / "labels"
    images_root = image_label_root / "extracted" / "images" / "images"
    vicon_audit = json.loads(vicon_audit_path.read_text(encoding="utf-8"))
    if vicon_audit.get("format") != "blindassist_revel_dynamic_vicon_trajectory_audit_v1":
        raise ValueError("unexpected Vicon audit format")
    if vicon_audit.get("source", {}).get("world_frame") != ["/vicon/world"]:
        raise ValueError("unexpected Vicon world frame")
    if vicon_audit.get("admission", {}).get("external_metric_person_sensor_trajectory_truth_admitted") is not True:
        raise ValueError("Vicon trajectory source is not admitted for source-only analysis")
    if bag.stat().st_size != vicon_audit.get("source", {}).get("bytes"):
        raise ValueError("bag size does not match the Vicon audit receipt")
    bag_sha256 = _sha256(bag)
    if bag_sha256 != vicon_audit.get("source", {}).get("sha256"):
        raise ValueError("bag SHA256 does not match the Vicon audit receipt")
    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if class_names != ["green-helmet", "yellow-helmet"]:
        raise ValueError(f"unexpected class map: {class_names}")
    stems = sorted((path.stem for path in images_root.glob("*.jpg")), key=int)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag) as reader:
        image_connection = reader.topics["/dvs/image_raw"].connections[0]
        image_timestamps = np.asarray([timestamp for _, timestamp, _ in reader.messages(connections=[image_connection])], dtype=np.int64)
        sensor = _extract_topic(reader, typestore, SENSOR_TOPIC)
        people = [_extract_topic(reader, typestore, topic) for topic in PERSON_TOPICS]
    if len(stems) != len(image_timestamps):
        raise ValueError("archive image count does not match bag image count")
    if selected_indices[0] < 0 or selected_indices[-1] >= len(stems):
        raise ValueError("detector selected index is outside the archive")

    selected_bag_times = image_timestamps[selected_indices]
    sensor_index = _nearest_indices(selected_bag_times, sensor["timestamps_ns"])
    sensor_position = sensor["positions"][sensor_index]
    sensor_quaternion = sensor["quaternions"][sensor_index]
    sensor_sync_ms = np.abs(sensor["timestamps_ns"][sensor_index] - selected_bag_times).astype(np.float64) / 1e6
    sensor_valid = (
        np.isfinite(sensor_position).all(axis=1)
        & np.isfinite(sensor_quaternion).all(axis=1)
        & (np.linalg.norm(sensor_position, axis=1) > 1e-9)
        & (np.linalg.norm(sensor_quaternion, axis=1) > 1e-6)
        & (sensor_sync_ms <= SYNC_MAX_DELTA_MS)
    )
    sensor_rotation = _rotation_matrix(sensor_quaternion)

    person_values: list[dict[str, Any]] = []
    for person in people:
        person_index = _nearest_indices(selected_bag_times, person["timestamps_ns"])
        position = person["positions"][person_index]
        quaternion = person["quaternions"][person_index]
        sync_ms = np.abs(person["timestamps_ns"][person_index] - selected_bag_times).astype(np.float64) / 1e6
        valid = (
            np.isfinite(position).all(axis=1)
            & np.isfinite(quaternion).all(axis=1)
            & (np.linalg.norm(position, axis=1) > 1e-9)
            & (np.linalg.norm(quaternion, axis=1) > 1e-6)
            & (sync_ms <= SYNC_MAX_DELTA_MS)
            & sensor_valid
        )
        local = np.einsum("nij,nj->ni", sensor_rotation.transpose(0, 2, 1), position - sensor_position)
        radial = _source_radial_motion(person, sensor)
        radial_pair_index = _bracketing_pair_indices(selected_bag_times, np.asarray(person["timestamps_ns"], dtype=np.int64))
        safe_pair_index = np.clip(radial_pair_index, 0, max(0, len(radial["valid"]) - 1))
        radial_available = (radial_pair_index >= 0) & radial["valid"][safe_pair_index] & valid
        person_values.append({
            "valid": valid,
            "local": local,
            "range": np.linalg.norm(local, axis=1),
            "sync_ms": sync_ms,
            "radial": radial,
            "radial_pair_index": radial_pair_index,
            "radial_available": radial_available,
        })

    box_records: list[dict[str, Any]] = []
    for frame_slot, frame in enumerate(frame_details):
        selected_index = frame["selected_index"]
        if frame["image_name"] != f"{stems[selected_index]}.jpg":
            raise ValueError("detector detail image does not match archive order")
        label_rows = []
        for line in (labels_root / f"{stems[selected_index]}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_id, cx, cy, width, height = line.split()
                label_rows.append((int(class_id), float(cx), float(cy), float(width), float(height)))
        if len(label_rows) != len(frame["ground_truth"]):
            raise ValueError("detector detail ground truth count does not match source labels")
        if frame.get("source_timestamp_ns") != int(stems[selected_index]):
            raise ValueError("detector detail source timestamp does not match archive timestamp")
        class_counts = Counter(row[0] for row in label_rows)
        for source_row, truth in zip(label_rows, frame["ground_truth"]):
            class_id, cx, cy, width, height = source_row
            source_xyxy = np.asarray([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
            if not np.allclose(source_xyxy, np.asarray(truth["xyxy_normalized"]), atol=1e-12, rtol=0):
                raise ValueError("detector detail box does not match source label")
            person = person_values[class_id]
            available = bool(person["valid"][frame_slot])
            pair_index = int(person["radial_pair_index"][frame_slot])
            ambiguous = class_counts[class_id] > 1
            motion_available = bool(person["radial_available"][frame_slot]) and not ambiguous
            if ambiguous:
                motion_unavailable_reason = "ambiguous_same_class_frame"
            elif pair_index < 0:
                motion_unavailable_reason = "no_strict_vicon_bracket"
            elif not available:
                motion_unavailable_reason = "current_source_vicon_unavailable"
            elif not person["radial"]["valid"][pair_index]:
                motion_unavailable_reason = "source_continuity_filter_rejected"
            else:
                motion_unavailable_reason = None
            radial = person["radial"]
            motion_state = str(radial["state"][pair_index]) if motion_available else None
            ttc_proxy = float(radial["ttc_proxy_s"][pair_index]) if motion_available and np.isfinite(radial["ttc_proxy_s"][pair_index]) else None
            box_records.append({
                "selected_index": selected_index,
                "archive_timestamp_ns": int(stems[selected_index]),
                "bag_image_timestamp_ns": int(selected_bag_times[frame_slot]),
                "class_id": class_id,
                "class_name": class_names[class_id],
                "stratum": truth["stratum"],
                "normalized_area": truth["normalized_area"],
                "matched_at_fixed_score": truth["matched_at_fixed_score"],
                "vicon_available": available,
                "sensor_sync_delta_ms": float(sensor_sync_ms[frame_slot]),
                "person_sync_delta_ms": float(person["sync_ms"][frame_slot]),
                "sensor_local_xyz_m": [float(value) for value in person["local"][frame_slot]] if available else None,
                "sensor_local_range_m": float(person["range"][frame_slot]) if available else None,
                "same_class_frame_ambiguous": ambiguous,
                "source_motion_available": motion_available,
                "source_motion_unavailable_reason": motion_unavailable_reason,
                "motion_pair_start_timestamp_ns": int(radial["start_timestamp_ns"][pair_index]) if motion_available else None,
                "motion_pair_end_timestamp_ns": int(radial["end_timestamp_ns"][pair_index]) if motion_available else None,
                "motion_pair_midpoint_timestamp_ns": int(radial["midpoint_timestamp_ns"][pair_index]) if motion_available else None,
                "image_to_motion_midpoint_delta_ms": abs(float(selected_bag_times[frame_slot] - radial["midpoint_timestamp_ns"][pair_index])) / 1e6 if motion_available else None,
                "motion_dt_s": float(radial["interval_s"][pair_index]) if motion_available else None,
                "source_range_start_m": float(radial["range_start_m"][pair_index]) if motion_available else None,
                "source_range_end_m": float(radial["range_end_m"][pair_index]) if motion_available else None,
                "source_range_midpoint_m": float(radial["range_midpoint_m"][pair_index]) if motion_available else None,
                "source_radial_range_rate_mps": float(radial["range_rate_mps"][pair_index]) if motion_available else None,
                "source_radial_motion": motion_state,
                "source_ttc_proxy_s": ttc_proxy,
                "source_ttc_proxy_unavailable_reason": None if ttc_proxy is not None else ("not_approaching" if motion_available else motion_unavailable_reason),
            })

    summary = _summarize(box_records, class_names)
    return {
        "format": "blindassist_revel_detector_vicon_failure_alignment_v2",
        "source": {
            "benchmark_sha256": _sha256(benchmark_path),
            "details_sha256": _sha256(details_path),
            "vicon_audit_sha256": _sha256(vicon_audit_path),
            "bag_sha256_from_audit": vicon_audit.get("source", {}).get("sha256"),
            "bag_sha256_verified": bag_sha256,
            "selected_indices_sha256": hashlib.sha256((",".join(str(value) for value in selected_indices) + "\n").encode("ascii")).hexdigest(),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "sync_max_delta_ms": SYNC_MAX_DELTA_MS,
            "motion_contract": {
                "timestamp_basis": "rosbag record time",
                "pair_selection": "strict native Vicon person-pose bracket around each bag image timestamp",
                "minimum_continuous_interval_s": MIN_CONTINUOUS_INTERVAL_S,
                "maximum_continuous_interval_s": MAX_CONTINUOUS_INTERVAL_S,
                "maximum_single_track_world_speed_mps": MAX_CONTINUOUS_WORLD_SPEED_MPS,
                "approach_recede_deadband_mps": RADIAL_DEADBAND_MPS,
                "range_rate_sign": "negative=approaching; positive=receding",
                "ttc_proxy_formula": "source_range_midpoint_m / -source_radial_range_rate_mps; approaching only",
                "ttc_proxy_bins_s": [name for _, _, name in TTC_PROXY_BINS],
                "offline_noncausal": True,
            },
            "compute_backend": {"name": "numpy", "device": "cpu", "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        },
        "summary": summary,
        "box_records": box_records,
        "admission": {
            "source_detector_range_stratification_admitted": True,
            "source_detector_radial_motion_stratification_admitted": True,
            "admitted_for": ["offline source-native radial range-rate, approach/recede, and TTC-proxy detector stratification"],
            "not_admitted_for": ["user-body distance", "physical assistive TTC", "body-local safe corridor", "assistive event truth", "on-device safety"],
            "reason": "helmet and event/LiDAR markers are source Vicon points; the noncausal bracketed constant-radial-rate proxy has no body envelope, closest-approach, acceleration, or assistive-event truth",
        },
        "production_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--image-label-root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--vicon-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-details-output", type=Path)
    args = parser.parse_args()
    report = align(args.bag_root, args.image_label_root, args.benchmark, args.details, args.vicon_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    motion_details_output = args.motion_details_output or args.output.with_name(args.output.stem + ".details.jsonl")
    motion_details_output.parent.mkdir(parents=True, exist_ok=True)
    motion_details_output.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for record in report["box_records"]),
        encoding="utf-8",
    )
    report["source"]["box_records_receipt"] = {
        "path": str(motion_details_output),
        "records": len(report["box_records"]),
        "sha256": _sha256(motion_details_output),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "boxes": report["summary"]["box_count"],
        "range_aligned": report["summary"]["vicon_aligned_box_count"],
        "motion_aligned": report["summary"]["source_motion_aligned_box_count"],
        "recall_by_motion": report["summary"]["recall_by_source_radial_motion"],
        "recall_by_ttc_proxy": report["summary"]["recall_by_source_ttc_proxy"],
    }, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
