#!/usr/bin/env python3
"""Groundtruth-only ADT geometry proposal producer.

This process never reads RGB/VRS, candidate signals, routes, lifecycle state, or
old-window outcomes. The 2D box file is reduced to boolean visibility receipts.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DATA_URL = "https://explorer.projectaria.com/data/adt"
DEVICE_STREAM = "214-1"
WINDOW_NS = 10_000_000_000
EPOCH_NS = 500_000_000
OBJECT_TOLERANCE_NS = 1_000
VISIBILITY_THRESHOLD = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("median of empty sequence")
    return float(statistics.median(materialized))


def quantile_nearest_rank(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile of empty sequence")
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return float(ordered[index])


def quaternion_rotation_wxyz(q: tuple[float, float, float, float]) -> list[list[float]]:
    if not all(math.isfinite(value) for value in q):
        raise ValueError("quaternion contains non-finite value")
    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("quaternion norm outside tolerance")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    for row in rotation:
        if abs(sum(value * value for value in row) - 1.0) > 1e-3:
            raise ValueError("rotation row is not unit")
    for left in range(3):
        for right in range(left + 1, 3):
            if abs(sum(rotation[left][i] * rotation[right][i] for i in range(3))) > 1e-3:
                raise ValueError("rotation rows are not orthogonal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-3:
        raise ValueError("rotation determinant is not +1")
    return rotation


def point_to_oriented_box_distance(
    point_world: tuple[float, float, float],
    center_world: tuple[float, float, float],
    rotation_world_from_object: list[list[float]],
    bounds: tuple[float, float, float, float, float, float],
) -> float:
    delta = [point_world[i] - center_world[i] for i in range(3)]
    local = [
        sum(rotation_world_from_object[row][axis] * delta[row] for row in range(3))
        for axis in range(3)
    ]
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    outside = (
        max(xmin - local[0], 0.0, local[0] - xmax),
        max(ymin - local[1], 0.0, local[1] - ymax),
        max(zmin - local[2], 0.0, local[2] - zmax),
    )
    return math.sqrt(sum(value * value for value in outside))


def nearest_index(times: list[int], query: int, tolerance: int) -> int | None:
    pos = bisect.bisect_left(times, query)
    candidates = [index for index in (pos - 1, pos) if 0 <= index < len(times)]
    if not candidates:
        return None
    distances = [abs(times[index] - query) for index in candidates]
    minimum = min(distances)
    if minimum > tolerance or distances.count(minimum) != 1:
        return None
    return candidates[distances.index(minimum)]


def load_visibility(zf: zipfile.ZipFile) -> dict[str, set[int]]:
    member = (
        "2d_bounding_box_with_skeleton.csv"
        if "2d_bounding_box_with_skeleton.csv" in zf.namelist()
        else "2d_bounding_box.csv"
    )
    visible: dict[str, set[int]] = defaultdict(set)
    with zf.open(member) as stream:
        header = stream.readline().decode("utf-8").rstrip("\r\n").split(",")
        expected = {
            "stream_id": 0,
            "object_uid": 1,
            "timestamp[ns]": 2,
            "visibility_ratio[%]": 7,
        }
        for field, index in expected.items():
            if header[index] != field:
                raise AssertionError(f"unexpected visibility schema: {member}")
        for raw_line in stream:
            fields = raw_line.decode("utf-8").rstrip("\r\n").split(",")
            if fields[0] != DEVICE_STREAM:
                continue
            if float(fields[7]) >= VISIBILITY_THRESHOLD:
                timestamp = int(fields[2])
                if timestamp in visible[fields[1]]:
                    raise ValueError("duplicate visibility identity/timestamp")
                visible[fields[1]].add(timestamp)
    return visible


def load_sequence(archive: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zf:
        metadata = json.load(zf.open("metadata.json"))
        if (
            metadata.get("dataset_version") != "2.0"
            or metadata.get("gt_time_domain") != "DEVICE_CAPTURE"
        ):
            raise ValueError("unsupported ADT time contract")
        instances = json.load(zf.open("instances.json"))
        visible = load_visibility(zf)

        with zf.open("aria_trajectory.csv") as stream:
            reader = csv.DictReader(line.decode("utf-8") for line in stream)
            device = []
            for row in reader:
                quaternion_rotation_wxyz(
                    (
                        float(row["qw_world_device"]),
                        float(row["qx_world_device"]),
                        float(row["qy_world_device"]),
                        float(row["qz_world_device"]),
                    )
                )
                device.append(
                    {
                        "time_ns": int(row["tracking_timestamp_us"]) * 1000,
                        "position": (
                            float(row["tx_world_device"]),
                            float(row["ty_world_device"]),
                            float(row["tz_world_device"]),
                        ),
                        "angular_speed": math.sqrt(
                            sum(
                                float(row[field]) ** 2
                                for field in (
                                    "angular_velocity_x_device",
                                    "angular_velocity_y_device",
                                    "angular_velocity_z_device",
                                )
                            )
                        ),
                    }
                )

        boxes: dict[str, tuple[float, float, float, float, float, float]] = {}
        with zf.open("3d_bounding_box.csv") as stream:
            reader = csv.DictReader(line.decode("utf-8") for line in stream)
            for row in reader:
                if row["object_uid"] in boxes:
                    raise ValueError("duplicate 3D box identity")
                bounds = tuple(
                    float(row[field])
                    for field in (
                        "p_local_obj_xmin[m]",
                        "p_local_obj_xmax[m]",
                        "p_local_obj_ymin[m]",
                        "p_local_obj_ymax[m]",
                        "p_local_obj_zmin[m]",
                        "p_local_obj_zmax[m]",
                    )
                )
                diagonal = math.sqrt(
                    (bounds[1] - bounds[0]) ** 2
                    + (bounds[3] - bounds[2]) ** 2
                    + (bounds[5] - bounds[4]) ** 2
                )
                if (
                    bounds[1] > bounds[0]
                    and bounds[3] > bounds[2]
                    and bounds[5] > bounds[4]
                    and diagonal >= 0.20
                ):
                    boxes[row["object_uid"]] = bounds

        static: dict[str, dict[str, Any]] = {}
        dynamic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with zf.open("scene_objects.csv") as stream:
            reader = csv.DictReader(line.decode("utf-8") for line in stream)
            for row in reader:
                uid = row["object_uid"]
                if uid not in boxes:
                    continue
                pose = {
                    "time_ns": int(row["timestamp[ns]"]),
                    "position": (
                        float(row["t_wo_x[m]"]),
                        float(row["t_wo_y[m]"]),
                        float(row["t_wo_z[m]"]),
                    ),
                    "rotation": quaternion_rotation_wxyz(
                        (
                            float(row["q_wo_w"]),
                            float(row["q_wo_x"]),
                            float(row["q_wo_y"]),
                            float(row["q_wo_z"]),
                        )
                    ),
                }
                if pose["time_ns"] == -1:
                    if uid in static:
                        raise ValueError("duplicate static object pose")
                    static[uid] = pose
                else:
                    dynamic[uid].append(pose)
        for rows in dynamic.values():
            rows.sort(key=lambda row: row["time_ns"])
            if len({row["time_ns"] for row in rows}) != len(rows):
                raise ValueError("duplicate dynamic object identity/timestamp")

    return {
        "metadata": metadata,
        "instances": instances,
        "visible": visible,
        "device": device,
        "boxes": boxes,
        "static": static,
        "dynamic": dynamic,
    }


def target_name(instances: dict[str, Any], uid: str) -> str:
    item = instances.get(uid, {})
    return str(item.get("instance_name") or item.get("prototype_name") or uid)


def build_target_series(
    source: dict[str, Any], uid: str, target_type: str
) -> list[dict[str, Any]]:
    visibility_times = sorted(source["visible"].get(uid, set()))
    if not visibility_times:
        return []
    dynamic_rows = source["dynamic"].get(uid, [])
    dynamic_times = [row["time_ns"] for row in dynamic_rows]
    series: list[dict[str, Any]] = []
    for device in source["device"]:
        time_ns = device["time_ns"]
        visible_index = nearest_index(visibility_times, time_ns, OBJECT_TOLERANCE_NS)
        if visible_index is None:
            continue
        if target_type == "STATIC_OBJECT":
            pose = source["static"][uid]
        else:
            pose_index = nearest_index(dynamic_times, time_ns, OBJECT_TOLERANCE_NS)
            if pose_index is None:
                continue
            pose = dynamic_rows[pose_index]
        distance = point_to_oriented_box_distance(
            device["position"], pose["position"], pose["rotation"], source["boxes"][uid]
        )
        series.append(
            {
                "time_ns": time_ns,
                "range_m": distance,
                "target_position": pose["position"],
            }
        )
    return series


def epoch_values(
    series: list[dict[str, Any]], start_ns: int
) -> tuple[list[float], list[float]] | None:
    ranges: list[list[float]] = [[] for _ in range(20)]
    closings: list[list[float]] = [[] for _ in range(20)]
    previous: dict[str, Any] | None = None
    for sample in series:
        if not start_ns <= sample["time_ns"] < start_ns + WINDOW_NS:
            continue
        epoch = int((sample["time_ns"] - start_ns) // EPOCH_NS)
        ranges[epoch].append(sample["range_m"])
        if previous is not None:
            dt = (sample["time_ns"] - previous["time_ns"]) / 1e9
            if 0 < dt <= 0.1:
                closing = -(
                    math.log(max(sample["range_m"], 0.20))
                    - math.log(max(previous["range_m"], 0.20))
                ) / dt
                closings[epoch].append(closing)
        previous = sample
    if any(len(values) < 10 for values in ranges):
        return None
    if any(len(values) < 9 for values in closings):
        return None
    return (
        [median(values) for values in ranges],
        [median(values) for values in closings],
    )


def device_window(source: dict[str, Any], start_ns: int) -> dict[str, float] | None:
    rows = [
        row
        for row in source["device"]
        if start_ns <= row["time_ns"] < start_ns + WINDOW_NS
    ]
    if len(rows) < 270:
        return None
    if rows[-1]["time_ns"] - rows[0]["time_ns"] < 9_900_000_000:
        return None
    gaps = [
        rows[index]["time_ns"] - rows[index - 1]["time_ns"]
        for index in range(1, len(rows))
    ]
    if gaps and max(gaps) > 100_000_000:
        return None
    displacement = math.dist(rows[0]["position"], rows[-1]["position"])
    path = sum(
        math.dist(rows[index - 1]["position"], rows[index]["position"])
        for index in range(1, len(rows))
    )
    return {
        "sample_count": float(len(rows)),
        "duration_s": (rows[-1]["time_ns"] - rows[0]["time_ns"]) / 1e9,
        "endpoint_displacement_m": displacement,
        "path_length_m": path,
        "angular_speed_median_rad_s": median(row["angular_speed"] for row in rows),
    }


def classify(
    target_type: str,
    target_series: list[dict[str, Any]],
    range_epochs: list[float],
    closing_epochs: list[float],
    device_metrics: dict[str, float],
    start_ns: int,
) -> list[str]:
    positive_samples: list[float] = []
    previous: dict[str, Any] | None = None
    window_samples = [
        sample
        for sample in target_series
        if start_ns <= sample["time_ns"] < start_ns + WINDOW_NS
    ]
    for sample in window_samples:
        if previous is not None:
            dt = (sample["time_ns"] - previous["time_ns"]) / 1e9
            if 0 < dt <= 0.1:
                closing = -(
                    math.log(max(sample["range_m"], 0.20))
                    - math.log(max(previous["range_m"], 0.20))
                ) / dt
                positive_samples.append(max(closing, 0.0))
        previous = sample

    r_start, r_end = range_epochs[0], range_epochs[-1]
    r_min = min(range_epochs)
    min_index = range_epochs.index(r_min)
    displacement = device_metrics["endpoint_displacement_m"]
    path = device_metrics["path_length_m"]
    angular = device_metrics["angular_speed_median_rad_s"]
    median_positive = median(max(value, 0.0) for value in closing_epochs)
    target_displacement = math.dist(
        window_samples[0]["target_position"], window_samples[-1]["target_position"]
    )

    cells: list[str] = []
    if (
        displacement <= 0.35
        and path <= 0.75
        and angular >= 0.25
        and abs(r_end - r_start) <= 0.25
        and quantile_nearest_rank(positive_samples, 0.90) <= 0.05
    ):
        cells.append("PURE_EGO_ROTATION_NO_CLOSING")
    if (
        target_type == "STATIC_OBJECT"
        and displacement >= 0.75
        and r_start - r_end >= 0.60
        and r_end <= 3.0
        and median_positive >= 0.025
    ):
        cells.append("EGO_APPROACH_STATIC_SURFACE")
    if (
        target_type == "TIMESTAMPED_OBJECT"
        and displacement <= 0.35
        and path <= 0.75
        and target_displacement >= 0.60
        and r_start - r_end >= 0.60
        and r_end <= 3.0
        and median_positive >= 0.025
    ):
        cells.append("STATIONARY_EGO_ACTIVE_TARGET_APPROACH")
    if (
        4 <= min_index <= 15
        and r_start - r_min >= 0.50
        and r_end - r_min >= 0.50
        and median(closing_epochs[:6]) >= 0.02
        and median(closing_epochs[-6:]) <= -0.02
        and abs(r_end - r_start) <= 0.50
    ):
        cells.append("LATERAL_PASS_NO_SUSTAINED_CLOSING")
    return cells


def analyze_sequence(
    sequence_id: str, archive: Path, component_id: str
) -> dict[str, Any]:
    source = load_sequence(archive)
    device_token = sequence_id.rsplit("_", 1)[-1]
    if not source["metadata"]["serial"].endswith(device_token):
        return {
            "sequence_id": sequence_id,
            "component_id": component_id,
            "status": "ABSTAIN_SERIAL_MISMATCH",
            "proposals": [],
        }
    targets = [
        ("STATIC_OBJECT", uid) for uid in source["static"] if uid in source["boxes"]
    ] + [
        ("TIMESTAMPED_OBJECT", uid)
        for uid in source["dynamic"]
        if uid in source["boxes"]
    ]
    proposals: list[dict[str, Any]] = []
    first_time = source["device"][0]["time_ns"]
    last_time = source["device"][-1]["time_ns"]
    window_starts = list(range(first_time, last_time - WINDOW_NS + 1, WINDOW_NS))
    for target_type, uid in sorted(targets):
        series = build_target_series(source, uid, target_type)
        if not series:
            continue
        for start_ns in window_starts:
            metrics = device_window(source, start_ns)
            if metrics is None:
                continue
            window_series = [
                sample
                for sample in series
                if start_ns <= sample["time_ns"] < start_ns + WINDOW_NS
            ]
            if len(window_series) < math.ceil(0.90 * metrics["sample_count"]):
                continue
            epochs = epoch_values(series, start_ns)
            if epochs is None:
                continue
            range_epochs, closing_epochs = epochs
            cells = classify(
                target_type,
                series,
                range_epochs,
                closing_epochs,
                metrics,
                start_ns,
            )
            for cell in cells:
                proposals.append(
                    {
                        "cell": cell,
                        "start_ns": start_ns,
                        "end_ns_exclusive": start_ns + WINDOW_NS,
                        "target_type": target_type,
                        "target_uid": uid,
                        "target_name": target_name(source["instances"], uid),
                        "device": metrics,
                        "range_epoch_median_m": range_epochs,
                        "closing_epoch_median_per_s": closing_epochs,
                        "r_start_m": range_epochs[0],
                        "r_end_m": range_epochs[-1],
                        "r_min_m": min(range_epochs),
                    }
                )
    chosen: list[dict[str, Any]] = []
    for cell in (
        "PURE_EGO_ROTATION_NO_CLOSING",
        "EGO_APPROACH_STATIC_SURFACE",
        "STATIONARY_EGO_ACTIVE_TARGET_APPROACH",
        "LATERAL_PASS_NO_SUSTAINED_CLOSING",
    ):
        eligible = [proposal for proposal in proposals if proposal["cell"] == cell]
        if eligible:
            chosen.append(
                sorted(
                    eligible,
                    key=lambda proposal: (
                        proposal["start_ns"],
                        proposal["target_type"],
                        proposal["target_uid"],
                    ),
                )[0]
            )
    return {
        "sequence_id": sequence_id,
        "component_id": component_id,
        "status": "GEOMETRY_PROPOSALS_COMPLETE",
        "archive_sha1": sha1(archive),
        "bbox_coordinate_field_access_count": 0,
        "proposals": chosen,
    }


def build_components(
    selected_ids: list[str], metadata_by_sequence: dict[str, dict[str, Any]]
) -> tuple[dict[str, str], dict[str, str]]:
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        all_ids = set(json.load(response))
    if len(all_ids) != 236:
        raise AssertionError(f"unexpected ADT sequence inventory: {len(all_ids)}")
    base_groups: dict[str, list[str]] = defaultdict(list)
    for sequence_id in all_ids:
        base_groups[sequence_id.rsplit("_", 1)[0]].append(sequence_id)
    invalid: dict[str, str] = {}
    component: dict[str, str] = {}
    for sequence_id in selected_ids:
        siblings = sorted(base_groups[sequence_id.rsplit("_", 1)[0]])
        concurrent = metadata_by_sequence[sequence_id].get("concurrent_sequence", "")
        members = set(siblings)
        if concurrent:
            if concurrent not in all_ids or concurrent not in metadata_by_sequence:
                invalid[sequence_id] = "ABSTAIN_CONCURRENT_REFERENCE_UNVERIFIED"
                continue
            reciprocal = metadata_by_sequence[concurrent].get("concurrent_sequence", "")
            if reciprocal != sequence_id:
                invalid[sequence_id] = "ABSTAIN_CONCURRENT_REFERENCE_NOT_RECIPROCAL"
                continue
            members.add(concurrent)
        component_id = hashlib.sha256(
            "\n".join(sorted(members)).encode("utf-8")
        ).hexdigest()[:16]
        component[sequence_id] = component_id
    return component, invalid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    acquisition = json.loads(args.acquisition.read_text(encoding="utf-8"))
    assert freeze["terminal"] == "ADT_GROUNDTRUTH_PRESCREEN_SELECTION_FROZEN"
    assert acquisition["terminal"] == "ADT_GROUNDTRUTH_PRESCREEN_PAYLOAD_ACQUIRED"
    assert acquisition["rgb_or_vrs_member_count"] == 0
    members = {row["sequence_id"]: row for row in acquisition["members"]}
    frozen_ids = {row["sequence_id"] for row in freeze["selections"]}
    if (
        len(acquisition["members"]) != 16
        or len(members) != 16
        or set(members) != frozen_ids
    ):
        raise AssertionError("acquisition members do not exactly match freeze")

    metadata_by_sequence: dict[str, dict[str, Any]] = {}
    for sequence_id, member in members.items():
        archive = Path(member["path"])
        if archive.stat().st_size != member["bytes"] or sha1(archive) != member["sha1"]:
            raise AssertionError(f"ADT member identity mismatch: {sequence_id}")
        with zipfile.ZipFile(archive) as zf:
            metadata_by_sequence[sequence_id] = json.load(zf.open("metadata.json"))

    components, invalid_components = build_components(
        list(members), metadata_by_sequence
    )
    sequence_results: list[dict[str, Any]] = []
    for selection in freeze["selections"]:
        sequence_id = selection["sequence_id"]
        if sequence_id in invalid_components:
            sequence_results.append(
                {
                    "sequence_id": sequence_id,
                    "status": invalid_components[sequence_id],
                    "proposals": [],
                }
            )
            continue
        sequence_results.append(
            analyze_sequence(
                sequence_id, Path(members[sequence_id]["path"]), components[sequence_id]
            )
        )

    counts = defaultdict(int)
    for result in sequence_results:
        for proposal in result["proposals"]:
            counts[proposal["cell"]] += 1
    receipt = {
        "schema_version": "adt_geometry_cell_prescreen_proposals_r0",
        "source_id": "ARIA_DIGITAL_TWIN",
        "cohort_role": "SOURCE_PRESCREEN_ONLY",
        "freeze_sha256": sha256(args.freeze),
        "acquisition_sha256": sha256(args.acquisition),
        "sequence_count": len(sequence_results),
        "component_count": len(set(components.values())),
        "accepted_eligible_object_proposal_counts": dict(sorted(counts.items())),
        "sequence_results": sequence_results,
        "visibility_contract": {
            "stream_id": DEVICE_STREAM,
            "minimum_visibility_ratio": VISIBILITY_THRESHOLD,
            "bbox_coordinate_field_access_count": 0,
        },
        "candidate_signal_read_count": 0,
        "old_window_or_outcome_read_count": 0,
        "rgb_or_vrs_read_count": 0,
        "role_split_frozen": False,
        "cell_review_complete": False,
        "skeleton_diagnostic_proposal_coverage": "NOT_IMPLEMENTED",
        "terminal": "ADT_GEOMETRY_CELL_PROPOSALS_READY_FOR_INDEPENDENT_REVIEW",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "VALID",
                "terminal": receipt["terminal"],
                "sequence_count": receipt["sequence_count"],
                "component_count": receipt["component_count"],
                "accepted_eligible_object_proposal_counts": receipt[
                    "accepted_eligible_object_proposal_counts"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
