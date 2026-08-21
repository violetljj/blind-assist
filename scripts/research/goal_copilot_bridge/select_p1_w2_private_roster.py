#!/usr/bin/env python3
"""Mechanically freeze the P1-W2 private roster from verified ADT truth."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


RGB_STREAM = "214-1"
SOURCE_VISIBILITY_MIN = 0.75
SOURCE_MIN_SIDE_PX = 24.0
PROBE_VISIBILITY_MIN = 0.50
PROBE_MIN_SIDE_PX = 16.0
REAPPEARANCE_GAP_NS = 500_000_000
ALIGNMENT_TOLERANCE_NS = 20_000_000
PROBE_STRATA = (
    "ROTATION_DOMINANT",
    "SMALL_TRANSLATION",
    "LARGE_TRANSLATION",
    "REAPPEARANCE",
    "SAME_SCENE_CONFUSER",
)


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def csv_rows(archive: zipfile.ZipFile, member: str) -> Iterable[dict[str, str]]:
    with archive.open(member) as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))


def nearest_index(times: list[int], timestamp: int) -> int | None:
    position = bisect.bisect_left(times, timestamp)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(times)]
    if not candidates:
        return None
    index = min(candidates, key=lambda item: abs(times[item] - timestamp))
    return index if abs(times[index] - timestamp) <= ALIGNMENT_TOLERANCE_NS else None


def video_timestamps(path: Path) -> list[int]:
    import av

    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        description = container.metadata.get("description")
        if len(streams) != 1 or not description:
            raise ValueError("RGB preview lacks one timestamped video stream")
        timestamps = [int(value) for value in json.loads(description)]
        if not timestamps or any(right < left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("RGB preview timestamps are empty or decrease")
        if streams[0].frames and streams[0].frames != len(timestamps):
            raise ValueError("RGB preview frame count differs from timestamp metadata")
        return timestamps


def bbox(row: dict[str, str]) -> dict[str, float]:
    return {
        "visibility": float(row["visibility_ratio[%]"]),
        "x_min": float(row["x_min[pixel]"]),
        "y_min": float(row["y_min[pixel]"]),
        "x_max": float(row["x_max[pixel]"]),
        "y_max": float(row["y_max[pixel]"]),
    }


def bbox_min_side(row: dict[str, float]) -> float:
    return min(row["x_max"] - row["x_min"], row["y_max"] - row["y_min"])


def bbox_list(row: dict[str, float]) -> list[float]:
    return [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]


def distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.sqrt(
        sum((right[key] - left[key]) ** 2 for key in ("tx_world_device", "ty_world_device", "tz_world_device"))
    )


def rotation_deg(left: dict[str, float], right: dict[str, float]) -> float:
    keys = ("qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device")
    dot = abs(sum(left[key] * right[key] for key in keys))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_parent(gt_path: Path, rgb_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(gt_path) as archive:
        names = set(archive.namelist())
        required = {"instances.json", "aria_trajectory.csv"}
        if missing := sorted(required - names):
            raise ValueError(f"required GT members missing: {missing}")
        bbox_member = (
            "2d_bounding_box_with_skeleton.csv"
            if "2d_bounding_box_with_skeleton.csv" in names
            else "2d_bounding_box.csv"
        )
        if bbox_member not in names:
            raise ValueError("2D bounding-box truth missing")
        raw_instances = json.load(archive.open("instances.json"))
        instances = {
            str(uid): value
            for uid, value in raw_instances.items()
            if isinstance(value, dict)
            and value.get("instance_type") == "object"
            and value.get("rigidity") == "rigid"
            and value.get("motion_type") == "static"
        }
        pose_keys = (
            "tx_world_device",
            "ty_world_device",
            "tz_world_device",
            "qx_world_device",
            "qy_world_device",
            "qz_world_device",
            "qw_world_device",
        )
        trajectory_rows = list(csv_rows(archive, "aria_trajectory.csv"))
        times = [int(row["tracking_timestamp_us"]) * 1000 for row in trajectory_rows]
        poses = {times[index]: {key: float(row[key]) for key in pose_keys} for index, row in enumerate(trajectory_rows)}
        boxes: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
        for row in csv_rows(archive, bbox_member):
            uid = str(row["object_uid"])
            if row["stream_id"] != RGB_STREAM or uid not in instances:
                continue
            timestamp = int(row["timestamp[ns]"])
            index = nearest_index(times, timestamp)
            if index is not None:
                boxes[uid][times[index]] = bbox(row)

    rgb_times = video_timestamps(rgb_path)
    aligned_times: list[int] = []
    frame_index: dict[int, int] = {}
    for timestamp in times:
        index = nearest_index(rgb_times, timestamp)
        if index is not None:
            aligned_times.append(timestamp)
            frame_index[timestamp] = index
    if len(aligned_times) < len(times) - 1:
        raise ValueError(f"only {len(aligned_times)}/{len(times)} GT timestamps align to RGB within 20ms")
    return {
        "instances": instances,
        "times": aligned_times,
        "poses": poses,
        "boxes": dict(boxes),
        "frame_index": frame_index,
        "rgb_frame_count": len(rgb_times),
    }


def confuser_pool(instances: dict[str, dict[str, Any]], target_uid: str) -> list[tuple[int, str]]:
    target = instances[target_uid]
    prototype = target.get("prototype_name")
    category_uid = target.get("category_uid")
    result: list[tuple[int, str]] = []
    for uid, instance in instances.items():
        if uid == target_uid:
            continue
        if prototype and instance.get("prototype_name") == prototype:
            result.append((0, uid))
        elif category_uid is not None and instance.get("category_uid") == category_uid:
            result.append((1, uid))
    return result


def usable_probe(row: dict[str, float] | None) -> bool:
    return bool(row and row["visibility"] >= PROBE_VISIBILITY_MIN and bbox_min_side(row) >= PROBE_MIN_SIDE_PX)


def reappearance_times(times: list[int], rows: dict[int, dict[str, float]]) -> set[int]:
    result: set[int] = set()
    gap_start: int | None = None
    seen_visible = False
    for timestamp in times:
        visible = usable_probe(rows.get(timestamp))
        if visible:
            if seen_visible and gap_start is not None and timestamp - gap_start >= REAPPEARANCE_GAP_NS:
                result.add(timestamp)
            seen_visible = True
            gap_start = None
        elif seen_visible and gap_start is None:
            gap_start = timestamp
    return result


def motion_strata(source_pose: dict[str, float], probe_pose: dict[str, float]) -> set[str]:
    translation = distance(source_pose, probe_pose)
    rotation = rotation_deg(source_pose, probe_pose)
    result = set()
    if translation <= 0.10 and rotation >= 15.0:
        result.add("ROTATION_DOMINANT")
    if 0.10 < translation <= 0.75:
        result.add("SMALL_TRANSLATION")
    if translation > 0.75:
        result.add("LARGE_TRANSLATION")
    return result


def select_parent(
    *,
    salt: str,
    sequence_id: str,
    parent_id: str,
    data: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    instances = data["instances"]
    times = data["times"]
    poses = data["poses"]
    boxes = data["boxes"]
    source_rows: list[tuple[str, int]] = []
    targets_with_confusers = 0
    for uid, rows in boxes.items():
        pool = confuser_pool(instances, uid)
        if pool:
            targets_with_confusers += 1
        for timestamp, row in rows.items():
            if (
                timestamp in data["frame_index"]
                and row["visibility"] >= SOURCE_VISIBILITY_MIN
                and bbox_min_side(row) >= SOURCE_MIN_SIDE_PX
            ):
                source_rows.append((uid, timestamp))
    if not source_rows:
        return "PARENT_NO_VALID_SOURCE", [], [], {"valid_source_candidates": 0, "targets_with_confuser_pool": targets_with_confusers}

    source_rows.sort(key=lambda item: stable_hash(salt, "source", sequence_id, item[0], item[1]))
    selected: tuple[str, int, dict[str, list[int]], dict[int, list[tuple[int, str]]]] | None = None
    for uid, source_timestamp in source_rows:
        pool = confuser_pool(instances, uid)
        if not pool:
            continue
        target_rows = boxes[uid]
        returns = reappearance_times(times, target_rows)
        by_stratum: dict[str, list[int]] = defaultdict(list)
        visible_confusers: dict[int, list[tuple[int, str]]] = {}
        for probe_timestamp in times:
            if probe_timestamp <= source_timestamp or not usable_probe(target_rows.get(probe_timestamp)):
                continue
            confusers = [
                (priority, other_uid)
                for priority, other_uid in pool
                if usable_probe(boxes.get(other_uid, {}).get(probe_timestamp))
            ]
            if confusers:
                visible_confusers[probe_timestamp] = confusers
                by_stratum["SAME_SCENE_CONFUSER"].append(probe_timestamp)
            for stratum in motion_strata(poses[source_timestamp], poses[probe_timestamp]):
                by_stratum[stratum].append(probe_timestamp)
            if probe_timestamp in returns:
                by_stratum["REAPPEARANCE"].append(probe_timestamp)
        has_viewpoint = any(by_stratum[name] for name in PROBE_STRATA[:-1])
        if by_stratum["SAME_SCENE_CONFUSER"] and has_viewpoint:
            selected = (uid, source_timestamp, dict(by_stratum), visible_confusers)
            break
    if selected is None:
        status = "PARENT_NO_CONFUSER_SUPPORT" if targets_with_confusers == 0 else "PARENT_INSUFFICIENT_PROBE_STRATA"
        return status, [], [], {"valid_source_candidates": len(source_rows), "targets_with_confuser_pool": targets_with_confusers}

    uid, source_timestamp, by_stratum, visible_confusers = selected
    selected_by_time: dict[int, set[str]] = defaultdict(set)
    for stratum in PROBE_STRATA:
        candidates = by_stratum.get(stratum, [])
        if not candidates:
            continue
        probe_timestamp = min(
            candidates,
            key=lambda timestamp: stable_hash(
                salt, "probe", sequence_id, uid, source_timestamp, stratum, timestamp
            ),
        )
        selected_by_time[probe_timestamp].add(stratum)

    provider_cases: list[dict[str, Any]] = []
    private_cases: list[dict[str, Any]] = []
    source_row = boxes[uid][source_timestamp]
    for ordinal, probe_timestamp in enumerate(sorted(selected_by_time), start=1):
        case_id = f"{parent_id}-pair-{ordinal:02d}"
        pool = visible_confusers.get(probe_timestamp, [])
        pool = sorted(
            pool,
            key=lambda item: (item[0], stable_hash(salt, "confuser", sequence_id, probe_timestamp, item[1])),
        )[:3]
        members = [("TRUE", uid, boxes[uid][probe_timestamp], -1), *[("CONFUSER", other, boxes[other][probe_timestamp], priority) for priority, other in pool]]
        members.sort(key=lambda item: stable_hash(salt, "shuffle", sequence_id, source_timestamp, probe_timestamp, item[1]))
        candidates = []
        truth_rows = []
        true_candidate_id = None
        for candidate_ordinal, (role, candidate_uid, row, priority) in enumerate(members, start=1):
            candidate_id = f"{case_id}-candidate-{candidate_ordinal:02d}"
            candidates.append({"candidate_id": candidate_id, "core_bbox_xyxy": bbox_list(row)})
            truth_rows.append(
                {
                    "candidate_id": candidate_id,
                    "object_uid": candidate_uid,
                    "role": role,
                    "confuser_priority": None if role == "TRUE" else ("SAME_PROTOTYPE" if priority == 0 else "SAME_CATEGORY"),
                }
            )
            if role == "TRUE":
                true_candidate_id = candidate_id
        provider_cases.append(
            {
                "case_id": case_id,
                "parent_id": parent_id,
                "source_frame_index": data["frame_index"][source_timestamp],
                "source_core_bbox_xyxy": bbox_list(source_row),
                "probe_frame_index": data["frame_index"][probe_timestamp],
                "candidates": candidates,
            }
        )
        private_cases.append(
            {
                "case_id": case_id,
                "sequence_id": sequence_id,
                "source_object_uid": uid,
                "source_timestamp_ns": source_timestamp,
                "probe_timestamp_ns": probe_timestamp,
                "true_candidate_id": true_candidate_id,
                "candidates": truth_rows,
                "translation_m": round(distance(poses[source_timestamp], poses[probe_timestamp]), 6),
                "rotation_deg": round(rotation_deg(poses[source_timestamp], poses[probe_timestamp]), 6),
                "support_buckets": sorted(selected_by_time[probe_timestamp]),
            }
        )
    diagnostics = {
        "valid_source_candidates": len(source_rows),
        "targets_with_confuser_pool": targets_with_confusers,
        "selected_probe_pairs": len(provider_cases),
    }
    return "PARENT_ROSTER_ELIGIBLE", provider_cases, private_cases, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {args.output_dir}")

    freeze_bytes = args.freeze.read_bytes()
    freeze = json.loads(freeze_bytes)
    acquisition_bytes = args.acquisition_receipt.read_bytes()
    acquisition = json.loads(acquisition_bytes)
    if acquisition.get("terminal") != "P1_W2_FRESH_SOURCE_PAYLOAD_ACQUIRED":
        raise RuntimeError("fresh source acquisition is not sealed")
    fresh = freeze["data"]["fresh_proxy"]
    if acquisition["selection_identity_sha256"] != fresh["selection_identity_sha256"]:
        raise RuntimeError("acquisition/freeze selection identity mismatch")
    member_by_key = {(row["sequence_id"], row["manifest_key"]): row for row in acquisition["members"]}
    for row in acquisition["members"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or file_digest(path, "sha1") != row["sha1"]:
            raise RuntimeError(f"acquired payload identity drift: {row['filename']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    provider_cases: list[dict[str, Any]] = []
    private_cases: list[dict[str, Any]] = []
    parent_accounting = []
    salt = fresh["selection_salt"]
    for ordinal, parent in enumerate(fresh["parents"], start=1):
        sequence_id = parent["sequence_id"]
        parent_id = f"p1w2-parent-{ordinal:02d}"
        rgb = member_by_key[(sequence_id, "video_main_rgb")]
        gt = member_by_key[(sequence_id, "main_groundtruth")]
        try:
            data = load_parent(Path(gt["path"]), Path(rgb["path"]))
            status, selected_provider, selected_private, diagnostics = select_parent(
                salt=salt, sequence_id=sequence_id, parent_id=parent_id, data=data
            )
            assets.append(
                {
                    "parent_id": parent_id,
                    "rgb_path": rgb["path"],
                    "rgb_sha256": rgb["sha256"],
                    "rgb_frame_count": data["rgb_frame_count"],
                }
            )
        except Exception as error:
            status = "PAYLOAD_OR_SCHEMA_FAILURE"
            selected_provider = []
            selected_private = []
            diagnostics = {"error_class": type(error).__name__, "error": str(error)}
        provider_cases.extend(selected_provider)
        private_cases.extend(selected_private)
        parent_accounting.append(
            {
                "parent_id": parent_id,
                "status": status,
                "selected_pair_count": len(selected_provider),
                "diagnostics": diagnostics,
            }
        )
        print(json.dumps({"parent_id": parent_id, "status": status, "pairs": len(selected_provider)}))

    support_counts = Counter(bucket for case in private_cases for bucket in case["support_buckets"])
    eligible_parents = sum(row["status"] == "PARENT_ROSTER_ELIGIBLE" for row in parent_accounting)
    missing_strata = [stratum for stratum in PROBE_STRATA if support_counts[stratum] == 0]
    data_support_ok = eligible_parents >= 6 and not missing_strata and support_counts["SAME_SCENE_CONFUSER"] >= 6
    terminal = "P1_W2_FRESH_PRIVATE_ROSTER_FROZEN" if data_support_ok else "P1_W2_NOT_EVALUABLE_DATA_SUPPORT"

    public = {
        "schema_version": "p1_w2_public_roster_v1",
        "protocol": "P1_W2_REFERENT_ANCHOR_INTERFACE_FEASIBILITY",
        "parent_denominator": 8,
        "eligible_parent_count": eligible_parents,
        "pair_count": len(provider_cases),
        "support_counts": {stratum: support_counts[stratum] for stratum in PROBE_STRATA},
        "missing_required_support": missing_strata,
        "parent_status_counts": dict(sorted(Counter(row["status"] for row in parent_accounting).items())),
        "terminal": terminal,
        "execution_authorized": False,
        "claim_ceiling": "FRESH_ADT_INDOOR_OBJECT_PROXY_ROSTER_ONLY_NO_EMPIRICAL_CAPABILITY",
    }
    provider = {
        "schema_version": "p1_w2_provider_input_v1",
        "truth_fields_present": False,
        "assets": assets,
        "cases": provider_cases,
        "model_matcher_identity_call_counts": {"model": 0, "matcher": 0, "identity": 0},
    }
    private = {
        "schema_version": "p1_w2_evaluator_private_truth_map_v1",
        "selection_authority": "ADT_INSTANCE_BBOX_VISIBILITY_AND_CAMERA_POSE_ONLY_NO_RGB_OR_MODEL_OUTPUT",
        "parents": [
            {
                "parent_id": f"p1w2-parent-{index:02d}",
                "sequence_id": parent["sequence_id"],
                "stratum": parent["stratum"],
            }
            for index, parent in enumerate(fresh["parents"], start=1)
        ],
        "cases": private_cases,
    }
    accounting = {
        "schema_version": "p1_w2_parent_accounting_v1",
        "fixed_parent_denominator": 8,
        "parents": parent_accounting,
    }
    paths = {
        "public_roster.json": public,
        "provider_input.json": provider,
        "evaluator_private_truth_map.json": private,
        "parent_accounting.json": accounting,
    }
    for filename, payload in paths.items():
        atomic_json(args.output_dir / filename, payload)
    hashes = {filename: file_digest(args.output_dir / filename) for filename in paths}
    receipt = {
        "schema_version": "p1_w2_private_roster_freeze_receipt_v1",
        "freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "acquisition_receipt_sha256": hashlib.sha256(acquisition_bytes).hexdigest(),
        "selection_identity_sha256": fresh["selection_identity_sha256"],
        "output_sha256": hashes,
        "parent_denominator": 8,
        "eligible_parent_count": eligible_parents,
        "pair_count": len(provider_cases),
        "support_counts": public["support_counts"],
        "terminal": terminal,
        "execution_authorized": False,
        "model_matcher_identity_call_counts": {"model": 0, "matcher": 0, "identity": 0},
    }
    atomic_json(args.output_dir / "roster_freeze_receipt.json", receipt)
    print(json.dumps({"terminal": terminal, "eligible_parents": eligible_parents, "pairs": len(provider_cases), "support_counts": public["support_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
