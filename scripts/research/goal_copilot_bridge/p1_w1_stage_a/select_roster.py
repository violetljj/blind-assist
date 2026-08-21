"""Select the P1-W1 Stage-A roster without reading any arm outcomes.

The selector consumes only the already-consumed P1-D0 episode truth and ADT
camera trajectory.  Public case IDs are opaque; the source/target mapping stays
in a separate evaluator-private file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_p1_w1_stage_a_roster_v1"
PROTOCOL_ID = "BLINDASSIST-P1-W1-STAGE-A-V1"
TRANSLATION_SMALL_MIN_M = 0.10
TRANSLATION_INVALID_MIN_M = 0.75
ROTATION_DOMINANT_MAX_TRANSLATION_M = 0.10
ROTATION_DOMINANT_MIN_DEG = 15.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory(path: Path) -> dict[int, dict[str, float]]:
    with zipfile.ZipFile(path) as archive:
        rows = csv.DictReader(io.TextIOWrapper(archive.open("aria_trajectory.csv"), encoding="utf-8"))
        return {
            int(row["tracking_timestamp_us"]) * 1000: {
                key: float(row[key])
                for key in (
                    "tx_world_device", "ty_world_device", "tz_world_device",
                    "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device",
                )
            }
            for row in rows
        }


def _rotation_supplements(path: Path, limit: int = 2) -> list[dict[str, Any]]:
    """Find low-translation turn-away/return windows using source truth only."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        bbox_member = "2d_bounding_box_with_skeleton.csv" if "2d_bounding_box_with_skeleton.csv" in names else "2d_bounding_box.csv"
        trajectory_rows = list(csv.DictReader(io.TextIOWrapper(archive.open("aria_trajectory.csv"), encoding="utf-8")))
        trajectory = [
            (
                int(row["tracking_timestamp_us"]) * 1000,
                {key: float(row[key]) for key in (
                    "tx_world_device", "ty_world_device", "tz_world_device",
                    "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device",
                )},
            )
            for row in trajectory_rows
        ]
        boxes: dict[str, dict[int, list[float]]] = {}
        for row in csv.DictReader(io.TextIOWrapper(archive.open(bbox_member), encoding="utf-8")):
            if row["stream_id"] != "214-1" or float(row["visibility_ratio[%]"]) < 0.10:
                continue
            timestamp = int(row["timestamp[ns]"])
            nearest = min(trajectory, key=lambda item: abs(item[0] - timestamp))[0]
            if abs(nearest - timestamp) > 20_000_000:
                continue
            boxes.setdefault(str(row["object_uid"]), {})[nearest] = [
                float(row["x_min[pixel]"]), float(row["y_min[pixel]"]),
                float(row["x_max[pixel]"]), float(row["y_max[pixel]"]),
            ]

    candidates = []
    window = 90
    edge = 12
    for start in range(0, len(trajectory) - window + 1, 3):
        rows = trajectory[start:start + window]
        origin = rows[0][1]
        max_translation = max(_distance(origin, row[1]) for row in rows)
        max_rotation = max(_rotation_deg(origin, row[1]) for row in rows)
        return_rotation = _rotation_deg(origin, rows[-1][1])
        if (
            max_translation > ROTATION_DOMINANT_MAX_TRANSLATION_M
            or max_rotation < ROTATION_DOMINANT_MIN_DEG
            or return_rotation > 10.0
        ):
            continue
        times = [row[0] for row in rows]
        for uid, visible in boxes.items():
            first = sum(timestamp in visible for timestamp in times[:edge])
            middle_missing = sum(timestamp not in visible for timestamp in times[edge:-edge])
            last = sum(timestamp in visible for timestamp in times[-edge:])
            if first < 6 or last < 6 or middle_missing < 12:
                continue
            candidates.append({
                "source_object_uid": uid,
                "start_timestamp_ns": times[0],
                "end_timestamp_ns": times[-1],
                "frame_count": window,
                "initial_target_bbox_xyxy": visible[next(timestamp for timestamp in times if timestamp in visible)],
                "motion_summary": {
                    "max_translation_from_start_m": round(max_translation, 6),
                    "max_rotation_from_start_deg": round(max_rotation, 6),
                    "return_rotation_from_start_deg": round(return_rotation, 6),
                },
            })
    candidates.sort(key=lambda row: (-row["motion_summary"]["max_rotation_from_start_deg"], row["start_timestamp_ns"], row["source_object_uid"]))
    selected = []
    used_uids = set()
    for candidate in candidates:
        if candidate["source_object_uid"] in used_uids:
            continue
        selected.append(candidate)
        used_uids.add(candidate["source_object_uid"])
        if len(selected) == limit:
            break
    return selected


def _nearest(rows: dict[int, dict[str, float]], timestamp_ns: int) -> dict[str, float]:
    timestamp = min(rows, key=lambda item: abs(item - timestamp_ns))
    if abs(timestamp - timestamp_ns) > 20_000_000:
        raise ValueError(f"trajectory alignment exceeds 20 ms at {timestamp_ns}")
    return rows[timestamp]


def _distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.sqrt(sum((right[key] - left[key]) ** 2 for key in ("tx_world_device", "ty_world_device", "tz_world_device")))


def _rotation_deg(left: dict[str, float], right: dict[str, float]) -> float:
    keys = ("qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device")
    dot = abs(sum(left[key] * right[key] for key in keys))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def classify_episode(episode: dict[str, Any], trajectory: dict[int, dict[str, float]]) -> tuple[list[str], dict[str, float]]:
    poses = [_nearest(trajectory, int(frame["timestamp_ns"])) for frame in episode["frames"]]
    origin = poses[0]
    max_translation = max(_distance(origin, pose) for pose in poses)
    max_rotation = max(_rotation_deg(origin, pose) for pose in poses)
    buckets = []
    if max_translation <= ROTATION_DOMINANT_MAX_TRANSLATION_M and max_rotation >= ROTATION_DOMINANT_MIN_DEG:
        buckets.append("ROTATION_DOMINANT")
    if TRANSLATION_SMALL_MIN_M < max_translation < TRANSLATION_INVALID_MIN_M:
        buckets.append("SMALL_TRANSLATION")
    if max_translation >= TRANSLATION_INVALID_MIN_M:
        buckets.append("TRANSLATION_BEYOND_TIER0")
    tags = set(episode["temporal_mode_tags"])
    if tags & {"TEMP_OCCLUSION", "OUT_OF_VIEW_RETURN", "LONG_LOSS", "REACQUISITION"}:
        buckets.append("OCCLUSION_OR_REAPPEARANCE")
    if episode["candidate_distractor_instance_ids"]:
        buckets.append("IDENTITY_CONFUSER")
    if any(not frame["target_visible"] for frame in episode["frames"]):
        buckets.append("OBSERVATION_LOSS")
    return buckets, {
        "max_translation_from_start_m": round(max_translation, 6),
        "max_rotation_from_start_deg": round(max_rotation, 6),
    }


def select(cohort_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {output_dir}")
    manifest_path = cohort_dir / "p1_d0_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("role") != "CONSUMED_DEVELOPMENT_ONLY":
        raise ValueError("Stage A only accepts the frozen consumed Development cohort")

    trajectory_by_source = {}
    source_by_id = {}
    source_receipts = []
    for source in manifest["sources"]:
        groundtruth = Path(source["groundtruth_path"])
        if sha256(groundtruth) != source["groundtruth_sha256"]:
            raise ValueError(f"groundtruth hash drift: {source['source_sequence_id']}")
        trajectory_by_source[source["source_sequence_id"]] = _trajectory(groundtruth)
        source_by_id[source["source_sequence_id"]] = (source, groundtruth)
        source_receipts.append({
            "source_sequence_sha256": hashlib.sha256(source["source_sequence_id"].encode()).hexdigest(),
            "groundtruth_sha256": source["groundtruth_sha256"],
            "rgb_video_sha256": source["rgb_video_sha256"],
        })

    episodes = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((cohort_dir / "episodes").glob("*.json"))]
    if [episode["episode_id"] for episode in episodes] != manifest["episode_ids"]:
        raise ValueError("episode order or identity drift")

    public_cases = []
    private_cases = []
    counts: Counter[str] = Counter()
    for ordinal, episode in enumerate(episodes, start=1):
        case_id = f"w1a-{ordinal:03d}"
        buckets, motion = classify_episode(episode, trajectory_by_source[episode["source_sequence_id"]])
        counts.update(buckets)
        public_cases.append({
            "case_id": case_id,
            "support_buckets": buckets,
            "frame_count": episode["frame_count"],
        })
        private_cases.append({
            "case_id": case_id,
            "source_episode_id": episode["episode_id"],
            "physical_target_id": episode["physical_target_id"],
            "motion_summary": motion,
            "support_buckets": buckets,
        })

    if counts["ROTATION_DOMINANT"] == 0:
        supplements = []
        for source_id in sorted(source_by_id):
            source, groundtruth = source_by_id[source_id]
            for candidate in _rotation_supplements(groundtruth):
                supplements.append((source_id, source, candidate))
        supplements.sort(key=lambda item: (-item[2]["motion_summary"]["max_rotation_from_start_deg"], item[0], item[2]["start_timestamp_ns"]))
        used_targets = set()
        for source_id, source, candidate in supplements:
            target_key = (source_id, candidate["source_object_uid"])
            if target_key in used_targets:
                continue
            ordinal = len(public_cases) + 1
            case_id = f"w1a-{ordinal:03d}"
            buckets = ["ROTATION_DOMINANT", "OCCLUSION_OR_REAPPEARANCE", "OBSERVATION_LOSS"]
            public_cases.append({"case_id": case_id, "support_buckets": buckets, "frame_count": candidate["frame_count"]})
            private_cases.append({
                "case_id": case_id,
                "source_sequence_id": source_id,
                "source_object_uid": candidate["source_object_uid"],
                "start_timestamp_ns": candidate["start_timestamp_ns"],
                "end_timestamp_ns": candidate["end_timestamp_ns"],
                "initial_target_bbox_xyxy": candidate["initial_target_bbox_xyxy"],
                "motion_summary": candidate["motion_summary"],
                "support_buckets": buckets,
                "selection_authority": "ADT_TRAJECTORY_AND_2D_BBOX_ONLY_NO_ARM_OUTPUTS",
            })
            counts.update(buckets)
            used_targets.add(target_key)
            if len(used_targets) == 2:
                break

    required = (
        "ROTATION_DOMINANT", "SMALL_TRANSLATION", "TRANSLATION_BEYOND_TIER0",
        "OCCLUSION_OR_REAPPEARANCE", "IDENTITY_CONFUSER", "OBSERVATION_LOSS", "GEOMETRY_DEGENERATE",
    )
    mechanics_fixtures = [{
        "case_id": "w1a-mechanics-001",
        "case_type": "SYNTHETIC_MECHANICS_ONLY",
        "support_buckets": ["GEOMETRY_DEGENERATE"],
        "fixture": "blank_current_frame_after_valid_textured_keyframe",
        "expected": "SPATIAL_ANCHOR_STALE_AND_ZERO_DIRECTIONAL_GUIDANCE",
    }]
    counts["GEOMETRY_DEGENERATE"] += len(mechanics_fixtures)
    missing = [bucket for bucket in required if counts[bucket] == 0]
    public = {
        "schema_version": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "claim_role": "CONSUMED_ADT_DEVELOPMENT_ONLY",
        "selection_rule": "ALL_15_FROZEN_P1_D0_EPISODES_PLUS_AT_MOST_2_ROTATION_SUPPORT_WINDOWS_FROM_SAME_SOURCES_NO_ARM_OUTCOME_ACCESS",
        "source_receipts": source_receipts,
        "thresholds": {
            "rotation_dominant_max_translation_m": ROTATION_DOMINANT_MAX_TRANSLATION_M,
            "rotation_dominant_min_deg": ROTATION_DOMINANT_MIN_DEG,
            "small_translation_open_interval_m": [TRANSLATION_SMALL_MIN_M, TRANSLATION_INVALID_MIN_M],
            "translation_beyond_tier0_min_m": TRANSLATION_INVALID_MIN_M,
        },
        "cases": public_cases,
        "mechanics_fixtures": mechanics_fixtures,
        "support_counts": {bucket: counts[bucket] for bucket in required},
        "missing_required_support": missing,
        "terminal": "STAGE_A_ROSTER_FROZEN" if not missing else "NOT_EVALUABLE_DATA_SUPPORT",
        "performance_execution_authorized": False,
        "stage_b_authorized": False,
        "claim_ceiling": "DATA_SUPPORT_AND_MECHANICS_ONLY_NO_EMPIRICAL_CAPABILITY",
    }
    private = {
        "schema_version": "blindassist_p1_w1_stage_a_private_truth_map_v1",
        "protocol_id": PROTOCOL_ID,
        "selection_authority": "ADT_TRAJECTORY_TEMPORAL_AND_INSTANCE_SUPPORT_ONLY_NO_ARM_OUTPUTS",
        "cases": private_cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "public_roster.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    (output_dir / "evaluator_private_truth_map.json").write_text(json.dumps(private, indent=2) + "\n", encoding="utf-8")
    return public


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.cohort_dir, args.output_dir)
    print(json.dumps({"terminal": result["terminal"], "support_counts": result["support_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
