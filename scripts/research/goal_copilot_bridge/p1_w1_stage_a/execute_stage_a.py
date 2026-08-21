"""One-shot P1-W1 Stage-A preparation, execution, and private evaluation."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .opencv_provider import FrozenRgbProvider
    from .stage_a import adjudicate_stage_a, step
except ImportError:  # Direct-file test discovery keeps this directory on sys.path.
    from opencv_provider import FrozenRgbProvider
    from stage_a import adjudicate_stage_a, step


EXPECTED_PUBLIC_ROSTER_SHA256 = "1969560ba8a3863ad4aef16fca9141602144a4b4555ee38c38ff49b6f62bef70"
EXPECTED_PRIVATE_MAP_SHA256 = "fd23bb01d928fdf97d65fa0f1d67868b85c0050108a9c632f8296d660f75aad8"
TRUTH_MATCH_IOU = 0.30
BEARING_TOLERANCE_DEG = 10.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: dict) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def _csv_rows(archive: zipfile.ZipFile, member: str):
    return csv.DictReader(io.TextIOWrapper(archive.open(member), encoding="utf-8"))


def load_truth(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        bbox_member = "2d_bounding_box_with_skeleton.csv" if "2d_bounding_box_with_skeleton.csv" in names else "2d_bounding_box.csv"
        poses = []
        for row in _csv_rows(archive, "aria_trajectory.csv"):
            poses.append({
                "timestamp_ns": int(row["tracking_timestamp_us"]) * 1000,
                "translation": [float(row[key]) for key in ("tx_world_device", "ty_world_device", "tz_world_device")],
            })
        times = [row["timestamp_ns"] for row in poses]
        boxes: dict[str, dict[int, list[float]]] = defaultdict(dict)
        for row in _csv_rows(archive, bbox_member):
            if row["stream_id"] != "214-1" or float(row["visibility_ratio[%]"]) < 0.10:
                continue
            timestamp = int(row["timestamp[ns]"])
            position = bisect.bisect_left(times, timestamp)
            candidates = [index for index in (position - 1, position) if 0 <= index < len(times)]
            if not candidates:
                continue
            index = min(candidates, key=lambda item: abs(times[item] - timestamp))
            if abs(times[index] - timestamp) > 20_000_000:
                continue
            boxes[str(row["object_uid"])][times[index]] = [
                float(row["x_min[pixel]"]), float(row["y_min[pixel]"]),
                float(row["x_max[pixel]"]), float(row["y_max[pixel]"]),
            ]
    return {"poses": poses, "pose_by_time": {row["timestamp_ns"]: row for row in poses}, "boxes": dict(boxes)}


def video_timestamps(path: Path) -> list[int]:
    import av
    with av.open(str(path)) as container:
        description = container.metadata.get("description")
        if not description:
            raise ValueError(f"missing video timestamps: {path}")
        values = [int(value) for value in json.loads(description)]
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("non-monotonic video timestamps")
    return values


def nearest_index(times: list[int], timestamp: int) -> int:
    position = bisect.bisect_left(times, timestamp)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(times)]
    index = min(candidates, key=lambda item: abs(times[item] - timestamp))
    if abs(times[index] - timestamp) > 20_000_000:
        raise ValueError(f"video alignment exceeds 20 ms: {timestamp}")
    return index


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def prepare(roster_dir: Path, cohort_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"formal run directory must be absent or empty: {output_dir}")
    public_roster_path = roster_dir / "public_roster.json"
    private_map_path = roster_dir / "evaluator_private_truth_map.json"
    if sha256(public_roster_path) != EXPECTED_PUBLIC_ROSTER_SHA256 or sha256(private_map_path) != EXPECTED_PRIVATE_MAP_SHA256:
        raise ValueError("frozen v3 roster hash drift")
    roster, truth_map = read_json(public_roster_path), read_json(private_map_path)
    cohort_manifest = read_json(cohort_dir / "p1_d0_manifest.json")
    source_rows = {row["source_sequence_id"]: row for row in cohort_manifest["sources"]}
    private_by_case = {row["case_id"]: row for row in truth_map["cases"]}
    episode_paths = {path.stem: path for path in (cohort_dir / "episodes").glob("*.json")}
    source_cache = {}
    video_time_cache = {}
    public_cases, private_cases = [], []

    for case in roster["cases"]:
        mapping = private_by_case[case["case_id"]]
        if "source_episode_id" in mapping:
            episode = read_json(episode_paths[mapping["source_episode_id"]])
            source_id = episode["source_sequence_id"]
            target_uid = episode["source_object_uid"]
            frames = [{
                "timestamp_ns": int(frame["timestamp_ns"]),
                "video_frame_index": int(frame["source_frame_index"]),
                "target_visible": bool(frame["target_visible"]),
                "target_bbox_xyxy": frame["target_bbox_xyxy"],
            } for frame in episode["frames"]]
        else:
            source_id = mapping["source_sequence_id"]
            target_uid = mapping["source_object_uid"]
            source = source_rows[source_id]
            gt_path = Path(source["groundtruth_path"])
            source_cache.setdefault(source_id, load_truth(gt_path))
            video_path = Path(source["rgb_video_path"])
            video_time_cache.setdefault(source_id, video_timestamps(video_path))
            truth = source_cache[source_id]
            target_boxes = truth["boxes"].get(target_uid, {})
            frames = []
            for pose in truth["poses"]:
                timestamp = pose["timestamp_ns"]
                if mapping["start_timestamp_ns"] <= timestamp <= mapping["end_timestamp_ns"]:
                    frames.append({
                        "timestamp_ns": timestamp,
                        "video_frame_index": nearest_index(video_time_cache[source_id], timestamp),
                        "target_visible": timestamp in target_boxes,
                        "target_bbox_xyxy": target_boxes.get(timestamp),
                    })
        init_index = next((index for index, frame in enumerate(frames) if frame["target_visible"]), None)
        if init_index is None:
            raise ValueError(f"{case['case_id']}: no P0 initialization opportunity")
        frames = frames[init_index:]
        source = source_rows[source_id]
        gt_path, video_path = Path(source["groundtruth_path"]), Path(source["rgb_video_path"])
        source_cache.setdefault(source_id, load_truth(gt_path))
        pose_by_time = source_cache[source_id]["pose_by_time"]
        origin = pose_by_time[frames[0]["timestamp_ns"]]["translation"]
        public_cases.append({
            "case_id": case["case_id"], "support_buckets": case["support_buckets"],
            "rgb_video_path": str(video_path), "rgb_video_sha256": source["rgb_video_sha256"],
            "initial_target_bbox_xyxy": frames[0]["target_bbox_xyxy"],
            "frames": [{"frame_id": f"{case['case_id']}-f{index:04d}", "video_frame_index": frame["video_frame_index"]}
                       for index, frame in enumerate(frames)],
        })
        private_cases.append({
            "case_id": case["case_id"], "source_sequence_id": source_id,
            "groundtruth_path": str(gt_path), "groundtruth_sha256": source["groundtruth_sha256"],
            "target_uid": target_uid,
            "frames": [{
                "timestamp_ns": frame["timestamp_ns"], "target_visible": frame["target_visible"],
                "target_bbox_xyxy": frame["target_bbox_xyxy"],
                "translation_from_initial_m": _distance(origin, pose_by_time[frame["timestamp_ns"]]["translation"]),
            } for frame in frames],
        })

    public_input = {"schema_version": "p1_w1_stage_a_public_input_v1", "cases": public_cases}
    private_truth = {"schema_version": "p1_w1_stage_a_private_truth_v1", "cases": private_cases}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "public_input.json", public_input)
    write_json(output_dir / "private_truth.json", private_truth)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema_version": "p1_w1_stage_a_run_manifest_v1", "protocol_id": "BLINDASSIST-P1-W1-STAGE-A-V1",
        "frozen_commit": commit, "public_roster_sha256": EXPECTED_PUBLIC_ROSTER_SHA256,
        "private_map_sha256": EXPECTED_PRIVATE_MAP_SHA256, "public_input_sha256": json_hash(public_input),
        "private_truth_sha256": json_hash(private_truth), "performance_outcome_accessed": False,
        "execution_budget": {"real_cases": 17, "passes_per_arm": 1, "external_model_calls": 0},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def decode_frames(video_path: Path, indices: list[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"video open failed: {video_path}")
    frames = []
    current = -1
    try:
        for target in indices:
            if current < 0 or target < current or target - current > 10:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                current = target
            while current <= target:
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"decode failed at frame {target}")
                current += 1
            frames.append(image)
    finally:
        capture.release()
    return frames


def execute(run_dir: Path) -> dict:
    manifest, public_input = read_json(run_dir / "manifest.json"), read_json(run_dir / "public_input.json")
    if manifest["public_input_sha256"] != json_hash(public_input):
        raise ValueError("public input hash drift")
    prediction_path = run_dir / "predictions.json"
    if prediction_path.exists():
        raise ValueError("predictions already exist; one-shot execution cannot resume or overwrite")
    results = []
    for case_index, case in enumerate(public_input["cases"], start=1):
        images = decode_frames(Path(case["rgb_video_path"]), [row["video_frame_index"] for row in case["frames"]])
        try:
            provider = FrozenRgbProvider(images[0], tuple(case["initial_target_bbox_xyxy"]))
            init_error = None
        except ValueError as error:
            provider, init_error = None, str(error)
        arms = {"C0": [], "W1-T0": []}
        previous = {"C0": "SUPPORTED", "W1-T0": "SUPPORTED"}
        if provider is not None:
            for frame_row, image in zip(case["frames"], images):
                evidence = provider.evidence(frame_row["frame_id"], image)
                for arm in arms:
                    snapshot = step(arm, case["case_id"], evidence, previous_observation_state=previous[arm])
                    arms[arm].append(snapshot)
                    previous[arm] = snapshot["observation_state"]
        results.append({"case_id": case["case_id"], "initialization_error": init_error, "arms": arms})
        print(json.dumps({"completed": case_index, "total": len(public_input["cases"]), "case_id": case["case_id"]}), flush=True)
    output = {
        "schema_version": "p1_w1_stage_a_predictions_v1", "public_input_sha256": manifest["public_input_sha256"],
        "truth_access": {"oracle_initializations": len(results), "post_initialization_gt_reads": 0, "future_frame_reads": 0},
        "cases": results,
    }
    write_json(prediction_path, output)
    return output


def iou(left: list[float], right: list[float]) -> float:
    x1, y1, x2, y2 = max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area, right_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1]), max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def evaluate(run_dir: Path, roster_dir: Path) -> dict:
    result_path = run_dir / "result.json"
    if result_path.exists():
        raise ValueError("result already exists; evaluation cannot overwrite")
    manifest, public_input = read_json(run_dir / "manifest.json"), read_json(run_dir / "public_input.json")
    private_truth, predictions = read_json(run_dir / "private_truth.json"), read_json(run_dir / "predictions.json")
    if manifest["private_truth_sha256"] != json_hash(private_truth) or predictions["public_input_sha256"] != manifest["public_input_sha256"]:
        raise ValueError("formal run binding drift")
    public_by_id = {row["case_id"]: row for row in public_input["cases"]}
    truth_by_id = {row["case_id"]: row for row in private_truth["cases"]}
    prediction_by_id = {row["case_id"]: row for row in predictions["cases"]}
    metrics = {}
    interface_errors = {row["case_id"]: row["initialization_error"] for row in predictions["cases"] if row["initialization_error"]}
    for arm in ("C0", "W1-T0"):
        value = {key: 0 for key in (
            "fabricated_observation", "single_channel_reacquisition", "stale_anchor_guidance_use",
            "post_initialization_truth_leakage", "future_frame_access", "false_reacquisition", "false_continuity",
            "identity_confirmed_reacquisition", "usable_anchor_coverage", "abstention_count", "honest_none_observation",
        )}
        bearing_ok = bearing_total = 0
        translation_checks = []
        for case_id, public_case in public_by_id.items():
            truth_case, prediction = truth_by_id[case_id], prediction_by_id[case_id]
            if prediction["initialization_error"]:
                continue
            gt = load_truth(Path(truth_case["groundtruth_path"]))
            target_uid = truth_case["target_uid"]
            snapshots = prediction["arms"][arm]
            for frame_truth, snapshot in zip(truth_case["frames"], snapshots):
                candidate = snapshot["candidate_region"]
                matched_uid = None
                if candidate is not None:
                    best = (0.0, None)
                    for uid, rows in gt["boxes"].items():
                        box = rows.get(frame_truth["timestamp_ns"])
                        if box is not None and iou(candidate, box) > best[0]:
                            best = (iou(candidate, box), uid)
                    if best[0] >= TRUTH_MATCH_IOU:
                        matched_uid = best[1]
                assertion = (
                    snapshot["observation_state"] == "SUPPORTED"
                    and snapshot["spatial_compatibility"] == "SUPPORTED"
                    and snapshot["independent_identity_confirmation"] == "SUPPORTED"
                )
                if assertion and matched_uid != target_uid:
                    value["false_continuity"] += 1
                if snapshot["reacquisition_status"] == "REACQUIRED":
                    value["identity_confirmed_reacquisition" if matched_uid == target_uid else "false_reacquisition"] += 1
                if snapshot["observation_state"] == "NONE" and not frame_truth["target_visible"]:
                    value["honest_none_observation"] += 1
                if frame_truth["target_visible"] and not (assertion and matched_uid == target_uid):
                    value["abstention_count"] += 1
                if snapshot["spatial_anchor_state"] == "GOOD":
                    value["usable_anchor_coverage"] += 1
                if snapshot["spatial_anchor_state"] != "GOOD" and snapshot["directional_guidance_authorized"]:
                    value["stale_anchor_guidance_use"] += 1
                if snapshot["observation_state"] == "NONE" and candidate is not None:
                    value["fabricated_observation"] += 1
                if snapshot["reacquisition_status"] == "REACQUIRED" and (
                    snapshot["spatial_compatibility"] != "SUPPORTED" or snapshot["independent_identity_confirmation"] != "SUPPORTED"
                ):
                    value["single_channel_reacquisition"] += 1
                if frame_truth["target_visible"] and snapshot["spatial_anchor_state"] == "GOOD":
                    truth_box = frame_truth["target_bbox_xyxy"]
                    truth_bearing = (((truth_box[0] + truth_box[2]) / 2) / images_width(public_case["rgb_video_path"]) - 0.5) * 90.0
                    bearing_total += 1
                    bearing_ok += abs(snapshot["bearing_estimate"] - truth_bearing) <= BEARING_TOLERANCE_DEG
            if "TRANSLATION_BEYOND_TIER0" in public_case["support_buckets"]:
                deadline = next((index for index, row in enumerate(truth_case["frames"]) if row["translation_from_initial_m"] >= 0.75), None)
                if deadline is not None:
                    translation_checks.append(snapshots[deadline]["spatial_anchor_state"] == "STALE")
        value["bearing_compatibility_rate"] = 0.0 if bearing_total == 0 else bearing_ok / bearing_total
        value["bearing_frames"] = bearing_total
        value["translation_overreach_timely_stale"] = bool(translation_checks) and all(translation_checks)
        value["geometry_degenerate_timely_stale"] = mechanics_fixture(arm)
        metrics[arm] = value
    support_counts = read_json(roster_dir / "public_roster.json")["support_counts"]
    terminal = "W1_T0_NOT_EVALUABLE_DATA_OR_INTERFACE" if interface_errors else adjudicate_stage_a(metrics["C0"], metrics["W1-T0"], support_counts)
    result = {
        "schema_version": "p1_w1_stage_a_result_v1", "terminal": terminal, "interface_errors": interface_errors,
        "support_counts": support_counts, "metrics": metrics, "claim_ceiling": "CONSUMED_ADT_DEVELOPMENT_ONLY_NO_PRODUCT_OR_SAFETY_AUTHORITY",
        "stage_b_authorized": False,
    }
    write_json(result_path, result)
    manifest["performance_outcome_accessed"] = True
    manifest["predictions_sha256"] = sha256(run_dir / "predictions.json")
    manifest["result_sha256"] = sha256(result_path)
    write_json(run_dir / "manifest.json", manifest)
    return result


_WIDTH_CACHE: dict[str, int] = {}


def images_width(path: str) -> int:
    if path not in _WIDTH_CACHE:
        capture = cv2.VideoCapture(path)
        _WIDTH_CACHE[path] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        capture.release()
    return _WIDTH_CACHE[path]


def mechanics_fixture(arm: str) -> bool:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(120):
        x, y = rng.integers(10, 310), rng.integers(10, 230)
        cv2.circle(image, (int(x), int(y)), 3, tuple(int(v) for v in rng.integers(80, 255, size=3)), -1)
    provider = FrozenRgbProvider(image, (80.0, 60.0, 240.0, 190.0))
    evidence = provider.evidence("fixture", np.zeros_like(image))
    snapshot = step(arm, "fixture-referent", evidence, previous_observation_state="SUPPORTED")
    return snapshot["spatial_anchor_state"] == "STALE" and not snapshot["directional_guidance_authorized"]


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluate"):
        item = sub.add_parser(name)
        item.add_argument("--run-dir", type=Path, required=True)
        item.add_argument("--roster-dir", type=Path, required=True)
        if name == "prepare":
            item.add_argument("--cohort-dir", type=Path, required=True)
    item = sub.add_parser("run")
    item.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.roster_dir, args.cohort_dir, args.run_dir)
    elif args.command == "run":
        value = execute(args.run_dir)
    else:
        value = evaluate(args.run_dir, args.roster_dir)
    print(json.dumps(value if args.command == "evaluate" else {"command": args.command, "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
