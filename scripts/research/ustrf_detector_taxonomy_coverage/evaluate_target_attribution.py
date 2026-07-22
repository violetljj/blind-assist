from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_host_canonical_coverage import RAW_BYTES_PER_FRAME, read_exact
from run_host_coverage import channels_by_prediction, iou


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_box(raw: np.ndarray, prediction: int, source_size: list[int], letterbox: dict) -> list[float] | None:
    source_width, source_height = source_size
    scale, dx, dy = letterbox["scale"], letterbox["dx"], letterbox["dy"]
    values = raw[:4, prediction].astype(np.float64)
    values = np.where(values <= 1.5, values * 320, values)
    cx, cy, width, height = values.tolist()
    box = [
        max(0.0, min(source_width, (cx - width / 2 - dx) / scale)),
        max(0.0, min(source_height, (cy - height / 2 - dy) / scale)),
        max(0.0, min(source_width, (cx + width / 2 - dx) / scale)),
        max(0.0, min(source_height, (cy + height / 2 - dy) / scale)),
    ]
    return box if box[2] - box[0] > 1 and box[3] - box[1] > 1 else None


def target_outcome(
    raw_output: np.ndarray,
    host_row: dict,
    target_box: list[float],
    labels: list[str],
    target_iou: float,
    diagnostic_floor: float,
    detection_threshold: float,
) -> tuple[str, str | None]:
    detections = [
        row for row in host_row["post_nms_detections_canonical_320"]
        if row["class_id"] == 0 and iou(target_box, row["box"]) >= target_iou
    ]
    if detections:
        return "matched_person", None
    raw = channels_by_prediction(raw_output, len(labels))
    scores = raw[4:, :]
    best_ids = np.argmax(scores, axis=0)
    best_scores = scores[best_ids, np.arange(scores.shape[1])]
    localized: list[tuple[int, int, float, float]] = []
    for prediction in np.flatnonzero(best_scores >= diagnostic_floor):
        box = source_box(raw, int(prediction), host_row["source_size"], host_row["letterbox"])
        if box is not None and iou(target_box, box) >= target_iou:
            localized.append((
                int(prediction), int(best_ids[prediction]),
                float(best_scores[prediction]), float(scores[0, prediction]),
            ))
    if any(person_score >= detection_threshold for _, _, _, person_score in localized):
        return "postprocess_or_nms_miss", None
    person_top = [row for row in localized if row[1] == 0 and row[3] >= diagnostic_floor]
    if person_top:
        return "below_035_person_score_miss", None
    other_top = [row for row in localized if row[1] != 0]
    if other_top:
        best = max(other_top, key=lambda row: row[2])
        return "taxonomy_confusion", labels[best[1]]
    return "localization_miss", None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite result: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_path = Path(config["parent_config_path"])
    if sha256(parent_path) != config["parent_config_sha256"]:
        raise ValueError("parent config hash mismatch")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    truth_path = Path(config["truth_path"])
    parity_path = Path(config["canonical_semantic_parity_path"])
    baseline = config["baseline"]
    bindings = [
        (truth_path, config["truth_sha256"]),
        (parity_path, config["canonical_semantic_parity_sha256"]),
        (Path(baseline["device_receipt_path"]), baseline["device_receipt_sha256"]),
        (Path(baseline["android_raw_path"]), baseline["android_raw_sha256"]),
        (Path(baseline["host_ledger_path"]), baseline["host_ledger_sha256"]),
    ]
    for path, expected in bindings:
        if sha256(path) != expected:
            raise ValueError(f"evidence hash mismatch: {path}")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if parity["G1b_canonical_semantic_parity"] != "pass" or parity["frame_count"] != 4594:
        raise ValueError("G1b canonical semantic parity is not closed")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    host = json.loads(Path(baseline["host_ledger_path"]).read_text(encoding="utf-8"))
    labels = Path(parent["detector"]["labels_path"]).read_text(encoding="utf-8").splitlines()
    target_iou = float(config["matching"]["target_iou_min"])
    negative_iou = float(config["matching"]["negative_all_person_iou_min"])
    diagnostic_floor = float(config["matching"]["diagnostic_pre_nms_floor"])
    detection_threshold = float(baseline["person_confidence_threshold"])
    target_frames: dict[tuple[str, str], tuple[dict, dict]] = {}
    negative_frames: dict[tuple[str, str], tuple[dict, dict]] = {}
    event_rows: dict[tuple[str, str], dict] = {}
    source_event_ids: dict[str, list[tuple[str, str]]] = {}
    for source in truth["sources"]:
        source_id = source["source_id"]
        source_event_ids[source_id] = []
        for event in source["target_events"]:
            key = (source_id, event["event_id"])
            source_event_ids[source_id].append(key)
            event_rows[key] = {
                "source_id": source_id,
                "event_id": event["event_id"],
                "target_person_identity": event["target_person_identity"],
                "critical": event["critical"],
                "alertable_frame": event["alertable_frame"],
                "passed_or_cleared_frame": event["passed_or_cleared_frame"],
                "visible_frame_count": 0,
                "matched_frame_count": 0,
                "first_matched_frame": None,
                "frame_outcomes": Counter(),
                "taxonomy_confusion_labels": Counter(),
            }
            for frame in event["frames"]:
                target_frames[(source_id, frame["frame_id"])] = (event, frame)
        for window in source["negative_windows"]:
            for frame in window["frames"]:
                negative_frames[(source_id, frame["frame_id"])] = (window, frame)
    negative_summary: dict[str, dict] = {
        source_id: {"frame_count": 0, "confirmed_absent_frame_count": 0, "person_detection_count": 0,
                    "matched_all_person_detection_count": 0, "false_detection_count": 0,
                    "false_detection_frame_count": 0, "windows": {}}
        for source_id in source_event_ids
    }
    with gzip.open(baseline["android_raw_path"], "rb") as raw_stream:
        for host_row in host["frames"]:
            identity = (host_row["source_id"], host_row["frame_id"])
            raw_output = np.frombuffer(
                read_exact(raw_stream, RAW_BYTES_PER_FRAME, str(identity)), dtype="<f4"
            ).reshape(1, 84, 2100)
            target_entry = target_frames.get(identity)
            if target_entry is not None:
                event, frame = target_entry
                if frame["visible_state"].startswith("visible_"):
                    key = (host_row["source_id"], event["event_id"])
                    summary = event_rows[key]
                    outcome, confusion = target_outcome(
                        raw_output, host_row, frame["target_bbox_xyxy"], labels,
                        target_iou, diagnostic_floor, detection_threshold,
                    )
                    summary["visible_frame_count"] += 1
                    summary["frame_outcomes"][outcome] += 1
                    if outcome == "matched_person":
                        summary["matched_frame_count"] += 1
                        current = int(host_row["frame_id"])
                        if summary["first_matched_frame"] is None:
                            summary["first_matched_frame"] = current
                    if confusion is not None:
                        summary["taxonomy_confusion_labels"][confusion] += 1
            negative_entry = negative_frames.get(identity)
            if negative_entry is not None:
                window, frame = negative_entry
                source_summary = negative_summary[host_row["source_id"]]
                window_summary = source_summary["windows"].setdefault(window["window_id"], {
                    "frame_count": 0, "confirmed_absent_frame_count": 0,
                    "person_detection_count": 0, "matched_all_person_detection_count": 0,
                    "false_detection_count": 0, "false_detection_frame_count": 0,
                })
                for summary in (source_summary, window_summary):
                    summary["frame_count"] += 1
                    summary["confirmed_absent_frame_count"] += int(frame["confirmed_absent"])
                truth_boxes = [row["bbox_xyxy"] for row in frame["all_person_boxes"]]
                person_detections = [row for row in host_row["post_nms_detections_canonical_320"] if row["class_id"] == 0]
                false_count = 0
                matched_count = 0
                for detection in person_detections:
                    matched = any(iou(detection["box"], box) >= negative_iou for box in truth_boxes)
                    matched_count += int(matched)
                    false_count += int(not matched)
                for summary in (source_summary, window_summary):
                    summary["person_detection_count"] += len(person_detections)
                    summary["matched_all_person_detection_count"] += matched_count
                    summary["false_detection_count"] += false_count
                    summary["false_detection_frame_count"] += int(false_count > 0)
        if raw_stream.read(1):
            raise ValueError("Android raw stream has trailing records")
    events: list[dict] = []
    for key, row in event_rows.items():
        row["event_covered"] = row["matched_frame_count"] > 0
        row["critical_miss"] = bool(row["critical"] and not row["event_covered"])
        row["first_match_delay_frames"] = (
            row["first_matched_frame"] - row["alertable_frame"] if row["first_matched_frame"] is not None else None
        )
        row["frame_outcomes"] = dict(row["frame_outcomes"])
        row["taxonomy_confusion_labels"] = dict(row["taxonomy_confusion_labels"])
        events.append(row)
    by_source: dict[str, dict] = {}
    for source_id, keys in source_event_ids.items():
        rows = [event_rows[key] for key in keys]
        covered = sum(int(row["event_covered"]) for row in rows)
        critical_misses = sum(int(row["critical_miss"]) for row in rows)
        outcomes = Counter()
        confusions = Counter()
        for row in rows:
            outcomes.update(row["frame_outcomes"])
            confusions.update(row["taxonomy_confusion_labels"])
        by_source[source_id] = {
            "target_event_count": len(rows),
            "target_event_covered_count": covered,
            "target_event_coverage": covered / len(rows),
            "critical_event_count": sum(int(row["critical"]) for row in rows),
            "critical_miss_count": critical_misses,
            "frame_outcomes": dict(outcomes),
            "taxonomy_confusion_labels": dict(confusions),
            "negative": negative_summary[source_id],
        }
    hard_gate_passed = all(
        row["target_event_coverage"] >= config["hard_gate"]["target_event_coverage_min_each_source"]
        and row["critical_miss_count"] <= config["hard_gate"]["critical_miss_max_each_source"]
        for row in by_source.values()
    )
    result = {
        "schema": "blindassist_ustrf_detector_target_attribution_result_r1",
        "authority": "benchmark_only_no_training_app_or_production_authority",
        "config_sha256": sha256(args.config),
        "truth_sha256": sha256(truth_path),
        "G1b_canonical_semantic_parity": parity["G1b_canonical_semantic_parity"],
        "legacy_raw_tensor_gate": parity["raw_gate_status"],
        "legacy_raw_within_tolerance_count": parity["raw_output_within_frozen_tolerance_count"],
        "frame_count": host["frame_count"],
        "sources": by_source,
        "events": sorted(events, key=lambda row: (row["source_id"], row["alertable_frame"])),
        "hard_gate_passed": hard_gate_passed,
        "decision": "STOP_DETECTOR_CHANGES_AND_REOPEN_T0_T3" if hard_gate_passed else "BASELINE_FAIL_OPEN_MAX_THREE_PREREGISTERED_CANDIDATES",
        "candidate_roster": config["hard_gate"]["candidate_roster"],
        "threshold_nms_route_event_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "hard_gate_passed": hard_gate_passed,
        "decision": result["decision"],
        "sources": {
            key: {
                "coverage": value["target_event_coverage"],
                "critical_miss_count": value["critical_miss_count"],
                "frame_outcomes": value["frame_outcomes"],
                "negative_false_detection_count": value["negative"]["false_detection_count"],
            }
            for key, value in by_source.items()
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
