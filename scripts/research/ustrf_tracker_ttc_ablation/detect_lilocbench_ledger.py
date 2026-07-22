from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ai_edge_litert.interpreter import Interpreter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, tuple[float, float, float]]:
    source_width, source_height = image.size
    scale = min(size / source_width, size / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    dx = (size - resized_width) / 2.0
    dy = (size - resized_height) / 2.0
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, (int(dx), int(dy)))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0), (scale, dx, dy)


def decode(output: np.ndarray, source_size: tuple[int, int], transform: tuple[float, float, float], labels: list[str], confidence: float, iou_threshold: float) -> tuple[list[dict[str, Any]], float, float]:
    source_width, source_height = source_size
    scale, dx, dy = transform
    raw = np.asarray(output).squeeze(axis=0)
    if raw.shape[0] > raw.shape[1]:
        raw = raw.T
    if raw.shape[1] < 5:
        raise ValueError(f"unexpected YOLO output shape: {output.shape}")
    raw_class_scores = raw[:, 4:4 + len(labels)]
    raw_all_class_max = float(np.max(raw_class_scores)) if raw_class_scores.size else 0.0
    raw_person_max = float(np.max(raw_class_scores[:, 0])) if raw_class_scores.shape[1] else 0.0
    boxes: list[dict[str, Any]] = []
    for row in raw:
        class_scores = row[4:4 + len(labels)]
        class_id = int(np.argmax(class_scores)) if class_scores.size else -1
        score = float(class_scores[class_id]) if class_id >= 0 else 0.0
        if class_id < 0 or score < confidence:
            continue
        values = row[:4].astype(np.float64)
        values = np.where(values <= 1.5, values * 320.0, values)
        cx, cy, width, height = values.tolist()
        left = max(0.0, min(float(source_width), (cx - width / 2.0 - dx) / scale))
        top = max(0.0, min(float(source_height), (cy - height / 2.0 - dy) / scale))
        right = max(0.0, min(float(source_width), (cx + width / 2.0 - dx) / scale))
        bottom = max(0.0, min(float(source_height), (cy + height / 2.0 - dy) / scale))
        if right - left <= 1.0 or bottom - top <= 1.0:
            continue
        boxes.append({"class_id": class_id, "label": labels[class_id], "confidence": score, "box": [left, top, right, bottom]})
    kept: list[dict[str, Any]] = []
    for candidate in sorted(boxes, key=lambda row: row["confidence"], reverse=True):
        def iou(first: list[float], second: list[float]) -> float:
            left = max(first[0], second[0]); top = max(first[1], second[1])
            right = min(first[2], second[2]); bottom = min(first[3], second[3])
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
            second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
            union = first_area + second_area - intersection
            return intersection / union if union > 0 else 0.0
        if any(candidate["class_id"] == other["class_id"] and iou(candidate["box"], other["box"]) > iou_threshold for other in kept):
            continue
        kept.append(candidate)
    return kept, raw_all_class_max, raw_person_max


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--source-name", choices=["dynamics_0", "lt_changes_dynamics_0"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    config = read_json(args.config)
    windows = read_json(args.windows)
    source = config["inputs"][args.source_name]
    if sha256(args.config) != windows["config_sha256"]:
        raise ValueError("window/config hash mismatch")
    model = Path(config["detector"]["model_path"])
    labels_path = Path(config["detector"]["labels_path"])
    if sha256(model) != config["detector"]["model_sha256"] or sha256(labels_path) != config["detector"]["labels_sha256"]:
        raise ValueError("detector model or labels hash mismatch")
    bundle_root = Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1") / ("dynamics_0-normalized-v1/lilocbench_dynamics_0_front" if args.source_name == "dynamics_0" else "lt_changes_dynamics_0-normalized-v1/lilocbench_lt_changes_dynamics_0_front")
    bundle = read_json(bundle_root / "bundle.json")
    frames_path = bundle_root / "frames.jsonl"
    if sha256(frames_path) != source["frames_sha256"] or bundle["source"]["source_id"] != source["source_id"]:
        raise ValueError("source bundle/frame hash mismatch")
    frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line]
    frame_by_id = {row["frame_id"]: row for row in frames}
    selected_ids: list[str] = []
    for window in windows["windows"]:
        if window["source_id"] != source["source_id"]:
            continue
        selected_ids.extend(f"{index:06d}" for index in range(int(window["start_frame"]), int(window["end_frame"]) + 1))
    selected_ids = sorted(set(selected_ids), key=lambda frame_id: int(frame_id))
    interpreter = Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    labels = labels_path.read_text(encoding="utf-8").splitlines()
    detections: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, frame_id in enumerate(selected_ids, start=1):
        frame = frame_by_id[frame_id]
        image_path = Path(bundle["source_root"]) / frame["rgb_path"]
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            input_data, transform = letterbox(rgb, int(config["detector"]["input_size"]))
            interpreter.set_tensor(input_detail["index"], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_detail["index"])
            all_detections, raw_all_class_max, raw_person_max = decode(output, rgb.size, transform, labels, float(config["detector"]["confidence_threshold"]), float(config["detector"]["iou_threshold"]))
        person_detections = [row for row in all_detections if row["label"] == "person"]
        detections.append({
            "frame_id": frame_id,
            "timestamp_s": frame["rgb_timestamp_s"],
            "detections": person_detections,
            "all_class_detection_count": len(all_detections),
            "all_class_max_confidence": max((row["confidence"] for row in all_detections), default=0.0),
            "person_max_confidence": max((row["confidence"] for row in person_detections), default=0.0),
            "raw_all_class_max_confidence": raw_all_class_max,
            "raw_person_max_confidence": raw_person_max,
        })
        if position % 100 == 0:
            print(f"{args.source_name} detector_frames={position}/{len(selected_ids)}")
    interpreter = None
    payload = {
        "schema": "blindassist_ustrf_tracker_ttc_detector_ledger_v1",
        "authority": "detector_input_only_no_event_truth_no_track_truth",
        "config_sha256": sha256(args.config),
        "windows_sha256": sha256(args.windows),
        "source_name": args.source_name,
        "source_id": source["source_id"],
        "frames_sha256": source["frames_sha256"],
        "model_sha256": config["detector"]["model_sha256"],
        "labels_sha256": config["detector"]["labels_sha256"],
        "detector_seconds": round(time.perf_counter() - started, 3),
        "frame_count": len(detections),
        "selected_frame_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest(),
        "candidate_alerts_visible": False,
        "event_truth_visible": False,
        "track_truth_visible": False,
        "frames": detections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"source": args.source_name, "frames": len(detections), "seconds": payload["detector_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
