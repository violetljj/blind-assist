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


SOURCE_BUNDLES = {
    "dynamics_0": "dynamics_0-normalized-v1/lilocbench_dynamics_0_front",
    "lt_changes_dynamics_0": "lt_changes_dynamics_0-normalized-v1/lilocbench_lt_changes_dynamics_0_front",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


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
    tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), (scale, dx, dy)


def channels_by_prediction(output: np.ndarray, label_count: int) -> np.ndarray:
    raw = np.asarray(output, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"unexpected YOLO output rank/shape: {raw.shape}")
    first, second = int(raw.shape[1]), int(raw.shape[2])
    required_channels = 4 + label_count
    if first == required_channels and second != required_channels:
        return raw[0]
    if second == required_channels and first != required_channels:
        return raw[0].T
    raise ValueError(f"ambiguous or incompatible YOLO output shape: {raw.shape}, labels={label_count}")


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0]); top = max(first[1], second[1])
    right = min(first[2], second[2]); bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def decode(
    output: np.ndarray,
    source_size: tuple[int, int],
    transform: tuple[float, float, float],
    labels: list[str],
    confidence: float,
    iou_threshold: float,
    input_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = channels_by_prediction(output, len(labels))
    if not np.isfinite(raw).all():
        raise ValueError("non-finite raw detector output")
    source_width, source_height = source_size
    scale, dx, dy = transform
    class_scores = raw[4:, :]
    best_ids = np.argmax(class_scores, axis=0)
    best_scores = class_scores[best_ids, np.arange(class_scores.shape[1])]
    person_scores = class_scores[0, :]
    top_prediction_indices = np.argsort(best_scores)[-10:][::-1]
    boxes: list[dict[str, Any]] = []
    for prediction in np.flatnonzero(best_scores >= confidence):
        class_id = int(best_ids[prediction])
        score = float(best_scores[prediction])
        values = raw[:4, prediction].astype(np.float64)
        values = np.where(values <= 1.5, values * input_size, values)
        cx, cy, width, height = values.tolist()
        left = max(0.0, min(float(source_width), (cx - width / 2.0 - dx) / scale))
        top = max(0.0, min(float(source_height), (cy - height / 2.0 - dy) / scale))
        right = max(0.0, min(float(source_width), (cx + width / 2.0 - dx) / scale))
        bottom = max(0.0, min(float(source_height), (cy + height / 2.0 - dy) / scale))
        if right - left <= 1.0 or bottom - top <= 1.0:
            continue
        boxes.append({
            "prediction_index": int(prediction),
            "class_id": class_id,
            "label": labels[class_id],
            "confidence": score,
            "box": [left, top, right, bottom],
        })
    kept: list[dict[str, Any]] = []
    for candidate in sorted(boxes, key=lambda row: (-row["confidence"], row["prediction_index"])):
        if any(candidate["class_id"] == other["class_id"] and iou(candidate["box"], other["box"]) > iou_threshold for other in kept):
            continue
        kept.append(candidate)
    diagnostics = {
        "raw_all_class_max_confidence": float(np.max(best_scores)),
        "raw_person_max_confidence": float(np.max(person_scores)),
        "raw_person_argmax_prediction": int(np.argmax(person_scores)),
        "pre_nms_candidate_count": len(boxes),
        "top_predictions": [
            {
                "prediction_index": int(index),
                "class_id": int(best_ids[index]),
                "label": labels[int(best_ids[index])],
                "confidence": float(best_scores[index]),
                "person_confidence": float(person_scores[index]),
            }
            for index in top_prediction_indices
        ],
    }
    return kept, diagnostics


def selected_frame_ids(windows: dict, source_id: str) -> list[str]:
    values: set[int] = set()
    for window in windows["windows"]:
        if window["source_id"] == source_id:
            values.update(range(int(window["start_frame"]), int(window["end_frame"]) + 1))
    return [f"{value:06d}" for value in sorted(values)]


def run(config_path: Path, source_name: str, output_path: Path) -> dict:
    config = read_json(config_path)
    detector = config["detector"]
    parent = config["parent"]
    if sha256(Path(parent["windows_path"])) != parent["windows_sha256"]:
        raise ValueError("window hash mismatch")
    if sha256(Path(detector["model_path"])) != detector["model_sha256"]:
        raise ValueError("model hash mismatch")
    if sha256(Path(detector["labels_path"])) != detector["labels_sha256"]:
        raise ValueError("labels hash mismatch")
    labels = Path(detector["labels_path"]).read_text(encoding="utf-8").splitlines()
    bundle_root = Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1") / SOURCE_BUNDLES[source_name]
    bundle = read_json(bundle_root / "bundle.json")
    frames_path = bundle_root / "frames.jsonl"
    frames = {row["frame_id"]: row for row in map(json.loads, frames_path.read_text(encoding="utf-8").splitlines())}
    windows = read_json(Path(parent["windows_path"]))
    ids = selected_frame_ids(windows, bundle["source"]["source_id"])
    expected_count = parent["source_frame_counts"][bundle["source"]["source_id"]]
    if len(ids) != expected_count:
        raise ValueError("source frame inventory mismatch")
    interpreter = Interpreter(model_path=detector["model_path"])
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["shape"].tolist() != detector["input_shape"] or output_detail["shape"].tolist() != detector["output_shape"]:
        raise ValueError("live tensor shape mismatch")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    person_frames = 0
    class_histogram: dict[str, int] = {}
    for position, frame_id in enumerate(ids, start=1):
        frame = frames[frame_id]
        image_path = Path(bundle["source_root"]) / frame["rgb_path"]
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            input_tensor, transform = letterbox(rgb, int(detector["input_shape"][1]))
            interpreter.set_tensor(input_detail["index"], input_tensor)
            interpreter.invoke()
            raw_output = interpreter.get_tensor(output_detail["index"])
            detections, diagnostics = decode(
                raw_output, rgb.size, transform, labels,
                float(detector["confidence_threshold"]), float(detector["nms_iou_threshold"]),
                int(detector["input_shape"][1]),
            )
        for detection in detections:
            class_histogram[detection["label"]] = class_histogram.get(detection["label"], 0) + 1
        if any(row["class_id"] == detector["person_class_index"] for row in detections):
            person_frames += 1
        rows.append({
            "frame_id": frame_id,
            "timestamp_s": frame["rgb_timestamp_s"],
            "image_sha256": sha256(image_path),
            "input_tensor_sha256": array_sha256(input_tensor),
            "raw_output_sha256": array_sha256(raw_output),
            "letterbox": {"scale": transform[0], "dx": transform[1], "dy": transform[2]},
            **diagnostics,
            "post_nms_detections": detections,
        })
        if position % 100 == 0:
            print(f"{source_name} coverage_frames={position}/{len(ids)}", flush=True)
    payload = {
        "schema": "blindassist_ustrf_detector_taxonomy_host_ledger_v1",
        "authority": "host_diagnostic_requires_android_parity",
        "config_sha256": sha256(config_path),
        "windows_sha256": parent["windows_sha256"],
        "source_name": source_name,
        "source_id": bundle["source"]["source_id"],
        "frame_count": len(rows),
        "model_sha256": detector["model_sha256"],
        "labels_sha256": detector["labels_sha256"],
        "input_shape": detector["input_shape"],
        "output_shape": detector["output_shape"],
        "output_layout": detector["output_layout"],
        "confidence_threshold": detector["confidence_threshold"],
        "nms_iou_threshold": detector["nms_iou_threshold"],
        "person_frame_count": person_frames,
        "class_histogram_post_nms": dict(sorted(class_histogram.items())),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "frames": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ValueError(f"refusing to overwrite output: {output_path}")
    output_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-name", choices=sorted(SOURCE_BUNDLES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.config, args.source_name, args.output)
    print(json.dumps({key: payload[key] for key in ("source_name", "frame_count", "person_frame_count", "elapsed_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
