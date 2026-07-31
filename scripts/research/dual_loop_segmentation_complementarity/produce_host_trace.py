"""Produce a frozen host YOLO trace for the image-space complementarity diagnostic.

This adapter uses the same YOLO11n FP16 TFLite asset and decoder contract as the
repository's Kotlin detector, but runs with LiteRT on the host.  It is a
Development reference trace only; it is not a QNN/device-parity or production
trace.
"""

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


SCHEMA_VERSION = "blindassist.dual_loop_segmentation_yolo_host_trace.v1"
RECEIPT_SCHEMA_VERSION = "blindassist.dual_loop_segmentation_yolo_host_trace_receipt.v1"
PROGRESS_SCHEMA_VERSION = "blindassist.dual_loop_segmentation_yolo_host_progress.v1"
MODEL_INPUT_SIZE = 320
MODEL_CHANNELS = 84
MODEL_PREDICTIONS = 2100
CONFIDENCE_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.45
LABEL_COUNT = 80
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "source_id",
    "frame_id",
    "source_capture_timestamp_ns",
    "image_path",
    "image_sha256",
    "width",
    "height",
}


class HostTraceInputError(ValueError):
    """Raised when a frozen input or detector contract is not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise HostTraceInputError(f"{path}:{line_number}: blank JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HostTraceInputError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise HostTraceInputError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    if not rows:
        raise HostTraceInputError(f"{path}: empty manifest")
    return rows


def _resolve_image(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (manifest_path.parent / candidate).resolve()


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(manifest_path)
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    previous_timestamp: int | None = None
    for row_number, row in enumerate(rows, start=1):
        missing = REQUIRED_MANIFEST_FIELDS - row.keys()
        if missing:
            raise HostTraceInputError(f"manifest row {row_number}: missing {sorted(missing)}")
        try:
            source_id = str(row["source_id"])
            frame_id = int(row["frame_id"])
            timestamp = int(row["source_capture_timestamp_ns"])
            image_sha = str(row["image_sha256"]).lower()
            width = int(row["width"])
            height = int(row["height"])
        except (TypeError, ValueError, KeyError) as exc:
            raise HostTraceInputError(f"manifest row {row_number}: invalid identity") from exc
        key = (source_id, frame_id, image_sha)
        if key in seen:
            raise HostTraceInputError(f"manifest row {row_number}: duplicate identity {key}")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise HostTraceInputError(f"manifest row {row_number}: timestamp is not increasing")
        if width <= 0 or height <= 0 or len(image_sha) != 64:
            raise HostTraceInputError(f"manifest row {row_number}: invalid dimensions or SHA")
        image_path = _resolve_image(manifest_path, str(row["image_path"]))
        if not image_path.is_file():
            raise HostTraceInputError(f"manifest row {row_number}: missing image {image_path}")
        actual_sha = sha256_file(image_path)
        if actual_sha.lower() != image_sha:
            raise HostTraceInputError(f"manifest row {row_number}: image hash mismatch {image_path}")
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise HostTraceInputError(
                    f"manifest row {row_number}: dimensions {image.size} != {(width, height)}"
                )
        observations.append(
            {
                "source_id": source_id,
                "frame_id": frame_id,
                "source_capture_timestamp_ns": timestamp,
                "image_sha256": image_sha,
                "image_path": image_path,
                "image_path_input": str(row["image_path"]),
                "width": width,
                "height": height,
            }
        )
        seen.add(key)
        previous_timestamp = timestamp
    return observations


def letterbox(image: Image.Image, input_size: int = MODEL_INPUT_SIZE) -> tuple[np.ndarray, tuple[float, float, float]]:
    source_width, source_height = image.size
    scale = min(input_size / source_width, input_size / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    dx = (input_size - resized_width) / 2.0
    dy = (input_size - resized_height) / 2.0
    resized = image.convert("RGB").resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    canvas.paste(resized, (int(dx), int(dy)))
    tensor = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
    return np.expand_dims(tensor, axis=0), (scale, dx, dy)


def _channels_by_prediction(output: np.ndarray, label_count: int) -> np.ndarray:
    raw = np.asarray(output, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise HostTraceInputError(f"unexpected YOLO output shape: {raw.shape}")
    first, second = int(raw.shape[1]), int(raw.shape[2])
    required_channels = 4 + label_count
    if first == required_channels and second != required_channels:
        return raw[0]
    if second == required_channels and first != required_channels:
        return raw[0].T
    raise HostTraceInputError(f"ambiguous YOLO output shape: {raw.shape}")


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def decode(
    output: np.ndarray,
    source_size: tuple[int, int],
    transform: tuple[float, float, float],
    labels: list[str],
    *,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    input_size: int = MODEL_INPUT_SIZE,
) -> list[dict[str, Any]]:
    if len(labels) != LABEL_COUNT:
        raise HostTraceInputError(f"expected {LABEL_COUNT} labels, got {len(labels)}")
    raw = _channels_by_prediction(output, len(labels))
    if not np.isfinite(raw).all():
        raise HostTraceInputError("non-finite raw detector output")
    source_width, source_height = source_size
    scale, dx, dy = transform
    class_scores = raw[4:, :]
    best_ids = np.argmax(class_scores, axis=0)
    best_scores = class_scores[best_ids, np.arange(class_scores.shape[1])]
    candidates: list[dict[str, Any]] = []
    for prediction in np.flatnonzero(best_scores >= confidence_threshold):
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
        candidates.append(
            {
                "prediction_index": int(prediction),
                "class_id": class_id,
                "label": labels[class_id],
                "confidence": score,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "frame_width": source_width,
                "frame_height": source_height,
                "source": "OBJECT_DETECTOR",
            }
        )
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: (-row["confidence"], row["prediction_index"])):
        candidate_box = [candidate[name] for name in ("left", "top", "right", "bottom")]
        if any(
            candidate["class_id"] == other["class_id"]
            and _iou(candidate_box, [other[name] for name in ("left", "top", "right", "bottom")])
            > iou_threshold
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _load_interpreter(model_path: Path, threads: int) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("ai-edge-litert is required for the host YOLO trace") from exc
    interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise HostTraceInputError("YOLO model must expose exactly one input and one output")
    input_detail, output_detail = input_details[0], output_details[0]
    input_shape = tuple(int(value) for value in input_detail["shape"])
    output_shape = tuple(int(value) for value in output_detail["shape"])
    if input_shape != (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3):
        raise HostTraceInputError(f"unexpected input shape: {input_shape}")
    if np.dtype(input_detail["dtype"]) != np.dtype(np.float32):
        raise HostTraceInputError(f"unexpected input dtype: {input_detail['dtype']}")
    if output_shape != (1, MODEL_CHANNELS, MODEL_PREDICTIONS):
        raise HostTraceInputError(f"unexpected output shape: {output_shape}")
    if np.dtype(output_detail["dtype"]) != np.dtype(np.float32):
        raise HostTraceInputError(f"unexpected output dtype: {output_detail['dtype']}")
    return interpreter, input_detail, output_detail


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_progress(path: Path, *, status: str, completed: int, total: int) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "status": status,
            "completed_frames": completed,
            "total_frames": total,
            "updated_at_unix_ms": int(time.time() * 1000),
        },
    )


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _build_receipt(
    *,
    repo_root: Path,
    manifest_path: Path,
    model_path: Path,
    labels_path: Path,
    output_path: Path,
    frame_count: int,
    source_ids: set[str],
    detection_total: int,
    model_sha: str,
    labels_sha: str,
    threads: int,
    elapsed_ms: float,
    recovered_from_complete_trace: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "COMPLETE",
        "authority": "DEVELOPMENT_HOST_REFERENCE_ONLY",
        "manifest_path": _relative_to_repo(manifest_path, repo_root),
        "manifest_sha256": sha256_file(manifest_path),
        "model_path": _relative_to_repo(model_path, repo_root),
        "model_sha256": model_sha,
        "labels_path": _relative_to_repo(labels_path, repo_root),
        "labels_sha256": labels_sha,
        "trace_path": _relative_to_repo(output_path, repo_root),
        "trace_sha256": sha256_file(output_path),
        "frame_count": frame_count,
        "source_ids": sorted(source_ids),
        "total_post_nms_detections": detection_total,
        "backend": "host_ai_edge_litert",
        "threads": threads,
        "input_shape": [1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3],
        "output_shape": [1, MODEL_CHANNELS, MODEL_PREDICTIONS],
        "decoder_contract": {
            "input_size": MODEL_INPUT_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "coordinate_rule": "value <= 1.5 is normalized by input_size, otherwise pixel coordinate",
            "nms_rule": "class-wise greedy NMS in source coordinates, suppress IoU > threshold",
        },
        "elapsed_ms": round(elapsed_ms, 3),
        "recovered_from_complete_trace": recovered_from_complete_trace,
    }


def finalize_existing_trace(
    *,
    repo_root: Path,
    manifest_path: Path,
    model_path: Path,
    labels_path: Path,
    output_path: Path,
    receipt_path: Path,
    progress_path: Path,
    threads: int,
) -> dict[str, Any]:
    """Seal a trace whose frame loop completed before receipt serialization failed."""
    if not output_path.is_file():
        raise HostTraceInputError(f"complete trace does not exist: {output_path}")
    if receipt_path.exists():
        raise HostTraceInputError(f"refusing to overwrite existing receipt: {receipt_path}")
    observations = load_manifest(manifest_path)
    model_sha = sha256_file(model_path)
    labels_sha = sha256_file(labels_path)
    detection_total = 0
    source_ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for index, (observation, line) in enumerate(zip(observations, handle, strict=False), start=1):
            if not line.strip():
                raise HostTraceInputError(f"trace row {index}: blank row")
            row = json.loads(line)
            expected_identity = {
                "schema_version": SCHEMA_VERSION,
                "source_id": observation["source_id"],
                "frame_id": observation["frame_id"],
                "source_capture_timestamp_ns": observation["source_capture_timestamp_ns"],
                "image_sha256": observation["image_sha256"],
                "detector_model_sha256": model_sha,
                "detector_labels_sha256": labels_sha,
                "input_size": MODEL_INPUT_SIZE,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "nms_iou_threshold": NMS_IOU_THRESHOLD,
            }
            for field, expected in expected_identity.items():
                if row.get(field) != expected:
                    raise HostTraceInputError(f"trace row {index}: {field} mismatch")
            detections = row.get("detections")
            if not isinstance(detections, list) or row.get("detection_count") != len(detections):
                raise HostTraceInputError(f"trace row {index}: detection count mismatch")
            for detection_index, detection in enumerate(detections):
                if not isinstance(detection, dict) or detection.get("source") != "OBJECT_DETECTOR":
                    raise HostTraceInputError(f"trace row {index} detection {detection_index}: invalid source")
                for field in ("left", "top", "right", "bottom"):
                    if not _finite(detection.get(field)):
                        raise HostTraceInputError(f"trace row {index} detection {detection_index}: invalid {field}")
                if int(detection.get("frame_width", -1)) != observation["width"] or int(
                    detection.get("frame_height", -1)
                ) != observation["height"]:
                    raise HostTraceInputError(f"trace row {index} detection {detection_index}: frame size mismatch")
            detection_total += len(detections)
            source_ids.add(observation["source_id"])
        if index != len(observations):
            raise HostTraceInputError(f"trace row count mismatch: expected {len(observations)}, got {index}")
        if next(handle, ""):
            raise HostTraceInputError("trace contains rows beyond the manifest")
    receipt = _build_receipt(
        repo_root=repo_root,
        manifest_path=manifest_path,
        model_path=model_path,
        labels_path=labels_path,
        output_path=output_path,
        frame_count=len(observations),
        source_ids=source_ids,
        detection_total=detection_total,
        model_sha=model_sha,
        labels_sha=labels_sha,
        threads=threads,
        elapsed_ms=0.0,
        recovered_from_complete_trace=True,
    )
    _write_json_atomic(receipt_path, receipt)
    _write_progress(progress_path, status="COMPLETE", completed=len(observations), total=len(observations))
    return receipt


def produce_trace(
    *,
    repo_root: Path,
    manifest_path: Path,
    model_path: Path,
    labels_path: Path,
    output_path: Path,
    receipt_path: Path,
    progress_path: Path,
    threads: int,
    progress_every: int,
) -> dict[str, Any]:
    if threads <= 0 or progress_every <= 0:
        raise ValueError("threads and progress_every must be positive")
    artifacts_root = (repo_root / "artifacts.local").resolve()
    for path in (output_path, receipt_path, progress_path):
        try:
            path.resolve().relative_to(artifacts_root)
        except ValueError as exc:
            raise HostTraceInputError(f"output must stay under artifacts.local: {path}") from exc
        if path.exists():
            raise HostTraceInputError(f"refusing to overwrite existing output: {path}")
    if not model_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("YOLO model and labels must exist")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != LABEL_COUNT:
        raise HostTraceInputError(f"expected {LABEL_COUNT} labels, got {len(labels)}")
    observations = load_manifest(manifest_path)
    model_sha = sha256_file(model_path)
    labels_sha = sha256_file(labels_path)
    interpreter, input_detail, output_detail = _load_interpreter(model_path, threads)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_trace = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_trace.exists():
        raise HostTraceInputError(f"refusing to reuse stale temporary output: {temporary_trace}")
    _write_progress(progress_path, status="RUNNING", completed=0, total=len(observations))
    started = time.perf_counter()
    detection_total = 0
    frame_rows = 0
    source_ids: set[str] = set()
    try:
        with temporary_trace.open("w", encoding="utf-8", newline="\n") as handle:
            for index, observation in enumerate(observations, start=1):
                with Image.open(observation["image_path"]) as image:
                    rgb = image.convert("RGB")
                    preprocess_start = time.perf_counter()
                    input_tensor, transform = letterbox(rgb)
                    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
                    interpreter.set_tensor(input_detail["index"], input_tensor)
                    inference_start = time.perf_counter()
                    interpreter.invoke()
                    inference_ms = (time.perf_counter() - inference_start) * 1000.0
                    raw_output = interpreter.get_tensor(output_detail["index"])
                    postprocess_start = time.perf_counter()
                    detections = decode(raw_output, rgb.size, transform, labels)
                    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "authority": "DEVELOPMENT_HOST_REFERENCE_ONLY",
                    "source_id": observation["source_id"],
                    "frame_id": observation["frame_id"],
                    "source_capture_timestamp_ns": observation["source_capture_timestamp_ns"],
                    "image_sha256": observation["image_sha256"],
                    "detector_backend": "host_ai_edge_litert",
                    "detector_model_sha256": model_sha,
                    "detector_labels_sha256": labels_sha,
                    "input_size": MODEL_INPUT_SIZE,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "nms_iou_threshold": NMS_IOU_THRESHOLD,
                    "detector_output_sha256": array_sha256(raw_output),
                    "detection_count": len(detections),
                    "detections": detections,
                    "detector_metrics": {
                        "preprocess_ms": round(preprocess_ms, 6),
                        "inference_ms": round(inference_ms, 6),
                        "postprocess_ms": round(postprocess_ms, 6),
                        "total_ms": round(preprocess_ms + inference_ms + postprocess_ms, 6),
                        "model_status": "model_loaded",
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                frame_rows += 1
                detection_total += len(detections)
                source_ids.add(observation["source_id"])
                if index == len(observations) or index % progress_every == 0:
                    _write_progress(progress_path, status="RUNNING", completed=index, total=len(observations))
        temporary_trace.replace(output_path)
    except BaseException:
        _write_progress(progress_path, status="FAILED", completed=frame_rows, total=len(observations))
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    trace_sha = sha256_file(output_path)
    receipt = _build_receipt(
        repo_root=repo_root,
        manifest_path=manifest_path,
        model_path=model_path,
        labels_path=labels_path,
        output_path=output_path,
        frame_count=frame_rows,
        source_ids=source_ids,
        detection_total=detection_total,
        model_sha=model_sha,
        labels_sha=labels_sha,
        threads=threads,
        elapsed_ms=elapsed_ms,
    )
    _write_json_atomic(receipt_path, receipt)
    _write_progress(progress_path, status="COMPLETE", completed=frame_rows, total=len(observations))
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Seal a fully written trace after a receipt-only failure; no inference is run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path.cwd().resolve()
    producer = finalize_existing_trace if args.finalize_existing else produce_trace
    kwargs = {
        "repo_root": repo_root,
        "manifest_path": args.manifest.resolve(),
        "model_path": args.model.resolve(),
        "labels_path": args.labels.resolve(),
        "output_path": args.output.resolve(),
        "receipt_path": args.receipt.resolve(),
        "progress_path": args.progress.resolve(),
        "threads": args.threads,
    }
    if args.finalize_existing:
        receipt = producer(**kwargs)
    else:
        receipt = producer(progress_every=args.progress_every, **kwargs)
    print(json.dumps({key: receipt[key] for key in ("status", "frame_count", "trace_sha256", "elapsed_ms")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
