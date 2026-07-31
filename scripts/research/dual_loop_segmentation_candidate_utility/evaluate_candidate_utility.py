"""Evaluate source-native pixel/component utility of the fixed segmentation reference.

This module is deliberately a host-side Development evaluator.  It consumes an
independently bound YOLO trace and source-native semantic masks; it never reads
event, risk, feedback, or central-obstruction labels.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import time
import tracemalloc
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

try:
    from .component_metrics import aggregate_confusion, component_metrics, component_records, pixel_metrics
    from .temporal_metrics import summarize_temporal
except ImportError:  # pragma: no cover - direct script execution
    from component_metrics import aggregate_confusion, component_metrics, component_records, pixel_metrics
    from temporal_metrics import summarize_temporal

from . import ALL_CLASSES, PRIMARY_HAZARD_CLASSES, PROTOCOL_ID


SCHEMA_VERSION = "blindassist.dual_loop_segmentation_candidate_utility_r0.result.v1"
FRAME_SCHEMA_VERSION = "blindassist.dual_loop_segmentation_candidate_utility_r0.frame.v1"
COMPONENT_SCHEMA_VERSION = "blindassist.dual_loop_segmentation_candidate_utility_r0.component.v1"
EXPECTED_TRACE_SCHEMAS = {
    "blindassist.dual_loop_segmentation_yolo_host_trace.v1",
    "blindassist.dual_loop_unseen_rank2_baseline_trace.v1",
}
FORBIDDEN_KEYS = {"risk", "feedback", "event", "central_obstruction_agent_labels"}
ANALYSIS_WIDTH = 256
ANALYSIS_HEIGHT = 256
CLASS_TO_ID = {name: index for index, name in enumerate(ALL_CLASSES)}


class CandidateUtilityInputError(ValueError):
    """Raised when a frozen identity, truth, or tensor contract is violated."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CandidateUtilityInputError(f"{path}:{line_number}: blank JSONL row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CandidateUtilityInputError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise CandidateUtilityInputError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    if not rows:
        raise CandidateUtilityInputError(f"{path}: empty JSONL input")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _forbidden_value(value: Any, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_lower = str(key).lower()
            if key_lower in FORBIDDEN_KEYS:
                return f"{path}.{key}"
            found = _forbidden_value(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _forbidden_value(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _identity(row: dict[str, Any]) -> tuple[str, int, str]:
    try:
        source_id = row.get("source_id") or row.get("session_id") or row["source"]["session_id"]
        frame_value = row["frame_id"] if "frame_id" in row else row["frame_index"]
        image_sha = str(row["image_sha256"]).lower()
        source_id = str(source_id)
        frame_id = int(frame_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateUtilityInputError(f"invalid frame identity: {row}") from exc
    if not source_id or len(image_sha) != 64:
        raise CandidateUtilityInputError(f"invalid frame identity: {row}")
    return source_id, frame_id, image_sha


def _resolve_data_path(value: str | Path, *, manifest_path: Path, dataset_root: Path | None) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    if dataset_root is not None:
        rooted = (dataset_root / candidate).resolve()
        if rooted.is_file():
            return rooted
    return (manifest_path.parent / candidate).resolve()


def _mask_from_path(path: Path, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
    except OSError as exc:
        raise CandidateUtilityInputError(f"cannot read semantic mask {path}") from exc
    if expected_shape is not None and array.shape != expected_shape:
        raise CandidateUtilityInputError(
            f"semantic mask dimensions mismatch for {path}: expected {expected_shape}, got {array.shape}"
        )
    if np.any(array > max(CLASS_TO_ID.values())):
        raise CandidateUtilityInputError(f"semantic mask contains unknown class ids: {path}")
    return array


def load_protocol(protocol_path: Path) -> dict[str, Any]:
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateUtilityInputError(f"cannot read protocol {protocol_path}") from exc
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise CandidateUtilityInputError(f"unexpected protocol id: {protocol.get('protocol_id')!r}")
    if protocol.get("status") != "DESIGN_FROZEN":
        raise CandidateUtilityInputError(f"protocol is not frozen: {protocol.get('status')!r}")
    if protocol.get("hazard_mask", {}).get("primary_classes") != list(PRIMARY_HAZARD_CLASSES):
        raise CandidateUtilityInputError("protocol hazard class order does not match the implementation")
    if protocol.get("analysis", {}).get("grid") != {
        "width": ANALYSIS_WIDTH,
        "height": ANALYSIS_HEIGHT,
        "projection": "nearest for source-native truth; clipped normalized boxes",
    }:
        raise CandidateUtilityInputError("protocol analysis grid/projection mismatch")
    return protocol


def load_manifest(
    manifest_path: Path,
    *,
    dataset_root: Path | None,
    split: str | None,
    require_truth: bool,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(manifest_path)
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for row_number, row in enumerate(rows, start=1):
        forbidden = _forbidden_value(row)
        if forbidden:
            raise CandidateUtilityInputError(f"manifest row {row_number}: forbidden input at {forbidden}")
        if split is not None and row.get("split") != split:
            continue
        source_id, frame_id, image_sha = _identity(row)
        key = (source_id, frame_id, image_sha)
        if key in seen:
            raise CandidateUtilityInputError(f"manifest row {row_number}: duplicate identity {key}")
        seen.add(key)
        timestamp_quality = "source_native"
        if "source_capture_timestamp_ns" in row:
            timestamp_ns = int(row["source_capture_timestamp_ns"])
        elif "frame_timestamp_ns" in row:
            timestamp_ns = int(row["frame_timestamp_ns"])
        else:
            timestamp_ns = frame_id
            timestamp_quality = "derived_frame_order"
        if timestamp_ns < 0:
            raise CandidateUtilityInputError(f"manifest row {row_number}: negative timestamp")
        image_path_value = row.get("image_path")
        if image_path_value is None:
            raise CandidateUtilityInputError(f"manifest row {row_number}: missing image_path")
        image_path = _resolve_data_path(image_path_value, manifest_path=manifest_path, dataset_root=dataset_root)
        if not image_path.is_file():
            raise CandidateUtilityInputError(f"manifest row {row_number}: missing image {image_path}")
        actual_image_sha = sha256_file(image_path)
        if actual_image_sha.lower() != image_sha:
            raise CandidateUtilityInputError(f"manifest row {row_number}: image SHA256 mismatch for {image_path}")
        try:
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
        except OSError as exc:
            raise CandidateUtilityInputError(f"manifest row {row_number}: unreadable image {image_path}") from exc
        width = int(row.get("width", actual_width))
        height = int(row.get("height", actual_height))
        if width <= 0 or height <= 0 or (width, height) != (actual_width, actual_height):
            raise CandidateUtilityInputError(f"manifest row {row_number}: image dimensions mismatch")

        truth_path: Path | None = None
        truth_sha: str | None = None
        label_authority = row.get("label_authority")
        if "semantic_mask_path" in row or require_truth:
            if label_authority != "source_ground_truth":
                raise CandidateUtilityInputError(
                    f"manifest row {row_number}: source-native truth authority required, got {label_authority!r}"
                )
            if not row.get("semantic_mask_path") or not row.get("semantic_mask_sha256"):
                raise CandidateUtilityInputError(f"manifest row {row_number}: source-native mask fields required")
            truth_path = _resolve_data_path(
                row["semantic_mask_path"], manifest_path=manifest_path, dataset_root=dataset_root
            )
            truth_sha = str(row["semantic_mask_sha256"]).lower()
            if not truth_path.is_file() or sha256_file(truth_path).lower() != truth_sha:
                raise CandidateUtilityInputError(f"manifest row {row_number}: semantic mask SHA256 mismatch")
            _mask_from_path(truth_path, expected_shape=(height, width))
        observations.append(
            {
                "source_id": source_id,
                "frame_id": frame_id,
                "image_sha256": image_sha,
                "source_capture_timestamp_ns": timestamp_ns,
                "timestamp_quality": timestamp_quality,
                "image_path": image_path,
                "width": width,
                "height": height,
                "semantic_mask_path": truth_path,
                "semantic_mask_sha256": truth_sha,
                "label_authority": label_authority,
                "scene_bucket": row.get("scene_bucket", source_id),
                "sequence_id": row.get("sequence_id", source_id),
                "split": row.get("split"),
            }
        )
    if not observations:
        raise CandidateUtilityInputError(f"{manifest_path}: no rows after split filter")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_source[observation["source_id"]].append(observation)
    for source_id, source_rows in by_source.items():
        previous_frame: int | None = None
        previous_time: int | None = None
        for observation in source_rows:
            frame_id = observation["frame_id"]
            timestamp_ns = observation["source_capture_timestamp_ns"]
            if previous_frame is not None and frame_id <= previous_frame:
                raise CandidateUtilityInputError(f"source {source_id}: frame ids must strictly increase")
            if previous_time is not None and timestamp_ns <= previous_time:
                raise CandidateUtilityInputError(f"source {source_id}: timestamps must strictly increase")
            previous_frame = frame_id
            previous_time = timestamp_ns
    return observations


def _validate_detection(detection: Any, *, row_number: int, index: int) -> dict[str, Any]:
    if not isinstance(detection, dict):
        raise CandidateUtilityInputError(f"trace row {row_number} detection {index}: expected object")
    required = {"left", "top", "right", "bottom", "frame_width", "frame_height", "source"}
    if required - detection.keys():
        raise CandidateUtilityInputError(f"trace row {row_number} detection {index}: missing fields")
    if detection.get("source") != "OBJECT_DETECTOR":
        raise CandidateUtilityInputError(f"trace row {row_number} detection {index}: non-detector source")
    result: dict[str, Any] = {}
    for field in ("left", "top", "right", "bottom"):
        if not _finite(detection[field]):
            raise CandidateUtilityInputError(f"trace row {row_number} detection {index}: non-finite {field}")
        result[field] = float(detection[field])
    result["frame_width"] = int(detection["frame_width"])
    result["frame_height"] = int(detection["frame_height"])
    result["source"] = "OBJECT_DETECTOR"
    if result["frame_width"] <= 0 or result["frame_height"] <= 0:
        raise CandidateUtilityInputError(f"trace row {row_number} detection {index}: invalid frame size")
    return result


def load_trace(trace_path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows = _read_jsonl(trace_path)
    traces: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        forbidden = _forbidden_value(row)
        if forbidden:
            raise CandidateUtilityInputError(f"trace row {row_number}: forbidden input at {forbidden}")
        if row.get("schema_version") not in EXPECTED_TRACE_SCHEMAS:
            raise CandidateUtilityInputError(f"trace row {row_number}: unsupported schema")
        source_id, frame_id, image_sha = _identity(row)
        key = (source_id, frame_id, image_sha)
        if key in traces:
            raise CandidateUtilityInputError(f"trace row {row_number}: duplicate identity {key}")
        if "source_capture_timestamp_ns" not in row:
            raise CandidateUtilityInputError(f"trace row {row_number}: timestamp required")
        detections = row.get("detections")
        if not isinstance(detections, list):
            raise CandidateUtilityInputError(f"trace row {row_number}: detections must be a list")
        traces[key] = {
            "source_id": source_id,
            "frame_id": frame_id,
            "image_sha256": image_sha,
            "source_capture_timestamp_ns": int(row["source_capture_timestamp_ns"]),
            "detections": [
                _validate_detection(item, row_number=row_number, index=index)
                for index, item in enumerate(detections)
            ],
            "detector_model_sha256": row.get("detector_model_sha256"),
            "detector_labels_sha256": row.get("detector_labels_sha256"),
        }
    return traces


def pair_inputs(
    manifest_rows: list[dict[str, Any]],
    trace_rows: dict[tuple[str, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_keys = {
        (row["source_id"], row["frame_id"], row["image_sha256"]): row for row in manifest_rows
    }
    if set(trace_rows) != set(manifest_keys):
        missing_trace = sorted(set(manifest_keys) - set(trace_rows))
        missing_manifest = sorted(set(trace_rows) - set(manifest_keys))
        raise CandidateUtilityInputError(
            f"exact pairing failed: missing_trace={missing_trace[:3]} missing_manifest={missing_manifest[:3]}"
        )
    pairs: list[dict[str, Any]] = []
    for key, manifest_row in manifest_keys.items():
        trace_row = trace_rows[key]
        if trace_row["source_capture_timestamp_ns"] != manifest_row["source_capture_timestamp_ns"]:
            raise CandidateUtilityInputError(f"timestamp mismatch for {key}")
        for detection in trace_row["detections"]:
            if (detection["frame_width"], detection["frame_height"]) != (
                manifest_row["width"],
                manifest_row["height"],
            ):
                raise CandidateUtilityInputError(f"detection dimensions mismatch for {key}")
        pairs.append({"manifest": manifest_row, "trace": trace_row})
    return pairs


def box_union_mask(
    detections: Iterable[dict[str, Any]],
    *,
    source_width: int,
    source_height: int,
    analysis_width: int = ANALYSIS_WIDTH,
    analysis_height: int = ANALYSIS_HEIGHT,
) -> np.ndarray:
    mask = np.zeros((analysis_height, analysis_width), dtype=bool)
    for detection in detections:
        left = max(0.0, min(float(source_width), detection["left"]))
        right = max(0.0, min(float(source_width), detection["right"]))
        top = max(0.0, min(float(source_height), detection["top"]))
        bottom = max(0.0, min(float(source_height), detection["bottom"]))
        if right <= left or bottom <= top:
            continue
        x0 = max(0, min(analysis_width, math.floor(left * analysis_width / source_width)))
        x1 = max(0, min(analysis_width, math.ceil(right * analysis_width / source_width)))
        y0 = max(0, min(analysis_height, math.floor(top * analysis_height / source_height)))
        y1 = max(0, min(analysis_height, math.ceil(bottom * analysis_height / source_height)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def resize_class_mask(mask: np.ndarray, *, width: int = ANALYSIS_WIDTH, height: int = ANALYSIS_HEIGHT) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("semantic mask must be two-dimensional")
    image = Image.fromarray(mask.astype(np.uint8), mode="L")
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8)


def _quantization(detail: dict[str, Any], label: str) -> tuple[float, int]:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if not _finite(scale) or float(scale) <= 0:
        raise CandidateUtilityInputError(f"{label}: positive quantization scale required")
    return float(scale), int(zero_point)


class TFLiteSegmenter:
    """Minimal fixed-contract TFLite runner with explicit quantization checks."""

    def __init__(self, model_path: Path, *, threads: int) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("TensorFlow is required for the segmentation evaluator") from exc
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise CandidateUtilityInputError("segmentation model must expose one input and one output")
        self.input_detail = inputs[0]
        self.output_detail = outputs[0]
        self.input_shape = tuple(int(value) for value in self.input_detail["shape"])
        self.output_shape = tuple(int(value) for value in self.output_detail["shape"])
        if self.input_shape != (1, ANALYSIS_HEIGHT, ANALYSIS_WIDTH, 3):
            raise CandidateUtilityInputError(f"unexpected segmentation input shape: {self.input_shape}")
        if self.output_shape != (1, ANALYSIS_HEIGHT, ANALYSIS_WIDTH, len(ALL_CLASSES)):
            raise CandidateUtilityInputError(f"unexpected segmentation output shape: {self.output_shape}")
        if np.dtype(self.input_detail["dtype"]) != np.dtype(np.int8):
            raise CandidateUtilityInputError("segmentation input must be int8")
        if np.dtype(self.output_detail["dtype"]) != np.dtype(np.int8):
            raise CandidateUtilityInputError("segmentation output must be int8")
        self.input_scale, self.input_zero = _quantization(self.input_detail, "segmentation input")
        self.output_scale, self.output_zero = _quantization(self.output_detail, "segmentation output")
        self.contract = {
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "input_dtype": str(np.dtype(self.input_detail["dtype"])),
            "output_dtype": str(np.dtype(self.output_detail["dtype"])),
            "input_quantization": {"scale": self.input_scale, "zero_point": self.input_zero},
            "output_quantization": {"scale": self.output_scale, "zero_point": self.output_zero},
        }

    def infer(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        preprocess_start = time.perf_counter()
        resized = image.convert("RGB").resize((ANALYSIS_WIDTH, ANALYSIS_HEIGHT), Image.Resampling.BILINEAR)
        rgb = np.asarray(resized, dtype=np.float32)
        tensor = np.clip(np.rint(rgb / self.input_scale + self.input_zero), -128, 127).astype(np.int8)[None, ...]
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        inference_start = time.perf_counter()
        self.interpreter.invoke()
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        raw = self.interpreter.get_tensor(self.output_detail["index"])
        scores = (raw.astype(np.float32) - self.output_zero) * self.output_scale
        scores = scores[0]
        if not np.isfinite(scores).all():
            raise CandidateUtilityInputError("segmentation output contains non-finite values")
        maximum = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(np.clip(scores - maximum, -80.0, 80.0))
        probabilities = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        ids = np.argmax(probabilities, axis=-1).astype(np.uint8)
        top_two = np.partition(probabilities, -2, axis=-1)[..., -2:]
        top1 = top_two[..., 1]
        top2 = top_two[..., 0]
        margin = top1 - top2
        return ids, top1.astype(np.float32), margin.astype(np.float32), {
            "preprocess_ms": float(preprocess_ms),
            "inference_ms": float(inference_ms),
            "segmentation_ms": float(preprocess_ms + inference_ms),
        }


def load_motion_trace(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_jsonl(path)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        forbidden = _forbidden_value(row)
        if forbidden:
            raise CandidateUtilityInputError(f"motion row {row_number}: forbidden input at {forbidden}")
        source_id = str(row.get("source_id", ""))
        frame_id = int(row.get("frame_id"))
        key = (source_id, frame_id)
        if key in result:
            raise CandidateUtilityInputError(f"motion row {row_number}: duplicate identity {key}")
        matrix = np.asarray(row.get("matrix_previous_to_current"), dtype=np.float64)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise CandidateUtilityInputError(f"motion row {row_number}: invalid 2x3 affine")
        result[key] = {
            "previous_source_id": str(row.get("previous_source_id", source_id)),
            "previous_frame_id": int(row.get("previous_frame_id")),
            "matrix_previous_to_current": matrix.tolist(),
        }
    return result


def _percentiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _pack_mask(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(np.uint8), axis=None).tobytes()).decode("ascii")


def _aggregate_components(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = sum(int(row["candidate_component_metrics"]["predicted_component_count"]) for row in rows)
    truth = sum(int(row["candidate_component_metrics"]["truth_component_count"]) for row in rows)
    hit_predicted = sum(int(row["candidate_component_metrics"]["hit_predicted_component_count"]) for row in rows)
    hit_truth = sum(int(row["candidate_component_metrics"]["hit_truth_component_count"]) for row in rows)
    false_count = sum(int(row["candidate_component_metrics"]["false_activation_component_count"]) for row in rows)
    return {
        "predicted_component_count": predicted,
        "truth_component_count": truth,
        "hit_predicted_component_count": hit_predicted,
        "hit_truth_component_count": hit_truth,
        "component_precision": float(hit_predicted / predicted) if predicted else (1.0 if truth == 0 else None),
        "component_recall": float(hit_truth / truth) if truth else (1.0 if predicted == 0 else None),
        "false_activation_component_count": false_count,
        "false_activation_components_per_frame": float(false_count / len(rows)) if rows else None,
    }


def _empty_runtime() -> dict[str, Any]:
    return {
        "segmentation_ms": _percentiles([]),
        "component_extraction_ms": _percentiles([]),
        "fusion_ms": _percentiles([]),
        "total_increment_ms": _percentiles([]),
        "peak_memory_bytes_if_available": None,
    }


def _attach_temporal_to_components(
    components: list[dict[str, Any]],
    temporal_by_source_class: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for row in components:
        summary = temporal_by_source_class.get((row["source_id"], row["class_name"]))
        if summary is None:
            continue
        assignments = summary.get("component_track_assignments", [])
        frame_ids = summary.get("_frame_ids", [])
        try:
            frame_index = frame_ids.index(row["frame_id"])
        except ValueError:
            continue
        component_index = int(row["component_index"])
        if frame_index >= len(assignments) or component_index >= len(assignments[frame_index]):
            continue
        track_id = assignments[frame_index][component_index]
        row["temporal_track_id"] = int(track_id) if track_id >= 0 else None
        if track_id is not None and track_id >= 0:
            track = next(
                (item for item in summary.get("component_tracks", []) if item["track_id"] == track_id),
                None,
            )
            row["persistence_frames"] = track["duration_frames"] if track else None
            row["flicker_track"] = bool(track and track["duration_frames"] <= 2)
        row["raw_adjacent_iou_median"] = summary.get("raw_adjacent_iou", {}).get("median")
        row["motion_warped_adjacent_iou_median"] = summary.get("motion_warped_adjacent_iou", {}).get("median")
        row["split_count_source_class"] = summary.get("split_count", 0)
        row["merge_count_source_class"] = summary.get("merge_count", 0)


def _build_temporal(
    temporal_inputs: dict[tuple[str, str], dict[str, Any]],
    motion_rows: dict[tuple[str, int], dict[str, Any]],
    *,
    match_iou: float,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    by_source_class: dict[tuple[str, str], dict[str, Any]] = {}
    for (source_id, class_name), value in sorted(temporal_inputs.items()):
        rows = value["rows"]
        source_rows = sorted(rows, key=lambda item: item["frame_id"])
        masks = [item["mask"] for item in source_rows]
        frame_ids = [int(item["frame_id"]) for item in source_rows]
        timestamps = [int(item["timestamp_ns"]) for item in source_rows]
        motion_warps: list[Sequence[Sequence[float]] | None] = []
        for previous, current in zip(source_rows, source_rows[1:]):
            motion = motion_rows.get((source_id, int(current["frame_id"])))
            if motion is None or motion["previous_source_id"] != source_id or motion["previous_frame_id"] != int(previous["frame_id"]):
                motion_warps.append(None)
            else:
                motion_warps.append(motion["matrix_previous_to_current"])
        summary = summarize_temporal(
            masks,
            frame_ids=frame_ids,
            timestamps_ns=timestamps,
            motion_warps=motion_warps,
            match_iou=match_iou,
            timestamps_are_source_native=all(
                item["timestamp_quality"] == "source_native" for item in source_rows
            ),
        )
        summary["_frame_ids"] = frame_ids
        by_source_class[(source_id, class_name)] = summary
    output: dict[str, Any] = {}
    for (source_id, class_name), summary in by_source_class.items():
        public = {key: value for key, value in summary.items() if not key.startswith("_")}
        output.setdefault(source_id, {})[class_name] = public
    return output, by_source_class


def _session_summary(
    frame_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        frame_by_source[row["source_id"]].append(row)
    for row in component_rows:
        component_by_source[row["source_id"]].append(row)
    result: list[dict[str, Any]] = []
    for source_id in sorted(frame_by_source):
        rows = frame_by_source[source_id]
        item: dict[str, Any] = {
            "source_id": source_id,
            "scene_bucket": rows[0]["scene_bucket"],
            "frame_count": len(rows),
            "timestamp_quality": rows[0]["timestamp_quality"],
            "runtime": {
                "segmentation_ms": _percentiles([row["runtime"]["segmentation_ms"] for row in rows]),
                "component_extraction_ms": _percentiles([row["runtime"]["component_extraction_ms"] for row in rows]),
                "fusion_ms": _percentiles([row["runtime"]["fusion_ms"] for row in rows]),
                "total_increment_ms": _percentiles([row["runtime"]["total_increment_ms"] for row in rows]),
            },
        }
        if rows[0].get("truth_available"):
            aggregates = {
                arm: aggregate_confusion([row["arms"][arm]["pixel"] for row in rows])
                for arm in ("A", "B", "C")
            }
            item["arms"] = aggregates
            item["delta_recall"] = (
                aggregates["C"]["recall"] - aggregates["A"]["recall"]
                if aggregates["C"]["recall"] is not None and aggregates["A"]["recall"] is not None
                else None
            )
            item["delta_false_positive_area_fraction"] = (
                aggregates["C"]["false_positive_area_fraction"]
                - aggregates["A"]["false_positive_area_fraction"]
            )
            item["candidate_components"] = _aggregate_components(rows)
        else:
            item["delta_recall"] = None
            item["delta_false_positive_area_fraction"] = None
        item["component_ledger_rows"] = len(component_by_source.get(source_id, []))
        result.append(item)
    return result


def run_evaluation(
    *,
    repo_root: Path,
    protocol_path: Path,
    manifest_path: Path,
    dataset_root: Path | None,
    trace_path: Path,
    model_path: Path,
    report_path: Path,
    frames_path: Path,
    components_path: Path,
    progress_path: Path,
    phase: str,
    split: str | None,
    motion_trace_path: Path | None,
    threads: int,
    frames_limit: int | None,
    progress_every: int,
) -> dict[str, Any]:
    if phase not in {"calibration", "formal", "temporal"}:
        raise ValueError(f"unsupported phase: {phase}")
    if threads <= 0 or progress_every <= 0:
        raise ValueError("threads and progress_every must be positive")
    protocol = load_protocol(protocol_path)
    require_truth = phase in {"calibration", "formal"}
    manifest_rows = load_manifest(
        manifest_path,
        dataset_root=dataset_root,
        split=split,
        require_truth=require_truth,
    )
    if frames_limit is not None:
        if frames_limit <= 0:
            raise ValueError("frames limit must be positive")
        manifest_rows = manifest_rows[:frames_limit]
    trace_rows = load_trace(trace_path)
    if frames_limit is not None:
        selected_keys = {
            (row["source_id"], row["frame_id"], row["image_sha256"]) for row in manifest_rows
        }
        trace_rows = {key: value for key, value in trace_rows.items() if key in selected_keys}
    pairs = pair_inputs(manifest_rows, trace_rows)
    motion_rows = load_motion_trace(motion_trace_path)
    for path in (report_path, frames_path, components_path, progress_path):
        if path.exists():
            raise CandidateUtilityInputError(f"refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model_sha = sha256_file(model_path)
    segmenter = TFLiteSegmenter(model_path, threads=threads)
    frame_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    temporal_inputs: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"rows": []})
    segmentation_ms: list[float] = []
    component_ms: list[float] = []
    fusion_ms: list[float] = []
    total_ms: list[float] = []
    tracemalloc.start()
    started = time.perf_counter()
    progress_payload = {
        "schema_version": "blindassist.dual_loop_segmentation_candidate_utility_r0.progress.v1",
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "status": "RUNNING",
        "completed_frames": 0,
        "total_frames": len(pairs),
    }
    _write_json(progress_path, progress_payload)
    try:
        with frames_path.open("w", encoding="utf-8", newline="\n") as frame_handle:
            for index, pair in enumerate(pairs, start=1):
                observation = pair["manifest"]
                trace = pair["trace"]
                total_start = time.perf_counter()
                with Image.open(observation["image_path"]) as image:
                    rgb = image.convert("RGB")
                    ids, confidence, margin, seg_timing = segmenter.infer(rgb)
                segmentation_ms.append(seg_timing["segmentation_ms"])
                detector_mask = box_union_mask(
                    trace["detections"],
                    source_width=observation["width"],
                    source_height=observation["height"],
                )
                segmentation_hazard = np.isin(ids, [CLASS_TO_ID[name] for name in PRIMARY_HAZARD_CLASSES])
                arms_masks = {
                    "A": detector_mask,
                    "B": segmentation_hazard,
                    "C": detector_mask | segmentation_hazard,
                }
                fusion_start = time.perf_counter()
                truth_ids: np.ndarray | None = None
                truth_hazard: np.ndarray | None = None
                if observation["semantic_mask_path"] is not None:
                    truth_ids = resize_class_mask(
                        _mask_from_path(
                            observation["semantic_mask_path"],
                            expected_shape=(observation["height"], observation["width"]),
                        )
                    )
                    truth_hazard = np.isin(truth_ids, [CLASS_TO_ID[name] for name in PRIMARY_HAZARD_CLASSES])
                arm_rows: dict[str, Any] = {}
                if truth_hazard is not None:
                    for arm, mask in arms_masks.items():
                        arm_rows[arm] = {
                            "name": {"A": "YOLO-only", "B": "Segmentation-only", "C": "YOLO + Segmentation"}[arm],
                            "pixel": pixel_metrics(mask, truth_hazard),
                        }
                fusion_elapsed = (time.perf_counter() - fusion_start) * 1000.0
                fusion_ms.append(float(fusion_elapsed))
                candidate_hazard = segmentation_hazard & ~detector_mask
                candidate_truth = truth_hazard & ~detector_mask if truth_hazard is not None else np.zeros_like(candidate_hazard)
                component_start = time.perf_counter()
                candidate_component_score = component_metrics(candidate_hazard, candidate_truth)
                arm_component_scores = (
                    {arm: component_metrics(mask, truth_hazard) for arm, mask in arms_masks.items()}
                    if truth_hazard is not None
                    else {}
                )
                local_components: list[dict[str, Any]] = []
                for class_name in PRIMARY_HAZARD_CLASSES:
                    class_mask = (ids == CLASS_TO_ID[class_name]) & ~detector_mask
                    truth_class = (
                        (truth_ids == CLASS_TO_ID[class_name]) & ~detector_mask
                        if truth_ids is not None
                        else np.zeros_like(class_mask)
                    )
                    local_components.extend(
                        component_records(
                            {class_name: class_mask},
                            truth_class,
                            detector_mask,
                            confidence,
                            margin,
                            source_id=observation["source_id"],
                            frame_id=observation["frame_id"],
                        )
                    )
                component_elapsed = (time.perf_counter() - component_start) * 1000.0
                component_ms.append(float(component_elapsed))
                total_increment = (time.perf_counter() - total_start) * 1000.0
                total_ms.append(float(total_increment))
                frame_key = {
                    "source_id": observation["source_id"],
                    "frame_id": observation["frame_id"],
                    "image_sha256": observation["image_sha256"],
                }
                frame_row: dict[str, Any] = {
                    "schema_version": FRAME_SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    **frame_key,
                    "source_capture_timestamp_ns": observation["source_capture_timestamp_ns"],
                    "timestamp_quality": observation["timestamp_quality"],
                    "scene_bucket": observation["scene_bucket"],
                    "sequence_id": observation["sequence_id"],
                    "truth_available": truth_hazard is not None,
                    "truth_label_authority": observation["label_authority"],
                    "truth_hazard_pixels": int(np.count_nonzero(truth_hazard)) if truth_hazard is not None else None,
                    "detector_box_union_pixels": int(np.count_nonzero(detector_mask)),
                    "segmentation_hazard_pixels": int(np.count_nonzero(segmentation_hazard)),
                    "candidate_hazard_pixels": int(np.count_nonzero(candidate_hazard)),
                    "segmentation_class_pixels": {
                        class_name: int(np.count_nonzero(ids == class_id))
                        for class_name, class_id in CLASS_TO_ID.items()
                    },
                    "segmentation_confidence": {
                        "top1_median": float(np.median(confidence)),
                        "top1_p10": float(np.percentile(confidence, 10)),
                        "top1_p90": float(np.percentile(confidence, 90)),
                        "top1_top2_margin_median": float(np.median(margin)),
                    },
                    "arms": arm_rows,
                    "arm_component_metrics": arm_component_scores,
                    "candidate_pixel_metrics": (
                        pixel_metrics(candidate_hazard, candidate_truth) if truth_hazard is not None else None
                    ),
                    "candidate_component_metrics": candidate_component_score,
                    "unknown_nonwalkable_ablation": (
                        pixel_metrics(ids == CLASS_TO_ID["unknown_nonwalkable"], truth_ids == CLASS_TO_ID["unknown_nonwalkable"])
                        if truth_ids is not None
                        else None
                    ),
                    "runtime": {
                        "segmentation_ms": float(seg_timing["segmentation_ms"]),
                        "segmentation_preprocess_ms": float(seg_timing["preprocess_ms"]),
                        "segmentation_inference_ms": float(seg_timing["inference_ms"]),
                        "component_extraction_ms": float(component_elapsed),
                        "fusion_ms": float(fusion_elapsed),
                        "total_increment_ms": float(total_increment),
                    },
                    "packed_masks": {
                        "shape": [ANALYSIS_HEIGHT, ANALYSIS_WIDTH],
                        "candidate_hazard": _pack_mask(candidate_hazard),
                        "candidate_boundary_step_curb": _pack_mask(
                            (ids == CLASS_TO_ID["boundary_step_curb"]) & ~detector_mask
                        ),
                        "candidate_obstacle": _pack_mask(
                            (ids == CLASS_TO_ID["obstacle"]) & ~detector_mask
                        ),
                    },
                }
                for local in local_components:
                    local["schema_version"] = COMPONENT_SCHEMA_VERSION
                    local["protocol_id"] = PROTOCOL_ID
                    local["runtime_component_extraction_ms"] = float(component_elapsed)
                component_rows.extend(local_components)
                for class_name, mask in (
                    ("candidate_hazard", candidate_hazard),
                    ("candidate_boundary_step_curb", (ids == CLASS_TO_ID["boundary_step_curb"]) & ~detector_mask),
                    ("candidate_obstacle", (ids == CLASS_TO_ID["obstacle"]) & ~detector_mask),
                ):
                    temporal_inputs[(observation["source_id"], class_name)]["rows"].append(
                        {
                            "frame_id": observation["frame_id"],
                            "timestamp_ns": observation["source_capture_timestamp_ns"],
                            "timestamp_quality": observation["timestamp_quality"],
                            "mask": mask.copy(),
                        }
                    )
                frame_rows.append(frame_row)
                frame_handle.write(json.dumps(frame_row, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n")
                if index == len(pairs) or index % progress_every == 0:
                    progress_payload["completed_frames"] = index
                    _write_json(progress_path, progress_payload)
    except BaseException:
        progress_payload["status"] = "FAILED"
        _write_json(progress_path, progress_payload)
        raise
    finally:
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    temporal_report, temporal_internal = _build_temporal(
        temporal_inputs,
        motion_rows,
        match_iou=float(protocol["analysis"]["temporal_match_iou"]),
    )
    _attach_temporal_to_components(component_rows, temporal_internal)
    _write_jsonl(components_path, component_rows)
    truth_available = bool(frame_rows and frame_rows[0]["truth_available"])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evidence_instance": PROTOCOL_ID,
        "status": "CALIBRATION_EVALUATED" if phase == "calibration" else "EVALUATED_UNVALIDATED",
        "authority": "DEVELOPMENT_HOST_REFERENCE_ONLY",
        "phase": phase,
        "truth_status": "source_native_pixel_truth" if truth_available else "not_available",
        "claim_ceiling": protocol["claim_ceiling"],
        "forbidden_claims": protocol["forbidden_claims"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "model_contract": segmenter.contract,
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "split": split,
        "frame_count": len(frame_rows),
        "source_ids": sorted({row["source_id"] for row in frame_rows}),
        "frames_path": str(frames_path),
        "components_path": str(components_path),
        "motion_trace_path": str(motion_trace_path) if motion_trace_path else None,
        "motion_trace_sha256": sha256_file(motion_trace_path) if motion_trace_path else None,
        "analysis_grid": {"width": ANALYSIS_WIDTH, "height": ANALYSIS_HEIGHT},
        "hazard_classes": list(PRIMARY_HAZARD_CLASSES),
        "unknown_nonwalkable_ablation": "reported_separately_and_excluded_from_default_utility",
        "arms": {"A": "YOLO-only", "B": "Segmentation-only", "C": "YOLO + Segmentation"},
        "summary": {},
        "temporal": temporal_report,
        "runtime": {
            "segmentation_ms": _percentiles(segmentation_ms),
            "component_extraction_ms": _percentiles(component_ms),
            "fusion_ms": _percentiles(fusion_ms),
            "total_increment_ms": _percentiles(total_ms),
            "peak_memory_bytes_if_available": int(peak_memory) if peak_memory else None,
            "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
        },
        "component_count": len(component_rows),
        "execution": {
            "threads": threads,
            "progress_every": progress_every,
            "motion_warp_input_supplied": bool(motion_rows),
            "motion_warped_iou_reported": any(
                value.get("motion_warp_available", False)
                for source in temporal_report.values()
                for value in source.values()
            ),
        },
        "forbidden_inputs_consumed": [],
    }
    if truth_available:
        arm_aggregate = {
            arm: aggregate_confusion([row["arms"][arm]["pixel"] for row in frame_rows])
            for arm in ("A", "B", "C")
        }
        candidate_pixel = aggregate_confusion([row["candidate_pixel_metrics"] for row in frame_rows])
        candidate_components = _aggregate_components(frame_rows)
        report["summary"] = {
            "arm_pixel_metrics": arm_aggregate,
            "candidate_pixel_metrics": candidate_pixel,
            "candidate_components": candidate_components,
            "delta_recall_C_minus_A": (
                arm_aggregate["C"]["recall"] - arm_aggregate["A"]["recall"]
                if arm_aggregate["C"]["recall"] is not None and arm_aggregate["A"]["recall"] is not None
                else None
            ),
            "delta_false_positive_area_fraction_C_minus_A": (
                arm_aggregate["C"]["false_positive_area_fraction"]
                - arm_aggregate["A"]["false_positive_area_fraction"]
            ),
            "false_activation_components_per_frame": candidate_components["false_activation_components_per_frame"],
        }
    else:
        report["summary"] = {
            "arm_pixel_metrics": None,
            "candidate_pixel_metrics": None,
            "candidate_components": None,
            "delta_recall_C_minus_A": None,
            "delta_false_positive_area_fraction_C_minus_A": None,
            "false_activation_components_per_frame": None,
        }
    report["session_summary"] = _session_summary(frame_rows, component_rows)
    report["progress_receipt"] = {
        "status": "COMPLETE",
        "completed_frames": len(frame_rows),
        "total_frames": len(pairs),
        "elapsed_ms": report["runtime"]["elapsed_ms"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    progress_payload.update({"status": "COMPLETE", "completed_frames": len(frame_rows)})
    _write_json(progress_path, progress_payload)
    _write_json(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "formal", "temporal"), required=True)
    parser.add_argument("--split")
    parser.add_argument("--motion-trace", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--components", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--frames-limit", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    report_path = args.report.resolve()
    frames_path = (args.frames or report_path.with_name("frames.jsonl")).resolve()
    components_path = (args.components or report_path.with_name("components.jsonl")).resolve()
    progress_path = (args.progress or report_path.with_name("progress.json")).resolve()
    report = run_evaluation(
        repo_root=repo_root,
        protocol_path=args.protocol.resolve(),
        manifest_path=args.manifest.resolve(),
        dataset_root=args.dataset_root.resolve() if args.dataset_root else None,
        trace_path=args.trace.resolve(),
        model_path=args.model.resolve(),
        report_path=report_path,
        frames_path=frames_path,
        components_path=components_path,
        progress_path=progress_path,
        phase=args.phase,
        split=args.split,
        motion_trace_path=args.motion_trace.resolve() if args.motion_trace else None,
        threads=args.threads,
        frames_limit=args.frames_limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "phase": report["phase"],
                "frame_count": report["frame_count"],
                "source_ids": report["source_ids"],
                "report_path": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
