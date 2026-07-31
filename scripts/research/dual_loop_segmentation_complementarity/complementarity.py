"""Run the fixed image-space YOLO/semantic-segmentation complementarity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

CLASS_NAMES = (
    "walkable",
    "boundary_step_curb",
    "obstacle",
    "unknown_nonwalkable",
)
SCHEMA_VERSION = "blindassist.dual_loop_segmentation_complementarity_r1.v1"
EVIDENCE_INSTANCE = "DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_R1"
EXPECTED_TRACE_SCHEMAS = {"blindassist.dual_loop_unseen_rank2_baseline_trace.v1"}
REQUIRED_MANIFEST_FIELDS = {
    "source_id",
    "frame_id",
    "source_capture_timestamp_ns",
    "image_path",
    "image_sha256",
    "width",
    "height",
}
REQUIRED_TRACE_FIELDS = {
    "schema_version",
    "source_id",
    "frame_id",
    "source_capture_timestamp_ns",
    "image_sha256",
    "detections",
}


class ComplementarityInputError(ValueError):
    """Raised when the frozen identity or tensor contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(repo_root: Path, value: str | Path, *, base_dir: Path | None = None) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return ((base_dir if base_dir is not None else repo_root) / candidate).resolve()


def ensure_artifact_path(repo_root: Path, value: Path) -> Path:
    artifacts_root = (repo_root / "artifacts.local").resolve()
    resolved = value.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as exc:
        raise ComplementarityInputError(f"output must stay under artifacts.local: {resolved}") from exc
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ComplementarityInputError(f"{path}:{line_number}: blank JSONL line")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ComplementarityInputError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ComplementarityInputError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    if not rows:
        raise ComplementarityInputError(f"{path}: empty JSONL input")
    return rows


def _identity(row: dict[str, Any]) -> tuple[str, int, str]:
    try:
        return str(row["source_id"]), int(row["frame_id"]), str(row["image_sha256"]).lower()
    except (KeyError, TypeError, ValueError) as exc:
        raise ComplementarityInputError(f"invalid frame identity: {row}") from exc


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_manifest(manifest_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(manifest_path)
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    previous_timestamp: int | None = None
    for row_number, row in enumerate(rows, start=1):
        missing = REQUIRED_MANIFEST_FIELDS - row.keys()
        if missing:
            raise ComplementarityInputError(f"manifest row {row_number}: missing {sorted(missing)}")
        key = _identity(row)
        if key in seen:
            raise ComplementarityInputError(f"manifest row {row_number}: duplicate identity {key}")
        seen.add(key)
        timestamp = int(row["source_capture_timestamp_ns"])
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ComplementarityInputError(f"manifest row {row_number}: non-increasing timestamp")
        previous_timestamp = timestamp
        width, height = int(row["width"]), int(row["height"])
        if width <= 0 or height <= 0:
            raise ComplementarityInputError(f"manifest row {row_number}: invalid dimensions")
        image_path = resolve_path(repo_root, str(row["image_path"]), base_dir=manifest_path.parent)
        if not image_path.is_file():
            raise ComplementarityInputError(f"manifest row {row_number}: missing image {image_path}")
        actual_sha = sha256_file(image_path)
        if actual_sha.lower() != key[2]:
            raise ComplementarityInputError(
                f"manifest row {row_number}: image hash mismatch for {image_path}"
            )
        observations.append(
            {
                "source_id": key[0],
                "frame_id": key[1],
                "image_sha256": key[2],
                "source_capture_timestamp_ns": timestamp,
                "image_path": image_path,
                "image_path_input": str(row["image_path"]),
                "width": width,
                "height": height,
            }
        )
    return observations


def _validate_detection(detection: Any, *, row_number: int, index: int) -> dict[str, float]:
    if not isinstance(detection, dict):
        raise ComplementarityInputError(f"trace row {row_number} detection {index}: expected object")
    required = {"left", "top", "right", "bottom", "frame_width", "frame_height", "source"}
    missing = required - detection.keys()
    if missing:
        raise ComplementarityInputError(
            f"trace row {row_number} detection {index}: missing {sorted(missing)}"
        )
    if detection["source"] != "OBJECT_DETECTOR":
        raise ComplementarityInputError(
            f"trace row {row_number} detection {index}: unexpected source {detection['source']!r}"
        )
    values = {name: float(detection[name]) for name in ("left", "top", "right", "bottom")}
    if not all(_finite(value) for value in values.values()):
        raise ComplementarityInputError(f"trace row {row_number} detection {index}: non-finite box")
    frame_width, frame_height = int(detection["frame_width"]), int(detection["frame_height"])
    if frame_width <= 0 or frame_height <= 0:
        raise ComplementarityInputError(f"trace row {row_number} detection {index}: invalid frame size")
    values["frame_width"] = float(frame_width)
    values["frame_height"] = float(frame_height)
    return values


def load_trace(trace_path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows = _read_jsonl(trace_path)
    traces: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        missing = REQUIRED_TRACE_FIELDS - row.keys()
        if missing:
            raise ComplementarityInputError(f"trace row {row_number}: missing {sorted(missing)}")
        if row["schema_version"] not in EXPECTED_TRACE_SCHEMAS:
            raise ComplementarityInputError(
                f"trace row {row_number}: unsupported schema {row['schema_version']!r}"
            )
        key = _identity(row)
        if key in traces:
            raise ComplementarityInputError(f"trace row {row_number}: duplicate identity {key}")
        timestamp = int(row["source_capture_timestamp_ns"])
        detections = row["detections"]
        if not isinstance(detections, list):
            raise ComplementarityInputError(f"trace row {row_number}: detections must be a list")
        traces[key] = {
            "source_id": key[0],
            "frame_id": key[1],
            "image_sha256": key[2],
            "source_capture_timestamp_ns": timestamp,
            "detections": [
                _validate_detection(item, row_number=row_number, index=index)
                for index, item in enumerate(detections)
            ],
        }
    return traces


def pair_inputs(manifest_rows: list[dict[str, Any]], trace_rows: dict[tuple[str, int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_keys = {
        (row["source_id"], row["frame_id"], row["image_sha256"]): row for row in manifest_rows
    }
    trace_keys = set(trace_rows)
    if manifest_keys.keys() != trace_keys:
        missing_trace = sorted(manifest_keys.keys() - trace_keys)
        missing_manifest = sorted(trace_keys - manifest_keys)
        raise ComplementarityInputError(
            f"exact pairing failed: missing_trace={missing_trace[:3]} missing_manifest={missing_manifest[:3]}"
        )
    pairs: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        key = (manifest_row["source_id"], manifest_row["frame_id"], manifest_row["image_sha256"])
        trace_row = trace_rows[key]
        if trace_row["source_capture_timestamp_ns"] != manifest_row["source_capture_timestamp_ns"]:
            raise ComplementarityInputError(f"timestamp mismatch for {key}")
        pairs.append({"manifest": manifest_row, "trace": trace_row})
    return pairs


def box_union_mask(
    detections: Iterable[dict[str, float]],
    *,
    source_width: int,
    source_height: int,
    analysis_width: int,
    analysis_height: int,
) -> np.ndarray:
    if source_width <= 0 or source_height <= 0 or analysis_width <= 0 or analysis_height <= 0:
        raise ValueError("all image dimensions must be positive")
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


def mask_iou(left: np.ndarray | None, right: np.ndarray) -> float | None:
    if left is None:
        return None
    if left.shape != right.shape:
        raise ValueError("mask shapes must match")
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def component_count(mask: np.ndarray) -> int:
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=np.uint8)
    _, count = ndimage.label(mask, structure=structure)
    return int(count)


def _quantization(detail: dict[str, Any], label: str) -> tuple[float, int]:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if not _finite(scale) or float(scale) <= 0:
        raise ComplementarityInputError(f"{label}: positive scalar quantization scale required")
    return float(scale), int(zero_point)


def _prepare_int8_rgb(image: Image.Image, shape: tuple[int, ...], scale: float, zero_point: int) -> np.ndarray:
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
        raise ComplementarityInputError(f"input tensor must be [1,H,W,3], got {shape}")
    resized = image.convert("RGB").resize((shape[2], shape[1]), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32)
    return np.clip(np.rint(rgb / scale + zero_point), -128, 127).astype(np.int8)[None, ...]


def _dequantize(raw: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if np.issubdtype(raw.dtype, np.integer):
        scale, zero_point = _quantization(detail, "output")
        return (raw.astype(np.float32) - zero_point) * scale
    return raw.astype(np.float32)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _progress(path: Path, *, completed: int, total: int, status: str) -> None:
    _write_json(
        path,
        {
            "schema_version": "blindassist.dual_loop_segmentation_complementarity_progress.v1",
            "evidence_instance": EVIDENCE_INSTANCE,
            "status": status,
            "completed_frames": completed,
            "total_frames": total,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _load_interpreter(model_path: Path, threads: int) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("TensorFlow is required for the TFLite complementarity diagnostic") from exc
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ComplementarityInputError("exactly one input and one output tensor are required")
    input_detail, output_detail = input_details[0], output_details[0]
    input_shape = tuple(int(value) for value in input_detail["shape"])
    output_shape = tuple(int(value) for value in output_detail["shape"])
    input_checks = {
        "input_shape_is_nhwc_rgb": len(input_shape) == 4 and input_shape[0] == 1 and input_shape[3] == 3,
        "input_dtype_is_int8": np.dtype(input_detail["dtype"]) == np.dtype(np.int8),
        "output_shape_is_nhwc": len(output_shape) == 4 and output_shape[0] == 1,
        "output_dtype_is_int8": np.dtype(output_detail["dtype"]) == np.dtype(np.int8),
        "output_has_four_classes": len(output_shape) == 4 and output_shape[3] == len(CLASS_NAMES),
    }
    if not all(input_checks.values()):
        raise ComplementarityInputError(f"interface contract failed: {input_checks}")
    input_scale, input_zero = _quantization(input_detail, "input")
    output_scale, output_zero = _quantization(output_detail, "output")
    contract = {
        "checks": input_checks,
        "input": {
            "shape": list(input_shape),
            "dtype": str(np.dtype(input_detail["dtype"])),
            "quantization": {"scale": input_scale, "zero_point": input_zero},
        },
        "output": {
            "shape": list(output_shape),
            "dtype": str(np.dtype(output_detail["dtype"])),
            "quantization": {"scale": output_scale, "zero_point": output_zero},
        },
    }
    return interpreter, input_detail, output_detail, {"input_shape": input_shape, "output_shape": output_shape, "contract": contract}


def _summarize_frame_rows(frame_rows: list[dict[str, Any]], class_names: tuple[str, ...]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_source[row["source_id"]].append(row)
    class_summary: dict[str, Any] = {}
    temporal_summary: dict[str, Any] = {}
    for class_name in class_names:
        class_summary[class_name] = {
            "segmentation_fraction": _percentiles(
                [row["segmentation"][class_name]["fraction"] for row in frame_rows]
            ),
            "uncovered_fraction": _percentiles(
                [row["segmentation"][class_name]["uncovered_fraction"] for row in frame_rows]
            ),
            "uncovered_nonempty_fraction": float(
                np.mean([row["segmentation"][class_name]["uncovered_pixels"] > 0 for row in frame_rows])
            ),
            "component_count": _percentiles(
                [float(row["segmentation"][class_name]["component_count"]) for row in frame_rows]
            ),
        }
        temporal_values = [
            row["segmentation"][class_name]["temporal_iou"]
            for row in frame_rows
            if row["segmentation"][class_name]["temporal_iou"] is not None
        ]
        temporal_summary[class_name] = {
            "adjacent_frame_count": len(temporal_values),
            "iou": _percentiles([float(value) for value in temporal_values]),
        }
    session_summary = []
    for source_id, rows in sorted(by_source.items()):
        session_summary.append(
            {
                "source_id": source_id,
                "frame_count": len(rows),
                "detector_coverage_fraction": _percentiles(
                    [row["detector"]["coverage_fraction"] for row in rows]
                ),
                "union_increment_fraction": _percentiles(
                    [row["fusion_geometry"]["union_increment_fraction"] for row in rows]
                ),
            }
        )
    return {
        "frame_count": len(frame_rows),
        "source_session_count": len(by_source),
        "class_summary": class_summary,
        "temporal_summary": temporal_summary,
        "session_summary": session_summary,
        "aggregation_rule": "session_first_then_equal_weight; frame_rows_descriptive_only",
    }


def run_diagnostic(
    *,
    repo_root: Path,
    manifest_path: Path,
    trace_path: Path,
    model_path: Path,
    report_path: Path,
    frames_path: Path,
    progress_path: Path,
    threads: int,
    progress_every: int = 250,
) -> dict[str, Any]:
    if threads <= 0 or progress_every <= 0:
        raise ValueError("threads and progress_every must be positive")
    manifest_rows = load_manifest(manifest_path, repo_root)
    trace_rows = load_trace(trace_path)
    pairs = pair_inputs(manifest_rows, trace_rows)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    interpreter, input_detail, output_detail, model_contract = _load_interpreter(model_path, threads)
    output_shape = model_contract["output_shape"]
    analysis_height, analysis_width, class_count = output_shape[1], output_shape[2], output_shape[3]
    input_shape = model_contract["input_shape"]
    input_scale, input_zero = _quantization(input_detail, "input")
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    previous_uncovered: dict[str, np.ndarray | None] = {name: None for name in CLASS_NAMES}
    durations_ms: list[float] = []
    inference_ms: list[float] = []
    preprocess_ms: list[float] = []
    finite_values = True
    class_total_pixels = np.zeros(class_count, dtype=np.int64)
    detector_total_pixels = 0
    started_all = time.perf_counter()
    _progress(progress_path, completed=0, total=len(pairs), status="RUNNING")
    try:
        with frames_path.open("w", encoding="utf-8", newline="\n") as frames_handle:
            for index, pair in enumerate(pairs, start=1):
                manifest_row, trace_row = pair["manifest"], pair["trace"]
                image_path = manifest_row["image_path"]
                for detection in trace_row["detections"]:
                    if int(detection["frame_width"]) != manifest_row["width"] or int(
                        detection["frame_height"]
                    ) != manifest_row["height"]:
                        raise ComplementarityInputError(
                            f"detection frame dimensions mismatch for {manifest_row['source_id']}:{manifest_row['frame_id']}"
                        )
                load_started = time.perf_counter()
                with Image.open(image_path) as image:
                    if image.size != (manifest_row["width"], manifest_row["height"]):
                        raise ComplementarityInputError(f"image dimensions mismatch: {image_path}")
                    tensor = _prepare_int8_rgb(image, input_shape, input_scale, input_zero)
                preprocess_ms.append(float((time.perf_counter() - load_started) * 1000.0))
                detector_mask = box_union_mask(
                    trace_row["detections"],
                    source_width=manifest_row["width"],
                    source_height=manifest_row["height"],
                    analysis_width=analysis_width,
                    analysis_height=analysis_height,
                )
                invoke_started = time.perf_counter()
                interpreter.set_tensor(input_detail["index"], tensor)
                interpreter.invoke()
                invoke_ms = float((time.perf_counter() - invoke_started) * 1000.0)
                total_ms = float((time.perf_counter() - load_started) * 1000.0)
                raw_output = interpreter.get_tensor(output_detail["index"])
                values = _dequantize(raw_output, output_detail)
                finite_values = finite_values and bool(np.isfinite(values).all())
                labels = np.argmax(raw_output[0], axis=-1)
                if labels.shape != (analysis_height, analysis_width):
                    raise ComplementarityInputError(f"unexpected argmax shape {labels.shape}")
                class_counts = np.bincount(labels.reshape(-1), minlength=class_count)
                class_total_pixels += class_counts
                detector_pixels = int(detector_mask.sum())
                detector_total_pixels += detector_pixels
                durations_ms.append(total_ms)
                inference_ms.append(invoke_ms)
                per_class: dict[str, Any] = {}
                for class_id, class_name in enumerate(CLASS_NAMES):
                    class_mask = labels == class_id
                    uncovered = np.logical_and(class_mask, np.logical_not(detector_mask))
                    per_class[class_name] = {
                        "class_id": class_id,
                        "pixels": int(class_mask.sum()),
                        "fraction": float(class_mask.mean()),
                        "uncovered_pixels": int(uncovered.sum()),
                        "uncovered_fraction": float(uncovered.mean()),
                        "component_count": component_count(uncovered),
                        "uncovered_nonempty": bool(uncovered.any()),
                        "temporal_iou": mask_iou(previous_uncovered[class_name], uncovered),
                    }
                    previous_uncovered[class_name] = uncovered
                total_pixels = analysis_width * analysis_height
                segmentation_union = np.ones((analysis_height, analysis_width), dtype=bool)
                frame_row = {
                    "source_id": manifest_row["source_id"],
                    "frame_id": manifest_row["frame_id"],
                    "source_capture_timestamp_ns": manifest_row["source_capture_timestamp_ns"],
                    "image_sha256": manifest_row["image_sha256"],
                    "source_width": manifest_row["width"],
                    "source_height": manifest_row["height"],
                    "analysis_width": analysis_width,
                    "analysis_height": analysis_height,
                    "detector": {
                        "box_count": len(trace_row["detections"]),
                        "covered_pixels": detector_pixels,
                        "coverage_fraction": float(detector_pixels / total_pixels),
                    },
                    "segmentation": per_class,
                    "fusion_geometry": {
                        "all_class_union_pixels": int(segmentation_union.sum()),
                        "all_class_union_fraction": float(segmentation_union.mean()),
                        "union_increment_pixels": int(
                            np.logical_and(segmentation_union, np.logical_not(detector_mask)).sum()
                        ),
                        "union_increment_fraction": float(1.0 - detector_pixels / total_pixels),
                        "all_class_union_covers_grid_by_construction": True,
                    },
                    "runtime_ms": {
                        "preprocess_and_load": preprocess_ms[-1],
                        "inference": invoke_ms,
                        "total": total_ms,
                    },
                }
                frame_rows.append(frame_row)
                frames_handle.write(json.dumps(frame_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if index == len(pairs) or index % progress_every == 0:
                    frames_handle.flush()
                    _progress(progress_path, completed=index, total=len(pairs), status="RUNNING")
    except Exception:
        _progress(progress_path, completed=len(frame_rows), total=len(pairs), status="FAILED")
        raise
    _progress(progress_path, completed=len(pairs), total=len(pairs), status="COMPLETE")
    if not finite_values:
        status = "FAIL_NONFINITE_OUTPUT"
    elif np.count_nonzero(class_total_pixels) <= 1:
        status = "STOPPED_SINGLE_CLASS_COLLAPSE"
    else:
        status = "COMPLETE_DEVELOPMENT_DIAGNOSTIC"
    report = {
        "schema_version": SCHEMA_VERSION,
        "evidence_instance": EVIDENCE_INSTANCE,
        "status": status,
        "authority": "DEVELOPMENT_ONLY / USER_AUTHORIZED / NO_EFFECT_AUTHORITY",
        "claim_ceiling": "image_space_mechanism_diagnostic_only",
        "user_authorization": "explicit_current_task_authorization_2026-07-31",
        "central_obstruction_truth_read": False,
        "risk_feedback_event_fields_read": False,
        "android_or_production_changed": False,
        "model_selection_performed": False,
        "model_comparison_performed": False,
        "fusion_effect_evaluated": False,
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": sha256_file(manifest_path),
            "role": "burned_development_rgb_input",
            "frame_count": len(manifest_rows),
            "source_ids": sorted({row["source_id"] for row in manifest_rows}),
        },
        "trace": {
            "path": trace_path.as_posix(),
            "sha256": sha256_file(trace_path),
            "role": "matched_yolo_reference_boxes_only",
            "frame_count": len(trace_rows),
        },
        "model": {
            "path": model_path.as_posix(),
            "sha256": sha256_file(model_path),
            "bytes": model_path.stat().st_size,
            "reference_name": "sanpo-v3-pretrained-weighted-best-int8-20260713",
            "role": "fixed_benchmark_only_reference",
            "comparison_rank": None,
        },
        "pairing": {
            "paired_frame_count": len(pairs),
            "not_evaluable_frame_count": 0,
            "interpolation_used": False,
            "nearest_frame_repair_used": False,
            "identity_key": "source_id + frame_id + image_sha256",
            "timestamp_exact_match": True,
        },
        "analysis": {
            "grid": {"width": analysis_width, "height": analysis_height},
            "classes": list(CLASS_NAMES),
            "box_projection": "normalized source box; floor left/top, ceil right/bottom; clip to grid",
            "estimands": {
                "detector_coverage_fraction": "|D_t| / |Omega|",
                "uncovered_fraction_t_k": "|S_t,k \\ D_t| / |Omega|",
                "union_increment_t": "|(D_t union all S_t,k) \\ D_t| / |Omega|",
            },
            "all_class_union_note": "all argmax class masks partition Omega, so union_increment is reported transparently and is not obstacle/risk evidence",
            "aggregation": "session-first then equal-weight; frame rows descriptive and not independent replicates",
        },
        "interface": model_contract["contract"],
        "runtime_ms": {
            "total": _percentiles(durations_ms),
            "inference": _percentiles(inference_ms),
            "preprocess_and_load": _percentiles(preprocess_ms),
            "threads": threads,
            "host_only": True,
        },
        "class_pixel_totals": {
            name: int(class_total_pixels[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "detector_coverage": {
            "mean_pixels_per_frame": float(detector_total_pixels / len(pairs)),
            "mean_fraction_per_frame": float(detector_total_pixels / (len(pairs) * analysis_width * analysis_height)),
        },
        "summary": _summarize_frame_rows(frame_rows, CLASS_NAMES),
        "artifacts": {
            "report": report_path.as_posix(),
            "frames": frames_path.as_posix(),
            "progress": progress_path.as_posix(),
        },
        "stop_checks": {
            "pairing_pass": len(pairs) == len(manifest_rows) == len(trace_rows),
            "finite_output_pass": finite_values,
            "single_class_collapse": np.count_nonzero(class_total_pixels) <= 1,
            "risk_truth_or_event_gate_applied": False,
        },
        "elapsed_host_seconds": float(time.perf_counter() - started_all),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-output", type=Path, required=True)
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path.cwd().resolve()
    manifest_path = resolve_path(repo_root, args.manifest)
    trace_path = resolve_path(repo_root, args.trace)
    model_path = resolve_path(repo_root, args.model)
    report_path = ensure_artifact_path(repo_root, resolve_path(repo_root, args.output))
    frames_path = ensure_artifact_path(repo_root, resolve_path(repo_root, args.frames_output))
    progress_value = args.progress_output or report_path.with_name("progress.json")
    progress_path = ensure_artifact_path(repo_root, resolve_path(repo_root, progress_value))
    report = run_diagnostic(
        repo_root=repo_root,
        manifest_path=manifest_path,
        trace_path=trace_path,
        model_path=model_path,
        report_path=report_path,
        frames_path=frames_path,
        progress_path=progress_path,
        threads=args.threads,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report_path.as_posix(),
                "frames": frames_path.as_posix(),
                "paired_frames": report["pairing"]["paired_frame_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
