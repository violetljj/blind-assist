"""Benchmark R2-P0 candidates and persist immutable per-frame stage timings."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from . import PROTOCOL_ID
from .canonicalizer import sha256_file
from .postprocess import filter_candidate_by_class, load_postprocess
from ..dual_loop_segmentation_candidate_utility import evaluate_candidate_utility as base
from ..dual_loop_segmentation_model_selection.benchmark_runtime import _load_interpreter


STAGES = (
    "preprocess",
    "tflite_inference",
    "output_dequantize_argmax",
    "component_extraction",
    "fusion_operator",
    "total_increment",
)
CLASS_TO_ID = {
    "walkable": 0,
    "boundary_step_curb": 1,
    "obstacle": 2,
    "unknown_nonwalkable": 3,
}


class RuntimeRowsError(ValueError):
    """Raised when runtime inputs or timing rows violate the frozen contract."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeRowsError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeRowsError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeRowsError(f"expected object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise RuntimeRowsError(f"zero-row runtime input: {path}")
    return rows


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0):
        raise RuntimeRowsError("runtime stage contains invalid values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _prepare(rgb: np.ndarray, scale: float, zero: int) -> np.ndarray:
    resized = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize((256, 256), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    return np.clip(np.rint(resized / scale + zero), -128, 127).astype(np.int8)[None, ...]


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(
    *,
    repo_root: Path,
    view_root: Path,
    role: str,
    trace_path: Path,
    model_path: Path,
    postprocess_path: Path,
    runtime_schema_path: Path,
    output_root: Path,
    threads: int,
    warmup_frames: int,
    measured_frames: int,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to((repo_root / "artifacts.local").resolve())
    except ValueError as exc:
        raise RuntimeRowsError("runtime output must stay under artifacts.local") from exc
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite runtime output: {output_root}")
    if threads <= 0 or warmup_frames < 0 or measured_frames <= 0:
        raise RuntimeRowsError("invalid runtime execution parameters")
    view_root = view_root.resolve()
    view_receipt = _read_json(view_root / "receipt.json")
    manifest_path = view_root / str(view_receipt["manifest"])
    if sha256_file(manifest_path) != view_receipt.get("manifest_sha256"):
        raise RuntimeRowsError("canonical view manifest identity mismatch")
    view_rows = [
        row for row in _read_jsonl(manifest_path)
        if row.get("role") == role
    ]
    if len(view_rows) != measured_frames:
        raise RuntimeRowsError(
            f"runtime role {role!r} must contain {measured_frames} rows, got {len(view_rows)}"
        )
    traces = base.load_trace(trace_path.resolve())
    postprocess = load_postprocess(postprocess_path.resolve())
    preloaded: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for row in view_rows:
        image_path = (repo_root / str(row["image_repo_relative_path"])).resolve()
        if sha256_file(image_path) != row.get("image_sha256"):
            raise RuntimeRowsError(f"{row['id']}: image SHA256 mismatch")
        with Image.open(image_path) as image:
            width, height = image.size
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        key = (str(row["source_id"]), int(row["frame_id"]), str(row["image_sha256"]))
        trace = traces.get(key)
        if trace is None:
            raise RuntimeRowsError(f"{row['id']}: missing YOLO trace")
        detector_mask = base.box_union_mask(
            trace["detections"],
            source_width=width,
            source_height=height,
        )
        preloaded.append((row, rgb, detector_mask))
    interpreter, input_detail, output_detail = _load_interpreter(model_path.resolve(), threads)
    input_scale, input_zero = (
        float(input_detail["quantization"][0]),
        int(input_detail["quantization"][1]),
    )
    output_scale, output_zero = (
        float(output_detail["quantization"][0]),
        int(output_detail["quantization"][1]),
    )

    def one(rgb: np.ndarray, detector_mask: np.ndarray, *, measure: bool) -> dict[str, float]:
        total_start = time.perf_counter_ns()
        preprocess_start = time.perf_counter_ns()
        tensor = _prepare(rgb, input_scale, input_zero)
        preprocess_end = time.perf_counter_ns()
        interpreter.set_tensor(input_detail["index"], tensor)
        inference_start = time.perf_counter_ns()
        interpreter.invoke()
        inference_end = time.perf_counter_ns()
        output_start = time.perf_counter_ns()
        raw = interpreter.get_tensor(output_detail["index"])[0]
        scores = (raw.astype(np.float32) - output_zero) * output_scale
        maximum = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(np.clip(scores - maximum, -80.0, 80.0))
        probabilities = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        ids = np.argmax(probabilities, axis=-1).astype(np.uint8)
        top_two = np.partition(probabilities, -2, axis=-1)[..., -2:]
        confidence = top_two[..., 1].astype(np.float32)
        margin = (top_two[..., 1] - top_two[..., 0]).astype(np.float32)
        output_end = time.perf_counter_ns()
        component_start = time.perf_counter_ns()
        candidate_by_class = filter_candidate_by_class(
            ids=ids,
            confidence=confidence,
            margin=margin,
            detector_mask=detector_mask,
            class_to_id=CLASS_TO_ID,
            config=postprocess,
        )
        candidate = candidate_by_class["boundary_step_curb"] | candidate_by_class["obstacle"]
        component_end = time.perf_counter_ns()
        fusion_start = time.perf_counter_ns()
        _ = detector_mask | candidate
        _ = candidate & ~detector_mask
        fusion_end = time.perf_counter_ns()
        total_end = time.perf_counter_ns()
        if not measure:
            return {}
        return {
            "preprocess": (preprocess_end - preprocess_start) / 1e6,
            "tflite_inference": (inference_end - inference_start) / 1e6,
            "output_dequantize_argmax": (output_end - output_start) / 1e6,
            "component_extraction": (component_end - component_start) / 1e6,
            "fusion_operator": (fusion_end - fusion_start) / 1e6,
            "total_increment": (total_end - total_start) / 1e6,
        }

    for index in range(warmup_frames):
        _, rgb, detector_mask = preloaded[index % len(preloaded)]
        one(rgb, detector_mask, measure=False)
    rows: list[dict[str, Any]] = []
    for index, (view_row, rgb, detector_mask) in enumerate(preloaded):
        timings = one(rgb, detector_mask, measure=True)
        if set(timings) != set(STAGES):
            raise RuntimeRowsError("runtime stage set mismatch")
        if any(not np.isfinite(value) or value < 0 for value in timings.values()):
            raise RuntimeRowsError("runtime row contains invalid timing")
        rows.append(
            {
                "schema_version": "blindassist.dual_loop_segmentation_r2_p0.runtime_row.v1",
                "protocol_id": PROTOCOL_ID,
                "candidate_id": postprocess["candidate_id"],
                "source_id": view_row["source_id"],
                "frame_id": int(view_row["frame_id"]),
                "image_sha256": view_row["image_sha256"],
                "measurement_index": index,
                "stages_ms": timings,
            }
        )
    runtime = {
        stage: _summary([float(row["stages_ms"][stage]) for row in rows])
        for stage in STAGES
    }
    output_root.mkdir(parents=True)
    rows_path = output_root / "runtime_rows.jsonl"
    _write_jsonl_atomic(rows_path, rows)
    report = {
        "schema_version": "blindassist.dual_loop_segmentation_r2_p0.runtime_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "RUNTIME_ROWS_COMPLETE_UNVALIDATED",
        "formal_authority": False,
        "candidate_id": postprocess["candidate_id"],
        "model_sha256": sha256_file(model_path.resolve()),
        "view_manifest_sha256": sha256_file(manifest_path),
        "trace_sha256": sha256_file(trace_path.resolve()),
        "postprocess_sha256": sha256_file(postprocess_path.resolve()),
        "runtime_harness_sha256": sha256_file(Path(__file__).resolve()),
        "runtime_row_schema_sha256": sha256_file(runtime_schema_path.resolve()),
        "runtime_rows_sha256": sha256_file(rows_path),
        "runtime_contract": {
            "threads": threads,
            "warmup_frames": warmup_frames,
            "measured_frames": measured_frames,
            "preloaded_inputs": True,
            "truth_pixels_read": False,
            "timed_stages_ms": list(STAGES),
            "excluded_from_timing": [
                "truth metrics",
                "file IO",
                "JSON logging",
                "validator",
                "cold interpreter construction",
            ],
        },
        "runtime": runtime,
    }
    _write_json_atomic(output_root / "report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--runtime-schema", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--measured-frames", type=int, default=200)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    value = run(
        repo_root=args.repo_root,
        view_root=args.view_root,
        role=args.role,
        trace_path=args.trace,
        model_path=args.model,
        postprocess_path=args.postprocess,
        runtime_schema_path=args.runtime_schema,
        output_root=args.output_root,
        threads=args.threads,
        warmup_frames=args.warmup_frames,
        measured_frames=args.measured_frames,
    )
    print(json.dumps({"status": value["status"], "runtime": value["runtime"]}))
