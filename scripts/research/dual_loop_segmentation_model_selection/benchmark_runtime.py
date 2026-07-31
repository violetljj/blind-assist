"""Benchmark the fixed R1 host runtime without truth or file-I/O timing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..dual_loop_segmentation_candidate_utility import evaluate_candidate_utility as _base
from ..dual_loop_segmentation_candidate_utility.component_metrics import connected_components
from .evaluate_model_selection import EXPECTED_GRID, HAZARD_CLASSES, PROTOCOL_ID, _load_r1_protocol


INPUT_SIZE = 256
CLASS_TO_ID = {name: index for index, name in enumerate(_base.ALL_CLASSES)}


def _percentiles(values: Sequence[float]) -> dict[str, Any]:
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


def _sha256(path: Path) -> str:
    return _base.sha256_file(path)


def _load_interpreter(model_path: Path, threads: int) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # pragma: no cover - fallback for older runtimes
        from tensorflow.lite.python.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("R1 runtime benchmark requires one input and one output")
    input_detail, output_detail = inputs[0], outputs[0]
    if tuple(input_detail["shape"]) != (1, INPUT_SIZE, INPUT_SIZE, 3) or np.dtype(input_detail["dtype"]) != np.dtype(np.int8):
        raise ValueError(f"unexpected input contract: {input_detail}")
    if tuple(output_detail["shape"]) != (1, INPUT_SIZE, INPUT_SIZE, 4) or np.dtype(output_detail["dtype"]) != np.dtype(np.int8):
        raise ValueError(f"unexpected output contract: {output_detail}")
    if float(input_detail["quantization"][0]) <= 0 or float(output_detail["quantization"][0]) <= 0:
        raise ValueError("runtime benchmark requires positive input/output quantization scales")
    return interpreter, input_detail, output_detail


def _prepare(rgb: np.ndarray, scale: float, zero: int) -> np.ndarray:
    resized = np.asarray(Image.fromarray(rgb, mode="RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.clip(np.rint(resized / scale + zero), -128, 127).astype(np.int8)[None, ...]


def run(
    *,
    protocol_path: Path,
    manifest_path: Path,
    trace_path: Path,
    model_path: Path,
    output_path: Path,
    threads: int,
    warmup_frames: int,
    measured_frames: int,
) -> dict[str, Any]:
    _load_r1_protocol(protocol_path)
    if not model_path.is_file() or not manifest_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError("runtime benchmark input is missing")
    if warmup_frames < 0 or measured_frames <= 0 or threads <= 0:
        raise ValueError("warmup_frames, measured_frames and threads are invalid")
    artifacts_root = (Path.cwd() / "artifacts.local").resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError("runtime output must remain under artifacts.local") from exc
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite runtime receipt: {output_path}")

    observations = _base.load_manifest(manifest_path, dataset_root=None, split=None, require_truth=False)
    traces = _base.load_trace(trace_path)
    pairs = _base.pair_inputs(observations, traces)
    if len(pairs) != measured_frames:
        raise ValueError(f"R1 runtime corpus must contain exactly {measured_frames} rows, got {len(pairs)}")
    preloaded: list[tuple[np.ndarray, np.ndarray]] = []
    for pair in pairs:
        observation, trace = pair["manifest"], pair["trace"]
        with Image.open(observation["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        detector_mask = _base.box_union_mask(
            trace["detections"], source_width=observation["width"], source_height=observation["height"]
        )
        preloaded.append((rgb, detector_mask))

    interpreter, input_detail, output_detail = _load_interpreter(model_path, threads)
    input_scale, input_zero = float(input_detail["quantization"][0]), int(input_detail["quantization"][1])
    output_scale, output_zero = float(output_detail["quantization"][0]), int(output_detail["quantization"][1])

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
        ids = np.argmax(scores, axis=-1).astype(np.uint8)
        hazard = np.isin(ids, [CLASS_TO_ID[name] for name in HAZARD_CLASSES])
        output_end = time.perf_counter_ns()
        component_start = time.perf_counter_ns()
        connected_components(hazard)
        component_end = time.perf_counter_ns()
        fusion_start = time.perf_counter_ns()
        _ = detector_mask | hazard
        _ = hazard & ~detector_mask
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
        rgb, detector_mask = preloaded[index % len(preloaded)]
        one(rgb, detector_mask, measure=False)
    measured: dict[str, list[float]] = {name: [] for name in ("preprocess", "tflite_inference", "output_dequantize_argmax", "component_extraction", "fusion_operator", "total_increment")}
    for rgb, detector_mask in preloaded:
        row = one(rgb, detector_mask, measure=True)
        for name, value in row.items():
            measured[name].append(value)
    runtime = {name: _percentiles(values) for name, values in measured.items()}
    result = {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.runtime_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "RUNTIME_BENCHMARK_COMPLETE",
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "corpus": {"frame_count": len(preloaded), "preload_inputs": True, "truth_pixels_read": False},
        "runtime_contract": {
            "device_class": "fixed_host_benchmark_machine",
            "threads": threads,
            "warmup_frames": warmup_frames,
            "measured_frames": measured_frames,
            "analysis_grid": EXPECTED_GRID,
            "timed_stages_ms": list(measured),
            "excluded_from_timing": ["truth metrics", "file IO", "JSON logging", "cold interpreter construction"],
        },
        "input_quantization": {"scale": input_scale, "zero_point": input_zero},
        "output_quantization": {"scale": output_scale, "zero_point": output_zero},
        "runtime": runtime,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--measured-frames", type=int, default=200)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = run(
        protocol_path=args.protocol.resolve(),
        manifest_path=args.manifest.resolve(),
        trace_path=args.trace.resolve(),
        model_path=args.model.resolve(),
        output_path=args.output.resolve(),
        threads=args.threads,
        warmup_frames=args.warmup_frames,
        measured_frames=args.measured_frames,
    )
    print(json.dumps({"status": result["status"], "model_sha256": result["model_sha256"], "runtime": result["runtime"]}, ensure_ascii=False))
