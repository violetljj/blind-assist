from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL = "app/src/main/assets/yolo11n_fp16_320.tflite"
DEFAULT_LAB_ROOT = ".downloads/detector-lab"
DEFAULT_DATASET_ROOT = ".downloads/detector-lab/datasets/coco8"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def configure_local_caches(project_root: Path) -> None:
    matplotlib_cache = project_root / ".cache" / "matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    matplotlib_cache.mkdir(parents=True, exist_ok=True)


def create_interpreter(model_path: Path) -> tuple[Any, str]:
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter(model_path=str(model_path)), "ai-edge-litert"
    except Exception:
        try:
            import tensorflow as tf

            return tf.lite.Interpreter(model_path=str(model_path)), "tensorflow"
        except Exception as tensorflow_error:
            raise RuntimeError(
                "Could not import ai-edge-litert or tensorflow to benchmark the TFLite model."
            ) from tensorflow_error


def dtype_name(dtype: Any) -> str:
    return getattr(dtype, "__name__", str(dtype).replace("<class '", "").replace("'>", ""))


def tensor_shape(tensor: dict[str, Any]) -> list[int]:
    shape = tensor.get("shape")
    return [int(part) for part in shape.tolist()] if hasattr(shape, "tolist") else list(shape)


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def default_models(project_root: Path, lab_root: Path) -> list[Path]:
    models = [resolve_path(project_root, DEFAULT_MODEL)]
    models.extend(sorted((lab_root / "exports").glob("*.tflite")))
    unique: list[Path] = []
    seen = set()
    for model in models:
        resolved = model.resolve()
        if model.is_file() and resolved not in seen:
            unique.append(model)
            seen.add(resolved)
    return unique


def image_paths(dataset_root: Path, limit: int) -> list[Path]:
    if not dataset_root.is_dir():
        return []
    paths = sorted(path for path in dataset_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:limit] if limit > 0 else paths


def prepare_input(input_detail: dict[str, Any], image_path: Path | None) -> np.ndarray:
    shape = tensor_shape(input_detail)
    dtype = input_detail["dtype"]
    if len(shape) != 4:
        raise ValueError(f"Only NHWC 4D inputs are supported, got {shape}")
    batch, height, width, channels = shape
    if batch != 1 or channels != 3:
        raise ValueError(f"Only [1,H,W,3] inputs are supported, got {shape}")

    if image_path is None:
        array = np.zeros((height, width, channels), dtype=np.float32)
    else:
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("RGB").resize((width, height)), dtype=np.float32)

    if dtype == np.float32:
        array = array / 255.0
    elif dtype == np.uint8:
        array = array.astype(np.uint8)
    elif dtype == np.int8:
        array = (array - 128).clip(-128, 127).astype(np.int8)
    else:
        array = array.astype(dtype)
    return np.expand_dims(array, axis=0)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[min(max(index, 0), len(ordered) - 1)]


def output_layout(shape: list[int]) -> str:
    if len(shape) != 3:
        return "unknown"
    dim1 = shape[1]
    dim2 = shape[2]
    if dim1 >= 5 and dim1 <= dim2:
        return "yolo_raw_channels_first"
    if dim2 >= 5 and dim2 < dim1:
        return "yolo_raw_channels_last"
    if dim2 >= 6 and dim1 <= 1000:
        return "end2end_detections"
    return "unknown"


def summarize_output(output: np.ndarray, confidence_threshold: float) -> dict[str, Any]:
    squeezed = np.asarray(output).squeeze(axis=0) if output.ndim >= 3 and output.shape[0] == 1 else output
    shape = list(output.shape)
    layout = output_layout(shape)
    result: dict[str, Any] = {
        "shape": shape,
        "layout": layout,
        "min": float(np.min(output)),
        "max": float(np.max(output)),
        "mean": float(np.mean(output)),
    }

    if layout == "yolo_raw_channels_first":
        scores = np.max(squeezed[4:, :], axis=0)
        result["rough_candidates_over_threshold"] = int(np.count_nonzero(scores >= confidence_threshold))
        result["top_score"] = float(np.max(scores)) if scores.size else 0.0
    elif layout == "yolo_raw_channels_last":
        scores = np.max(squeezed[:, 4:], axis=1)
        result["rough_candidates_over_threshold"] = int(np.count_nonzero(scores >= confidence_threshold))
        result["top_score"] = float(np.max(scores)) if scores.size else 0.0
    elif layout == "end2end_detections":
        score_column = squeezed[:, 4]
        result["rough_candidates_over_threshold"] = int(np.count_nonzero(score_column >= confidence_threshold))
        result["top_score"] = float(np.max(score_column)) if score_column.size else 0.0
    else:
        result["rough_candidates_over_threshold"] = None
        result["top_score"] = None
    return result


def benchmark_model(
    model_path: Path,
    images: list[Path],
    warmup: int,
    runs: int,
    confidence_threshold: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    interpreter, backend = create_interpreter(model_path)
    interpreter.allocate_tensors()
    init_ms = (time.perf_counter() - start) * 1000.0

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    image_inputs = images if images else [None]
    prepared = [prepare_input(input_detail, path) for path in image_inputs]

    for i in range(warmup):
        input_data = prepared[i % len(prepared)]
        interpreter.set_tensor(input_detail["index"], input_data)
        interpreter.invoke()

    timings: list[float] = []
    output_summaries: list[dict[str, Any]] = []
    for i in range(runs):
        input_data = prepared[i % len(prepared)]
        run_start = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], input_data)
        interpreter.invoke()
        timings.append((time.perf_counter() - run_start) * 1000.0)
        if len(output_summaries) < min(3, len(image_inputs)):
            output = interpreter.get_tensor(output_detail["index"])
            output_summaries.append(summarize_output(output, confidence_threshold))

    return {
        "model": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "backend": backend,
        "input": {
            "name": str(input_detail.get("name", "")),
            "shape": tensor_shape(input_detail),
            "dtype": dtype_name(input_detail.get("dtype")),
        },
        "output": {
            "name": str(output_detail.get("name", "")),
            "shape": tensor_shape(output_detail),
            "dtype": dtype_name(output_detail.get("dtype")),
            "layout": output_layout(tensor_shape(output_detail)),
        },
        "init_ms": round(init_ms, 3),
        "runs": runs,
        "warmup": warmup,
        "timing_ms": {
            "mean": round(statistics.fmean(timings), 3) if timings else 0.0,
            "p50": round(percentile(timings, 0.50), 3),
            "p95": round(percentile(timings, 0.95), 3),
            "min": round(min(timings), 3) if timings else 0.0,
            "max": round(max(timings), 3) if timings else 0.0,
        },
        "output_summaries": output_summaries,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Detector Benchmark Result",
        "",
        f"- Dataset root: `{payload['dataset_root']}`",
        f"- Image count: `{payload['image_count']}`",
        f"- Warmup / runs: `{payload['warmup']}` / `{payload['runs']}`",
        "",
        "| Model | Backend | Input | Output | Size MB | P50 ms | P95 ms | Status |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in payload["results"]:
        if result.get("status") == "failed":
            lines.append(
                f"| `{Path(result['model']).name}` | - | - | - | - | - | - | failed: {result['error']} |"
            )
            continue
        size_mb = result["size_bytes"] / (1024 * 1024)
        lines.append(
            "| `{name}` | {backend} | `{input_shape}` {input_dtype} | `{output_shape}` {layout} | "
            "{size:.2f} | {p50:.3f} | {p95:.3f} | ok |".format(
                name=Path(result["model"]).name,
                backend=result["backend"],
                input_shape=result["input"]["shape"],
                input_dtype=result["input"]["dtype"],
                output_shape=result["output"]["shape"],
                layout=result["output"]["layout"],
                size=size_mb,
                p50=result["timing_ms"]["p50"],
                p95=result["timing_ms"]["p95"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- COCO8 is a smoke dataset for pipeline validation only.",
            "- Rough candidate counts are pre-NMS or layout-level summaries, not safety-quality conclusions.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local TFLite detector candidates.")
    parser.add_argument("models", nargs="*", help="TFLite model paths. Defaults to app asset plus detector-lab exports.")
    parser.add_argument("--lab-root", default=DEFAULT_LAB_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--image-limit", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--output-dir", help="Optional output directory for benchmark.json and benchmark.md.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    configure_local_caches(project_root)
    lab_root = resolve_path(project_root, args.lab_root)
    dataset_root = resolve_path(project_root, args.dataset_root)
    models = [resolve_path(project_root, value) for value in args.models] if args.models else default_models(project_root, lab_root)
    images = image_paths(dataset_root, args.image_limit)

    results = []
    for model_path in models:
        try:
            print(f"benchmark_model={model_path}")
            results.append(
                benchmark_model(
                    model_path=model_path,
                    images=images,
                    warmup=args.warmup,
                    runs=args.runs,
                    confidence_threshold=args.confidence_threshold,
                )
            )
        except Exception as error:
            results.append(
                {
                    "model": str(model_path),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    payload = {
        "dataset_root": str(dataset_root),
        "image_count": len(images),
        "warmup": args.warmup,
        "runs": args.runs,
        "results": results,
    }

    if args.output_dir:
        output_dir = resolve_path(project_root, args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = project_root / f"test-artifacts.local-detector-benchmark-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "benchmark.json", payload)
    (output_dir / "benchmark.md").write_text(markdown_report(payload), encoding="utf-8")
    print(f"benchmark_json={output_dir / 'benchmark.json'}")
    print(f"benchmark_md={output_dir / 'benchmark.md'}")


if __name__ == "__main__":
    main()
