#!/usr/bin/env python3
"""Convert a verified fixed-shape DA V2 Metric ONNX graph to FP32 TFLite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validation_input(height: int, width: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    rgb = np.stack((xx, yy, 0.5 * (xx + yy)), axis=0)[None]
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((rgb - mean) / std)


def interpreter_for(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    return Interpreter(model_path=str(path))


def canonical_depth(value: np.ndarray, height: int, width: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (1, height, width):
        return array[0]
    if array.shape == (1, 1, height, width):
        return array[0, 0]
    if array.shape == (1, height, width, 1):
        return array[0, :, :, 0]
    raise ValueError(f"unsupported TFLite output shape: {array.shape}")


def run_tflite(path: Path, nchw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    interpreter = interpreter_for(path)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"expected one input/output, got {len(inputs)}/{len(outputs)}")
    input_detail = inputs[0]
    shape = tuple(int(item) for item in input_detail["shape"])
    if shape == nchw.shape:
        value = nchw
        layout = "NCHW"
    elif shape == (nchw.shape[0], nchw.shape[2], nchw.shape[3], nchw.shape[1]):
        value = np.transpose(nchw, (0, 2, 3, 1)).copy()
        layout = "NHWC"
    else:
        raise ValueError(f"unexpected TFLite input shape: {shape}")
    interpreter.set_tensor(input_detail["index"], value.astype(input_detail["dtype"]))
    interpreter.invoke()
    raw = interpreter.get_tensor(outputs[0]["index"])
    return raw, {
        "input_name": input_detail["name"],
        "input_shape": list(shape),
        "input_dtype": str(input_detail["dtype"]),
        "input_layout": layout,
        "output_name": outputs[0]["name"],
        "output_shape": [int(item) for item in outputs[0]["shape"]],
        "output_dtype": str(outputs[0]["dtype"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--expected-onnx-sha256", required=True)
    parser.add_argument("--output-tflite", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    actual_onnx_hash = sha256(args.onnx)
    if actual_onnx_hash != args.expected_onnx_sha256.upper():
        raise ValueError(
            f"ONNX hash mismatch: expected {args.expected_onnx_sha256}, got {actual_onnx_hash}"
        )
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    shape = tuple(int(item) for item in model_input.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise ValueError(f"expected fixed NCHW RGB input, got {shape}")
    _, _, height, width = shape

    work_dir = args.work_dir.resolve()
    conversion_dir = work_dir / "onnx2tf"
    if conversion_dir.exists():
        shutil.rmtree(conversion_dir)
    conversion_dir.mkdir(parents=True)
    converter_onnx = work_dir / "model.converter-input.onnx"
    slim_onnx = work_dir / "model.slim.onnx"
    shutil.copy2(args.onnx, converter_onnx)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "onnxslim",
            str(converter_onnx),
            str(slim_onnx),
            "--model-check",
        ],
        cwd=work_dir,
        check=True,
    )
    # onnx2tf 1.28.8 unconditionally reads this sample for fixed rank-4 image
    # inputs.  Materialize it locally to keep conversion offline and deterministic.
    np.save(
        work_dir / "calibration_image_sample_data_20x128x128x3_float32.npy",
        np.zeros((20, 128, 128, 3), dtype=np.float32),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "onnx2tf",
            "-i",
            str(slim_onnx),
            "-o",
            str(conversion_dir),
            "-k",
            model_input.name,
            "-rtpo",
            "Erf",
            "-dsm",
            "-n",
        ],
        cwd=work_dir,
        check=True,
    )
    candidates = sorted(conversion_dir.glob("*_float32.tflite"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one FP32 TFLite output, got {candidates}")
    args.output_tflite.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], args.output_tflite)

    sample = validation_input(height, width)
    expected = session.run(None, {model_input.name: sample})[0]
    slim_session = ort.InferenceSession(str(slim_onnx), providers=["CPUExecutionProvider"])
    slim_actual = slim_session.run(None, {model_input.name: sample})[0]
    slim_difference = np.abs(expected.astype(np.float64) - slim_actual.astype(np.float64))
    raw_actual, contract = run_tflite(args.output_tflite, sample)
    actual = canonical_depth(raw_actual, height, width)
    reference = canonical_depth(expected, height, width)
    difference = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    receipt = {
        "schema": "hftf_dav2_metric_onnx_to_tflite_r0",
        "onnx_path": str(args.onnx.resolve()),
        "onnx_sha256": actual_onnx_hash,
        "tflite_path": str(args.output_tflite.resolve()),
        "tflite_sha256": sha256(args.output_tflite),
        "tflite_size_bytes": args.output_tflite.stat().st_size,
        "slim_onnx_sha256": sha256(slim_onnx),
        "original_onnx_slim_parity": {
            "max_abs_difference_m": float(np.max(slim_difference)),
            "mean_abs_difference_m": float(np.mean(slim_difference)),
        },
        "contract": contract,
        "host_onnx_tflite_parity": {
            "max_abs_difference_m": float(np.max(difference)),
            "mean_abs_difference_m": float(np.mean(difference)),
            "p95_abs_difference_m": float(np.quantile(difference, 0.95)),
        },
        "versions": {
            "onnxruntime": ort.__version__,
            "onnxslim": importlib.metadata.version("onnxslim"),
            "onnx2tf": importlib.metadata.version("onnx2tf"),
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
