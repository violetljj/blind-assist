#!/usr/bin/env python3
"""Prepare deterministic float input and ORT reference for a DA V2 Metric DLC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scripts.research.hftf.deployment.depthart.export_depth_anything_v2_metric_onnx import (
    make_validation_input,
    validate_input_shape,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--repetitions", type=int, default=6)
    args = parser.parse_args()

    validate_input_shape(args.input_height, args.input_width)
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    if not args.onnx.is_file():
        parser.error(f"ONNX does not exist: {args.onnx}")

    import onnxruntime as ort

    tensor = make_validation_input(args.input_height, args.input_width)
    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    output = np.asarray(session.run(["depth_m"], {"image": tensor})[0])
    expected_shape = (1, args.input_height, args.input_width)
    if output.shape != expected_shape:
        raise ValueError(f"expected output {expected_shape}, got {output.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "gradient-image-normalized-f32.raw"
    reference_path = args.output_dir / "reference-depth-m-f32.raw"
    input_path.write_bytes(tensor.astype(np.float32).tobytes(order="C"))
    reference_path.write_bytes(output.astype(np.float32).tobytes(order="C"))
    remote_input = f"{args.remote_root.rstrip('/')}/{input_path.name}"
    input_list = args.output_dir / "input-list-device.txt"
    input_list.write_text(
        "\n".join(f"image:={remote_input}" for _ in range(args.repetitions)) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "hftf_depth_anything_v2_metric_qairt_input_r0",
        "role": "deployment parity and performance canary only",
        "input_shape_nchw": list(tensor.shape),
        "input_dtype": "float32",
        "output_shape": list(output.shape),
        "reference_output_dtype": "float32",
        "repetitions": args.repetitions,
        "onnx_sha256": sha256(args.onnx),
        "input_sha256": sha256(input_path),
        "reference_output_sha256": sha256(reference_path),
        "input_size_bytes": input_path.stat().st_size,
        "reference_output_size_bytes": reference_path.stat().st_size,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
