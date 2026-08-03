#!/usr/bin/env python3
"""Export a fixed-shape Depth Anything V2 Small Metric graph to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PATCH_SIZE = 14
MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
    "max_depth": 20.0,
}


def validate_input_shape(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("input dimensions must be positive")
    if height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(f"input dimensions must be divisible by {PATCH_SIZE}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def make_validation_input(height: int, width: int) -> np.ndarray:
    """Return deterministic ImageNet-normalized NCHW input."""
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    rgb = np.stack((xx, yy, 0.5 * (xx + yy)), axis=0)[None]
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((rgb - mean) / std)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--output-onnx", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    validate_input_shape(args.input_height, args.input_width)
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    metric_root = args.repo / "metric_depth"
    if not metric_root.is_dir():
        parser.error(f"metric_depth source does not exist: {metric_root}")
    sys.path.insert(0, str(metric_root))

    import onnx
    import onnxruntime as ort
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(args.device)
    model = DepthAnythingV2(**MODEL_CONFIG)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    validation = make_validation_input(args.input_height, args.input_width)
    tensor = torch.from_numpy(validation).to(device)
    with torch.inference_mode():
        expected = model(tensor).detach().cpu().numpy()

    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.onnx.export(
        model,
        tensor,
        str(args.output_onnx),
        input_names=["image"],
        output_names=["depth_m"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    export_seconds = time.perf_counter() - started

    checked = onnx.load(str(args.output_onnx), load_external_data=True)
    onnx.checker.check_model(checked)
    session = ort.InferenceSession(
        str(args.output_onnx), providers=["CPUExecutionProvider"]
    )
    started = time.perf_counter()
    actual = session.run(["depth_m"], {"image": validation})[0]
    ort_seconds = time.perf_counter() - started
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))

    receipt = {
        "schema": "hftf_depth_anything_v2_metric_onnx_export_r0",
        "source_role": "metric-depth observer candidate",
        "source_repo": str(args.repo.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "input_name": "image",
        "input_shape_nchw": [1, 3, args.input_height, args.input_width],
        "input_semantics": "RGB float32 normalized by ImageNet mean/std",
        "output_name": "depth_m",
        "output_shape": list(actual.shape),
        "output_semantics": "Hypersim-trained metric depth in metres, max 20 m",
        "onnx_path": str(args.output_onnx.resolve()),
        "onnx_sha256": sha256(args.output_onnx),
        "onnx_size_bytes": args.output_onnx.stat().st_size,
        "opset": 17,
        "validation": {
            "input_kind": "deterministic normalized gradient",
            "max_abs_difference_m": float(np.max(difference)),
            "mean_abs_difference_m": float(np.mean(difference)),
            "pytorch_output_min_m": float(np.min(expected)),
            "pytorch_output_max_m": float(np.max(expected)),
            "onnx_output_min_m": float(np.min(actual)),
            "onnx_output_max_m": float(np.max(actual)),
        },
        "timing": {
            "export_seconds": export_seconds,
            "ort_cpu_validation_seconds": ort_seconds,
        },
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
