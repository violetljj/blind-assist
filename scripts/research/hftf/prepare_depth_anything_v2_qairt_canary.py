#!/usr/bin/env python3
"""Prepare deterministic float or quantized input for Qualcomm DA V2 DLCs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

INPUT_SHAPE = (1, 518, 518, 3)
INPUT_SCALE = 0.00002101432801282499
INPUT_ZERO_POINT = 9979


def quantize_input(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != INPUT_SHAPE:
        raise ValueError(f"expected {INPUT_SHAPE}, got {values.shape}")
    quantized = np.rint(values / INPUT_SCALE + INPUT_ZERO_POINT)
    return np.clip(quantized, 0, 65535).astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument(
        "--dtype", choices=("uint16", "float32"), default="uint16"
    )
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")

    x = np.linspace(0.0, 1.0, INPUT_SHAPE[2], dtype=np.float32)
    y = np.linspace(0.0, 1.0, INPUT_SHAPE[1], dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    image = np.stack((xx, yy, 0.5 * (xx + yy)), axis=-1)[None]
    tensor = quantize_input(image) if args.dtype == "uint16" else image

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "u16" if args.dtype == "uint16" else "f32"
    raw_path = args.output_dir / f"gradient-image-{suffix}.raw"
    raw_path.write_bytes(tensor.tobytes(order="C"))
    remote_raw = f"{args.remote_root.rstrip('/')}/{raw_path.name}"
    input_list = args.output_dir / "input-list-device.txt"
    input_list.write_text(
        "\n".join(f"image:={remote_raw}" for _ in range(args.repetitions))
        + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "hftf_depth_anything_v2_qairt_input_r0",
        "role": "relative-depth performance canary only",
        "shape": list(INPUT_SHAPE),
        "dtype": args.dtype,
        "scale": INPUT_SCALE if args.dtype == "uint16" else None,
        "zero_point": INPUT_ZERO_POINT if args.dtype == "uint16" else None,
        "repetitions": args.repetitions,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest().upper(),
        "raw_size_bytes": raw_path.stat().st_size,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
