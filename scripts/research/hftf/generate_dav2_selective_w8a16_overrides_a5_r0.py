#!/usr/bin/env python3
"""Generate the single frozen DA V2 A5 W8A16 override and receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

STATIC_LINEAR = re.compile(
    r"^/blocks\.(?:[0-9]|1[01])/(?:attn/qkv|attn/proj|mlp/fc1|mlp/fc2)/MatMul$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def select_static_linear_weights(model: onnx.ModelProto) -> list[tuple[str, np.ndarray, str]]:
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    selected = []
    for node in model.graph.node:
        if STATIC_LINEAR.fullmatch(node.name):
            if len(node.input) != 2 or node.input[1] not in initializers:
                raise ValueError(f"A5 expected static MatMul weight: {node.name}")
            weight = np.asarray(initializers[node.input[1]], dtype=np.float32)
            if weight.ndim != 2:
                raise ValueError(f"A5 expected rank-2 weight: {node.input[1]}")
            selected.append((node.input[1], weight, node.name))
    selected.sort(key=lambda item: item[2])
    if len(selected) != 48 or len({item[0] for item in selected}) != 48:
        raise ValueError(f"A5 expected exactly 48 unique static linear weights, got {len(selected)}")
    return selected


def symmetric_int8_encoding(name: str, weight: np.ndarray) -> dict[str, Any]:
    maximum = np.max(np.abs(weight), axis=0)
    scale = np.maximum(maximum / 127.0, np.finfo(np.float32).tiny).astype(np.float64)
    return {
        "name": name,
        "output_dtype": "int8",
        "y_scale": scale.tolist(),
        "y_zero_point": [0] * int(scale.size),
        "axis": 1,
    }


def build_override(model: onnx.ModelProto) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = select_static_linear_weights(model)
    encodings = []
    records = []
    for tensor_name, weight, node_name in selected:
        encoding = symmetric_int8_encoding(tensor_name, weight)
        encodings.append(encoding)
        records.append(
            {
                "node_name": node_name,
                "tensor_name": tensor_name,
                "shape": list(weight.shape),
                "axis": 1,
                "scale_count": len(encoding["y_scale"]),
                "weight_sha256": hashlib.sha256(weight.tobytes(order="C")).hexdigest().upper(),
            }
        )
    return {"version": "2.0.0", "encodings": encodings}, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--expected-onnx-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    if sha256_file(args.onnx) != args.expected_onnx_sha256.upper():
        raise ValueError("A5 ONNX hash mismatch")
    model = onnx.load(str(args.onnx), load_external_data=True)
    override, records = build_override(model)
    args.output_root.mkdir(parents=True)
    override_path = args.output_root / "dav2_a5_selective_w8a16_overrides.json"
    override_path.write_text(json.dumps(override, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "blindassist_dav2_selective_w8a16_a5_r0_override_receipt",
        "onnx_sha256": sha256_file(args.onnx),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "override_sha256": sha256_file(override_path),
        "quantized_tensor_count": len(records),
        "quantized_scope": "12 blocks x qkv/proj/fc1/fc2 static weights only",
        "activation_precision": "float16",
        "unencoded_tensor_rule": "QAIRT target processing treats missing encodings as float16",
        "records": records,
    }
    receipt_path = args.output_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
