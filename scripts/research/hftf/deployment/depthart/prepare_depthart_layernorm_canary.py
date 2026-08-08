#!/usr/bin/env python3
"""Create deterministic DepthArtLayerNorm inputs and float32 host oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


SHAPE = (1, 8, 128)


def reference(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = (x - mean).astype(np.float32)
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    return (centered / np.sqrt(variance + np.float32(epsilon)) * weight + bias).astype(np.float32)


def make_model() -> onnx.ModelProto:
    node = helper.make_node(
        "DepthArtLayerNorm", ["x", "weight", "bias"], ["y"],
        name="DepthArtLayerNormG4CCanary", domain="com.depthart", epsilon=1e-5,
    )
    model = helper.make_model(
        helper.make_graph(
            [node], "depthart_layernorm_g4c_canary",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(SHAPE)),
             helper.make_tensor_value_info("weight", TensorProto.FLOAT, [SHAPE[-1]]),
             helper.make_tensor_value_info("bias", TensorProto.FLOAT, [SHAPE[-1]])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(SHAPE))],
        ),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.depthart", 1)],
        ir_version=10,
    )
    return model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"output root already exists: {root}")
    root.mkdir(parents=True)
    model_path = root / "layernorm-canary.onnx"
    model = make_model()
    onnx.checker.check_model(model, full_check=False, check_custom_domain=False)
    onnx.save(model, model_path)

    rng = np.random.default_rng(args.seed)
    base_weight = rng.uniform(0.5, 1.5, SHAPE[-1]).astype(np.float32)
    base_bias = rng.uniform(-0.2, 0.2, SHAPE[-1]).astype(np.float32)
    cases = {
        "nominal": (rng.standard_normal(SHAPE).astype(np.float32), base_weight, base_bias),
        "centered_low_variance": ((rng.standard_normal(SHAPE).astype(np.float32) * np.float32(1e-3)), base_weight, base_bias),
        "varied_scale": ((rng.standard_normal(SHAPE).astype(np.float32) * np.float32(12.0)), base_weight * np.float32(1.7), base_bias * np.float32(2.0)),
    }
    input_lines: list[str] = []
    records: list[dict[str, object]] = []
    for name, (x, weight, bias) in cases.items():
        input_dir = root / "inputs" / name
        oracle_dir = root / "oracle" / name
        input_dir.mkdir(parents=True)
        oracle_dir.mkdir(parents=True)
        entries = []
        for tensor_name, value in (("x", x), ("weight", weight), ("bias", bias)):
            path = input_dir / f"{tensor_name}.raw"
            value.tofile(path)
            entries.append(f"{tensor_name}:={path.relative_to(root).as_posix()}")
        expected = reference(x, weight, bias)
        expected_path = oracle_dir / "y.raw"
        expected.tofile(expected_path)
        input_lines.append(" ".join(entries))
        records.append({"name": name, "expected_path": expected_path.relative_to(root).as_posix(), "expected_sha256": sha256(expected_path)})
    (root / "input-list.txt").write_text("\n".join(input_lines) + "\n", encoding="utf-8")
    receipt = {
        "schema": "blindassist_depthart_layernorm_g4c_canary", "schema_version": 1,
        "status": "HOST_ORACLE_READY_RUNTIME_NOT_EVALUATED", "seed": args.seed,
        "input_shape": list(SHAPE), "output_shape": list(SHAPE), "epsilon": 1e-5,
        "onnx_sha256": sha256(model_path), "cases": records,
        "comparison": {"rtol": 3e-5, "atol": 3e-6},
        "authority": "Synthetic operator oracle only; no QNN load, HTP execution, full graph, performance, safety, or production claim.",
    }
    (root / "canary-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
