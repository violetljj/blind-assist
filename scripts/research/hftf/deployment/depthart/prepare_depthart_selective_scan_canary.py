#!/usr/bin/env python3
"""Create a deterministic one-op SelectiveScan DLC input/oracle fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


SHAPES = {
    "u": (1, 48, 196),
    "delta": (1, 48, 196),
    "A": (48, 8),
    "B": (1, 4, 8, 196),
    "C": (1, 4, 8, 196),
    "D": (48,),
    "delta_bias": (48,),
}


def reference(values: dict[str, np.ndarray]) -> np.ndarray:
    u = values["u"]
    active_delta = np.logaddexp(
        np.float32(0.0), values["delta"] + values["delta_bias"][None, :, None]
    ).astype(np.float32)
    matrix_b = np.repeat(values["B"], u.shape[1] // values["B"].shape[1], axis=1)
    matrix_c = np.repeat(values["C"], u.shape[1] // values["C"].shape[1], axis=1)
    state = np.zeros((u.shape[0], u.shape[1], values["A"].shape[1]), dtype=np.float32)
    result = np.empty_like(u)
    for step in range(u.shape[2]):
        dt = active_delta[:, :, step]
        state = (
            np.exp(dt[:, :, None] * values["A"][None]).astype(np.float32) * state
            + dt[:, :, None] * matrix_b[:, :, :, step] * u[:, :, step, None]
        ).astype(np.float32)
        result[:, :, step] = (
            np.sum(state * matrix_c[:, :, :, step], axis=-1, dtype=np.float32)
            + values["D"][None] * u[:, :, step]
        )
    return result


def make_model() -> onnx.ModelProto:
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))
        for name, shape in SHAPES.items()
    ]
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(SHAPES["u"]))
    node = helper.make_node(
        "SelectiveScan",
        list(SHAPES),
        ["y"],
        name="DepthArtSelectiveScanG4Canary",
        domain="com.depthart",
        delta_softplus=1,
        out_float=0,
    )
    return helper.make_model(
        helper.make_graph([node], "depthart_selective_scan_g4_canary", inputs, [output]),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.depthart", 1)],
        ir_version=10,
    )


def cases(seed: int) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)

    def nominal() -> dict[str, np.ndarray]:
        return {
            "u": (rng.standard_normal(SHAPES["u"]) * 0.1).astype(np.float32),
            "delta": (rng.standard_normal(SHAPES["delta"]) * 0.1).astype(np.float32),
            "A": -rng.uniform(0.05, 1.0, SHAPES["A"]).astype(np.float32),
            "B": (rng.standard_normal(SHAPES["B"]) * 0.1).astype(np.float32),
            "C": (rng.standard_normal(SHAPES["C"]) * 0.1).astype(np.float32),
            "D": (rng.standard_normal(SHAPES["D"]) * 0.1).astype(np.float32),
            "delta_bias": (rng.standard_normal(SHAPES["delta_bias"]) * 0.2).astype(np.float32),
        }

    result = {"nominal": nominal()}
    accumulation = nominal()
    accumulation["A"].fill(np.float32(-0.002))
    accumulation["delta"].fill(np.float32(0.25))
    accumulation["delta_bias"].fill(np.float32(0.0))
    result["accumulation"] = accumulation
    softplus_extremes = nominal()
    ramp = np.linspace(-25.0, 25.0, SHAPES["delta"][-1], dtype=np.float32)
    softplus_extremes["delta"][:] = ramp
    softplus_extremes["A"] = -rng.uniform(0.1, 0.8, SHAPES["A"]).astype(np.float32)
    softplus_extremes["delta_bias"].fill(np.float32(0.0))
    result["softplus_extremes"] = softplus_extremes
    return result


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

    model_path = root / "selective-scan-canary.onnx"
    model = make_model()
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    records = []
    input_lines = []
    for case_name, values in cases(args.seed).items():
        input_dir = root / "inputs" / case_name
        oracle_dir = root / "oracle" / case_name
        input_dir.mkdir(parents=True)
        oracle_dir.mkdir(parents=True)
        entries = []
        for name, value in values.items():
            path = input_dir / f"{name}.raw"
            value.tofile(path)
            relative = path.relative_to(root).as_posix()
            entries.append(f"{name}:={relative}")
        expected = reference(values)
        expected_path = oracle_dir / "y.raw"
        expected.tofile(expected_path)
        input_lines.append(" ".join(entries))
        records.append(
            {
                "name": case_name,
                "expected_path": expected_path.relative_to(root).as_posix(),
                "expected_sha256": sha256(expected_path),
                "expected_min": float(expected.min()),
                "expected_max": float(expected.max()),
                "expected_finite": bool(np.isfinite(expected).all()),
            }
        )
    (root / "input-list.txt").write_text("\n".join(input_lines) + "\n", encoding="utf-8")
    receipt = {
        "schema": "blindassist_depthart_selective_scan_g4_canary",
        "schema_version": 1,
        "status": "HOST_ORACLE_READY_RUNTIME_NOT_EVALUATED",
        "seed": args.seed,
        "contract": {name: list(shape) for name, shape in SHAPES.items()},
        "output_shape": list(SHAPES["u"]),
        "onnx_sha256": sha256(model_path),
        "cases": records,
        "comparison": {
            "metrics": ["max_abs", "mean_abs", "max_rel", "p50_abs", "p95_abs", "p99_abs", "per_step_max_abs"],
            "initial_development_tolerance": {"rtol": 3e-5, "atol": 3e-6},
        },
        "authority": "Synthetic deterministic operator oracle only; no QNN load, HTP execution, device, full-graph, latency, thermal, Android, safety, or production claim.",
    }
    (root / "canary-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
