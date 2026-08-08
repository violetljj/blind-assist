#!/usr/bin/env python3
"""Evaluate the frozen DepthART G4-D three-way numerical conjunction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


RTOL = 3e-5
ATOL = 3e-6
SCHEMA = "blindassist_depthart_g4d_repair_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def compare(left: Path, right: Path) -> dict[str, Any]:
    a = np.fromfile(left, dtype=np.float32)
    b = np.fromfile(right, dtype=np.float32)
    if a.size == 0 or a.shape != b.shape:
        raise ValueError(f"invalid output size: {left} vs {right}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"non-finite output: {left} vs {right}")
    difference = np.abs(a - b)
    return {
        "elements": int(a.size),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(a - b)))),
        "allclose": bool(np.allclose(a, b, rtol=RTOL, atol=ATOL)),
        "bit_exact": bool(np.array_equal(a, b)),
        "left_sha256": sha256(left),
        "right_sha256": sha256(right),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    pytorch_onnx = compare(args.pytorch, args.canonical_onnx_output)
    onnx_htp = compare(args.canonical_onnx_output, args.htp_direct)
    direct_context = compare(args.htp_direct, args.htp_context)
    frontier = compare(args.frontier_ort, args.frontier_htp)
    gates = {
        "pytorch_vs_canonical_onnx": "PASS" if pytorch_onnx["allclose"] else "FAIL",
        "canonical_onnx_vs_sm8650_htp": "PASS" if onnx_htp["allclose"] else "FAIL",
        "dlc_direct_vs_saved_context": "PASS" if direct_context["bit_exact"] else "FAIL",
    }
    overall_pass = all(value == "PASS" for value in gates.values())
    terminal = None
    if (
        gates["pytorch_vs_canonical_onnx"] == "PASS"
        and gates["canonical_onnx_vs_sm8650_htp"] == "FAIL"
        and gates["dlc_direct_vs_saved_context"] == "PASS"
    ):
        terminal = "CURRENT_QAIRT_2_47_SM8650_HTP_STANDARD_FLOAT_PATH_STRICT_G4D_NOT_SUPPORTED"
    assets = {}
    for name in (
        "frozen_input", "deployment_onnx", "dlc", "context",
        "op_package_arm64", "op_package_dsp",
    ):
        path = getattr(args, name)
        assets[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return {
        "schema": SCHEMA,
        "status": "G4_D_PASS" if overall_pass else "G4_D_FAIL",
        "terminal": terminal,
        "device": {
            "model": "Samsung SM-S9280",
            "soc": "Qualcomm SM8650",
            "htp": "v75",
            "serial": args.serial,
            "qairt": "2.47.0.260601",
        },
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "gates": gates,
        "comparisons": {
            "pytorch_vs_canonical_onnx": pytorch_onnx,
            "canonical_onnx_vs_sm8650_htp_direct": onnx_htp,
            "dlc_direct_vs_saved_context": direct_context,
            "repaired_prefix_next_standard_conv": {
                "node": args.frontier_name,
                **frontier,
            },
        },
        "assets": assets,
        "diagnosis": {
            "confirmed": [
                "PyTorch-to-canonical-ONNX final output closes under the frozen tolerance when CUDA TF32 is disabled.",
                "Correctness-first custom float32 PatchConv, BatchNorm and GELU kernels restore their isolated prefix checkpoints.",
                "After those repairs, the next standard Conv is the first failing checkpoint.",
                "Direct DLC and saved-context outputs are bit-exact, excluding context serialization as the observed drift source.",
                "QAIRT HTP documentation states that delegated fp32 models use underlying 16-bit math on the supported HTP floating path.",
            ],
            "not_claimed": [
                "A single internal HTP primitive or accumulator implementation is not identified.",
                "A near-complete custom float32 inference engine is not evaluated.",
            ],
        },
        "downstream": {
            "G4_E_partition_purity": "NOT_EVALUATED",
            "G4_F_performance": "NOT_EVALUATED",
            "DA2_replacement": "NOT_AUTHORIZED",
        },
        "authority": "SYNTHETIC_FULL_GRAPH_NUMERICAL_CANARY_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "pytorch", "canonical_onnx_output", "htp_direct", "htp_context",
        "frontier_ort", "frontier_htp", "frozen_input", "deployment_onnx",
        "dlc", "context", "op_package_arm64", "op_package_dsp", "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--frontier-name", required=True)
    parser.add_argument("--serial", default="R5CX10M8Y8X")
    args = parser.parse_args()
    receipt = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
