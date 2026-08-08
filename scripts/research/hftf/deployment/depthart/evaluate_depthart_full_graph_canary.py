#!/usr/bin/env python3
"""Evaluate the frozen DepthART full-graph numerical canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "blindassist_depthart_g4d_full_graph_parity_receipt_v1"
RTOL = 3e-5
ATOL = 3e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> np.ndarray:
    value = np.fromfile(path, dtype=np.float32)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"invalid float32 output: {path}")
    return value


def compare(left: Path, right: Path) -> dict[str, object]:
    a = load(left)
    b = load(right)
    if a.shape != b.shape:
        raise ValueError(f"size mismatch: {left} vs {right}")
    difference = np.abs(a - b)
    relative = difference / np.maximum(np.abs(a), 1e-6)
    return {
        "left": left.name,
        "right": right.name,
        "elements": int(a.size),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(a - b)))),
        "max_rel": float(relative.max()),
        "mean_rel": float(relative.mean()),
        "bit_exact": bool(np.array_equal(a, b)),
        "allclose": bool(np.allclose(a, b, rtol=RTOL, atol=ATOL)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--custom-dlc", type=Path, required=True)
    parser.add_argument("--primitive-dlc", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.evidence_dir.resolve()
    host = root / "host"
    frontier = root / "qnn-first-frontier"
    comparisons = {
        "pytorch_vs_onnx_primitive": compare(
            host / "pytorch-depth.raw", host / "onnx-primitive-depth.raw"
        ),
        "pytorch_vs_qnn_htp_context": compare(
            host / "pytorch-depth.raw", host / "qnn-context-depth.raw"
        ),
        "onnx_primitive_vs_qnn_htp_context": compare(
            host / "onnx-primitive-depth.raw", host / "qnn-context-depth.raw"
        ),
        "qnn_htp_direct_vs_context": compare(
            host / "qnn-custom-depth.raw", host / "qnn-context-depth.raw"
        ),
    }
    first_frontier = {
        "pre_first_selective_scan": compare(
            root / "ort-0.raw",
            frontier / "_network_0_network_0_2_op_Reshape_2_output_0.raw",
        ),
        "first_selective_scan_output": compare(
            root / "ort-1.raw",
            frontier / "_network_0_network_0_2_op_SelectiveScan_output_0.raw",
        ),
        "first_post_scan_layernorm": compare(
            root / "ort-2.raw",
            frontier
            / "_network_0_network_0_2_op_out_norm_LayerNormalization_output_0.raw",
        ),
    }
    strict_pass = bool(
        comparisons["pytorch_vs_onnx_primitive"]["allclose"]
        and comparisons["pytorch_vs_qnn_htp_context"]["allclose"]
    )
    receipt = {
        "schema": SCHEMA,
        "status": (
            "PASS_SM8650_V75_SYNTHETIC_FULL_GRAPH_NUMERICAL_CANARY"
            if strict_pass
            else "FAIL_SM8650_V75_FULL_GRAPH_NUMERICAL_PARITY"
        ),
        "device": {
            "model": "Samsung SM-S9280",
            "soc": "Qualcomm SM8650",
            "marketing_soc": "Snapdragon 8 Gen 3",
            "htp": "v75",
            "serial": "R5CX10M8Y8X",
        },
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "canary_authority": "SYNTHETIC_FULL_GRAPH_NUMERICAL_CANARY_ONLY",
        "comparisons": comparisons,
        "first_divergence_localization": first_frontier,
        "primitive_htp_reference": {
            "status": "FAIL_GRAPH_FINALIZE_1002",
            "qnn_ir_ops": 21440,
        },
        "assets": {
            "custom_dlc": {
                "bytes": args.custom_dlc.stat().st_size,
                "sha256": sha256(args.custom_dlc),
            },
            "primitive_dlc": {
                "bytes": args.primitive_dlc.stat().st_size,
                "sha256": sha256(args.primitive_dlc),
            },
            "context": {
                "bytes": args.context.stat().st_size,
                "sha256": sha256(args.context),
            },
        },
        "interpretation": {
            "full_model_htp_parity": "FAIL",
            "direct_dlc_vs_saved_context": "BIT_EXACT",
            "divergence_exists_before_first_custom_op": True,
            "custom_op_isolated_parity_prior_gate": "G4_B_PASS_SM8650_V75",
            "next_frontier": "FULL_GRAPH_HTP_FLOAT_PRECISION_AND_STANDARD_OP_PATH",
        },
        "explicit_exclusions": [
            "REAL_SCENE_TASK_QUALITY",
            "CLEARANCE_SAFETY",
            "TEMPORAL_QUALITY",
            "PERFORMANCE",
            "PRODUCTIZATION",
            "DA2_REPLACEMENT",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
