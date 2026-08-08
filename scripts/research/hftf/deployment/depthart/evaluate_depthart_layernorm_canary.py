#!/usr/bin/env python3
"""Compare qnn-net-run DepthArtLayerNorm outputs with the frozen host oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-root", required=True, type=Path)
    parser.add_argument("--qnn-output-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--rtol", type=float, default=3e-5)
    parser.add_argument("--atol", type=float, default=3e-6)
    args = parser.parse_args()
    source = json.loads((args.canary_root / "canary-receipt.json").read_text(encoding="utf-8"))
    shape = tuple(source["output_shape"])
    records = []
    all_pass = True
    for index, case in enumerate(source["cases"]):
        expected = np.fromfile(args.canary_root / case["expected_path"], dtype=np.float32).reshape(shape)
        candidates = sorted((args.qnn_output_root / f"Result_{index}").glob("*.raw"))
        if len(candidates) != 1:
            raise ValueError(f"expected one output for case {index}, found {len(candidates)}")
        actual = np.fromfile(candidates[0], dtype=np.float32).reshape(shape)
        absolute = np.abs(actual - expected)
        passed = bool(np.isfinite(actual).all() and np.allclose(actual, expected, rtol=args.rtol, atol=args.atol))
        all_pass &= passed
        records.append({"case": case["name"], "pass": passed, "max_abs": float(absolute.max()), "mean_abs": float(absolute.mean()), "p99_abs": float(np.percentile(absolute, 99))})
    receipt = {
        "schema": "blindassist_depthart_layernorm_g4c_operator_parity", "schema_version": 1,
        "status": "PASS" if all_pass else "FAIL_NUMERICAL_PARITY", "rtol": args.rtol, "atol": args.atol,
        "cases": records,
        "authority": "LayerNorm operator parity only; full graph, partition purity, performance, safety, and production remain unevaluated.",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
