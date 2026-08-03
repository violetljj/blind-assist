#!/usr/bin/env python3
"""Compare one QAIRT DA V2 Metric output with its ORT reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    difference = np.abs(
        reference.astype(np.float64) - candidate.astype(np.float64)
    )
    denominator = np.maximum(np.abs(reference.astype(np.float64)), 1e-6)
    return {
        "mean_abs_difference_m": float(np.mean(difference)),
        "max_abs_difference_m": float(np.max(difference)),
        "mean_relative_abs_difference": float(np.mean(difference / denominator)),
        "candidate_min_m": float(np.min(candidate)),
        "candidate_max_m": float(np.max(candidate)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--candidate-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shape = (1, args.input_height, args.input_width)
    reference = np.fromfile(args.reference, dtype=np.float32).reshape(shape)
    candidate = np.fromfile(args.candidate, dtype=args.candidate_dtype).reshape(shape)
    result = {
        "schema": "hftf_depth_anything_v2_metric_qairt_parity_r0",
        "shape": list(shape),
        "candidate_dtype": args.candidate_dtype,
        **compare(reference, candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
