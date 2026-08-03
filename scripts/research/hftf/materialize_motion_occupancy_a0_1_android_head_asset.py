#!/usr/bin/env python3
"""Materialize deterministic consumed feature rows for Android head parity."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_collision_risk_field_a1 import frozen_probability
from evaluate_motion_conditioned_occupancy_a0 import (
    FEATURE_NAMES,
    build_rows,
    extract_motion,
    sha256,
)


SCHEMA = "blindassist_hftf_motion_occupancy_a0_1_android_head_asset_r0"


def serialize_rows(
    features: np.ndarray, groups: np.ndarray, model: dict[str, Any]
) -> bytes:
    if features.shape != (len(groups), len(FEATURE_NAMES)):
        raise ValueError("feature/group shape mismatch")
    header = ["sequence_id", *FEATURE_NAMES, "expected_probability"]
    lines = ["\t".join(header)]
    for group, row in zip(groups.tolist(), features, strict=True):
        probability = frozen_probability(row, model)
        values = [str(group), *(format(float(value), ".17g") for value in row)]
        values.append(format(probability, ".17g"))
        lines.append("\t".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")


def materialize(
    report_path: Path,
    model_path: Path,
    raft_weights: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    motion = extract_motion(report["frames"], raft_weights)
    features, _, groups = build_rows([report], motion)
    raw = serialize_rows(features, groups, model)
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(compressed)
    return {
        "schema": SCHEMA,
        "rows": len(groups),
        "windows": len(set(groups.tolist())),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "source_report_sha256": sha256(report_path).upper(),
        "model_sha256": sha256(model_path).upper(),
        "raft_sha256": sha256(raft_weights).upper(),
        "output": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    receipt = materialize(args.report, args.model, args.raft_weights, args.output)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
