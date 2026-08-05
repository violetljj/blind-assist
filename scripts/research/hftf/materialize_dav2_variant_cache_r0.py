#!/usr/bin/env python3
"""Materialize one frozen DA V2 model variant into canonical aligned depth."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_dav2_model_variant_gate_r0 import sha256_file
from produce_external_rgb_metric_depth_observations import (
    DepthAnythingV2MetricSource,
)


def finalize_memmap(array: np.memmap, partial: Path, final: Path) -> None:
    array.flush()
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        memory_map.close()
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dav2-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.p1_protocol.read_text(encoding="utf-8"))
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    if sha256_file(args.roster) != protocol["roster_sha256"]:
        raise ValueError("P1 roster hash mismatch")
    expected_checkpoint_sha256 = (
        args.expected_checkpoint_sha256
        or protocol["baseline"]["checkpoint_sha256"]
    ).upper()
    if sha256_file(args.checkpoint) != expected_checkpoint_sha256:
        raise ValueError("checkpoint hash mismatch")
    rows = roster["rows"]
    expected_shape = tuple(protocol["cohort"]["aligned_depth_shape"])
    if expected_shape != (len(rows), 480, 640):
        raise ValueError("unexpected P1 aligned-depth shape")

    args.output_root.mkdir(parents=True)
    final_path = args.output_root / "aligned_depth_f16.npy"
    partial_path = args.output_root / "aligned_depth_f16.partial.npy"
    output = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float16, shape=expected_shape
    )
    source = DepthAnythingV2MetricSource(
        args.dav2_repo.resolve(),
        args.checkpoint.resolve(),
        args.device,
        args.input_size,
        args.precision,
    )
    latencies: list[float] = []
    frame_receipts = []
    for index, row in enumerate(rows):
        rgb_path = (
            args.source_root
            / str(row["sequence_root"])
            / str(row["rgb_path"])
        )
        if sha256_file(rgb_path) != row["rgb_sha256"]:
            raise ValueError(f"RGB hash mismatch: {row['frame_id']}")
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (480, 640):
            raise OSError(f"cannot decode canonical RGB: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        depth, _metadata = source.infer(rgb, row)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if depth.shape != (480, 640) or not np.all(np.isfinite(depth)):
            raise ValueError(f"invalid aligned depth: {row['frame_id']}")
        output[index] = depth.astype(np.float16)
        latencies.append(latency_ms)
        frame_receipts.append(
            {
                "index": index,
                "frame_id": row["frame_id"],
                "latency_ms": latency_ms,
            }
        )
    del source
    finalize_memmap(output, partial_path, final_path)
    sorted_latency = sorted(latencies)
    manifest = {
        "schema": "blindassist_dav2_model_variant_cache_r0",
        "candidate_id": args.candidate_id,
        "p1_protocol_sha256": sha256_file(args.p1_protocol),
        "roster_sha256": sha256_file(args.roster),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "runtime": {
            "device": args.device,
            "input_size": args.input_size,
            "precision": args.precision,
        },
        "aligned_depth": {
            "path": str(final_path.resolve()),
            "shape": list(expected_shape),
            "dtype": "float16",
            "sha256": sha256_file(final_path),
        },
        "host_materialization_latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted_latency[round(0.95 * (len(sorted_latency) - 1))],
            "claim_ceiling": "host cache materialization diagnostic only; not Android App latency",
        },
        "frames": frame_receipts,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
