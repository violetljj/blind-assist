#!/usr/bin/env python3
"""Materialize hash-bound 518px DA V2 teacher targets for A2 distillation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--dav2-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_TEACHER_CACHE_OR_STUDENT_TRAINING":
        raise ValueError("A2 protocol is not frozen")
    if sha256_file(args.source_manifest) != protocol["data"][
        "source_manifest_sha256"
    ]:
        raise ValueError("source manifest hash mismatch")
    if source_manifest.get("roster_lock_sha256") != protocol["data"][
        "source_roster_lock_sha256"
    ]:
        raise ValueError("source roster lock mismatch")
    if sha256_file(args.checkpoint) != protocol["teacher"]["checkpoint_sha256"]:
        raise ValueError("teacher checkpoint hash mismatch")
    dpt_path = args.dav2_repo / "metric_depth/depth_anything_v2/dpt.py"
    if sha256_file(dpt_path) != protocol["teacher"]["source_dpt_sha256"]:
        raise ValueError("teacher source hash mismatch")
    records = source_manifest.get("records")
    if not isinstance(records, list) or len(records) != 3000:
        raise ValueError("expected exactly 3000 source identities")
    roles = Counter(str(row["role"]) for row in records)
    if roles != {"train": 2400, "validation": 600}:
        raise ValueError(f"unexpected source roles: {roles}")

    args.output_root.mkdir(parents=True)
    cache_path = args.output_root / "teacher_depth_f16.npy"
    partial_path = args.output_root / "teacher_depth_f16.partial.npy"
    cache = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float16, shape=(3000, 192, 256)
    )
    teacher = DepthAnythingV2MetricSource(
        args.dav2_repo.resolve(),
        args.checkpoint.resolve(),
        "cuda",
        518,
        "fp16",
    )
    receipts = []
    latencies = []
    for index, row in enumerate(records):
        rgb_path = (
            args.rgb_root
            / str(row["video_id"])
            / "lowres_wide"
            / f"{row['frame_stem']}.png"
        )
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (192, 256):
            raise OSError(f"cannot decode frozen distillation RGB: {rgb_path}")
        started = time.perf_counter()
        depth, _metadata = teacher.infer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), row)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if depth.shape != (192, 256) or not np.all(np.isfinite(depth)):
            raise ValueError(f"invalid teacher depth: {row['frame_stem']}")
        cache[index] = depth.astype(np.float16)
        latencies.append(latency_ms)
        receipts.append(
            {
                "index": index,
                "frame_id": str(row["frame_stem"]),
                "video_id": str(row["video_id"]),
                "parent_id": str(row["parent_id"]),
                "role": str(row["role"]),
                "rgb_path": str(rgb_path.resolve()),
                "rgb_sha256": sha256_file(rgb_path),
            }
        )
    del teacher
    finalize_memmap(cache, partial_path, cache_path)
    sorted_latency = sorted(latencies)
    manifest = {
        "schema": "blindassist_dav2_distillation_teacher_r0",
        "protocol_sha256": sha256_file(args.protocol),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "truth_inputs_opened": False,
        "teacher_depth": {
            "path": str(cache_path.resolve()),
            "shape": [3000, 192, 256],
            "dtype": "float16",
            "sha256": sha256_file(cache_path),
        },
        "role_counts": dict(sorted(roles.items())),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted_latency[round(0.95 * (len(sorted_latency) - 1))],
        },
        "records": receipts,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
