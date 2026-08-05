#!/usr/bin/env python3
"""Materialize the frozen A4 student on P1 RGB before opening P1 truth."""

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
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dav2_temporal_mobile_student_r0 import (
    TemporalMobileDepthStudent,
    normalize_bgr_batch,
    parameter_count,
)
from evaluate_dav2_model_variant_gate_r0 import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a4-protocol", type=Path, required=True)
    parser.add_argument("--p1-protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.a4_protocol.read_text(encoding="utf-8"))
    p1_protocol = json.loads(args.p1_protocol.read_text(encoding="utf-8"))
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    if sha256_file(args.p1_protocol) != protocol["parent_p1_r1"]["parent_r0_protocol_sha256"]:
        raise ValueError("A4 parent P1 R0 hash mismatch")
    if sha256_file(args.roster) != p1_protocol["roster_sha256"]:
        raise ValueError("A4 P1 roster hash mismatch")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256.upper():
        raise ValueError("A4 checkpoint hash mismatch")
    model_source = SCRIPT_DIR / "dav2_temporal_mobile_student_r0.py"
    materializer_source = Path(__file__).resolve()
    if sha256_file(model_source) != protocol["implementation"]["model_source_sha256"]:
        raise ValueError("A4 model source hash mismatch")
    if sha256_file(materializer_source) != protocol["implementation"]["materializer_source_sha256"]:
        raise ValueError("A4 materializer source hash mismatch")
    model = TemporalMobileDepthStudent(pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    parameters = parameter_count(model)
    rows = roster["rows"]
    shape = tuple(p1_protocol["cohort"]["aligned_depth_shape"])
    args.output_root.mkdir(parents=True)
    partial = args.output_root / "aligned_depth_f16.partial.npy"
    final = args.output_root / "aligned_depth_f16.npy"
    output = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
    latencies = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            rgb_path = args.source_root / str(row["sequence_root"]) / str(row["rgb_path"])
            if sha256_file(rgb_path) != row["rgb_sha256"]:
                raise ValueError(f"A4 P1 RGB hash mismatch: {row['frame_id']}")
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (480, 640):
                raise OSError(f"cannot decode A4 P1 RGB: {rgb_path}")
            image = torch.from_numpy(bgr.transpose(2, 0, 1).copy())
            normalized = normalize_bgr_batch([image]).cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                depth = model(normalized, (480, 640))[0]
            torch.cuda.synchronize()
            value = depth.float().cpu().numpy()
            if value.shape != (480, 640) or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid A4 P1 depth: {row['frame_id']}")
            output[index] = value.astype(np.float16)
            latencies.append((time.perf_counter() - started) * 1000.0)
    output.flush()
    memory_map = getattr(output, "_mmap", None)
    if memory_map is not None:
        memory_map.close()
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, final)
    ordered = sorted(latencies)
    manifest = {
        "schema": "blindassist_dav2_rgbd_mobile_student_a4_p1_cache",
        "a4_protocol_sha256": sha256_file(args.a4_protocol),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "parameter_count": parameters,
        "p1_truth_opened_during_materialization": False,
        "aligned_depth": {"path": str(final.resolve()), "shape": list(shape), "dtype": "float16", "sha256": sha256_file(final)},
        "host_cuda_latency_ms": {"mean": statistics.fmean(latencies), "median": statistics.median(latencies), "p95": ordered[round(0.95 * (len(ordered) - 1))], "claim_ceiling": "host CUDA diagnostic only; not Android App latency"},
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
