#!/usr/bin/env python3
"""Materialize the frozen A3 student on P1 RGB before opening sensor truth."""

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
    parser.add_argument("--a3-protocol", type=Path, required=True)
    parser.add_argument("--p1-protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    a3_protocol = json.loads(args.a3_protocol.read_text(encoding="utf-8"))
    p1_protocol = json.loads(args.p1_protocol.read_text(encoding="utf-8"))
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    if sha256_file(args.p1_protocol) != a3_protocol[
        "parent_p1_protocol_sha256"
    ]:
        raise ValueError("A3 parent P1 protocol hash mismatch")
    if sha256_file(args.roster) != p1_protocol["roster_sha256"]:
        raise ValueError("A3 P1 roster hash mismatch")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256.upper():
        raise ValueError("A3 checkpoint hash mismatch")
    model_source = SCRIPT_DIR / "dav2_temporal_mobile_student_r0.py"
    if sha256_file(model_source) != a3_protocol["implementation"][
        "model_source_sha256"
    ]:
        raise ValueError("A3 model source hash mismatch")
    materializer_source = Path(__file__).resolve()
    if sha256_file(materializer_source) != a3_protocol["implementation"][
        "materializer_source_sha256"
    ]:
        raise ValueError("A3 materializer source hash mismatch")
    model = TemporalMobileDepthStudent(pretrained=False).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    parameters = parameter_count(model)
    if parameters > int(a3_protocol["model"]["maximum_parameters"]):
        raise ValueError("A3 parameter cap exceeded")
    rows = roster["rows"]
    expected_shape = tuple(p1_protocol["cohort"]["aligned_depth_shape"])
    if expected_shape != (len(rows), 480, 640):
        raise ValueError("A3 P1 cache shape mismatch")
    args.output_root.mkdir(parents=True)
    final_path = args.output_root / "aligned_depth_f16.npy"
    partial_path = args.output_root / "aligned_depth_f16.partial.npy"
    output = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float16, shape=expected_shape
    )
    latencies = []
    receipts = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            rgb_path = (
                args.source_root
                / str(row["sequence_root"])
                / str(row["rgb_path"])
            )
            if sha256_file(rgb_path) != row["rgb_sha256"]:
                raise ValueError(f"A3 P1 RGB hash mismatch: {row['frame_id']}")
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (480, 640):
                raise OSError(f"cannot decode A3 P1 RGB: {rgb_path}")
            image = torch.from_numpy(bgr.transpose(2, 0, 1).copy())
            normalized = normalize_bgr_batch([image]).cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                depth = model(normalized, (480, 640))[0]
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            depth_numpy = depth.float().cpu().numpy()
            if depth_numpy.shape != (480, 640) or not np.all(
                np.isfinite(depth_numpy)
            ):
                raise ValueError(f"invalid A3 P1 depth: {row['frame_id']}")
            output[index] = depth_numpy.astype(np.float16)
            latencies.append(latency_ms)
            receipts.append(
                {
                    "index": index,
                    "frame_id": row["frame_id"],
                    "latency_ms": latency_ms,
                }
            )
    finalize_memmap(output, partial_path, final_path)
    sorted_latency = sorted(latencies)
    manifest = {
        "schema": "blindassist_dav2_temporal_mobile_student_a3_p1_cache",
        "a3_protocol_sha256": sha256_file(args.a3_protocol),
        "p1_protocol_sha256": sha256_file(args.p1_protocol),
        "roster_sha256": sha256_file(args.roster),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_source_sha256": sha256_file(model_source),
        "materializer_source_sha256": sha256_file(materializer_source),
        "truth_inputs_opened": False,
        "parameter_count": parameters,
        "aligned_depth": {
            "path": str(final_path.resolve()),
            "shape": list(expected_shape),
            "dtype": "float16",
            "sha256": sha256_file(final_path),
        },
        "host_cuda_latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted_latency[round(0.95 * (len(sorted_latency) - 1))],
            "claim_ceiling": "host CUDA diagnostic only; not Android App latency",
        },
        "frames": receipts,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
