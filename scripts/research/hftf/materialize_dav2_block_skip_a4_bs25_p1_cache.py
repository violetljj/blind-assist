#!/usr/bin/env python3
"""Materialize the frozen DA V2 A4-BS25 block-skip candidate on P1 RGB."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_dav2_model_variant_gate_r0 import sha256_file
from produce_external_rgb_metric_depth_observations import (
    DepthAnythingV2MetricSource,
)

FROZEN_SKIP_INDICES = (3, 7, 11)


def apply_frozen_block_skip(model: Any) -> None:
    """Replace exactly the frozen fourth, eighth, and twelfth ViT-S blocks."""
    import torch

    blocks = model.pretrained.blocks
    if model.pretrained.chunked_blocks or len(blocks) != 12:
        raise ValueError("A4-BS25 requires the unchunked 12-block DA V2 ViT-S")
    if model.intermediate_layer_idx["vits"] != [2, 5, 8, 11]:
        raise ValueError("A4-BS25 intermediate feature indices drifted")
    for index in FROZEN_SKIP_INDICES:
        blocks[index] = torch.nn.Identity()
    identities = tuple(
        index for index, block in enumerate(blocks) if isinstance(block, torch.nn.Identity)
    )
    if identities != FROZEN_SKIP_INDICES:
        raise AssertionError(f"unexpected A4-BS25 identity blocks: {identities}")


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
    parser.add_argument("--p1-r0-protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dav2-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    p1 = json.loads(args.p1_r0_protocol.read_text(encoding="utf-8"))
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    bindings = protocol["bindings"]
    if sha256_file(Path(__file__).resolve()) != bindings["materializer_source_sha256"]:
        raise ValueError("A4-BS25 materializer hash mismatch")
    if sha256_file(args.p1_r0_protocol) != protocol["parent_p1_r0_protocol_sha256"]:
        raise ValueError("A4-BS25 P1 R0 protocol hash mismatch")
    if sha256_file(args.roster) != p1["roster_sha256"]:
        raise ValueError("A4-BS25 roster hash mismatch")
    if sha256_file(args.checkpoint) != bindings["checkpoint_sha256"]:
        raise ValueError("A4-BS25 checkpoint hash mismatch")
    if tuple(protocol["candidate"]["skip_zero_based_block_indices"]) != FROZEN_SKIP_INDICES:
        raise ValueError("A4-BS25 frozen skip indices mismatch")

    rows = roster["rows"]
    shape = tuple(p1["cohort"]["aligned_depth_shape"])
    if shape != (len(rows), 480, 640):
        raise ValueError("A4-BS25 unexpected P1 aligned-depth shape")

    source = DepthAnythingV2MetricSource(
        args.dav2_repo.resolve(),
        args.checkpoint.resolve(),
        args.device,
        518,
        "fp16",
    )
    apply_frozen_block_skip(source.model)
    args.output_root.mkdir(parents=True)
    partial = args.output_root / "aligned_depth_f16.partial.npy"
    final = args.output_root / "aligned_depth_f16.npy"
    output = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
    latencies: list[float] = []
    for index, row in enumerate(rows):
        rgb_path = args.source_root / str(row["sequence_root"]) / str(row["rgb_path"])
        if sha256_file(rgb_path) != row["rgb_sha256"]:
            raise ValueError(f"A4-BS25 P1 RGB hash mismatch: {row['frame_id']}")
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (480, 640):
            raise OSError(f"cannot decode A4-BS25 RGB: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        depth, _ = source.infer(rgb, row)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if depth.shape != (480, 640) or not np.all(np.isfinite(depth)):
            raise ValueError(f"invalid A4-BS25 depth: {row['frame_id']}")
        output[index] = depth.astype(np.float16)
        latencies.append(latency_ms)
    del source
    finalize_memmap(output, partial, final)
    ordered = sorted(latencies)
    manifest = {
        "schema": "blindassist_dav2_block_skip_a4_bs25_p1_cache",
        "candidate_id": "a4-bs25-dav2-vits-fp16-518-skip-3-7-11",
        "protocol_sha256": sha256_file(args.protocol),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "p1_truth_opened_during_materialization": False,
        "runtime": {
            "device": args.device,
            "input_size": 518,
            "precision": "fp16",
            "skip_zero_based_block_indices": list(FROZEN_SKIP_INDICES),
        },
        "aligned_depth": {
            "path": str(final.resolve()),
            "shape": list(shape),
            "dtype": "float16",
            "sha256": sha256_file(final),
        },
        "host_materialization_latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": ordered[round(0.95 * (len(ordered) - 1))],
            "claim_ceiling": "host CUDA cache materialization diagnostic only; not Android latency",
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
