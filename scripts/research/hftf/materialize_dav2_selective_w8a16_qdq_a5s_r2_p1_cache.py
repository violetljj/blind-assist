#!/usr/bin/env python3
"""Materialize A5S R2 QDQ output on frozen P1 RGB without reading truth."""

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
import onnxruntime as ort
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_dav2_model_variant_gate_r0 import sha256_file


def load_preprocessor(repo: Path, checkpoint: Path) -> Any:
    sys.path.insert(0, str(repo / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        max_depth=20.0,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


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
    parser.add_argument("--qdq-onnx", type=Path, required=True)
    parser.add_argument("--dav2-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    p1 = json.loads(args.p1_r0_protocol.read_text(encoding="utf-8"))
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    bindings = protocol["bindings"]
    if sha256_file(args.qdq_onnx) != bindings["qdq_onnx_sha256"]:
        raise ValueError("A5S R2 QDQ hash mismatch")
    if sha256_file(Path(__file__).resolve()) != bindings["materializer_source_sha256"]:
        raise ValueError("A5S R2 materializer hash mismatch")
    if sha256_file(args.p1_r0_protocol) != protocol["parent_p1_r0_protocol_sha256"]:
        raise ValueError("A5S R2 P1 R0 hash mismatch")
    if sha256_file(args.roster) != p1["roster_sha256"]:
        raise ValueError("A5S R2 roster hash mismatch")
    if sha256_file(args.checkpoint) != p1["baseline"]["checkpoint_sha256"]:
        raise ValueError("A5S R2 checkpoint hash mismatch")
    rows = roster["rows"]
    shape = tuple(p1["cohort"]["aligned_depth_shape"])
    model = load_preprocessor(args.dav2_repo.resolve(), args.checkpoint.resolve())
    session = ort.InferenceSession(
        str(args.qdq_onnx.resolve()),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    if session.get_inputs()[0].shape != [1, 3, 518, 686]:
        raise ValueError("A5S R2 ONNX input shape mismatch")
    args.output_root.mkdir(parents=True)
    partial = args.output_root / "aligned_depth_f16.partial.npy"
    final = args.output_root / "aligned_depth_f16.npy"
    output = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
    latencies = []
    for index, row in enumerate(rows):
        rgb_path = args.source_root / str(row["sequence_root"]) / str(row["rgb_path"])
        if sha256_file(rgb_path) != row["rgb_sha256"]:
            raise ValueError(f"A5S R2 P1 RGB hash mismatch: {row['frame_id']}")
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (480, 640):
            raise OSError(f"cannot decode A5S R2 RGB: {rgb_path}")
        tensor, original_shape = model.image2tensor(bgr, input_size=518)
        if original_shape != (480, 640):
            raise ValueError("A5S R2 preprocess original-shape mismatch")
        normalized = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        started = time.perf_counter()
        raw = np.asarray(session.run(["depth_m"], {"image": normalized})[0], dtype=np.float32)
        latency_ms = (time.perf_counter() - started) * 1000.0
        aligned = torch.nn.functional.interpolate(
            torch.from_numpy(raw[:, None]),
            size=(480, 640),
            mode="bilinear",
            align_corners=True,
        )[0, 0].numpy()
        if aligned.shape != (480, 640) or not np.all(np.isfinite(aligned)):
            raise ValueError(f"invalid A5S R2 depth: {row['frame_id']}")
        output[index] = aligned.astype(np.float16)
        latencies.append(latency_ms)
    finalize_memmap(output, partial, final)
    ordered = sorted(latencies)
    manifest = {
        "schema": "blindassist_dav2_selective_w8a16_a5s_r2_p1_cache",
        "protocol_sha256": sha256_file(args.protocol),
        "qdq_onnx_sha256": sha256_file(args.qdq_onnx),
        "p1_truth_opened_during_materialization": False,
        "aligned_depth": {
            "path": str(final.resolve()),
            "shape": list(shape),
            "dtype": "float16",
            "sha256": sha256_file(final),
        },
        "host_ort_latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": ordered[round(0.95 * (len(ordered) - 1))],
            "claim_ceiling": "host CPU QDQ quality materialization only; not Android latency",
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
