"""Materialize S0 depth on the frozen 120-frame development roster."""

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

from clearance_student_mobile_r0 import ClearanceStudentMobileR0, normalize_bgr_batch  # noqa: E402
from evaluate_dav2_model_variant_gate_r0 import sha256_file  # noqa: E402
from train_clearance_student_mobile_r0 import load_encoder_weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "clearance-student-mobile-r0":
        raise ValueError("S0 protocol mismatch")
    gate = json.loads(args.gate_protocol.read_text(encoding="utf-8"))
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    if sha256_file(args.roster) != gate["roster_sha256"]:
        raise ValueError("roster hash mismatch")
    if len(roster.get("rows", [])) != 120:
        raise ValueError("S0 cache requires exactly 120 frozen development frames")
    args.output_root.mkdir(parents=True)
    final = args.output_root / "aligned_depth_f16.npy"
    partial = args.output_root / "aligned_depth_f16.partial.npy"
    output = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(120, 480, 640))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClearanceStudentMobileR0(pretrained=False).to(device)
    load_encoder_weights(model, args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    latencies: list[float] = []
    with torch.inference_mode():
        for index, row in enumerate(roster["rows"]):
            rgb_path = args.source_root / str(row["sequence_root"]) / str(row["rgb_path"])
            if sha256_file(rgb_path) != row["rgb_sha256"]:
                raise ValueError(f"RGB hash mismatch: {row['frame_id']}")
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (480, 640):
                raise OSError(f"cannot decode S0 RGB: {rgb_path}")
            image = normalize_bgr_batch([torch.from_numpy(bgr.transpose(2, 0, 1).copy())]).to(device)
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(image, (480, 640))["metric_depth"][0]
            elapsed = (time.perf_counter() - started) * 1000.0
            depth = prediction.float().cpu().numpy()
            if depth.shape != (480, 640) or not np.all(np.isfinite(depth)):
                raise ValueError(f"invalid S0 depth: {row['frame_id']}")
            output[index] = depth.astype(np.float16)
            latencies.append(elapsed)
    output.flush()
    mmap = getattr(output, "_mmap", None)
    if mmap is not None:
        mmap.close()
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, final)
    ordered = sorted(latencies)
    manifest = {
        "schema": "blindassist_clearance_student_mobile_r0_cache",
        "candidate_id": "S0_CLEARANCE_STUDENT_MOBILE_R0",
        "protocol_sha256": sha256_file(args.protocol),
        "gate_protocol_sha256": sha256_file(args.gate_protocol),
        "roster_sha256": sha256_file(args.roster),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "p1_truth_opened_during_materialization": False,
        "runtime": {"device": str(device), "input_size": 384, "precision": "fp16_autocast"},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "aligned_depth": {"path": str(final.resolve()), "shape": [120, 480, 640], "dtype": "float16", "sha256": sha256_file(final)},
        "host_materialization_latency_ms": {"mean": statistics.fmean(latencies), "median": statistics.median(latencies), "p95": ordered[round(0.95 * (len(ordered) - 1))], "claim_ceiling": "host CUDA diagnostic only; not Android App latency"},
        "frames": [{"index": i, "frame_id": row["frame_id"], "latency_ms": latencies[i]} for i, row in enumerate(roster["rows"])],
        "terminal": "CLEARANCE_STUDENT_MOBILE_R0_CACHE_COMPLETE_DEVELOPMENT_ONLY",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
