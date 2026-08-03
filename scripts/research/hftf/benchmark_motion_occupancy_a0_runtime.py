#!/usr/bin/env python3
"""Measure PC component cost for the supported A0.1 current-occupancy route."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_small

from evaluate_collision_risk_field_a1 import two_d_clearances
from evaluate_metric3d_clearance_field_a0 import clearance_field
from evaluate_motion_conditioned_occupancy_a0 import (
    EXPECTED_RAFT_SHA256,
    _image_tensor,
    build_rows,
    sha256,
)
from produce_external_rgb_metric_depth_observations import (
    UniDepthSource,
    intrinsics_matrix,
)


SCHEMA = "blindassist_hftf_motion_occupancy_a0_runtime_pc"


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("timing distribution is empty")
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(ordered)),
        "median_ms": float(np.median(ordered)),
        "p95_ms": float(np.quantile(ordered, 0.95)),
        "maximum_ms": float(np.max(ordered)),
    }


def benchmark(
    report: dict[str, Any],
    model: dict[str, Any],
    raft_weights: Path,
    source: UniDepthSource,
    intrinsics_values: list[float],
    warmup_frames: int = 10,
) -> dict[str, Any]:
    if sha256(raft_weights) != EXPECTED_RAFT_SHA256:
        raise ValueError("unexpected RAFT-small checkpoint")
    row = {"intrinsics_fx_fy_cx_cy": intrinsics_values}
    intrinsics = intrinsics_matrix(row)
    fx, _, cx, _ = intrinsics_values
    inference_ms = []
    geometry_ms = []
    corridor_ms = []
    valid_geometry = 0
    for index, frame in enumerate(report["frames"]):
        bgr = cv2.imread(frame["frame_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(frame["frame_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        depth, metadata = source.infer(rgb, row)
        infer_elapsed = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        field = clearance_field(
            depth,
            intrinsics,
            confidence_map=metadata.get("confidence_map"),
        )
        geometry_elapsed = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        two_d_clearances(depth, fx, cx)
        corridor_elapsed = (time.perf_counter() - started) * 1000.0
        if index >= warmup_frames:
            inference_ms.append(infer_elapsed)
            geometry_ms.append(geometry_elapsed)
            corridor_ms.append(corridor_elapsed)
            valid_geometry += field["status"] == "VALID"

    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in report["frames"]:
        by_sequence[str(frame["sequence_id"])].append(frame)
    pairs = []
    for frames in by_sequence.values():
        frames.sort(key=lambda value: float(value["timestamp"]))
        pairs.extend(zip(frames, frames[1:]))
    raft = raft_small(weights=None, progress=False)
    raft.load_state_dict(torch.load(raft_weights, map_location="cpu", weights_only=True), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raft.to(device).eval()
    raft_ms = []
    batch_size = 16
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(pairs), batch_size)):
            batch = pairs[start : start + batch_size]
            started = time.perf_counter()
            previous = torch.stack([_image_tensor(pair[0]["frame_path"]) for pair in batch]).to(device)
            current = torch.stack([_image_tensor(pair[1]["frame_path"]) for pair in batch]).to(device)
            raft(previous, current)[-1].cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_per_pair = (time.perf_counter() - started) * 1000.0 / len(batch)
            if batch_index > 0:
                raft_ms.extend([elapsed_per_pair] * len(batch))

    # Use already computed frozen motion features from the report evaluator for
    # a stable CPU head measurement; model-head cost is independent of image IO.
    from evaluate_motion_conditioned_occupancy_a0 import extract_motion

    motion = extract_motion(report["frames"], raft_weights)
    started = time.perf_counter()
    x, _, _ = build_rows([report], motion)
    feature_build_ms = (time.perf_counter() - started) * 1000.0
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights_intercept_then_features"], dtype=np.float64)
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    repetitions = 1000
    started = time.perf_counter()
    for _ in range(repetitions):
        1.0 / (1.0 + np.exp(-np.clip(design @ weights, -40, 40)))
    head_total_ms = (time.perf_counter() - started) * 1000.0

    return {
        "schema": SCHEMA,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "frames_total": len(report["frames"]),
        "frames_timed_after_warmup": len(inference_ms),
        "geometry_valid_fraction_after_warmup": valid_geometry / len(inference_ms),
        "raft_pairs_total": len(pairs),
        "unidepth_inference": distribution(inference_ms),
        "clearance_3d_cpu": distribution(geometry_ms),
        "corridor_2d_cpu": distribution(corridor_ms),
        "raft_pair_pipeline_after_first_batch": distribution(raft_ms),
        "feature_build_total_ms": feature_build_ms,
        "probability_head_opportunities": len(x),
        "probability_head_microseconds_per_opportunity": (
            head_total_ms * 1000.0 / (repetitions * len(x))
        ),
        "scope": "PC offline component timing; not Android or external-camera evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--unidepth-repo", type=Path, required=True)
    parser.add_argument("--unidepth-model-name", default="lpiccinelli/unidepth-v2-vits14")
    parser.add_argument("--unidepth-resolution-level", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--intrinsics-fx-fy-cx-cy", nargs=4, type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = UniDepthSource(
        args.unidepth_repo,
        args.unidepth_model_name,
        args.unidepth_resolution_level,
        args.device,
    )
    result = benchmark(
        json.loads(args.report.read_text(encoding="utf-8")),
        json.loads(args.model.read_text(encoding="utf-8")),
        args.raft_weights,
        source,
        args.intrinsics_fx_fy_cx_cy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
