#!/usr/bin/env python3
"""Materialize hash-bound dense depth and frozen DA feature caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from common import (
    REPO_ROOT,
    frame_key,
    load_json,
    report_frames,
    resolve,
    sha256,
    write_json_new,
)

HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
DEPENDENCY_DIR = (
    REPO_ROOT / "artifacts.local/vendor/python-packages-hftf-metric-depth-r0"
)
sys.path.insert(0, str(DEPENDENCY_DIR))
sys.path.insert(0, str(HFTF_DIR))

from evaluate_metric3d_clearance_field_a0 import clearance_field
from produce_external_rgb_metric_depth_observations import (
    DepthAnythingV2MetricSource,
    Metric3DPytorchSource,
    intrinsics_matrix,
)

SCHEMA = "blindassist_metric_depth_dense_cache_r0"
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "docs/research/hftf/DENSE_METRIC_DEPTH_PROPAGATION_R0_PROTOCOL_2026-08-03.json"
)


def validate_bound_inputs(
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        protocol.get("schema")
        != "blindassist_dense_metric_depth_propagation_r0_protocol"
    ):
        raise ValueError("unexpected dense propagation protocol")
    if protocol.get("status") != "FROZEN_BEFORE_DENSE_OUTPUT_MATERIALIZATION":
        raise ValueError("dense propagation protocol is not frozen")
    reports = []
    for key in ("metric_report", "fast_report"):
        receipt = protocol["inputs"][key]
        path = resolve(str(receipt["path"]))
        if sha256(path) != str(receipt["sha256"]).upper():
            raise ValueError(f"{key} hash mismatch")
        reports.append(load_json(path))
    metric, fast = reports
    if [frame_key(row) for row in report_frames(metric)] != [
        frame_key(row) for row in report_frames(fast)
    ]:
        raise ValueError("bound reports have different frames")
    metadata_by_path: dict[str, dict[str, Any]] = {}
    for receipt in protocol["inputs"]["frame_manifests"]:
        path = resolve(str(receipt["path"]))
        if sha256(path) != str(receipt["sha256"]).upper():
            raise ValueError(f"frame manifest hash mismatch: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metadata_by_path[str(Path(str(row["frame_path"])).resolve())] = row
    for report in (metric, fast):
        for frame in report["frames"]:
            key = str(Path(str(frame["frame_path"])).resolve())
            metadata = metadata_by_path.get(key)
            if metadata is None:
                raise ValueError(f"no bound frame metadata for {key}")
            frame["intrinsics_fx_fy_cx_cy"] = list(metadata["intrinsics_fx_fy_cx_cy"])
    return metric, fast


def validate_model_inputs(protocol: dict[str, Any]) -> dict[str, Path]:
    inputs = protocol["inputs"]
    paths = {
        "metric_repo": resolve(inputs["metric3d"]["repo"]),
        "metric_checkpoint": resolve(inputs["metric3d"]["checkpoint"]),
        "dav2_repo": resolve(inputs["dav2_metric"]["repo"]),
        "dav2_checkpoint": resolve(inputs["dav2_metric"]["checkpoint"]),
    }
    if sha256(paths["metric_checkpoint"]) != inputs["metric3d"]["checkpoint_sha256"]:
        raise ValueError("Metric3D checkpoint hash mismatch")
    if sha256(paths["dav2_checkpoint"]) != inputs["dav2_metric"]["checkpoint_sha256"]:
        raise ValueError("DA V2 checkpoint hash mismatch")
    dpt = paths["dav2_repo"] / "metric_depth/depth_anything_v2/dpt.py"
    if sha256(dpt) != inputs["dav2_metric"]["source_dpt_sha256"]:
        raise ValueError("DA V2 source hash mismatch")
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(paths["metric_repo"]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != inputs["metric3d"]["repo_commit"]:
        raise ValueError("Metric3D repository commit mismatch")
    return paths


def dav2_depth_and_cls(
    source: DepthAnythingV2MetricSource,
    bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = source.model
    image, (height, width) = model.image2tensor(bgr, source.input_size)
    patch_height, patch_width = image.shape[-2] // 14, image.shape[-1] // 14
    with (
        source.torch.inference_mode(),
        source.torch.autocast(
            device_type=source.device.type,
            dtype=source.torch.float16,
            enabled=source.precision == "fp16",
        ),
    ):
        features = model.pretrained.get_intermediate_layers(
            image,
            model.intermediate_layer_idx[model.encoder],
            return_class_token=True,
        )
        depth = model.depth_head(features, patch_height, patch_width) * model.max_depth
        depth = source.torch.nn.functional.interpolate(
            depth,
            (height, width),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        cls = features[-1][1][0]
    return (
        depth.float().cpu().numpy().astype(np.float32),
        cls.float().cpu().numpy().astype(np.float32),
    )


def clearance_difference(
    depth: np.ndarray,
    frame: dict[str, Any],
) -> float:
    generated = clearance_field(depth, intrinsics_matrix(frame))
    expected = frame["candidate"]
    differences = []
    if generated.get("status") != expected.get("status"):
        return float("inf")
    for band in ("left", "center", "right"):
        left = generated.get("bands", {}).get(band, {}).get("clearance_m")
        right = expected.get("bands", {}).get(band, {}).get("clearance_m")
        if left is None or right is None:
            if left != right:
                return float("inf")
        else:
            differences.append(abs(float(left) - float(right)))
    return max(differences, default=0.0)


def new_memmap(
    path: Path, shape: tuple[int, ...], dtype: Any
) -> tuple[np.memmap, Path]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    if partial.exists():
        raise FileExistsError(f"partial output already exists: {partial}")
    return np.lib.format.open_memmap(
        partial, mode="w+", dtype=dtype, shape=shape
    ), partial


def finalize_memmap(array: np.memmap, partial: Path, final: Path) -> None:
    array.flush()
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        memory_map.close()
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, final)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")

    protocol = load_json(args.protocol)
    metric_report, fast_report = validate_bound_inputs(protocol)
    paths = validate_model_inputs(protocol)
    metric_frames = report_frames(metric_report)
    fast_frames = report_frames(fast_report)
    expected_shape = tuple(int(value) for value in protocol["dense_cache"]["shape"])
    if expected_shape != (len(metric_frames), 480, 640):
        raise ValueError("unexpected frozen cache shape")
    args.output_root.mkdir(parents=True)
    da_path = args.output_root / "dav2_depth_f16.npy"
    metric_path = args.output_root / "metric3d_depth_f16.npy"
    feature_path = args.output_root / "dav2_layer11_cls_f32.npy"
    da_cache, da_partial = new_memmap(da_path, expected_shape, np.float16)
    metric_cache, metric_partial = new_memmap(metric_path, expected_shape, np.float16)
    feature_cache, feature_partial = new_memmap(
        feature_path, (len(metric_frames), 384), np.float32
    )

    da_source = DepthAnythingV2MetricSource(
        paths["dav2_repo"],
        paths["dav2_checkpoint"],
        "cuda",
        input_size=int(protocol["inputs"]["dav2_metric"]["input_size"]),
        precision=str(protocol["inputs"]["dav2_metric"]["precision"]),
    )
    da_latencies = []
    da_parity = []
    for index, frame in enumerate(fast_frames):
        bgr = cv2.imread(str(frame["frame_path"]), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (480, 640):
            raise OSError(f"cannot decode frozen RGB: {frame['frame_path']}")
        started = time.perf_counter()
        depth, cls = dav2_depth_and_cls(da_source, bgr)
        da_latencies.append((time.perf_counter() - started) * 1000.0)
        if not np.all(np.isfinite(depth)) or not np.all(np.isfinite(cls)):
            raise ValueError("DA V2 produced non-finite cache values")
        da_cache[index] = depth.astype(np.float16)
        feature_cache[index] = cls
        da_parity.append(clearance_difference(depth, frame))
    del da_source
    import torch

    torch.cuda.empty_cache()
    finalize_memmap(da_cache, da_partial, da_path)
    finalize_memmap(feature_cache, feature_partial, feature_path)

    metric_source = Metric3DPytorchSource(
        paths["metric_repo"],
        paths["metric_checkpoint"],
        "cuda",
        precision=str(protocol["inputs"]["metric3d"]["precision"]),
    )
    metric_latencies = []
    metric_parity = []
    for index, frame in enumerate(metric_frames):
        bgr = cv2.imread(str(frame["frame_path"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        depth, _metadata = metric_source.infer(rgb, frame)
        metric_latencies.append((time.perf_counter() - started) * 1000.0)
        if not np.all(np.isfinite(depth)):
            raise ValueError("Metric3D produced non-finite cache values")
        metric_cache[index] = depth.astype(np.float16)
        metric_parity.append(clearance_difference(depth, frame))
    finalize_memmap(metric_cache, metric_partial, metric_path)

    maximum_allowed = float(
        protocol["dense_cache"][
            "maximum_recomputed_clearance_difference_from_bound_reports_m"
        ]
    )
    maximum_da_parity = max(da_parity)
    maximum_metric_parity = max(metric_parity)
    if maximum_da_parity > maximum_allowed or maximum_metric_parity > maximum_allowed:
        raise ValueError(
            "dense cache clearance parity failed: "
            f"DA={maximum_da_parity}, Metric3D={maximum_metric_parity}"
        )
    rows = [
        {
            "index": index,
            "sequence_id": str(metric["sequence_id"]),
            "timestamp": float(metric["timestamp"]),
            "frame_path": str(metric["frame_path"]),
            "intrinsics_fx_fy_cx_cy": list(metric["intrinsics_fx_fy_cx_cy"]),
        }
        for index, (metric, fast) in enumerate(
            zip(metric_frames, fast_frames, strict=True)
        )
    ]
    report = {
        "schema": SCHEMA,
        "protocol_sha256": sha256(args.protocol),
        "data_role": protocol["authority"]["data_role"],
        "truth_depth_materialized": False,
        "rows": rows,
        "outputs": {
            "dav2_depth": {"path": str(da_path.resolve()), "sha256": sha256(da_path)},
            "metric3d_depth": {
                "path": str(metric_path.resolve()),
                "sha256": sha256(metric_path),
            },
            "dav2_layer11_cls": {
                "path": str(feature_path.resolve()),
                "sha256": sha256(feature_path),
            },
        },
        "parity": {
            "maximum_allowed_clearance_difference_m": maximum_allowed,
            "dav2_max_clearance_difference_m": maximum_da_parity,
            "metric3d_max_clearance_difference_m": maximum_metric_parity,
            "passed": True,
        },
        "latency_ms": {
            "dav2_mean": float(np.mean(da_latencies)),
            "dav2_p95": float(np.quantile(da_latencies, 0.95)),
            "metric3d_mean": float(np.mean(metric_latencies)),
            "metric3d_p95": float(np.quantile(metric_latencies, 0.95)),
        },
        "claim_ceiling": "consumed Development dense cache; non-keyframe Metric3D is forbidden to scheme-2 candidate generation",
    }
    write_json_new(args.output_root / "manifest.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
