#!/usr/bin/env python3
"""Export the frozen UniDepthV2-S camera-input route and verify ONNX parity.

This is a deployment canary only.  It consumes one already-materialized RGB row,
does not read outcome labels, and does not evaluate or tune model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_first_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                break
        else:
            raise ValueError(f"empty source manifest: {path}")
    required = {"frame_path", "intrinsics_fx_fy_cx_cy"}
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"source row is missing: {missing}")
    return row


def prepare_inputs(
    frame_path: Path,
    intrinsics: list[float],
    pixels_min: int,
    pixels_max: int,
    resolution_level: int,
    device: str,
):
    import torch
    import torch.nn.functional as functional
    import torchvision.transforms.v2.functional as transforms
    from unidepth.models.unidepthv2.unidepthv2 import (
        get_paddings,
        get_resize_factor,
    )
    from unidepth.utils.camera import BatchCamera, Pinhole
    from unidepth.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD

    bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(frame_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
    _, _, height, width = image.shape

    interval = (pixels_max - pixels_min) / 10
    pixels_bounds = (
        resolution_level * interval + pixels_min,
        (resolution_level + 1) * interval + pixels_min,
    )
    paddings, (padded_height, padded_width) = get_paddings(
        (height, width), (0.5, 2.5)
    )
    pad_left, pad_right, pad_top, pad_bottom = paddings
    resize_factor, (new_height, new_width) = get_resize_factor(
        (padded_height, padded_width), pixels_bounds
    )

    image = transforms.normalize(
        image.float() / 255.0,
        mean=IMAGENET_DATASET_MEAN,
        std=IMAGENET_DATASET_STD,
    )
    image = functional.pad(
        image, (pad_left, pad_right, pad_top, pad_bottom), value=0.0
    )
    image = functional.interpolate(
        image, size=(new_height, new_width), mode="bilinear", align_corners=False
    )

    fx, fy, cx, cy = (float(value) for value in intrinsics)
    matrix = torch.tensor(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    camera = BatchCamera.from_camera(Pinhole(K=matrix))
    camera = camera.crop(
        left=-pad_left, top=-pad_top, right=-pad_right, bottom=-pad_bottom
    )
    camera = camera.resize(resize_factor).to(device)
    rays = camera.get_rays(shapes=(1, new_height, new_width))
    return image.to(device), rays, {
        "original_shape_hw": [height, width],
        "paddings_lrtb": list(paddings),
        "resize_factor": resize_factor,
        "model_shape_hw": [new_height, new_width],
    }


def tensor_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    relative = delta / np.maximum(np.abs(reference.astype(np.float64)), 1e-6)
    max_index = np.unravel_index(int(delta.argmax()), delta.shape)
    return {
        "shape": list(reference.shape),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "candidate_min": float(candidate.min()),
        "candidate_max": float(candidate.max()),
        "max_abs": float(delta.max()),
        "max_abs_reference": float(reference[max_index]),
        "max_abs_candidate": float(candidate[max_index]),
        "max_abs_relative": float(relative[max_index]),
        "mean_abs": float(delta.mean()),
        "p99_abs": float(np.quantile(delta, 0.99)),
        "max_relative": float(relative.max()),
        "mean_relative": float(relative.mean()),
        "finite": bool(np.isfinite(candidate).all()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-onnx", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution-level", type=int, default=0)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--p99-abs-tolerance", type=float, default=5e-3)
    parser.add_argument("--max-relative-tolerance", type=float, default=2e-2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(args.vendor_repo.resolve()))

    import onnx
    import onnxruntime as ort
    import torch
    from safetensors.torch import load_file
    from unidepth.models.unidepthv2.export import UniDepthV2ONNXcam

    config_path = args.vendor_repo / "configs" / "config_v2_vits14.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["export"] = True
    row = load_first_row(args.source_manifest)
    image, rays, preprocessing = prepare_inputs(
        Path(row["frame_path"]),
        row["intrinsics_fx_fy_cx_cy"],
        int(config["data"]["augmentations"]["shape_constraints"]["pixels_min"]),
        int(config["data"]["augmentations"]["shape_constraints"]["pixels_max"]),
        args.resolution_level,
        args.device,
    )

    model = UniDepthV2ONNXcam(config)
    load_info = model.load_state_dict(load_file(str(args.checkpoint)), strict=False)
    if load_info.missing_keys or load_info.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={load_info.missing_keys}, "
            f"unexpected={load_info.unexpected_keys}"
        )
    model = model.to(args.device).eval()
    with torch.inference_mode():
        torch_outputs = tuple(value.detach().cpu().numpy() for value in model(image, rays))

    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    export_started = time.perf_counter()
    torch.onnx.export(
        model,
        (image, rays),
        str(args.output_onnx),
        input_names=["rgbs", "rays"],
        output_names=["pts_3d", "confidence", "intrinsics"],
        opset_version=args.opset,
        dynamic_axes={
            "rgbs": {0: "batch"},
            "rays": {0: "batch"},
            "pts_3d": {0: "batch"},
            "confidence": {0: "batch"},
            "intrinsics": {0: "batch"},
        },
        dynamo=False,
    )
    export_seconds = time.perf_counter() - export_started

    checked = onnx.load(str(args.output_onnx), load_external_data=True)
    onnx.checker.check_model(checked)
    session_started = time.perf_counter()
    session = ort.InferenceSession(
        str(args.output_onnx), providers=["CPUExecutionProvider"]
    )
    session_creation_seconds = time.perf_counter() - session_started
    ort_started = time.perf_counter()
    ort_outputs = session.run(
        None,
        {
            "rgbs": image.detach().cpu().numpy(),
            "rays": rays.detach().cpu().numpy(),
        },
    )
    ort_inference_seconds = time.perf_counter() - ort_started
    names = ["pts_3d", "confidence", "intrinsics"]
    parity = {
        name: tensor_delta(reference, candidate)
        for name, reference, candidate in zip(names, torch_outputs, ort_outputs)
    }
    passed = all(
        item["finite"]
        and item["p99_abs"] <= args.p99_abs_tolerance
        and item["max_relative"] <= args.max_relative_tolerance
        for item in parity.values()
    )

    report = {
        "schema_version": 1,
        "terminal": "PASS" if passed else "FAIL",
        "scope": "deployment_export_parity_only_no_quality_claim",
        "source_role": "consumed_rgb_no_outcome_labels_read",
        "model": "lpiccinelli/unidepth-v2-vits14",
        "camera_input": "normalized_unit_rays",
        "resolution_level": args.resolution_level,
        "opset": args.opset,
        "parity_gate": {
            "p99_abs_tolerance": args.p99_abs_tolerance,
            "max_relative_tolerance": args.max_relative_tolerance,
            "all_outputs_must_be_finite": True,
        },
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "frame_path": str(Path(row["frame_path"]).resolve()),
        "frame_sha256": sha256(Path(row["frame_path"])),
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx_path": str(args.output_onnx.resolve()),
        "onnx_sha256": sha256(args.output_onnx),
        "onnx_size_bytes": args.output_onnx.stat().st_size,
        "preprocessing": preprocessing,
        "parity": parity,
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "torch_device": args.device,
            "ort_providers": session.get_providers(),
            "export_seconds": export_seconds,
            "ort_session_creation_seconds": session_creation_seconds,
            "ort_single_inference_seconds": ort_inference_seconds,
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
