#!/usr/bin/env python3
"""Freeze a deterministic full-graph DepthART parity canary and PyTorch oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .export_depthart_camera_external import ExternalCameraMetric, install_timm_compat
except ImportError:
    from export_depthart_camera_external import ExternalCameraMetric, install_timm_compat


SCHEMA = "blindassist_depthart_full_graph_canary_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def procedural_bgr(resolution: int) -> np.ndarray:
    """Return a fixed, spatially varied uint8 image without an external asset."""
    y, x = np.indices((resolution, resolution), dtype=np.uint32)
    blue = (3 * x + 5 * y + ((x // 17) ^ (y // 13)) * 11) % 256
    green = (7 * x + 2 * y + ((x * y) % 97)) % 256
    red = (x + 9 * y + (((x // 29) + (y // 31)) % 2) * 73) % 256
    return np.stack((blue, green, red), axis=-1).astype(np.uint8)


def write_raw(path: Path, value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value, dtype=np.float32)
    array.tofile(path)
    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": "float32",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=448)
    args = parser.parse_args()

    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    install_timm_compat()
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from common import make_K, preprocess  # type: ignore
    from depthart_selective_scan import install_depthart  # type: ignore
    from model import load_model  # type: ignore
    from network import tvimblock  # type: ignore

    resolution = args.resolution
    bgr = procedural_bgr(resolution)
    K = make_K(500.0, 500.0, resolution / 2.0, resolution / 2.0)
    image, intrinsics = preprocess(bgr, K, resolution, resolution)
    model = load_model(checkpoint, "S", "indoor", "cuda").eval()
    install_depthart(tvimblock)
    wrapper = ExternalCameraMetric(model).cuda().eval()
    image = image.cuda()
    intrinsics = intrinsics.cuda()
    cameras = model.cam_embedder(intrinsics, resolution, resolution, "cuda")
    with torch.inference_mode():
        reference = model(image, intrinsics)
        external = wrapper(image, *cameras)
    camera_prompt_max_abs = float((reference - external).abs().max().item())
    if camera_prompt_max_abs > 1e-5:
        raise RuntimeError(
            f"external camera prompt parity failed: max_abs={camera_prompt_max_abs}"
        )

    cv2.imwrite(str(output_dir / "procedural-canary.png"), bgr)
    arrays = {
        "image": image.detach().cpu().numpy(),
        "camera_prompt_4": cameras[0].detach().cpu().numpy(),
        "camera_prompt_8": cameras[1].detach().cpu().numpy(),
        "camera_prompt_16": cameras[2].detach().cpu().numpy(),
        "camera_prompt_32": cameras[3].detach().cpu().numpy(),
        "depth": external.detach().cpu().numpy(),
    }
    files: dict[str, dict[str, object]] = {}
    for name, array in arrays.items():
        filename = "pytorch-depth.raw" if name == "depth" else f"{name}.raw"
        files[name] = write_raw(output_dir / filename, array)

    input_line = " ".join(
        f"{name}:={files[name]['path']}"
        for name in (
            "image",
            "camera_prompt_4",
            "camera_prompt_8",
            "camera_prompt_16",
            "camera_prompt_32",
        )
    )
    (output_dir / "input-list.txt").write_text(input_line + "\n", encoding="utf-8")
    receipt = {
        "schema": SCHEMA,
        "authority": "SYNTHETIC_FULL_GRAPH_NUMERICAL_CANARY_ONLY",
        "explicit_exclusions": [
            "REAL_SCENE_TASK_QUALITY",
            "CLEARANCE_SAFETY",
            "TEMPORAL_QUALITY",
            "PERFORMANCE",
            "PRODUCTIZATION",
        ],
        "generator": "fixed_integer_formula_v1",
        "resolution": resolution,
        "intrinsics_fx_fy_cx_cy": [500.0, 500.0, resolution / 2.0, resolution / 2.0],
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        },
        "procedural_png_sha256": sha256(output_dir / "procedural-canary.png"),
        "camera_prompt_parity_max_abs": camera_prompt_max_abs,
        "files": files,
    }
    receipt_path = output_dir / "canary-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
