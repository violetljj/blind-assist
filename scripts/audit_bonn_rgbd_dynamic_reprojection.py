#!/usr/bin/env python3
"""CUDA source-native depth/pose reprojection audit for Bonn Dynamic RGB-D.

This reports temporal residuals in the supplied camera coordinate system.  Dynamic objects and
occlusions are intentionally retained in the residual distribution; no residual is relabelled as
an assistive event or object track.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import audit_bonn_rgbd_dynamic_source as source


def _pairs(root: Path) -> list[dict[str, Any]]:
    rgb = source._index(root / "rgb.txt", 2)
    depth = source._index(root / "depth.txt", 2)
    pose = source._index(root / "groundtruth.txt", 8)
    rows = []
    for timestamp, (rgb_name,) in rgb:
        depth_row = source._nearest(depth, timestamp); pose_row = source._nearest(pose, timestamp)
        rows.append({"timestamp": timestamp, "rgb": rgb_name, "depth": depth_row[1][0], "pose": pose_row[1]})
    return rows


def audit(root: Path, pair_count: int = 24, stride: int = 4) -> dict[str, Any]:
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Bonn temporal reprojection audit")
    if pair_count < 1 or stride < 1:
        raise ValueError("pair_count and stride must be positive")
    rows = _pairs(root)
    indexes = np.unique(np.linspace(0, len(rows) - 2, num=min(pair_count, len(rows) - 1), dtype=int))
    fx, fy, cx, cy = source.INTRINSICS
    residual_medians: list[float] = []; residual_p95s: list[float] = []; fractions: list[float] = []
    device = torch.device("cuda")
    for index in indexes:
        first, second = rows[int(index)], rows[int(index) + 1]
        depth1 = torch.as_tensor(np.array(Image.open(root / first["depth"]), dtype=np.uint16, copy=True), dtype=torch.float32, device=device) / source.DEPTH_SCALE
        depth2 = torch.as_tensor(np.array(Image.open(root / second["depth"]), dtype=np.uint16, copy=True), dtype=torch.float32, device=device) / source.DEPTH_SCALE
        height, width = depth1.shape
        vv, uu = torch.meshgrid(torch.arange(0, height, stride, device=device), torch.arange(0, width, stride, device=device), indexing="ij")
        z1 = depth1[vv, uu]
        x1 = (uu.float() - cx) * z1 / fx; y1 = (vv.float() - cy) * z1 / fy
        points = torch.stack((x1, y1, z1), dim=-1).reshape(-1, 3)
        pose1 = source._matrix(np.asarray([float(value) for value in first["pose"][:3]], dtype=np.float32), np.asarray([float(value) for value in first["pose"][3:]], dtype=np.float32))
        pose2 = source._matrix(np.asarray([float(value) for value in second["pose"][:3]], dtype=np.float32), np.asarray([float(value) for value in second["pose"][3:]], dtype=np.float32))
        relative = torch.as_tensor(np.linalg.inv(pose2) @ pose1, device=device)
        transformed = points @ relative[:3, :3].T + relative[:3, 3]
        z2 = transformed[:, 2]
        u2 = torch.round(fx * transformed[:, 0] / z2.clamp_min(1e-6) + cx).long()
        v2 = torch.round(fy * transformed[:, 1] / z2.clamp_min(1e-6) + cy).long()
        valid = (z1.reshape(-1) > 0) & (z2 > 0) & (u2 >= 0) & (u2 < width) & (v2 >= 0) & (v2 < height)
        target = torch.zeros_like(z2); target[valid] = depth2[v2[valid], u2[valid]]
        valid &= target > 0
        residual = torch.abs(target[valid] - z2[valid])
        if residual.numel() == 0:
            raise ValueError(f"no valid reprojections for pair {index}")
        fractions.append(float(valid.float().mean().item()))
        residual_medians.append(float(residual.median().item()))
        residual_p95s.append(float(torch.quantile(residual, .95).item()))
    report = {
        "format": "blindassist_bonn_rgbd_dynamic_reprojection_audit_v1", "source_sequence": root.name,
        "sampled_consecutive_pairs": len(indexes), "sample_stride_pixels": stride,
        "aggregate": {"median_valid_projection_fraction": float(np.median(fractions)), "median_pair_median_abs_depth_residual_m": float(np.median(residual_medians)), "median_pair_p95_abs_depth_residual_m": float(np.median(residual_p95s))},
        "interpretation": "Residuals contain source-native moving-box pixels, occlusions, association error, and pose/depth noise; they are not dynamic-object labels.",
        "ustrf_geometry_input_admitted": False, "reason": "no sensor-body extrinsics, body-local ground plane, or assistive event truth", "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)}, "production_authority": False,
    }
    qa = root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "bonn_rgbd_dynamic_reprojection_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=24)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    report = audit(args.root, args.pair_count, args.stride)
    print(json.dumps(report["aggregate"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
