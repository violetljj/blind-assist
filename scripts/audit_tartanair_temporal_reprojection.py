#!/usr/bin/env python3
"""GPU audit of source-native TartanAir depth/pose temporal reprojection.

This intentionally remains in the dataset camera/world convention.  It does not
manufacture a USTRF body frame, a local ground plane, or a safety event label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _manifest(root: Path) -> dict[str, Any]:
    text = (root / "slice_manifest.json").read_text(encoding="utf-8").strip()
    return json.loads(text.removesuffix("\\n"))


def _pair_metrics(
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    first_pose_cam_to_world: np.ndarray,
    second_pose_cam_to_world: np.ndarray,
    intrinsics: np.ndarray,
    torch: Any,
) -> dict[str, float | int]:
    """Reproject frame zero into frame one using TartanAir x-right/y-down/z-forward."""
    device = torch.device("cuda")
    height, width = first_depth.shape
    first = torch.as_tensor(first_depth, device=device)
    second = torch.as_tensor(second_depth, device=device)
    pose0 = torch.as_tensor(first_pose_cam_to_world, device=device)
    pose1 = torch.as_tensor(second_pose_cam_to_world, device=device)
    k = torch.as_tensor(intrinsics, device=device)
    v, u = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    z0 = first.reshape(-1)
    x0 = ((u.reshape(-1) - k[0, 2]) * z0 / k[0, 0])
    y0 = ((v.reshape(-1) - k[1, 2]) * z0 / k[1, 1])
    points0 = torch.stack((x0, y0, z0), dim=1)
    world = points0 @ pose0[:3, :3].T + pose0[:3, 3]
    points1 = (world - pose1[:3, 3]) @ pose1[:3, :3]
    z1_predicted = points1[:, 2]
    u1 = torch.round(points1[:, 0] * k[0, 0] / z1_predicted + k[0, 2]).long()
    v1 = torch.round(points1[:, 1] * k[1, 1] / z1_predicted + k[1, 2]).long()
    inside = (z1_predicted > 0) & (u1 >= 0) & (u1 < width) & (v1 >= 0) & (v1 < height)
    observed = torch.zeros_like(z1_predicted)
    observed[inside] = second[v1[inside], u1[inside]]
    valid = inside & torch.isfinite(observed) & (observed > 0)
    residual = torch.abs(observed[valid] - z1_predicted[valid])
    relative = residual / z1_predicted[valid].clamp_min(1e-6)
    if residual.numel() == 0:
        return {"valid_projection_count": 0, "valid_projection_fraction": 0.0, "median_abs_depth_residual_m": float("inf"), "p95_abs_depth_residual_m": float("inf"), "median_relative_depth_residual": float("inf"), "relative_outlier_fraction_gt_10pct": 1.0}
    return {
        "valid_projection_count": int(valid.sum().item()),
        "valid_projection_fraction": float(valid.float().mean().item()),
        "median_abs_depth_residual_m": float(residual.median().item()),
        "p95_abs_depth_residual_m": float(torch.quantile(residual, .95).item()),
        "median_relative_depth_residual": float(relative.median().item()),
        "relative_outlier_fraction_gt_10pct": float((relative > .10).float().mean().item()),
    }


def audit(root: Path, require_cuda: bool = True) -> dict[str, Any]:
    import torch

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not torch.cuda.is_available():
        raise RuntimeError("this audit is intentionally GPU-only")
    manifest = _manifest(root)
    trajectory = root / manifest["trajectory"]
    frame_ids = manifest["frame_ids"]
    rows: list[dict[str, Any]] = []
    for previous, current in zip(frame_ids, frame_ids[1:]):
        previous_camera = np.load(trajectory / f"{previous}_cam.npz")
        current_camera = np.load(trajectory / f"{current}_cam.npz")
        previous_intrinsics = np.asarray(previous_camera["camera_intrinsics"], dtype=np.float32)
        current_intrinsics = np.asarray(current_camera["camera_intrinsics"], dtype=np.float32)
        if not np.allclose(previous_intrinsics, current_intrinsics, atol=1e-6):
            raise ValueError(f"intrinsics changed between {previous} and {current}")
        metrics = _pair_metrics(
            np.load(trajectory / f"{previous}_depth.npy"),
            np.load(trajectory / f"{current}_depth.npy"),
            np.asarray(previous_camera["camera_pose"], dtype=np.float32),
            np.asarray(current_camera["camera_pose"], dtype=np.float32),
            previous_intrinsics,
            torch,
        )
        rows.append({"first_frame_id": previous, "second_frame_id": current, **metrics})
    medians = np.asarray([row["median_abs_depth_residual_m"] for row in rows], dtype=np.float64)
    p95s = np.asarray([row["p95_abs_depth_residual_m"] for row in rows], dtype=np.float64)
    valid_fractions = np.asarray([row["valid_projection_fraction"] for row in rows], dtype=np.float64)
    report = {
        "format": "blindassist_tartanair_temporal_reprojection_audit_v1",
        "source": manifest["source_archive"],
        "trajectory": manifest["trajectory"],
        "pair_count": len(rows),
        "pairs": rows,
        "aggregate": {
            "median_pair_median_abs_depth_residual_m": float(np.median(medians)),
            "p95_pair_p95_abs_depth_residual_m": float(np.quantile(p95s, .95)),
            "median_valid_projection_fraction": float(np.median(valid_fractions)),
        },
        "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)},
        "source_pose_direction": "cam2world",
        "source_camera_axes": "x_right_y_down_z_forward",
        "ustrf_geometry_input_admitted": False,
        "reason": "source-native reprojection measures depth/pose consistency only; body mapping, local ground receipt, and safety event truth remain unbound",
    }
    qa = root / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "temporal_reprojection_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps({"pairs": report["pair_count"], **report["aggregate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
