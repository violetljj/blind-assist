#!/usr/bin/env python3
"""Fail-closed structural audit for an extracted TartanAir RGB-depth-pose slice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def audit(root: Path, require_cuda: bool) -> dict[str, Any]:
    manifest_text = (root / "slice_manifest.json").read_text(encoding="utf-8").strip()
    # Older extraction receipts may end with a literal escaped newline from a shell handoff.
    manifest = json.loads(manifest_text.removesuffix("\\n"))
    trajectory = root / manifest["trajectory"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    depths: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    intrinsics: np.ndarray | None = None
    for frame_id in manifest["frame_ids"]:
        depth_path = trajectory / f"{frame_id}_depth.npy"
        camera_path = trajectory / f"{frame_id}_cam.npz"
        rgb_path = trajectory / f"{frame_id}_rgb.png"
        if not (depth_path.is_file() and camera_path.is_file() and rgb_path.is_file()):
            errors.append(f"missing_modalities:{frame_id}")
            continue
        depth = np.load(depth_path)
        camera = np.load(camera_path)
        pose = np.asarray(camera["camera_pose"], dtype=np.float64)
        current_intrinsics = np.asarray(camera["camera_intrinsics"], dtype=np.float64)
        if depth.ndim != 2 or depth.dtype != np.float32 or not np.isfinite(depth).all() or (depth <= 0).any():
            errors.append(f"invalid_depth:{frame_id}")
        if pose.shape != (4, 4) or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6): errors.append(f"invalid_pose_shape:{frame_id}")
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-4): errors.append(f"invalid_pose_rotation:{frame_id}")
        if intrinsics is None: intrinsics = current_intrinsics
        elif not np.allclose(intrinsics, current_intrinsics, atol=1e-6): errors.append(f"intrinsics_changed:{frame_id}")
        depths.append(depth); translations.append(pose[:3, 3]); rows.append({"frame_id": frame_id, "depth_path": str(depth_path.relative_to(root))})
    motion = [float(np.linalg.norm(b - a)) for a, b in zip(translations, translations[1:])]
    backend: dict[str, Any] = {"name": "numpy", "cuda": False}
    if require_cuda:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
        values = torch.as_tensor(np.stack(depths), device="cuda")
        depth_stats = {"minimum_m": float(values.min().cpu()), "median_m": float(values.median().cpu()), "positive_fraction": float((values > 0).float().mean().cpu())}
        backend = {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)}
    else:
        values = np.stack(depths); depth_stats = {"minimum_m": float(values.min()), "median_m": float(np.median(values)), "positive_fraction": float((values > 0).mean())}
    report = {
        "format": "blindassist_tartanair_slice_structural_audit_v1", "ok": not errors, "errors": errors,
        "source": manifest["source_archive"], "trajectory": manifest["trajectory"], "frame_count": len(rows),
        "depth_shape": list(depths[0].shape) if depths else None, "intrinsics": intrinsics.tolist() if intrinsics is not None else None,
        "relative_translation_meters": motion, "depth_stats": depth_stats, "compute_backend": backend,
        "official_pose_direction": "cam2world", "official_camera_axes": "x_right_y_down_z_forward",
        "ustrf_body_mapping_admitted": False,
        "ustrf_body_mapping_reason": "cam2world and optical axes are source-defined, but conversion to USTRF body-up and a local ground-plane receipt is not independently bound in this slice",
    }
    qa = root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "structural_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = audit(args.root, args.require_cuda)
    print(json.dumps({"ok": report["ok"], "frames": report["frame_count"], "mapping_admitted": report["ustrf_body_mapping_admitted"]}))
    return 0 if report["ok"] else 1

if __name__ == "__main__": raise SystemExit(main())
