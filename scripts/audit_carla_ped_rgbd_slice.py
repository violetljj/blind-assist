#!/usr/bin/env python3
"""CUDA structural audit for a synchronized CARLA pedestrian RGB-D camera slice.

This verifies source-native modality/time/pose coherence only.  It does not assert a human-body
mount, local ground plane, target trajectory, traversability, drop, or a USTRF safety input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _files(root: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in root.glob("*.camera.json"):
        result.append((int(path.name.rsplit("_", 1)[1].split(".")[0]), path))
    return sorted(result)


def audit(root: Path) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for CARLA RGB-D slice audit")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    depths: list[np.ndarray] = []
    intrinsics: np.ndarray | None = None
    timestamps: list[float] = []
    translations: list[np.ndarray] = []
    handedness: list[float] = []
    configurations: list[dict[str, Any]] = []
    for frame_id, camera_path in _files(root):
        base = camera_path.name.removesuffix(".camera.json")
        depth_path = root / f"{base}.depth.npy"
        rgb_path = root / f"{base}.rgb.png"
        metadata_path = root / f"{base}.metadata.json"
        if not (depth_path.is_file() and rgb_path.is_file() and metadata_path.is_file()):
            errors.append(f"missing_modalities:{frame_id}")
            continue
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        depth = np.load(depth_path)
        c2w = np.asarray(camera["extrinsic"]["c2w"], dtype=np.float64)
        w2c = np.asarray(camera["extrinsic"]["w2c"], dtype=np.float64)
        current_intrinsics = np.asarray(camera["intrinsic"]["K"], dtype=np.float64)
        if depth.ndim != 2 or depth.dtype != np.float32 or not np.isfinite(depth).all() or (depth <= 0).any():
            errors.append(f"invalid_depth:{frame_id}")
        if c2w.shape != (4, 4) or w2c.shape != (4, 4) or not np.allclose(c2w @ w2c, np.eye(4), atol=1e-5):
            errors.append(f"invalid_extrinsic_inverse:{frame_id}")
        # CARLA's supplied camera matrix may include the image-axis reflection
        # (det=-1).  For a source-native audit, an orthonormal, invertible frame
        # is valid; handedness must remain explicit until a body-frame adapter is
        # independently received and tested.
        determinant = np.linalg.det(c2w[:3, :3])
        if not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(abs(determinant), 1.0, atol=1e-5):
            errors.append(f"invalid_rotation:{frame_id}")
        if intrinsics is None:
            intrinsics = current_intrinsics
        elif not np.allclose(intrinsics, current_intrinsics, atol=1e-6):
            errors.append(f"intrinsics_changed:{frame_id}")
        if int(camera["frame_id"]) != frame_id or int(metadata["frame_id"]) != frame_id:
            errors.append(f"frame_id_binding:{frame_id}")
        depths.append(depth); timestamps.append(float(camera["timestamp"])); translations.append(c2w[:3, 3]); handedness.append(float(determinant)); configurations.append(metadata["config"])
        rows.append({"frame_id": frame_id, "depth_path": depth_path.name, "rgb_path": rgb_path.name, "timestamp": float(camera["timestamp"])})
    if not rows:
        raise ValueError("no valid CARLA RGB-D rows")
    intervals = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    if any(interval <= 0.0 for interval in intervals): errors.append("timestamps_not_monotonic")
    if len({json.dumps(config, sort_keys=True) for config in configurations}) != 1: errors.append("config_changed")
    configuration = configurations[0]
    device = torch.device("cuda")
    all_depth = torch.as_tensor(np.stack(depths), device=device)
    motion = [float(np.linalg.norm(later - earlier)) for earlier, later in zip(translations, translations[1:])]
    report = {
        "format": "blindassist_carla_ped_rgbd_slice_audit_v1",
        "ok": not errors,
        "errors": errors,
        "frame_count": len(rows),
        "frames": rows,
        "depth": {"shape": list(depths[0].shape), "dtype": str(depths[0].dtype), "minimum_m": float(all_depth.min().item()), "median_m": float(all_depth.median().item()), "maximum_m": float(all_depth.max().item())},
        "intrinsics": intrinsics.tolist() if intrinsics is not None else None,
        "timestamp": {"source_field": "camera.timestamp", "median_interval_seconds": float(np.median(intervals)) if intervals else None, "min_interval_seconds": float(min(intervals)) if intervals else None, "max_interval_seconds": float(max(intervals)) if intervals else None, "metadata_capture_fps": configuration["capture"]["fps"], "metadata_frame_skip": configuration["capture"]["frame_skip"]},
        "camera_motion": {"median_translation_m": float(np.median(motion)) if motion else 0.0, "max_translation_m": float(max(motion)) if motion else 0.0},
        "source_pose_convention": "camera c2w/w2c supplied by dataset; camera optical/depth convention and handedness have not been independently bound to USTRF body coordinates",
        "camera_coordinate_handedness": "left_handed_or_image_reflected" if handedness[0] < 0.0 else "right_handed",
        "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)},
        "source_rgbd_pose_sequence_admitted": not errors,
        "ustrf_metric_geometry_input_admitted": False,
        "reason": "no independent device/human mount receipt, local ground-plane truth, object trajectory truth, or drop/head event labels",
    }
    qa = root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "carla_rgbd_slice_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps({"ok": report["ok"], "frames": report["frame_count"], "interval_s": report["timestamp"]["median_interval_seconds"], "source_admitted": report["source_rgbd_pose_sequence_admitted"]}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
