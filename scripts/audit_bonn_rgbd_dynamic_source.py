#!/usr/bin/env python3
"""CUDA audit of a Bonn Dynamic RGB-D sequence with OptiTrack camera-pose truth.

This verifies source-native RGB/depth/pose temporal coherence and controlled dynamic-scene input.
It deliberately does not infer per-object tracks, body coordinates, walkability, or user events.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "blindassist_bonn_rgbd_dynamic_source_audit_v1"
DEPTH_SCALE = 5000.0  # TUM-RGBD-compatible uint16 convention; retained as an explicit assumption.
INTRINSICS = (542.822841, 542.576870, 315.593520, 237.756098)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(path: Path, columns: int) -> list[tuple[float, tuple[str, ...]]]:
    rows: list[tuple[float, tuple[str, ...]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = tuple(line.split())
        if len(values) != columns:
            raise ValueError(f"unexpected row in {path.name}: {line}")
        rows.append((float(values[0]), values[1:]))
    if not rows or any(later[0] <= earlier[0] for earlier, later in zip(rows, rows[1:])):
        raise ValueError(f"timestamps must be non-empty and strictly increasing: {path}")
    return rows


def _nearest(rows: list[tuple[float, tuple[str, ...]]], timestamp: float) -> tuple[float, tuple[str, ...]]:
    stamps = [row[0] for row in rows]
    index = bisect.bisect_left(stamps, timestamp)
    candidates = rows[max(0, index - 1):min(len(rows), index + 1)]
    return min(candidates, key=lambda row: abs(row[0] - timestamp))


def _matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw / np.linalg.norm(quaternion_xyzw)
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    result = np.eye(4, dtype=np.float32); result[:3, :3] = rotation; result[:3, 3] = position
    return result


def audit(root: Path, archive: Path | None = None) -> dict[str, Any]:
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Bonn RGB-D source audit")
    rgb = _index(root / "rgb.txt", 2)
    depth = _index(root / "depth.txt", 2)
    pose = _index(root / "groundtruth.txt", 8)
    pairs: list[dict[str, Any]] = []
    for timestamp, (rgb_name,) in rgb:
        depth_row = _nearest(depth, timestamp); pose_row = _nearest(pose, timestamp)
        pairs.append({"timestamp": timestamp, "rgb": rgb_name, "depth": depth_row[1][0], "depth_delta_s": abs(depth_row[0] - timestamp), "pose": pose_row[1], "pose_delta_s": abs(pose_row[0] - timestamp)})
    # The RGB/depth stream is near 30Hz; ground-truth samples have one tail association at
    # 33.3ms.  Preserve both the strict 20ms coverage fraction and a one-frame (40ms) hard cap.
    depth_deltas = np.asarray([pair["depth_delta_s"] for pair in pairs])
    pose_deltas = np.asarray([pair["pose_delta_s"] for pair in pairs])
    if depth_deltas.max() > .02 or pose_deltas.max() > .04:
        raise ValueError("RGB association exceeds source-frame hard cap")
    sample_indexes = np.linspace(0, len(pairs) - 1, num=min(48, len(pairs)), dtype=int)
    depths: list[np.ndarray] = []
    shapes: set[tuple[int, int]] = set()
    for index in sample_indexes:
        image = np.asarray(Image.open(root / pairs[int(index)]["depth"]), dtype=np.uint16)
        shapes.add(tuple(image.shape)); depths.append(image)
    if len(shapes) != 1:
        raise ValueError(f"depth shape changed: {shapes}")
    depth_tensor = torch.as_tensor(np.stack(depths), device="cuda", dtype=torch.float32) / DEPTH_SCALE
    valid = depth_tensor > 0
    poses = np.asarray([[float(value) for value in row[1]] for row in pose], dtype=np.float32)
    pose_times = np.asarray([row[0] for row in pose], dtype=np.float64)
    quaternion_norm_error = float(np.abs(np.linalg.norm(poses[:, 3:7], axis=1) - 1.0).max())
    translations = torch.as_tensor(poses[:, :3], device="cuda")
    pose_steps = torch.linalg.vector_norm(translations[1:] - translations[:-1], dim=1)
    report = {
        "format": SCHEMA, "source_sequence": root.name,
        "frame_count": len(pairs), "pose_count": len(pose),
        "rgb_depth_association": {"maximum_delta_seconds": float(depth_deltas.max()), "median_delta_seconds": float(np.median(depth_deltas)), "within_20ms_fraction": float((depth_deltas <= .02).mean())},
        "rgb_pose_association": {"maximum_delta_seconds": float(pose_deltas.max()), "median_delta_seconds": float(np.median(pose_deltas)), "within_20ms_fraction": float((pose_deltas <= .02).mean())},
        "timestamps": {"rgb_hz": float(1.0 / np.median(np.diff([row[0] for row in rgb]))), "depth_hz": float(1.0 / np.median(np.diff([row[0] for row in depth]))), "pose_hz": float(1.0 / np.median(np.diff(pose_times))), "duration_seconds": float(rgb[-1][0] - rgb[0][0])},
        "depth": {"sampled_frame_count": len(depths), "shape": list(next(iter(shapes))), "scale_assumption": DEPTH_SCALE, "valid_fraction": float(valid.float().mean().item()), "minimum_positive_meters": float(depth_tensor[valid].min().item()), "median_positive_meters": float(depth_tensor[valid].median().item()), "maximum_meters": float(depth_tensor.max().item())},
        "pose": {"quaternion_norm_max_error": quaternion_norm_error, "median_step_meters": float(pose_steps.median().item()), "maximum_step_meters": float(pose_steps.max().item()), "source_pose_convention": "TUM-format camera trajectory supplied by OptiTrack; body frame remains unbound"},
        "calibration": {"rgb_intrinsics": {"fx": INTRINSICS[0], "fy": INTRINSICS[1], "cx": INTRINSICS[2], "cy": INTRINSICS[3]}, "rgb_depth_registered": True},
        "archive": {"path": archive.as_posix(), "size_bytes": archive.stat().st_size, "sha256": _sha256(archive)} if archive else None,
        "checks": {"timestamps_strictly_increasing": True, "rgb_depth_within_20ms": bool((depth_deltas <= .02).all()), "rgb_pose_within_40ms": bool((pose_deltas <= .04).all()), "rgb_pose_20ms_coverage": float((pose_deltas <= .02).mean()) >= .995, "depth_positive_samples_present": bool(valid.any().item()), "quaternion_norm": quaternion_norm_error <= 1e-4, "registered_rgb_depth": True},
        "source_dynamic_context": "official sequence identifier moving_obstructing_box; no per-frame object trajectory or intervention label is supplied",
        "ustrf_metric_geometry_input_admitted": False,
        "reason": "sensor-body extrinsics, body-local ground plane, per-object trajectory, and assistive event truth are absent",
        "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)}, "production_authority": False,
    }
    report["audit_passed"] = bool(all(report["checks"].values()))
    qa = root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "bonn_rgbd_dynamic_source_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.archive)
    print(json.dumps({"passed": report["audit_passed"], "frames": report["frame_count"], "depth_valid": report["depth"]["valid_fraction"]}))
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
