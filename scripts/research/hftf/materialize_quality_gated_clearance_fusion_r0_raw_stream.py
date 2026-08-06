#!/usr/bin/env python3
"""Materialize the frozen R0 raw geometry stream for three new parents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_raw_geometry_stream"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def bound(path: Path, expected: str) -> None:
    require(path.is_file(), f"missing bound file: {path}")
    require(sha256_file(path) == expected.upper(), f"SHA mismatch: {path}")


def state(value: float | None, valid: bool) -> str:
    if not valid or value is None or not math.isfinite(float(value)):
        return "UNKNOWN"
    return "OCCUPIED" if float(value) <= 1.5 else "CLEAR"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--arkit-root", type=Path, required=True)
    parser.add_argument("--a2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "overwrite forbidden")
    root = args.repo_root.resolve()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    catalog = json.loads(args.source_catalog.read_text(encoding="utf-8"))
    require(protocol.get("schema") == "blindassist_quality_gated_clearance_fusion_r0_raw_stream_materialization_protocol", "materialization protocol schema drift")
    require(catalog.get("schema") == "blindassist_quality_gated_clearance_fusion_r0_source_catalog", "source catalog schema drift")
    require(catalog.get("labels_opened") is False, "labels boundary violated")
    checkpoint = args.a2_checkpoint.resolve()
    bound(checkpoint, protocol["a2_binding"]["checkpoint_sha256"])
    # All static joins and output absence checks happen before ML imports.
    try:
        import cv2
        import torch
        sys.path.insert(0, str((root / "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main/metric_depth").resolve()))
        from depth_anything_v2.dpt import DepthAnythingV2
        from evaluate_metric3d_clearance_field_a0 import clearance_field
    except Exception as error:
        raise RuntimeError(f"runtime unavailable after static checks: {type(error).__name__}: {error}") from error
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    tum_root = args.tum_root.resolve()
    arkit_root = args.arkit_root.resolve()
    tum_intrinsics = [535.4, 539.2, 320.1, 247.6]
    rows = []
    by_parent = {}
    for frame in catalog["frames"]:
        by_parent.setdefault(frame["parent_id"], []).append(frame)
    for parent_id, frames in sorted(by_parent.items()):
        tar_path = tum_root / f"{parent_id}.tgz"
        tar = tarfile.open(tar_path, "r:gz") if parent_id != "381644" else None
        zips = {}
        if parent_id == "381644":
            videos = sorted({str(row["video_id"]) for row in frames})
            for video in videos:
                base = arkit_root / "raw" / "Validation" / video
                zips[video] = tuple(zipfile.ZipFile(base / name) for name in ("lowres_wide.zip", "lowres_depth.zip", "confidence.zip"))
        try:
            for index, frame in enumerate(sorted(frames, key=lambda row: row["timestamp_ns"])):
                if parent_id == "381644":
                    rgb_zip, depth_zip, conf_zip = zips[str(frame["video_id"])]
                    rgb_bytes = rgb_zip.read(frame["rgb_member"])
                    depth_bytes = depth_zip.read(frame["depth_member"])
                    depth_scale = 0.001
                    intrinsics = tum_intrinsics
                    confidence = None
                else:
                    prefix = parent_id + "/"
                    rgb_bytes = tar.extractfile(prefix + frame["rgb_member"]).read()
                    depth_bytes = tar.extractfile(prefix + frame["depth_member"]).read()
                    depth_scale = 1.0 / 5000.0
                    intrinsics = tum_intrinsics
                    confidence = None
                bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
                raw_depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                require(bgr is not None and raw_depth is not None, "media decode failed")
                metric = np.asarray(raw_depth, dtype=np.float32) * depth_scale
                tensor, _ = model.image2tensor(bgr, 392)
                with torch.inference_mode():
                    pred = model(tensor.to(device))
                pred = torch.nn.functional.interpolate(pred[:, None], size=metric.shape, mode="bilinear", align_corners=True)[0, 0].float().cpu().numpy()
                valid = np.isfinite(metric) & (metric > 0.1) & (metric <= 20.0) & np.isfinite(pred) & (pred > 0.1)
                disagreement = float(np.mean(np.abs(np.log(np.clip(pred[valid], .1, 20)) - np.log(np.clip(metric[valid], .1, 20))))) if bool(valid.any()) else None
                field = clearance_field(pred, np.asarray([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]], dtype=np.float32), confidence_map=confidence)
                clearances = [field.get("bands", {}).get(name, {}).get("clearance_m") for name in ("left", "center", "right")] if field.get("status") == "VALID" else [None, None, None]
                valids = [value is not None and math.isfinite(float(value)) for value in clearances]
                rows.append({"schema": SCHEMA, "frame_id": frame["frame_id"], "parent_id": frame["parent_id"], "video_id": frame["video_id"], "timestamp_ns": frame["timestamp_ns"], "raw_clearance_m": clearances, "raw_geometry_valid": valids, "raw_geometry_state": [state(value, ok) for value, ok in zip(clearances, valids)], "tof_valid": bool(valid.any()), "teacher_age_s": 0.0, "frozen_a2_disagreement": disagreement, "rgb_sha256": sha256_bytes(rgb_bytes), "metric_depth_sha256": sha256_bytes(depth_bytes), "geometry_status": field.get("status")})
        finally:
            if tar is not None:
                tar.close()
            for handles in zips.values():
                for handle in handles:
                    handle.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": SCHEMA, "frame_count": len(rows), "sha256": sha256_bytes(payload), "model_loaded": True, "p3_model_constructed": False, "optimizer_constructed": False, "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
