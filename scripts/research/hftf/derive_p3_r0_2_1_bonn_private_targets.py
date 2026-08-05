#!/usr/bin/env python3
"""Derive private P3 R0.2.1 truth from locked Bonn RGB-D identities."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_metric3d_clearance_field_a0 import clearance_field
from p3_r0_2_1_sealing_common import exact_fields, exclusive_write, load_json, materialization_receipt, pretty_bytes, require, sha256_file, verify_bound_file
from prepare_bonn_rgbd_metric_depth_manifest import BONN_INTRINSICS, normalize_depth_image, read_tum_index


REQUEST_SCHEMA = "blindassist_p3_r0_2_1_bonn_private_target_request"
PRIVATE_SCHEMA = "blindassist_p3_r0_2_1_private_holdout_targets"
RECEIPT_SCHEMA = "blindassist_p3_r0_2_1_private_target_materialization_receipt"
PUBLIC_FRAME_FIELDS = {"frame_id", "video_id", "parent_id", "timestamp_ns", "sealed_target_id", "rgb_identity", "rgb_sha256"}


def depth_index(sequence: Path) -> tuple[list[tuple[float, Path]], list[float]]:
    rows = [(stamp, path) for stamp, path in read_tum_index(sequence / "depth.txt") if (sequence / path).is_file()]
    return rows, [row[0] for row in rows]


def nearest_depth(rows: list[tuple[float, Path]], stamps: list[float], rgb_timestamp: float) -> tuple[float, Path]:
    index = bisect.bisect_left(stamps, rgb_timestamp)
    candidates = [row for i in (index - 1, index) if 0 <= i < len(rows) for row in [rows[i]]]
    require(bool(candidates), "no valid depth candidate")
    return min(candidates, key=lambda row: abs(row[0] - rgb_timestamp))


def state_for(clearance: float | None) -> str:
    if clearance is None:
        return "UNKNOWN_GROUND"
    return "OCCUPIED" if clearance <= 1.5 else "CLEAR"


def derive_frame(sequence: Path, depth_rows: list[tuple[float, Path]], depth_stamps: list[float], frame: dict[str, Any]) -> dict[str, Any]:
    exact_fields(frame, PUBLIC_FRAME_FIELDS, "public frame")
    rgb_path = sequence / frame["rgb_identity"]
    require(rgb_path.is_file() and sha256_file(rgb_path) == frame["rgb_sha256"], "RGB identity mismatch")
    rgb_timestamp = float(str(frame["frame_id"]).rsplit(":", 1)[-1])
    depth_timestamp, depth_relative = nearest_depth(depth_rows, depth_stamps, rgb_timestamp)
    require(abs(depth_timestamp - rgb_timestamp) <= 0.05, "RGB-depth association drift")
    depth_path = sequence / depth_relative
    raw = normalize_depth_image(cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED), depth_path)
    depth = raw.astype(np.float32) / 5000.0
    fx, fy, cx, cy = BONN_INTRINSICS
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    field = clearance_field(depth, intrinsics)
    bands = []
    for name in ("left", "center", "right"):
        value = field.get("bands", {}).get(name, {}).get("clearance_m") if field.get("status") == "VALID" else None
        clearance = float(value) if value is not None else None
        bands.append({"clearance_m": clearance, "geometry_state": state_for(clearance), "geometry_target_valid": True})
    return frame | {
        "independent_metric_sensor_valid": True,
        "truth_depth_ref": depth_relative.as_posix(),
        "truth_depth_sha256": sha256_file(depth_path),
        "rgb_depth_timestamp_delta_ns": int(round(abs(depth_timestamp - rgb_timestamp) * 1_000_000_000)),
        "bands": bands,
    }


def build(repo_root: Path, request: dict[str, Any], source_path: Path) -> None:
    exact_fields(request, {"schema", "protocol", "public_holdout_manifest", "bonn_dataset_root", "producer_sha256", "outputs"}, "request")
    require(request["schema"] == REQUEST_SCHEMA, "request schema drift")
    require(sha256_file(source_path) == request["producer_sha256"], "producer SHA mismatch")
    protocol_path = verify_bound_file(repo_root, request["protocol"], "protocol")
    public_path = verify_bound_file(repo_root, request["public_holdout_manifest"], "public holdout manifest")
    public = load_json(public_path)
    require(public.get("role") == "public_holdout" and public.get("outcomes_opened") is False, "public holdout boundary drift")
    root = (repo_root / request["bonn_dataset_root"]).resolve()
    require(root.is_dir(), "Bonn dataset root missing")
    private_clips = []
    depth_by_parent: dict[str, tuple[list[tuple[float, Path]], list[float]]] = {}
    for clip in public["clips"]:
        sequence = root / clip["parent_id"]
        if clip["parent_id"] not in depth_by_parent:
            depth_by_parent[clip["parent_id"]] = depth_index(sequence)
        depth_rows, depth_stamps = depth_by_parent[clip["parent_id"]]
        frames = [derive_frame(sequence, depth_rows, depth_stamps, frame) for frame in clip["frames"]]
        private_clips.append({key: clip[key] for key in ("clip_id", "video_id", "parent_id")} | {"frames": frames})
    value = {"schema": PRIVATE_SCHEMA, "protocol_sha256": sha256_file(protocol_path), "clips": private_clips}
    exact_fields(request["outputs"], {"private_targets", "receipt"}, "outputs")
    payload = pretty_bytes(value)
    outputs = {"private_targets": (request["outputs"]["private_targets"], payload)}
    receipt = materialization_receipt(RECEIPT_SCHEMA, request["producer_sha256"], {"protocol": sha256_file(protocol_path), "public_holdout_manifest": sha256_file(public_path)}, outputs)
    exclusive_write(repo_root / outputs["private_targets"][0], payload)
    exclusive_write(repo_root / request["outputs"]["receipt"], pretty_bytes(receipt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root.resolve(), json.loads(args.request.read_text(encoding="utf-8")), Path(__file__).resolve())


if __name__ == "__main__":
    main()
