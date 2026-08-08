#!/usr/bin/env python3
"""Materialize the frozen D0 RGB roster into shared multi-input quantizer raws."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from .export_depthart_camera_external import install_timm_compat
except ImportError:
    from export_depthart_camera_external import install_timm_compat


ROSTER_SCHEMA = "blindassist_depthart_task_preserving_d0_tum_calibration_roster_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_raw(path: Path, value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=np.float32)
    array.tofile(path)
    return {
        "path": str(path.resolve()),
        "shape": list(array.shape),
        "dtype": "float32",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "finite": bool(np.isfinite(array).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--depthart-source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=448)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    if roster.get("schema") != ROSTER_SCHEMA or roster.get("status") != "FROZEN_BEFORE_CALIBRATION_MATERIALIZATION":
        raise ValueError("calibration roster is not frozen")
    if roster.get("r2_arkit_roster_accessed") is not False:
        raise ValueError("R2 cohort boundary changed")

    install_timm_compat()
    source = args.depthart_source.resolve()
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    from common import make_K, preprocess  # type: ignore
    from model import load_model  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("frozen calibration materializer requires CUDA")
    model = load_model(args.checkpoint.resolve(), "S", "indoor", device).eval()
    args.output_root.mkdir(parents=True)
    input_lines: list[str] = []
    records: list[dict[str, Any]] = []
    input_names = ("image", "camera_prompt_4", "camera_prompt_8", "camera_prompt_16", "camera_prompt_32")
    for index, row in enumerate(roster["rows"]):
        rgb_path = args.source_root / row["sequence_root"] / row["rgb_path"]
        if sha256(rgb_path) != row["rgb_sha256"]:
            raise ValueError(f"RGB hash mismatch: {row['calibration_id']}")
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode RGB: {rgb_path}")
        fx, fy, cx, cy = row["intrinsics_fx_fy_cx_cy"]
        image, intrinsics = preprocess(
            bgr, make_K(fx, fy, cx, cy), args.resolution, args.resolution
        )
        cameras = model.cam_embedder(
            intrinsics.to(device), args.resolution, args.resolution, device
        )
        arrays = {
            "image": image.detach().cpu().numpy(),
            "camera_prompt_4": cameras[0].detach().cpu().numpy(),
            "camera_prompt_8": cameras[1].detach().cpu().numpy(),
            "camera_prompt_16": cameras[2].detach().cpu().numpy(),
            "camera_prompt_32": cameras[3].detach().cpu().numpy(),
        }
        frame_root = args.output_root / f"frame-{index:03d}"
        frame_root.mkdir()
        files = {
            name: write_raw(frame_root / f"{name}.raw", arrays[name])
            for name in input_names
        }
        input_lines.append(" ".join(f"{name}:={files[name]['path']}" for name in input_names))
        records.append({
            "index": index,
            "calibration_id": row["calibration_id"],
            "rgb_sha256": row["rgb_sha256"],
            "files": files,
        })

    input_list = args.output_root / "calibration-input-list.txt"
    input_list.write_text("\n".join(input_lines) + "\n", encoding="utf-8")
    receipt = {
        "schema": "blindassist_depthart_task_preserving_d0_calibration_materialization_receipt_v1",
        "protocol_id": "DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN",
        "status": "CALIBRATION_INPUTS_MATERIALIZED_NO_MODEL_OUTCOMES",
        "device": device,
        "frames": len(records),
        "arms": ["D0_W8A16_R0", "D0_INT8_R0"],
        "roster_sha256": sha256(args.roster),
        "checkpoint_sha256": sha256(args.checkpoint),
        "input_list": {
            "path": str(input_list.resolve()),
            "bytes": input_list.stat().st_size,
            "sha256": sha256(input_list),
        },
        "records": records,
        "r2_cohort_accessed": False,
        "task_truth_accessed": False,
        "model_outcomes_accessed": False,
        "authority": "Shared D0 quantizer calibration inputs only.",
    }
    receipt_path = args.output_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "frames": receipt["frames"],
        "input_list_sha256": receipt["input_list"]["sha256"],
        "receipt_sha256": sha256(receipt_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
