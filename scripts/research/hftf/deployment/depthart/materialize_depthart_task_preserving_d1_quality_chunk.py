#!/usr/bin/env python3
"""Materialize one frozen D1 reference/truth/input chunk after explicit activation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (  # noqa: E402
    TruthReaderPolicy,
    derive_assistive_truth,
    load_manifest_frame,
    parse_trajectory,
)
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    load_json,
    require,
    sha256_file,
)
from scripts.research.hftf.deployment.depthart.export_depthart_camera_external import (  # noqa: E402
    install_timm_compat,
)


INPUT_NAMES = ("image", "camera_prompt_4", "camera_prompt_8", "camera_prompt_16", "camera_prompt_32")
BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
TASK_HORIZON_M = 2.0


def chunk_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    chunk_size = int(protocol["execution"]["chunk_size_frames"])
    require(chunk_size == 50 and 300 % chunk_size == 0, "chunk size drift")
    return [
        {"chunk_index": session_index * (300 // chunk_size) + start // chunk_size,
         "session_index": session_index, "visit_id": identity["visit_id"],
         "video_id": identity["video_id"], "frame_start": start, "frame_stop": start + chunk_size}
        for session_index, identity in enumerate(protocol["cohort"]["ordered_sessions"])
        for start in range(0, 300, chunk_size)
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"output already exists: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_raw(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"output already exists: {path}")
    array = np.ascontiguousarray(value, dtype=np.float32)
    with temporary.open("xb") as handle:
        array.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "shape": list(array.shape), "dtype": "float32"}


def source_videos(manifest: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {str(row["video_id"]): row for row in manifest["videos"]}
    rows = []
    for identity in protocol["cohort"]["ordered_sessions"]:
        video = lookup[str(identity["video_id"])]
        require(str(video["visit_id"]) == str(identity["visit_id"]), "visit/video binding drift")
        require(video.get("eligible") is True and int(video["selected_frame_count"]) == 300,
                "selected video eligibility drift")
        rows.append(video)
    return rows


def clearance_payload(band: dict[str, Any] | None) -> dict[str, Any]:
    band = band or {}
    value = band.get("clearance_m")
    occupied = band.get("occupied_by_horizon", {})
    if value is not None and np.isfinite(value):
        return {"clearance_valid": True, "clearance_m": min(float(value), TASK_HORIZON_M)}
    if all(occupied.get(str(horizon)) is False for horizon in HORIZONS):
        return {"clearance_valid": True, "clearance_m": TASK_HORIZON_M}
    return {"clearance_valid": False, "clearance_m": None}


def state(value: bool | None) -> str:
    if value is None:
        return "UNKNOWN_GROUND"
    return "OCCUPIED" if value else "CLEAR"


def observation(identity: dict[str, Any], orientation: dict[str, Any], intrinsics_tensor: np.ndarray,
                truth: dict[str, Any], reference: dict[str, Any], reference_raw: dict[str, Any],
                inputs: dict[str, Any]) -> dict[str, Any]:
    bands = []
    for name in BANDS:
        truth_band = truth.get("bands", {}).get(name)
        reference_band = reference.get("bands", {}).get(name)
        cells = []
        for horizon in HORIZONS:
            cells.append({
                "horizon_m": horizon,
                "truth": {"state": state((truth_band or {}).get("occupied_by_horizon", {}).get(str(horizon)))},
                "reference": {"state": state((reference_band or {}).get("occupied_by_horizon", {}).get(str(horizon)))},
            })
        bands.append({"band": name, "truth": clearance_payload(truth_band),
                      "reference": clearance_payload(reference_band), "cells": cells})
    return {
        "parent_id": identity["visit_id"], "session_id": identity["video_id"],
        "frame_id": identity["frame_stem"],
        "frame_index": identity["frame_index"],
        "timestamp_ns": int(Decimal(identity["frame_stem"].rsplit("_", 1)[1]) * 1_000_000_000),
        "orientation": "portrait", "orientation_index": int(orientation["rotation_index"]),
        "up_camera": orientation["up_camera"], "intrinsics_tensor": intrinsics_tensor.tolist(), "bands": bands,
        "reference_depth": reference_raw, "candidate_inputs": inputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    args = parser.parse_args()
    protocol_path, activation_path, manifest_path = (args.protocol.resolve(), args.activation_receipt.resolve(),
                                                      args.source_manifest.resolve())
    protocol, activation, manifest = (load_json(protocol_path), load_json(activation_path), load_json(manifest_path))
    require(protocol.get("protocol_id") == "DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN", "protocol drift")
    require(activation.get("status") == "OUTCOME_ACCESS_ACTIVATED" and activation.get("execution_authorized") is True,
            "D1 outcome access is not activated")
    require(activation["protocol_sha256"] == sha256_file(protocol_path), "activation/protocol SHA drift")
    require(protocol["bindings"]["development_manifest"]["sha256"] == sha256_file(manifest_path),
            "Development manifest SHA drift")
    require(protocol["bindings"]["reference_checkpoint"]["sha256"] == sha256_file(args.checkpoint.resolve()),
            "reference checkpoint SHA drift")
    require(protocol["bindings"]["materializer"]["sha256"] == sha256_file(Path(__file__)),
            "materializer SHA drift")
    frozen_policy = json.loads(json.dumps(asdict(TruthReaderPolicy())))
    require(protocol["task_postprocess"]["truth_reader_policy"] == frozen_policy,
            "truth reader policy drift")
    chunks = chunk_schedule(protocol)
    require(0 <= args.chunk_index < len(chunks), "chunk index outside frozen schedule")
    chunk = chunks[args.chunk_index]
    chunk_root = args.output_root.resolve() / f"chunk-{args.chunk_index:02d}"
    receipt_path = chunk_root / "materialization-receipt.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        require(receipt["protocol_sha256"] == sha256_file(protocol_path), "resume protocol drift")
        for record in receipt["records"]:
            for item in (*record["candidate_inputs"].values(), record["reference_depth"]):
                path = Path(item["path"])
                require(path.is_file() and path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"],
                        f"resume file drift: {path}")
        print(json.dumps({"chunk": args.chunk_index, "status": "RESUMED_VALID", "frames": len(receipt["records"])}))
        return 0
    require(not chunk_root.exists(), f"incomplete chunk exists without receipt: {chunk_root}")
    chunk_root.mkdir(parents=True)

    install_timm_compat()
    source = args.source_root.resolve()
    require(subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True,
                           text=True, check=True).stdout.strip() == protocol["reference"]["source_git_commit"],
            "DepthART source commit drift")
    require(not subprocess.run(["git", "-C", str(source), "status", "--short"], capture_output=True,
                               text=True, check=True).stdout.strip(), "DepthART source tree is dirty")
    versions = protocol["reference"]["host_runtime"]
    require(torch.__version__ == versions["torch"] and torch.version.cuda == versions["cuda"] and
            cv2.__version__ == versions["opencv"] and np.__version__ == versions["numpy"],
            "host inference runtime drift")
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from common import preprocess  # type: ignore
    from depthart_selective_scan import install_depthart  # type: ignore
    from model import load_model  # type: ignore
    from network import tvimblock  # type: ignore

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    model = load_model(args.checkpoint.resolve(), "S", "indoor", "cuda").eval()
    install_depthart(tvimblock)
    videos = source_videos(manifest, protocol)
    video = videos[int(chunk["session_index"])]
    require(str(video["video_id"]) == str(chunk["video_id"]), "chunk/video schedule drift")
    trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
    records = []
    input_lines = []
    for frame_index in range(int(chunk["frame_start"]), int(chunk["frame_stop"])):
        frame = load_manifest_frame(video, frame_index, trajectory, TruthReaderPolicy())
        require(int(frame["orientation"]["rotation_index"]) in (1, 3), "non-portrait frame crossed D1 gate")
        rgb = np.asarray(frame["rgb_upright"], dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image, intrinsics = preprocess(bgr, np.asarray(frame["intrinsics_upright"], dtype=np.float32), 448, 448)
        require(tuple(image.shape) == (1, 3, 608, 448), "tensor shape drift")
        image, intrinsics = image.cuda(), intrinsics.cuda()
        cameras = model.cam_embedder(intrinsics, 608, 448, "cuda")
        with torch.inference_mode():
            reference_depth = model(image, intrinsics)
        depth = reference_depth.detach().cpu().numpy().astype(np.float32, copy=False)
        require(depth.shape == (1, 608, 448) and np.all(np.isfinite(depth)), "reference depth invalid")
        k_tensor = intrinsics.detach().cpu().numpy()[0]
        reference_geometry = derive_assistive_truth(
            depth[0], np.full((608, 448), 2, dtype=np.uint8), k_tensor,
            np.asarray(frame["orientation"]["up_camera"], dtype=np.float64), TruthReaderPolicy(),
        )
        stem = frame["identity"]["frame_stem"]
        frame_root = chunk_root / "inputs" / stem
        arrays = {"image": image.detach().cpu().numpy(),
                  **{name: value.detach().cpu().numpy() for name, value in zip(INPUT_NAMES[1:], cameras, strict=True)}}
        input_receipts = {name: atomic_raw(frame_root / f"{name}.raw", arrays[name]) for name in INPUT_NAMES}
        reference_raw = atomic_raw(chunk_root / "reference-depth" / f"{stem}.raw", depth)
        records.append(observation(frame["identity"], frame["orientation"], k_tensor, frame["truth"],
                                   reference_geometry, reference_raw, input_receipts))
        input_lines.append(" ".join(f"{name}:=inputs/{stem}/{name}.raw" for name in INPUT_NAMES))
        print(json.dumps({"chunk": args.chunk_index, "frame": frame_index, "stem": stem}), flush=True)

    input_list = chunk_root / "input-list.txt"
    input_list.write_text("\n".join(input_lines) + "\n", encoding="utf-8", newline="\n")
    receipt = {
        "schema": "blindassist_depthart_task_preserving_d1_quality_chunk_materialization_v1",
        "protocol_sha256": sha256_file(protocol_path), "activation_receipt_sha256": sha256_file(activation_path),
        "development_manifest_sha256": sha256_file(manifest_path), "chunk": chunk,
        "input_list": {"path": str(input_list.resolve()), "bytes": input_list.stat().st_size,
                       "sha256": sha256_file(input_list)},
        "records": records, "truth_reference_outcome_accessed": True,
        "candidate_outcome_accessed": False, "r2_cohort_accessed": False,
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps({"chunk": args.chunk_index, "status": "MATERIALIZED", "frames": len(records),
                      "receipt_sha256": sha256_file(receipt_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
