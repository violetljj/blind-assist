#!/usr/bin/env python3
"""Materialize hash-bound exact timestamp frames for the selected R1.2b GPU candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
from PIL import Image

from contract import load_json, sha256_file, write_json
from diagnostic_contract import require


TRANSPORT_SCHEMA = "blindassist_ustrf_crosscam_exact_frame_transport_preregistration_v1"
INPUT_SCHEMA = "blindassist_ustrf_crosscam_exact_frame_android_input_v1"
REMOTE_ROOT = "ustrf-crosscam-r12b"


def raw_rgb_sha256(path: Path) -> str:
    with Image.open(path) as image:
        payload = image.convert("RGB").tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--transport-contract", type=Path, required=True)
    parser.add_argument("--r12a-input", type=Path, required=True)
    parser.add_argument("--r12a-staging-receipt", type=Path, required=True)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    contract = load_json(args.transport_contract)
    require(contract.get("schema") == TRANSPORT_SCHEMA, "transport contract schema mismatch")
    require(sha256_file(args.r12a_input) == contract["parent"]["r12a_android_input_sha256"], "R1.2a input drift")
    decoder = contract["transport"]["decoder"]
    if decoder == "ffmpeg 8.1.2 essentials build":
        require(args.ffmpeg is not None, "ffmpeg path required by transport contract")
        require(sha256_file(args.ffmpeg) == contract["transport"]["decoder_executable_sha256"], "ffmpeg drift")
    elif decoder == "opencv VideoCapture CAP_PROP_POS_MSEC":
        require(cv2.__version__ == contract["transport"]["decoder_version"], "OpenCV version drift")
    else:
        raise ValueError(f"unsupported decoder: {decoder}")
    parent = load_json(args.r12a_input)
    staging = load_json(args.r12a_staging_receipt)
    canary = load_json(args.canary_manifest)
    require(parent["dataset_role"] == "seen_diagnostic_not_held_out", "parent role drift")
    require(parent["authority"]["new_held_out_read"] is False, "parent opened new held-out")
    host_by_remote = {row["device_relative_path"].replace("\\", "/"): Path(row["host_path"]).resolve()
                      for row in staging["files"]}
    sample_period = int(contract["transport"]["regular_sample_period_ms"])
    output_sources, staged_files = [], []
    frame_lookup: dict[tuple[str, int], Path] = {}
    for source in parent["sources"]:
        video_remote = source["video_path"].replace("\\", "/")
        video = host_by_remote.get(video_remote)
        require(video is not None and video.is_file(), f"{source['event_id']}: host video missing")
        require(sha256_file(video) == source["video_sha256"], f"{source['event_id']}: video SHA mismatch")
        start_ms, end_ms = map(int, source["clip_window_ms"])
        timestamps = set(range(start_ms, end_ms + 1, sample_period))
        timestamps.update(int(row["timestamp_ms"]) for row in source["target_anchors"]
                          if start_ms <= int(row["timestamp_ms"]) <= end_ms)
        frame_rows = []
        capture = cv2.VideoCapture(str(video)) if decoder == "opencv VideoCapture CAP_PROP_POS_MSEC" else None
        if capture is not None:
            require(capture.isOpened(), f"{source['event_id']}: OpenCV cannot open source")
        try:
            for timestamp_ms in sorted(timestamps):
                frame_path = args.output_dir / "frames" / source["event_id"] / f"f{timestamp_ms:09d}.png"
                require(not frame_path.exists(), f"refusing to overwrite exact frame: {frame_path}")
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                if capture is not None:
                    capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
                    ok, image = capture.read()
                    require(ok and image is not None, f"{source['event_id']}/{timestamp_ms}: OpenCV decode failed")
                    require(cv2.imwrite(str(frame_path), image), f"{source['event_id']}/{timestamp_ms}: PNG write failed")
                else:
                    command = [str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-ss",
                               f"{timestamp_ms / 1000:.3f}", "-i", str(video), "-map", "0:v:0",
                               "-frames:v", "1", "-vsync", "0", str(frame_path)]
                    completed = subprocess.run(command, capture_output=True, text=True)
                    require(completed.returncode == 0 and frame_path.is_file(),
                            f"{source['event_id']}/{timestamp_ms}: ffmpeg failed: {completed.stderr[-500:]}")
                with Image.open(frame_path) as image:
                    width, height = image.size
                remote = f"{REMOTE_ROOT}/frames/{source['event_id']}/{frame_path.name}"
                frame_rows.append({"timestamp_ms": timestamp_ms, "image_path": remote,
                                   "image_sha256": sha256_file(frame_path), "width": width, "height": height})
                staged_files.append({"host_path": str(frame_path.resolve()), "device_relative_path": remote,
                                     "sha256": sha256_file(frame_path)})
                frame_lookup[(source["event_id"], timestamp_ms)] = frame_path
        finally:
            if capture is not None:
                capture.release()
        row = dict(source)
        row.pop("video_path", None)
        row["source_video_sha256"] = row.pop("video_sha256")
        row["frame_transport"] = "hash_bound_exact_timestamp_png_v1"
        row["frames"] = frame_rows
        output_sources.append(row)
    candidate = contract["selected_candidate"]
    android_input = {
        "schema": INPUT_SCHEMA,
        "contract_id": contract["contract_id"],
        "transport_contract_sha256": sha256_file(args.transport_contract),
        "dataset_role": "seen_diagnostic_not_held_out",
        "candidate": candidate,
        "replay_contract": parent["replay_contract"],
        "device_gate": parent["device_gate"],
        "event_gate": parent["event_gate"],
        "sources": output_sources,
        "authority": {"new_held_out_read": False, "r13_sources_read": False, "benchmark_only": True,
                      "training_authorized": False, "production_model_replacement_authorized": False},
    }
    input_path = args.output_dir / "android-input" / "android_r12b_exact_frame_input.json"
    write_json(input_path, android_input)
    input_remote = f"{REMOTE_ROOT}/input.json"
    staged_files.append({"host_path": str(input_path.resolve()), "device_relative_path": input_remote,
                         "sha256": sha256_file(input_path)})
    host_audit_rows = []
    for frame in canary["frames"]:
        exact = frame_lookup[(frame["event_id"], int(frame["timestamp_ms"]))]
        static = (args.canary_manifest.parent / frame["image_asset"].replace("ustrf_r12_detector/", "")).resolve()
        require(static.is_file(), f"canary image missing: {static}")
        with Image.open(static) as static_image, Image.open(exact) as exact_image:
            size_parity = static_image.size == exact_image.size
        host_audit_rows.append({"event_id": frame["event_id"], "frame_id": frame["frame_id"],
                                "timestamp_ms": frame["timestamp_ms"], "size_parity": size_parity,
                                "static_rgb_sha256": raw_rgb_sha256(static),
                                "exact_rgb_sha256": raw_rgb_sha256(exact),
                                "pixel_exact": raw_rgb_sha256(static) == raw_rgb_sha256(exact)})
    write_json(args.output_dir / "host_exact_frame_audit.json", {
        "schema": "blindassist_ustrf_crosscam_exact_frame_host_audit_v1",
        "transport_contract_sha256": sha256_file(args.transport_contract),
        "canary_frame_count": len(host_audit_rows),
        "size_parity_count": sum(row["size_parity"] for row in host_audit_rows),
        "pixel_exact_count": sum(row["pixel_exact"] for row in host_audit_rows),
        "frames": host_audit_rows,
        "target_status_parity_requires_device_test": True,
    })
    write_json(args.output_dir / "android-input" / "host_staging_receipt.json", {
        "schema": "blindassist_ustrf_crosscam_exact_frame_staging_v1",
        "input_sha256": sha256_file(input_path), "new_held_out_read": False,
        "source_count": len(output_sources), "frame_count": sum(len(row["frames"]) for row in output_sources),
        "files": staged_files,
    })
    print(json.dumps({"ok": True, "input": str(input_path), "input_sha256": sha256_file(input_path),
                      "source_count": len(output_sources), "frame_count": sum(len(row["frames"]) for row in output_sources)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
