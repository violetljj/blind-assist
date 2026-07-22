from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

from run_host_coverage import array_sha256, decode, read_json, sha256


RGB_BYTES_PER_FRAME = 320 * 320 * 3
RAW_FLOATS_PER_FRAME = 84 * 2100
RAW_BYTES_PER_FRAME = RAW_FLOATS_PER_FRAME * 4


def read_exact(stream: gzip.GzipFile, count: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ValueError(f"truncated canonical stream while reading {label}: missing {remaining} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def run(
    config_path: Path,
    manifest_path: Path,
    device_receipt_path: Path,
    canonical_rgb_path: Path,
    canonical_raw_path: Path,
    source_bundle_root: Path,
    output_path: Path,
) -> dict:
    config = read_json(config_path)
    detector = config["detector"]
    manifest = read_json(manifest_path)
    device = read_json(device_receipt_path)
    if manifest.get("frame_count") != config["parent"]["frame_count"]:
        raise ValueError("manifest frame count mismatch")
    if device.get("frame_count") != manifest["frame_count"] or device.get("failure_count") != 0:
        raise ValueError("device receipt is incomplete")
    if device.get("input_manifest_sha256") != sha256(manifest_path):
        raise ValueError("device receipt manifest binding mismatch")
    input_stream = device.get("canonical_input_stream", {})
    raw_stream = device.get("canonical_raw_stream", {})
    if input_stream.get("bytes_per_frame_uncompressed") != RGB_BYTES_PER_FRAME:
        raise ValueError("canonical RGB record size mismatch")
    if raw_stream.get("bytes_per_frame_uncompressed") != RAW_BYTES_PER_FRAME:
        raise ValueError("canonical raw record size mismatch")
    if sha256(canonical_rgb_path) != input_stream.get("compressed_sha256"):
        raise ValueError("canonical RGB compressed hash mismatch")
    if sha256(canonical_raw_path) != raw_stream.get("compressed_sha256"):
        raise ValueError("canonical raw compressed hash mismatch")
    if sha256(Path(detector["model_path"])) != detector["model_sha256"]:
        raise ValueError("model hash mismatch")
    labels = Path(detector["labels_path"]).read_text(encoding="utf-8").splitlines()
    # Match the frozen Android benchmark runtime. XNNPACK reduction order is
    # thread-count dependent, so a host default of one thread is not a valid
    # raw-output parity comparison against Android's four-thread interpreter.
    interpreter = Interpreter(model_path=detector["model_path"], num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    device_rows = device["frames"]
    manifest_rows = manifest["frames"]
    if len(device_rows) != len(manifest_rows):
        raise ValueError("device/manifest row count mismatch")
    rgb_digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    rows: list[dict] = []
    input_exact = raw_exact = within_tolerance = 0
    max_abs_all = max_rel_all = 0.0
    started = time.perf_counter()
    with gzip.open(canonical_rgb_path, "rb") as rgb_stream, gzip.open(canonical_raw_path, "rb") as raw_device_stream:
        for index, (expected, actual) in enumerate(zip(manifest_rows, device_rows, strict=True), start=1):
            identity = (expected["source_name"], expected["frame_id"])
            if identity != (actual["source_name"], actual["frame_id"]):
                raise ValueError(f"frame order mismatch: {identity}")
            rgb_bytes = read_exact(rgb_stream, RGB_BYTES_PER_FRAME, f"{identity}/rgb")
            raw_bytes = read_exact(raw_device_stream, RAW_BYTES_PER_FRAME, f"{identity}/raw")
            rgb_digest.update(rgb_bytes)
            raw_digest.update(raw_bytes)
            input_tensor = np.frombuffer(rgb_bytes, dtype=np.uint8).astype(np.float32)
            input_tensor /= np.float32(255.0)
            input_tensor = input_tensor.reshape((1, 320, 320, 3))
            input_hash = array_sha256(input_tensor)
            input_matches = input_hash == actual["android_input_tensor_sha256"]
            input_exact += int(input_matches)
            if not input_matches:
                raise ValueError(f"canonical input reconstruction mismatch: {identity}")
            interpreter.set_tensor(input_detail["index"], input_tensor)
            interpreter.invoke()
            host_raw = interpreter.get_tensor(output_detail["index"])
            device_raw = np.frombuffer(raw_bytes, dtype="<f4").reshape((1, 84, 2100))
            host_raw_hash = array_sha256(host_raw)
            raw_matches = host_raw_hash == actual["android_raw_output_sha256"]
            raw_exact += int(raw_matches)
            absolute = np.abs(host_raw.astype(np.float64) - device_raw.astype(np.float64))
            denominator = np.maximum(np.abs(device_raw.astype(np.float64)), 1e-12)
            relative = absolute / denominator
            max_abs = float(np.max(absolute))
            max_rel = float(np.max(relative))
            tolerance_pass = bool(np.all((absolute <= 1e-5) | (relative <= 1e-4)))
            within_tolerance += int(tolerance_pass)
            max_abs_all = max(max_abs_all, max_abs)
            max_rel_all = max(max_rel_all, max_rel)
            source_image = source_bundle_root / expected["image_path"]
            if sha256(source_image) != expected["image_sha256"]:
                raise ValueError(f"source image hash mismatch: {identity}")
            with Image.open(source_image) as image:
                source_size = image.size
            scale = min(320.0 / source_size[0], 320.0 / source_size[1])
            resized_width = max(1, int(source_size[0] * scale))
            resized_height = max(1, int(source_size[1] * scale))
            transform = (scale, (320 - resized_width) / 2.0, (320 - resized_height) / 2.0)
            detections, diagnostics = decode(
                host_raw,
                source_size,
                transform,
                labels,
                float(detector["confidence_threshold"]),
                float(detector["nms_iou_threshold"]),
                320,
            )
            rows.append({
                "source_name": identity[0],
                "source_id": expected["source_id"],
                "frame_id": identity[1],
                "source_size": list(source_size),
                "letterbox": {"scale": transform[0], "dx": transform[1], "dy": transform[2]},
                "android_input_tensor_sha256": actual["android_input_tensor_sha256"],
                "host_input_tensor_sha256": input_hash,
                "input_tensor_exact_match": input_matches,
                "android_raw_output_sha256": actual["android_raw_output_sha256"],
                "host_raw_output_sha256": host_raw_hash,
                "raw_output_exact_match": raw_matches,
                "raw_output_tolerance_pass": tolerance_pass,
                "raw_output_max_abs_error": max_abs,
                "raw_output_max_rel_error": max_rel,
                **diagnostics,
                "post_nms_detections_canonical_320": detections,
            })
            if index % 100 == 0:
                print(f"canonical_host_frames={index}/{len(manifest_rows)}", flush=True)
        if rgb_stream.read(1) or raw_device_stream.read(1):
            raise ValueError("canonical stream has trailing records")
    if rgb_digest.hexdigest() != input_stream.get("uncompressed_sha256"):
        raise ValueError("canonical RGB uncompressed hash mismatch")
    if raw_digest.hexdigest() != raw_stream.get("uncompressed_sha256"):
        raise ValueError("canonical raw uncompressed hash mismatch")
    frame_count = len(rows)
    payload = {
        "schema": "blindassist_ustrf_detector_canonical_host_ledger_v1",
        "authority": "benchmark_only_target_attribution_input",
        "config_sha256": sha256(config_path),
        "device_receipt_sha256": sha256(device_receipt_path),
        "input_manifest_sha256": sha256(manifest_path),
        "canonical_rgb_compressed_sha256": sha256(canonical_rgb_path),
        "canonical_raw_compressed_sha256": sha256(canonical_raw_path),
        "frame_count": frame_count,
        "input_tensor_exact_match_count": input_exact,
        "raw_output_exact_match_count": raw_exact,
        "raw_output_within_frozen_tolerance_count": within_tolerance,
        "raw_output_global_max_abs_error": max_abs_all,
        "raw_output_global_max_rel_error": max_rel_all,
        "G1_android_host_parity": "pass" if input_exact == frame_count and within_tolerance == frame_count else "fail",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "frames": rows,
    }
    if output_path.exists():
        raise ValueError(f"refusing to overwrite output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device-receipt", type=Path, required=True)
    parser.add_argument("--canonical-rgb", type=Path, required=True)
    parser.add_argument("--canonical-raw", type=Path, required=True)
    parser.add_argument("--source-bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        args.config, args.manifest, args.device_receipt, args.canonical_rgb, args.canonical_raw,
        args.source_bundle_root, args.output,
    )
    print(json.dumps({key: payload[key] for key in (
        "frame_count", "input_tensor_exact_match_count", "raw_output_exact_match_count",
        "raw_output_within_frozen_tolerance_count", "G1_android_host_parity", "elapsed_seconds",
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
