#!/usr/bin/env python3
"""Decode exact Android Canvas/TFLite raw tensors into the frozen detector ledger."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

DETECTOR_DIR = Path(__file__).resolve().parents[1] / "ustrf_detector_taxonomy_coverage"
sys.path.insert(0, str(DETECTOR_DIR))
from run_host_coverage import decode  # noqa: E402

from contract import load_json, sha256_file, validate_prereg


RAW_BYTES_PER_FRAME = 84 * 2100 * 4


def read_exact(stream: gzip.GzipFile, count: int, label: str) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError(f"truncated canonical raw stream at {label}: missing {remaining} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--truth-windows", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device-receipt", required=True, type=Path)
    parser.add_argument("--canonical-raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replacement", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite holdout App detector ledger")
    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    execution = config["candidate_execution_contract"]
    detector = execution["app_detector"]
    replacement = load_json(args.replacement) if args.replacement else None
    planned_output = (
        replacement["planned_outputs"]["app_detector_ledger"]
        if replacement
        else detector["planned_output_path"]
    )
    if args.output.resolve() != (repo / planned_output).resolve():
        raise RuntimeError("App detector output differs from preregistration")
    truth = load_json(args.truth_windows)
    manifest = load_json(args.manifest)
    device = load_json(args.device_receipt)
    if truth.get("selection_authority") is not True or truth.get("admitted_source_count") != 2:
        raise RuntimeError("holdout truth/windows are not admitted for selection")
    if replacement:
        replacement_sha = sha256_file(args.replacement)
        if truth.get("replacement_preregistration_sha256") != replacement_sha:
            raise RuntimeError("replacement truth-window binding mismatch")
        if manifest.get("replacement_preregistration_sha256") != replacement_sha:
            raise RuntimeError("replacement manifest binding mismatch")
    if manifest.get("truth_windows_sha256") != sha256_file(args.truth_windows):
        raise RuntimeError("device manifest is not bound to frozen truth windows")
    if device.get("input_manifest_sha256") != sha256_file(args.manifest):
        raise RuntimeError("device receipt manifest binding mismatch")
    if device.get("failure_count") != 0 or device.get("frame_count") != manifest.get("frame_count"):
        raise RuntimeError("Android detector export is incomplete")
    raw_contract = device.get("canonical_raw_stream", {})
    if raw_contract.get("bytes_per_frame_uncompressed") != RAW_BYTES_PER_FRAME:
        raise RuntimeError("canonical raw record size drifted")
    if sha256_file(args.canonical_raw) != raw_contract.get("compressed_sha256"):
        raise RuntimeError("canonical raw compressed hash mismatch")
    if manifest.get("model_sha256") != detector["model_sha256"] or manifest.get("labels_sha256") != detector["labels_sha256"]:
        raise RuntimeError("device manifest detector binding drifted")
    labels = (repo / detector["labels_path"]).read_text(encoding="utf-8").splitlines()
    grouped: dict[tuple[str, str], list[dict]] = {}
    raw_digest = __import__("hashlib").sha256()
    device_frames = device.get("frames", [])
    if len(device_frames) != len(manifest["frames"]):
        raise RuntimeError("device receipt frame inventory length mismatch")
    with gzip.open(args.canonical_raw, "rb") as stream:
        for row, device_row in zip(manifest["frames"], device_frames, strict=True):
            identity = (row["source_id"], row["sequence_id"], row["frame_id"])
            if identity != (
                device_row.get("source_name"),
                device_row.get("sequence_id"),
                device_row.get("frame_id"),
            ):
                raise RuntimeError(f"device raw frame order mismatch: {identity}")
            raw_bytes = read_exact(stream, RAW_BYTES_PER_FRAME, f"{row['sequence_id']}/{row['frame_id']}")
            raw_digest.update(raw_bytes)
            raw_sha256 = __import__("hashlib").sha256(raw_bytes).hexdigest()
            if raw_sha256 != device_row.get("android_raw_output_sha256"):
                raise RuntimeError(f"device raw frame hash mismatch: {identity}")
            raw = np.frombuffer(raw_bytes, dtype="<f4").reshape((1, 84, 2100))
            width, height = row["source_size"]
            scale = min(320.0 / float(width), 320.0 / float(height))
            resized_width = max(1, int(width * scale))
            resized_height = max(1, int(height * scale))
            transform = (scale, (320 - resized_width) / 2.0, (320 - resized_height) / 2.0)
            detections, _ = decode(
                raw,
                (int(width), int(height)),
                transform,
                labels,
                float(detector["confidence_threshold"]),
                float(detector["nms_iou_threshold"]),
                320,
            )
            grouped.setdefault((row["source_id"], row["sequence_id"]), []).append({
                "frame_id": row["frame_id"],
                "source_capture_timestamp_ns": row["source_capture_timestamp_ns"],
                "image_sha256": row["image_sha256"],
                "source_size": row["source_size"],
                "android_raw_output_sha256": raw_sha256,
                "person_detections": [item for item in detections if item["class_id"] == detector["person_class_index"]],
            })
        if stream.read(1):
            raise RuntimeError("canonical raw stream has trailing records")
    if raw_digest.hexdigest() != raw_contract.get("uncompressed_sha256"):
        raise RuntimeError("canonical raw uncompressed hash mismatch")
    sources = []
    for source_id in sorted({key[0] for key in grouped}):
        sequences = [
            {"sequence_id": sequence_id, "frame_count": len(grouped[(source_id, sequence_id)]), "frames": grouped[(source_id, sequence_id)]}
            for candidate_source, sequence_id in sorted(grouped)
            if candidate_source == source_id
        ]
        sources.append({"source_id": source_id, "sequence_count": len(sequences), "sequences": sequences})
    payload = {
        "schema": "blindassist_crowdbot_holdout_app_detector_ledger_r1",
        "authority": "exact_android_canvas_tflite_candidate_input_not_shadow_or_production_authority",
        "candidate_outputs_executed": False,
        "truth_windows_sha256": sha256_file(args.truth_windows),
        "device_manifest_sha256": sha256_file(args.manifest),
        "device_receipt_sha256": sha256_file(args.device_receipt),
        "canonical_raw_sha256": sha256_file(args.canonical_raw),
        "config_sha256": sha256_file(args.config),
        "replacement_preregistration_sha256": sha256_file(args.replacement) if args.replacement else None,
        "model_sha256": detector["model_sha256"],
        "labels_sha256": detector["labels_sha256"],
        "association_arm": "T0",
        "sources": sources,
        "android_shadow_authority": False,
        "candidate_h2_authority": False,
        "production_authority": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "HOLDOUT_EXACT_ANDROID_APP_DETECTOR_COMPLETE",
        "source_count": len(sources),
        "frame_count": sum(sequence["frame_count"] for source in sources for sequence in source["sequences"]),
        "output_sha256": sha256_file(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
