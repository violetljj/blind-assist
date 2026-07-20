#!/usr/bin/env python3
"""Freeze full-video one-second DINOv2 features before visual window review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

import build_public_video_dinov2_prospective_contract as contract_builder
import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_dinov2_regional_pair_probe as dino
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_dinov2_full_feature_report_v1"


def schedule_timestamps(duration_ms: int, interval_ms: int = 1000) -> list[int]:
    if duration_ms <= 0 or interval_ms <= 0:
        raise ValueError("duration and interval must be positive")
    return list(range(0, duration_ms, interval_ms))


def scheduled_frame_index(timestamp_ms: int, fps: float, frame_count: int) -> int:
    if timestamp_ms < 0 or fps <= 0 or frame_count <= 0:
        raise ValueError("invalid scheduled-frame inputs")
    return min(int(round(timestamp_ms * fps / 1000.0)), frame_count - 1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.video, args.source_registry, args.contract, args.model_dir, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = lifecycle.verify_json_sidecar(args.contract.resolve())
    if contract.get("schema") != contract_builder.SCHEMA or contract.get("authorizations", {}).get("feature_extraction") is not True:
        raise ValueError("invalid DINOv2 prospective contract")
    frozen = contract["frozen_feature_contract"]
    if common.sha256_file(args.model_dir / "model.safetensors") != frozen["model_weights_sha256"]:
        raise ValueError("model weights drift")
    if common.sha256_file(args.model_dir / "config.json") != frozen["model_config_sha256"] or common.sha256_file(args.model_dir / "preprocessor_config.json") != frozen["preprocessor_config_sha256"]:
        raise ValueError("model configuration drift")
    direction = contract["frozen_prototype"]["direction"]
    if contract_builder.direction_sha256(direction) != contract["frozen_prototype"]["direction_sha256"]:
        raise ValueError("prototype direction drift")
    registry = common.load_json(args.source_registry.resolve())
    source = registry.get("source") or registry
    required = ("source_id", "source_page_url", "license", "reuse_allowed")
    if any(key not in source for key in required) or source.get("reuse_allowed") is not True:
        raise ValueError("source registry lacks item-level reuse authorization")
    video_sha = common.sha256_file(args.video.resolve())
    if source.get("video_sha256") not in (None, video_sha):
        raise ValueError("source registry video hash mismatch")

    capture = cv2.VideoCapture(str(args.video.resolve()))
    if not capture.isOpened():
        raise ValueError("cannot open video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("invalid video timing metadata")
    duration_ms = int(round(frame_count / fps * 1000.0))
    timestamps = schedule_timestamps(duration_ms)
    frames = []
    decoded_timestamps = []
    for timestamp in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp))
        ok, frame = capture.read()
        if not ok or frame is None:
            frame_index = scheduled_frame_index(timestamp, fps, frame_count)
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise ValueError(f"failed to decode scheduled timestamp: {timestamp}")
        frames.append(frame)
        decoded_timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
    capture.release()
    teacher = dino.FrozenDinoV2(args.model_dir.resolve())
    vectors = teacher.extract(frames, batch_size=args.batch_size)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": common.sha256_file(args.contract),
        "video": {"path": str(args.video.resolve()), "sha256": video_sha, "fps": fps, "frame_count": frame_count, "duration_ms": duration_ms},
        "source": {"source_id": source["source_id"], "source_page_url": source["source_page_url"], "license": source["license"], "reuse_allowed": True, "registry_sha256": common.sha256_file(args.source_registry)},
        "sampling": {"interval_ms": 1000, "half_open_schedule": True, "scheduled_sample_count": len(timestamps), "review_windows_received": False},
        "feature_contract": frozen,
        "prototype_direction_sha256": contract["frozen_prototype"]["direction_sha256"],
        "samples": [
            {"timestamp_ms": timestamp, "decoded_timestamp_ms": decoded, "vector": vector.tolist()}
            for timestamp, decoded, vector in zip(timestamps, decoded_timestamps, vectors)
        ],
        "hazard_or_lifecycle_verdict_emitted": False,
        "evidence_limit": "Full-video frozen features only. No visual review window or event verdict was accepted by this extractor."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run(args)
    print(json.dumps({"ok": True, "sample_count": payload["sampling"]["scheduled_sample_count"], "video_sha256": payload["video"]["sha256"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
