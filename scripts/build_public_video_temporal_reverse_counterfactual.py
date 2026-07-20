#!/usr/bin/env python3
"""Build an isolated temporal-reversal counterfactual from licensed public video.

The output is a discovery-only transform.  It may test whether a frozen causal
exit heuristic responds when a real obstruction recedes, but it is never human
event truth, calibration data, blind evidence, or production authorization.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import scan_public_video_prompt_free_exit_candidates as discovery


SCHEMA = "blindassist_public_video_temporal_reverse_counterfactual_v1"


def reversed_timestamp_mapping(timestamps_ms: Sequence[int]) -> list[dict[str, int]]:
    if not timestamps_ms:
        raise ValueError("temporal reverse requires at least one timestamp")
    ordered = [int(value) for value in timestamps_ms]
    if any(current <= previous for previous, current in zip(ordered, ordered[1:])):
        raise ValueError("source timestamps must be strictly increasing")
    return [
        {
            "output_frame_index": output_index,
            "synthetic_timestamp_ms": int(round(output_index * 1000.0 / max(1, len(ordered) - 1)))
            if len(ordered) == 1
            else 0,
            "original_source_timestamp_ms": source_timestamp_ms,
        }
        for output_index, source_timestamp_ms in enumerate(reversed(ordered))
    ]


def assign_synthetic_timestamps(
    mapping: Sequence[dict[str, int]], *, target_fps: float
) -> list[dict[str, int]]:
    if target_fps <= 0:
        raise ValueError("target FPS must be positive")
    return [
        {
            **row,
            "synthetic_timestamp_ms": int(round(index * 1000.0 / target_fps)),
        }
        for index, row in enumerate(mapping)
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = (
        args.source_registry,
        args.output_video,
        args.output_registry,
        args.output_receipt,
    )
    for path in paths:
        mil.reject_independent_direction(path)
    for path in (args.output_video, args.output_registry, args.output_receipt):
        if path.exists():
            raise ValueError(f"refusing to overwrite output: {path}")

    registry = common.load_json(args.source_registry)
    sources = discovery.validate_registry(registry, args.source_registry.resolve())
    matches = [source for source in sources if source["source_id"] == args.source_id]
    if len(matches) != 1:
        raise ValueError(f"source ID must match exactly one registry entry: {args.source_id}")
    source = matches[0]
    video_path = Path(source["local_video_path"])

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("source video timing metadata is invalid")
    duration_ms = int(round(frame_count * 1000.0 / source_fps))
    if not (0 <= args.start_ms < args.end_ms <= duration_ms):
        capture.release()
        raise ValueError(
            f"requested window must satisfy 0 <= start < end <= {duration_ms} ms"
        )

    step_ms = 1000.0 / args.target_fps
    timestamps_ms: list[int] = []
    frames: list[Any] = []
    timestamp = float(args.start_ms)
    try:
        while timestamp <= args.end_ms + 0.5:
            source_timestamp_ms = int(round(timestamp))
            capture.set(cv2.CAP_PROP_POS_MSEC, float(source_timestamp_ms))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot decode source timestamp {source_timestamp_ms} ms")
            timestamps_ms.append(source_timestamp_ms)
            frames.append(frame)
            timestamp += step_ms
    finally:
        capture.release()
    if len(frames) < 2:
        raise RuntimeError("temporal reverse decoded fewer than two frames")

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.target_fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output video: {args.output_video}")
    try:
        for frame in reversed(frames):
            writer.write(frame)
    finally:
        writer.release()

    mapping = assign_synthetic_timestamps(
        reversed_timestamp_mapping(timestamps_ms),
        target_fps=args.target_fps,
    )
    synthetic_source_id = (
        f"{args.source_id}__temporal_reverse_{args.start_ms}_{args.end_ms}"
    )
    output_registry = {
        "schema": discovery.REGISTRY_SCHEMA,
        "sources": [
            {
                "source_id": synthetic_source_id,
                "local_video_path": str(args.output_video.resolve()),
                "commons_title": source["commons_title"],
                "commons_page_url": source["commons_page_url"],
                "author": source["author"],
                "license": source["license"],
                "synthetic_transform": "temporal_reverse",
                "parent_source_id": args.source_id,
                "parent_video_sha256": common.sha256_file(video_path),
                "discovery_only": True,
            }
        ],
    }
    args.output_registry.parent.mkdir(parents=True, exist_ok=True)
    args.output_registry.write_text(
        json.dumps(output_registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(str(args.output_registry) + ".sha256").write_text(
        common.sha256_file(args.output_registry) + "\n", encoding="ascii"
    )

    receipt = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(args.source_registry.resolve()),
        "source_registry_sha256": common.sha256_file(args.source_registry),
        "parent_source": source,
        "parent_video_sha256": common.sha256_file(video_path),
        "transform": {
            "kind": "temporal_reverse",
            "window_start_ms": args.start_ms,
            "window_end_ms": args.end_ms,
            "target_fps": args.target_fps,
            "output_frame_count": len(frames),
            "frame_mapping": mapping,
        },
        "output_video": str(args.output_video.resolve()),
        "output_video_sha256": common.sha256_file(args.output_video),
        "output_registry": str(args.output_registry.resolve()),
        "output_registry_sha256": common.sha256_file(args.output_registry),
        "counterfactual_contract": {
            "real_pixels_reordered_only": True,
            "synthetic_temporal_direction": True,
            "human_event_truth_present": False,
            "training_execution_authorized": False,
            "calibration_authorized": False,
            "blind_evaluation_authorized": False,
            "production_model_replacement_authorized": False,
        },
        "evidence_limit": "Discovery-only temporal counterfactual for a frozen exit heuristic; it cannot enter the real evaluation denominator.",
    }
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(str(args.output_receipt) + ".sha256").write_text(
        common.sha256_file(args.output_receipt) + "\n", encoding="ascii"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.start_ms < 0 or args.end_ms <= args.start_ms or args.target_fps <= 0:
        parser.error("window and target FPS must be positive")
    return args


def main() -> int:
    try:
        receipt = build(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "output_frame_count": receipt["transform"]["output_frame_count"],
                "output_video_sha256": receipt["output_video_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
