#!/usr/bin/env python3
"""Cache aligned THOR-MAGNI RGB histories for trainable students."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    DEFAULT_SAMPLES,
    load_jsonl,
    sha256,
)


def rgb_uint8(frame_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        (224, 128),
        interpolation=cv2.INTER_LINEAR,
    )
    if resized.shape != (128, 224, 3) or resized.dtype != np.uint8:
        raise ValueError("Unexpected resized RGB tensor")
    return resized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = Path(str(args.output) + ".json")
    if args.output.exists() or report_path.exists():
        raise ValueError("Refusing to overwrite THOR RGB history cache")
    partial_path = Path(str(args.output) + ".partial.npy")
    if partial_path.exists():
        partial_path.unlink()

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 1078:
        raise ValueError("Expected the fixed 1,078-sample THOR corpus")
    requested: dict[
        str,
        dict[int, list[tuple[int, int]]],
    ] = defaultdict(lambda: defaultdict(list))
    expected_hashes: dict[str, str] = {}
    for sample_index, record in enumerate(records):
        video = str(Path(record["video_path"]).resolve())
        expected_hashes[video] = str(record["video_sha256"])
        for history_index, frame in enumerate(
            record["history_scene_frames"]
        ):
            requested[video][int(frame)].append(
                (sample_index, history_index)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(records), 5, 128, 224, 3),
    )
    filled = np.zeros((len(records), 5), dtype=bool)
    video_rows = []
    for video_text in sorted(requested):
        video_path = Path(video_text)
        if sha256(video_path) != expected_hashes[video_text]:
            raise ValueError(f"Video hash mismatch: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Unable to open video: {video_path}")
        frame_number = 0
        requested_frame_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                destinations = requested[video_text].get(frame_number)
                if not destinations:
                    continue
                value = rgb_uint8(frame)
                requested_frame_count += 1
                for sample_index, history_index in destinations:
                    cache[sample_index, history_index] = value
                    filled[sample_index, history_index] = True
        finally:
            capture.release()
        if requested_frame_count != len(requested[video_text]):
            raise ValueError(
                f"Missing requested frames in video: {video_path}"
            )
        video_rows.append(
            {
                "video_path": video_text,
                "decoded_frame_count": frame_number,
                "requested_unique_frame_count": requested_frame_count,
            }
        )
    if not np.all(filled):
        missing = np.argwhere(~filled)
        raise ValueError(f"Unfilled RGB history entries: {missing[:10]}")
    cache.flush()
    del cache
    partial_path.replace(args.output)

    report: dict[str, Any] = {
        "schema": (
            "blindassist_hftf_stage_c_d10_thor_magni_"
            "trainable_rgb_history_cache_v0"
        ),
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "video_count": len(requested),
        },
        "design": {
            "sample_order": "lexicographic sample_id",
            "shape": [len(records), 5, 128, 224, 3],
            "dtype": "uint8",
            "resize": "OpenCV bilinear after BGR-to-RGB",
            "sample_ids": [
                record["sample_id"] for record in records
            ],
        },
        "videos": video_rows,
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "shape": report["design"]["shape"],
                "bytes": report["output"]["bytes"],
                "sha256": report["output"]["sha256"],
                "report_path": str(report_path.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
