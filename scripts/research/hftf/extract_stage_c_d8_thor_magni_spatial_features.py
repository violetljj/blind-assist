#!/usr/bin/env python3
"""Extract frozen MobileNet spatial maps for THOR-MAGNI history samples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    DEFAULT_PRETRAINED,
    DEFAULT_SAMPLES,
    MobileNetFeatures,
    frame_tensor,
    load_jsonl,
    sha256,
)


def flush_batch(
    model: MobileNetFeatures,
    device: torch.device,
    batch_frames: list[torch.Tensor],
    batch_keys: list[tuple[str, int]],
    output: dict[tuple[str, int], np.ndarray],
) -> None:
    if not batch_frames:
        return
    with torch.inference_mode():
        values = model.features(
            torch.stack(batch_frames).to(device, non_blocking=True)
        )
    if values.shape[1:] != (576, 4, 7):
        raise ValueError(
            f"Unexpected MobileNet spatial feature shape: {values.shape}"
        )
    for key, feature in zip(batch_keys, values.cpu().numpy()):
        output[key] = feature.astype(np.float16)
    batch_frames.clear()
    batch_keys.clear()


def extract(
    records: list[dict[str, Any]],
    pretrained: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    requested: dict[str, set[int]] = defaultdict(set)
    expected_hashes: dict[str, str] = {}
    for record in records:
        video_path = str(Path(record["video_path"]).resolve())
        requested[video_path].update(
            int(value) for value in record["history_scene_frames"]
        )
        expected_hashes[video_path] = str(record["video_sha256"])

    model = MobileNetFeatures(pretrained).to(device).eval()
    by_key: dict[tuple[str, int], np.ndarray] = {}
    video_rows = []
    for video_text in sorted(requested):
        video_path = Path(video_text)
        if sha256(video_path) != expected_hashes[video_text]:
            raise ValueError(f"Video hash mismatch: {video_path}")
        wanted = requested[video_text]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Unable to open video: {video_path}")
        batch_frames: list[torch.Tensor] = []
        batch_keys: list[tuple[str, int]] = []
        frame_number = 0
        captured = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                if frame_number not in wanted:
                    continue
                batch_frames.append(frame_tensor(frame))
                batch_keys.append((video_text, frame_number))
                captured += 1
                if len(batch_frames) >= batch_size:
                    flush_batch(
                        model,
                        device,
                        batch_frames,
                        batch_keys,
                        by_key,
                    )
            flush_batch(
                model,
                device,
                batch_frames,
                batch_keys,
                by_key,
            )
        finally:
            capture.release()
        if captured != len(wanted):
            missing = sorted(
                index
                for index in wanted
                if (video_text, index) not in by_key
            )
            raise ValueError(
                f"Missing requested frames for {video_path}: {missing[:10]}"
            )
        video_rows.append(
            {
                "video_path": video_text,
                "requested_frame_count": len(wanted),
                "decoded_frame_count": frame_number,
            }
        )

    matrix = np.stack(
        [
            np.stack(
                [
                    by_key[
                        (
                            str(Path(record["video_path"]).resolve()),
                            int(frame),
                        )
                    ]
                    for frame in record["history_scene_frames"]
                ]
            )
            for record in records
        ]
    )
    return matrix, {
        "videos": video_rows,
        "unique_frame_count": len(by_key),
        "feature_shape": list(matrix.shape),
        "storage_dtype": str(matrix.dtype),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists() or Path(str(args.output) + ".json").exists():
        raise ValueError("Refusing to overwrite spatial feature cache")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 1078:
        raise ValueError("Expected the fixed 1,078-sample THOR corpus")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    features, extraction = extract(
        records,
        args.pretrained,
        device,
        args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_ids=np.asarray(
            [record["sample_id"] for record in records]
        ),
        features=features,
    )
    report = {
        "schema": (
            "blindassist_hftf_stage_c_d8_thor_magni_"
            "mobilenet_spatial_feature_cache_v0"
        ),
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "device": str(device),
        "extraction": extraction,
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
        },
    }
    report_path = Path(str(args.output) + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "device": str(device),
                "feature_shape": list(features.shape),
                "feature_sha256": report["output"]["sha256"],
                "report_path": str(report_path.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
