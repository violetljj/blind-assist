#!/usr/bin/env python3
"""Extract a fixed-rate RGB video manifest for the A0.1 candidate CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def sample_frame_indices(source_fps: float, frame_count: int, target_fps: float) -> list[int]:
    if source_fps <= 0 or frame_count <= 0 or target_fps <= 0:
        raise ValueError("FPS and frame count must be positive")
    duration = frame_count / source_fps
    indices = []
    sample = 0
    while sample / target_fps < duration:
        index = min(frame_count - 1, round((sample / target_fps) * source_fps))
        if not indices or index != indices[-1]:
            indices.append(index)
        sample += 1
    return indices


def prepare(
    video: Path,
    frames_dir: Path,
    manifest: Path,
    sequence_id: str,
    target_fps: float,
    intrinsics: list[float],
) -> dict:
    if manifest.exists():
        raise FileExistsError(manifest)
    if frames_dir.exists() and any(frames_dir.iterdir()):
        raise FileExistsError(f"frames directory is not empty: {frames_dir}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(source_fps, frame_count, target_fps)
    wanted = set(indices)
    rows = []
    source_index = 0
    sample_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index in wanted:
            path = (frames_dir / f"frame_{sample_index:06d}.jpg").resolve()
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"cannot write frame: {path}")
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "timestamp": sample_index / target_fps,
                    "frame_path": str(path),
                    "intrinsics_fx_fy_cx_cy": intrinsics,
                }
            )
            sample_index += 1
        source_index += 1
    capture.release()
    if len(rows) != len(indices):
        raise RuntimeError(f"decoded {len(rows)} of {len(indices)} requested frames")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "video": str(video.resolve()),
        "source_fps": source_fps,
        "target_fps": target_fps,
        "source_frames": frame_count,
        "extracted_frames": len(rows),
        "manifest": str(manifest.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--intrinsics-fx-fy-cx-cy", nargs=4, type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.video,
                args.frames_dir,
                args.manifest,
                args.sequence_id,
                args.target_fps,
                args.intrinsics_fx_fy_cx_cy,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
