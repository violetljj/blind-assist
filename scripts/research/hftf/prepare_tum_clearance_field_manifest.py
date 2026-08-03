#!/usr/bin/env python3
"""Materialize fixed RGB frames for a TUM clearance-field experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_bonn_rgbd_metric_depth_manifest import (
    associate_nearest,
    read_tum_index,
    sample_timestamp_pairs,
)


def prepare(
    sequence_root: Path,
    output: Path,
    sequence_id: str,
    starts_s: list[float],
    duration_s: float,
    target_fps: float,
    intrinsics: list[float],
) -> dict:
    associated = associate_nearest(
        read_tum_index(sequence_root / "rgb.txt"),
        read_tum_index(sequence_root / "depth.txt"),
        0.02,
    )
    rows = []
    for start_s in starts_s:
        sampled = sample_timestamp_pairs(
            associated,
            start_s=start_s,
            duration_s=duration_s,
            target_fps=target_fps,
        )
        if len(sampled) < 7:
            raise ValueError(f"window {start_s:g}s has fewer than seven paired frames")
        first_timestamp = sampled[0][0]
        tag = f"{int(round(start_s)):03d}"
        for frame_index, (rgb_timestamp, rgb_relative, _, _) in enumerate(sampled):
            rows.append(
                {
                    "sequence_id": f"{sequence_id}-{tag}",
                    "frame_index": frame_index,
                    "timestamp_ns": int(round((rgb_timestamp - first_timestamp) * 1e9)),
                    "frame_path": str((sequence_root / rgb_relative).resolve()),
                    "intrinsics_fx_fy_cx_cy": intrinsics,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {
        "manifest": str(output.resolve()),
        "windows_s": starts_s,
        "duration_s": duration_s,
        "target_fps": target_fps,
        "rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--start-s", type=float, nargs="+", required=True)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--intrinsics-fx-fy-cx-cy", type=float, nargs=4, required=True)
    args = parser.parse_args()
    receipt = prepare(
        args.sequence_root,
        args.output,
        args.sequence_id,
        args.start_s,
        args.duration_s,
        args.target_fps,
        args.intrinsics_fx_fy_cx_cy,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
