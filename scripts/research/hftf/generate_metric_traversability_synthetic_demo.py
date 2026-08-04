#!/usr/bin/env python3
"""Generate a reproducible synthetic software demo, never algorithm evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from metric_traversability_field import AlertMapper, build_metric_traversability_field
from render_metric_traversability_field_demo import render
from run_external_rgb_clearance_sidecar import assess_image_quality, write_visualization_assets


def synthetic_scene(wall_distance_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 240, 320
    fx = fy = 240.0
    cx, cy = width / 2.0, height / 2.0
    depth = np.full((height, width), np.nan, dtype=np.float64)
    camera_height = 1.20
    for row in range(round(cy) + 8, height):
        depth[row, :] = fy * camera_height / (row - cy)
    depth[55:172, 135:185] = wall_distance_m
    intrinsics = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    rgb = np.full((height, width, 3), (62, 58, 52), dtype=np.uint8)
    cv2.rectangle(rgb, (0, 172), (width, height), (80, 105, 75), -1)
    cv2.rectangle(rgb, (135, 55), (185, 172), (80, 80, 210), -1)
    cv2.putText(
        rgb,
        "SYNTHETIC SOFTWARE DEMO - NOT EVIDENCE",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return rgb, depth, intrinsics


def generate(output_dir: Path, fps: float) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    records = []
    previous_field = None
    previous_timestamp = None
    scale = {
        "status": "VALID",
        "scale": 1.0,
        "anchor_age_ns": 0,
        "anchor_source": "SYNTHETIC_SOFTWARE_FIXTURE_NOT_EVIDENCE",
    }
    for frame_index, distance in enumerate((1.8, 1.55, 1.30, 1.05, 0.85)):
        timestamp = frame_index * 100_000_000
        bgr, depth, intrinsics = synthetic_scene(distance)
        field = build_metric_traversability_field(
            depth,
            intrinsics,
            metric_scale=scale,
            source_model="SYNTHETIC_SOFTWARE_FIXTURE_NOT_EVIDENCE",
            timestamp_ns=timestamp,
            previous_field=previous_field,
            previous_timestamp_ns=previous_timestamp,
            image_quality=assess_image_quality(bgr),
        )
        assets = write_visualization_assets(
            bgr, depth, assets_dir, "synthetic-not-evidence", frame_index
        )
        records.append(
            {
                "schema": "hftf_metric_traversability_synthetic_software_demo_r0",
                "frame_index": frame_index,
                "timestamp_ns": timestamp,
                "metric_traversability_field": field,
                "shadow_demo_alert_projection": AlertMapper().map(field),
                "visualization_assets": assets,
                "synthetic": True,
                "evidence_authority": False,
            }
        )
        previous_field = field
        previous_timestamp = timestamp
    jsonl = output_dir / "synthetic_software_demo.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    render_summary = render(
        records,
        frames_dir=output_dir / "frames",
        video_path=output_dir / "synthetic_software_demo.mp4",
        fps=fps,
    )
    summary = {
        "schema": "hftf_metric_traversability_synthetic_software_demo_summary_r0",
        "status": "SYNTHETIC_SOFTWARE_DEMO_RENDERED",
        "synthetic": True,
        "evidence_authority": False,
        "records": str(jsonl.resolve()),
        "render": render_summary,
        "claim_ceiling": "software and display mechanism demonstration only; not model evidence",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.fps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
