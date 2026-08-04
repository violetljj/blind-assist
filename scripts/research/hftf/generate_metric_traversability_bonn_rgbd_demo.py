#!/usr/bin/env python3
"""Render a real Bonn RGB-D teacher demo of MetricTraversabilityField mechanics.

Registered sensor depth is intentionally used as a display-only geometry teacher.
This is not RGB-model evidence, a sparse-scale result, or a safety evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from metric_traversability_field import AlertMapper, build_metric_traversability_field
from prepare_bonn_rgbd_metric_depth_manifest import (
    associate_nearest,
    normalize_depth_image,
    read_tum_index,
)
from render_metric_traversability_field_demo import render
from run_external_rgb_clearance_sidecar import (
    assess_image_quality,
    write_research_depth_artifact,
    write_visualization_assets,
)


SCHEMA = "hftf_metric_traversability_bonn_rgbd_teacher_demo_r0"
DEPTH_UNITS_PER_METER = 5000.0
MAX_RGB_DEPTH_DELTA_S = 0.02
SOURCE_ROLE = "SOURCE_AUTHORITATIVE_REGISTERED_RGBD_DEPTH_TEACHER_DISPLAY_ONLY"


def _load_manifest(path: Path, maximum_frames: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if maximum_frames is not None:
        rows = rows[:maximum_frames]
    if not rows:
        raise ValueError("manifest contains no rows")
    return rows


def _depth_pairs(sequence_root: Path) -> dict[Path, tuple[float, Path]]:
    pairs = associate_nearest(
        read_tum_index(sequence_root / "rgb.txt"),
        read_tum_index(sequence_root / "depth.txt"),
        MAX_RGB_DEPTH_DELTA_S,
    )
    return {
        (sequence_root / rgb_relative).resolve(): (
            abs(depth_timestamp - rgb_timestamp),
            (sequence_root / depth_relative).resolve(),
        )
        for rgb_timestamp, rgb_relative, depth_timestamp, depth_relative in pairs
    }


def generate(
    manifest: Path,
    output_dir: Path,
    *,
    fps: float,
    maximum_frames: int | None,
) -> dict[str, Any]:
    rows = _load_manifest(manifest, maximum_frames)
    first_frame = Path(rows[0]["frame_path"]).resolve()
    sequence_root = first_frame.parent.parent
    pairs = _depth_pairs(sequence_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    depth_dir = output_dir / "depth"
    records = []
    previous_field = None
    previous_timestamp = None
    maximum_delta_s = 0.0
    for ordinal, row in enumerate(rows):
        frame_path = Path(row["frame_path"]).resolve()
        if frame_path not in pairs:
            raise ValueError(f"no registered depth pair for {frame_path}")
        delta_s, depth_path = pairs[frame_path]
        maximum_delta_s = max(maximum_delta_s, delta_s)
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise OSError(f"failed to decode RGB frame: {frame_path}")
        depth_raw = normalize_depth_image(depth_raw, depth_path)
        if depth_raw.shape != bgr.shape[:2]:
            raise ValueError(
                f"registered RGB/depth shape mismatch: {bgr.shape[:2]} vs {depth_raw.shape}"
            )
        depth_m = depth_raw.astype(np.float32) / DEPTH_UNITS_PER_METER
        depth_m[depth_raw == 0] = np.nan
        intrinsics_values = [float(value) for value in row["intrinsics_fx_fy_cx_cy"]]
        intrinsics = np.asarray(
            [
                [intrinsics_values[0], 0.0, intrinsics_values[2]],
                [0.0, intrinsics_values[1], intrinsics_values[3]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        timestamp_ns = int(row.get("timestamp_ns", ordinal * round(1e9 / fps)))
        image_quality = assess_image_quality(bgr)
        metric_receipt = {
            "status": "VALID",
            "scale": 1.0,
            "anchor_age_ns": 0,
            "anchor_source": SOURCE_ROLE,
        }
        field = build_metric_traversability_field(
            depth_m,
            intrinsics,
            metric_scale=metric_receipt,
            source_model=SOURCE_ROLE,
            timestamp_ns=timestamp_ns,
            previous_field=previous_field,
            previous_timestamp_ns=previous_timestamp,
            image_quality=image_quality,
        )
        assets = write_visualization_assets(
            bgr,
            depth_m,
            assets_dir,
            str(row["sequence_id"]),
            int(row.get("frame_index", ordinal)),
        )
        depth_artifact = write_research_depth_artifact(
            depth_raw,
            depth_m,
            depth_dir,
            str(row["sequence_id"]),
            int(row.get("frame_index", ordinal)),
        )
        record = {
            "schema": SCHEMA,
            "sequence_id": row["sequence_id"],
            "frame_index": int(row.get("frame_index", ordinal)),
            "timestamp_ns": timestamp_ns,
            "rgb_path": str(frame_path),
            "registered_depth_path": str(depth_path),
            "rgb_depth_timestamp_delta_s": delta_s,
            "intrinsics_fx_fy_cx_cy": intrinsics_values,
            "source_role": SOURCE_ROLE,
            "metric_traversability_field": field,
            "shadow_demo_alert_projection": AlertMapper().map(field),
            "visualization_assets": assets,
            "research_depth_artifact": depth_artifact,
            "real_rgbd": True,
            "rgb_model_inference": False,
            "algorithm_evidence_authority": False,
            "claim_ceiling": (
                "real registered RGB-D teacher and geometry-mechanism display only; "
                "not RGB-model, sparse-scale, alert, navigation, or safety evidence"
            ),
        }
        records.append(record)
        previous_field = field
        previous_timestamp = timestamp_ns

    sidecar = output_dir / "sidecar.jsonl"
    sidecar.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    render_summary = render(
        records,
        frames_dir=output_dir / "frames",
        video_path=output_dir / "bonn_rgbd_teacher_demo.mp4",
        fps=fps,
    )
    valid = sum(
        record["metric_traversability_field"]["status"] == "VALID"
        for record in records
    )
    summary = {
        "schema": "hftf_metric_traversability_bonn_rgbd_teacher_demo_summary_r0",
        "status": "REAL_RGBD_TEACHER_GEOMETRY_DEMO_RENDERED",
        "source_manifest": str(manifest.resolve()),
        "sequence_root": str(sequence_root),
        "frames": len(records),
        "valid_fields": valid,
        "unknown_fields": len(records) - valid,
        "maximum_rgb_depth_timestamp_delta_s": maximum_delta_s,
        "source_role": SOURCE_ROLE,
        "real_rgbd": True,
        "rgb_model_inference": False,
        "algorithm_evidence_authority": False,
        "sidecar": str(sidecar.resolve()),
        "render": render_summary,
        "claim_ceiling": (
            "real registered RGB-D teacher and geometry-mechanism display only; "
            "not RGB-model, sparse-scale, alert, navigation, or safety evidence"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.manifest,
                args.output_dir,
                fps=args.fps,
                maximum_frames=args.max_frames,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
