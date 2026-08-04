#!/usr/bin/env python3
"""Render a four-panel DEVELOPMENT_ONLY MetricTraversabilityField demo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


WIDTH = 1280
HEIGHT = 720
PANEL_W = 620
PANEL_H = 300
BACKGROUND = (24, 29, 34)
UNKNOWN = (105, 105, 105)
CLEAR = (55, 180, 90)
OCCUPIED = (50, 65, 220)


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            field = row.get("metric_traversability_field", row)
            if field.get("schema") != "blindassist_hftf_metric_traversability_field_r0":
                raise ValueError(f"{path}:{line_number}: unsupported field schema")
            records.append(row)
    if not records:
        raise ValueError("no MetricTraversabilityField records")
    return records


def _fit_image(image: np.ndarray | None, width: int, height: int, label: str) -> np.ndarray:
    panel = np.full((height, width, 3), 45, dtype=np.uint8)
    if image is None:
        cv2.putText(panel, "NO DISPLAY ASSET", (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, UNKNOWN, 2)
    else:
        scale = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        top = (height - resized.shape[0]) // 2
        left = (width - resized.shape[1]) // 2
        panel[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (85, 95, 105), 1)
    cv2.putText(panel, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)
    return panel


def _asset(row: dict[str, Any], key: str) -> np.ndarray | None:
    assets = row.get("visualization_assets") or {}
    value = assets.get(key)
    if not value:
        return None
    return cv2.imread(str(value), cv2.IMREAD_COLOR)


def _state_color(state: str) -> tuple[int, int, int]:
    if state == "OCCUPIED_OBSERVED":
        return OCCUPIED
    if state == "CLEAR_OBSERVED":
        return CLEAR
    return UNKNOWN


def _source_badge(row: dict[str, Any]) -> str | None:
    source_role = str(row.get("source_role", ""))
    if "TEACHER" in source_role:
        return "REAL RGB-D TEACHER | NOT RGB MODEL INFERENCE"
    return None


def _draw_bev(field: dict[str, Any], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 38, dtype=np.uint8)
    cv2.putText(panel, "BEV BODY-SWEEP ENVELOPE", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2)
    origin = (width // 2, height - 20)
    pixels_per_meter = (height - 50) / 4.0
    for horizon in (1.0, 1.5, 2.0, 3.0, 4.0):
        radius = round(horizon * pixels_per_meter)
        cv2.ellipse(panel, origin, (radius, radius), 0, 220, 320, (80, 85, 90), 1)
        cv2.putText(panel, f"{horizon:g}m", (origin[0] + 5, origin[1] - radius + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (155, 160, 165), 1)
    half_width_px = round(0.42 * pixels_per_meter)
    cv2.rectangle(panel, (origin[0] - half_width_px, origin[1] - round(2.0 * pixels_per_meter)), (origin[0] + half_width_px, origin[1]), (60, 110, 75), 1)
    cv2.circle(panel, origin, 8, (230, 210, 70), -1)

    if field.get("status") != "VALID":
        cv2.putText(panel, "UNKNOWN", (origin[0] - 75, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, UNKNOWN, 3)
        reasons = ", ".join(field.get("unknown_reasons", []))[:72]
        cv2.putText(panel, reasons, (12, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, UNKNOWN, 1)
        return panel

    for envelope in field.get("sweep_envelopes", []):
        horizon = float(envelope["horizon_m"])
        for cell in envelope["directions"]:
            angle = math.radians(float(cell["theta_deg"]))
            endpoint = (
                round(origin[0] + math.sin(angle) * horizon * pixels_per_meter),
                round(origin[1] - math.cos(angle) * horizon * pixels_per_meter),
            )
            cv2.circle(panel, endpoint, 3, _state_color(str(cell["state"])), -1)
    for item in field.get("clearance_profile", []):
        distance = item.get("nearest_intrusion_m")
        if distance is None:
            continue
        angle = math.radians(float(item["theta_deg"]))
        point = (
            round(origin[0] + math.sin(angle) * float(distance) * pixels_per_meter),
            round(origin[1] - math.cos(angle) * float(distance) * pixels_per_meter),
        )
        cv2.circle(panel, point, 5, OCCUPIED, -1)
    cv2.putText(panel, "GREEN=OBSERVED CLEAR  RED=INTRUSION  GRAY=UNKNOWN", (12, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (205, 205, 205), 1)
    return panel


def _draw_profile(field: dict[str, Any], alert: dict[str, Any], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 38, dtype=np.uint8)
    cv2.putText(panel, "DIRECTIONAL OBSERVED CLEARANCE", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2)
    left, right, top, bottom = 55, width - 20, 38, height - 60
    cv2.rectangle(panel, (left, top), (right, bottom), (100, 105, 110), 1)
    for distance in (1.0, 1.5, 2.0, 3.0, 4.0):
        y = round(bottom - distance / 4.0 * (bottom - top))
        cv2.line(panel, (left, y), (right, y), (65, 70, 75), 1)
        cv2.putText(panel, f"{distance:g}m", (5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (165, 170, 175), 1)
    profile = field.get("clearance_profile", [])
    points = []
    for item in profile:
        theta = float(item["theta_deg"])
        x = round(left + (theta + 40.0) / 80.0 * (right - left))
        distance = item.get("nearest_intrusion_m")
        if distance is None:
            cv2.circle(panel, (x, bottom), 3, UNKNOWN, -1)
            points.append(None)
        else:
            y = round(bottom - min(float(distance), 4.0) / 4.0 * (bottom - top))
            cv2.circle(panel, (x, y), 4, OCCUPIED, -1)
            points.append((x, y))
    previous = None
    for point in points:
        if point is not None and previous is not None:
            cv2.line(panel, previous, point, (80, 160, 230), 2)
        previous = point
    cv2.putText(panel, "-40 deg", (left - 5, bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(panel, "0", ((left + right) // 2, bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(panel, "+40 deg", (right - 50, bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    status = str(alert.get("status", "NO_ALERT_PROJECTION"))
    cv2.putText(panel, f"SHADOW ALERT: {status}", (12, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (225, 225, 225), 1)
    cv2.putText(panel, "NOT A SAFE-DIRECTION OR NAVIGATION OUTPUT", (12, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (85, 175, 245), 1)
    return panel


def render_frame(row: dict[str, Any]) -> np.ndarray:
    field = row.get("metric_traversability_field", row)
    alert = row.get("shadow_demo_alert_projection", {})
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    canvas[55 : 55 + PANEL_H, 10 : 10 + PANEL_W] = _fit_image(_asset(row, "rgb_path"), PANEL_W, PANEL_H, "RGB DISPLAY ASSET")
    canvas[55 : 55 + PANEL_H, 650 : 650 + PANEL_W] = _fit_image(_asset(row, "metric_depth_heatmap_path"), PANEL_W, PANEL_H, "CALIBRATED DEPTH HEATMAP")
    canvas[375 : 375 + PANEL_H, 10 : 10 + PANEL_W] = _draw_bev(field, PANEL_W, PANEL_H)
    canvas[375 : 375 + PANEL_H, 650 : 650 + PANEL_W] = _draw_profile(field, alert, PANEL_W, PANEL_H)
    cv2.putText(canvas, "BlindAssist MetricTraversabilityField R0", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2)
    cv2.putText(canvas, "DEVELOPMENT_ONLY | SHADOW_DEMO | NO_NAVIGATION_AUTHORITY", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 180, 245), 1)
    source_badge = _source_badge(row)
    if source_badge is not None:
        cv2.putText(canvas, source_badge, (650, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 210, 245), 1)
    cv2.putText(canvas, f"frame={row.get('frame_index', '?')}  status={field.get('status', 'UNKNOWN')}", (930, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    return canvas


def render(
    records: list[dict[str, Any]],
    *,
    frames_dir: Path | None,
    video_path: Path | None,
    fps: float,
) -> dict[str, Any]:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if frames_dir is None and video_path is None:
        raise ValueError("at least one output is required")
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {video_path}")
    valid = 0
    try:
        for index, row in enumerate(records):
            frame = render_frame(row)
            field = row.get("metric_traversability_field", row)
            valid += int(field.get("status") == "VALID")
            if frames_dir is not None:
                path = frames_dir / f"frame_{index:06d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"failed to write {path}")
            if writer is not None:
                writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    return {
        "schema": "blindassist_hftf_metric_traversability_demo_render_r0",
        "status": "RENDERED_DEVELOPMENT_ONLY_SHADOW_DEMO",
        "frames": len(records),
        "valid_fields": valid,
        "unknown_fields": len(records) - valid,
        "frames_dir": str(frames_dir.resolve()) if frames_dir is not None else None,
        "video_path": str(video_path.resolve()) if video_path is not None else None,
        "claim_ceiling": "display mechanism only; rendered frames are not algorithm evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = render(
        load_records(args.input),
        frames_dir=args.frames_dir,
        video_path=args.video,
        fps=args.fps,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
