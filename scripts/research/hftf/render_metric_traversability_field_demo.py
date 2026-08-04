#!/usr/bin/env python3
"""Render a four-panel DEVELOPMENT_ONLY MetricTraversabilityField demo."""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720
PANEL_W = 616
TOP_PANEL_H = 278
BOTTOM_PANEL_H = 286
BACKGROUND = (17, 21, 27)
CARD = (29, 35, 44)
GRID = (57, 65, 76)
TEXT = (240, 243, 247)
MUTED = (164, 172, 184)
UNKNOWN = (118, 124, 133)
CLEAR = (72, 196, 112)
OCCUPIED = (66, 82, 235)
ACCENT = (231, 181, 67)


@lru_cache(maxsize=32)
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


def _text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    size = max(10, round(scale * 31))
    font = _font(size, thickness >= 2)
    scratch = Image.new("RGBA", (1, 1))
    bbox = ImageDraw.Draw(scratch).textbbox((0, 0), value, font=font)
    width = max(1, bbox[2] - bbox[0] + 2)
    height = max(1, bbox[3] - bbox[1] + 2)
    patch = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text(
        (-bbox[0] + 1, -bbox[1] + 1),
        value,
        font=font,
        fill=(color[2], color[1], color[0], 255),
    )
    x = max(0, origin[0])
    y = max(0, origin[1] - height)
    width = min(width, image.shape[1] - x)
    height = min(height, image.shape[0] - y)
    if width <= 0 or height <= 0:
        return
    rgba = np.asarray(patch)[:height, :width]
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    foreground = rgba[:, :, :3][:, :, ::-1].astype(np.float32)
    target = image[y : y + height, x : x + width].astype(np.float32)
    image[y : y + height, x : x + width] = np.round(
        foreground * alpha + target * (1.0 - alpha)
    ).astype(np.uint8)


def _scene_title(row: dict[str, Any]) -> str:
    sequence = str(row.get("sequence_id", "real-rgbd-sequence")).lower()
    if "obstructing-box" in sequence:
        return "场景：移动纸箱近距离遮挡"
    if "tracking2" in sequence:
        return "场景：人物横穿开放通道"
    if "tracking" in sequence:
        return "场景：人物与移动机器人动态交互"
    return str(row.get("sequence_id", "REAL RGB-D SEQUENCE")).upper()[:54]


def _status_color(status: str) -> tuple[int, int, int]:
    return CLEAR if status == "VALID" else (80, 175, 235)


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
    panel = np.full((height, width, 3), CARD, dtype=np.uint8)
    if image is None:
        _text(panel, "NO DISPLAY ASSET", (40, height // 2), 0.72, UNKNOWN, 2)
    else:
        cover_scale = max(width / image.shape[1], height / image.shape[0])
        cover = cv2.resize(
            image,
            (max(1, round(image.shape[1] * cover_scale)), max(1, round(image.shape[0] * cover_scale))),
            interpolation=cv2.INTER_AREA,
        )
        cover_top = max(0, (cover.shape[0] - height) // 2)
        cover_left = max(0, (cover.shape[1] - width) // 2)
        cover = cover[cover_top : cover_top + height, cover_left : cover_left + width]
        cover = cv2.GaussianBlur(cover, (31, 31), 0)
        panel[:] = (cover.astype(np.float32) * 0.30).astype(np.uint8)
        scale = min((width - 28) / image.shape[1], (height - 20) / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        top = (height - resized.shape[0]) // 2
        left = (width - resized.shape[1]) // 2
        panel[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        cv2.rectangle(
            panel,
            (left - 1, top - 1),
            (left + resized.shape[1], top + resized.shape[0]),
            (95, 105, 117),
            1,
            cv2.LINE_AA,
        )
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (width, 36), (12, 16, 21), -1)
    cv2.addWeighted(overlay, 0.78, panel, 0.22, 0, panel)
    _text(panel, label, (14, 25), 0.55, TEXT, 1)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (66, 75, 87), 1, cv2.LINE_AA)
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


def _friendly_alert(alert: dict[str, Any], field_status: str) -> str:
    if field_status != "VALID":
        return "SCENE  /  NO JUDGMENT"
    status = str(alert.get("status", ""))
    if status == "CENTER_RISK":
        return "SCENE  /  NEAR INTRUSION"
    if status == "SILENT_NO_NEAR_INTRUSION_OBSERVED":
        return "SCENE  /  NO NEAR INTRUSION OBSERVED"
    return "SCENE  /  OBSERVATION ONLY"


def _friendly_shadow_state(alert: dict[str, Any], field_status: str) -> str:
    if field_status != "VALID":
        return "WITHHELD / UNKNOWN"
    status = str(alert.get("status", ""))
    if status == "CENTER_RISK":
        return "NEAR INTRUSION"
    if status == "SILENT_NO_NEAR_INTRUSION_OBSERVED":
        return "NO NEAR INTRUSION OBSERVED"
    return "OBSERVATION ONLY"


def _nearest_intrusion(field: dict[str, Any]) -> float | None:
    distances = [
        float(item["nearest_intrusion_m"])
        for item in field.get("clearance_profile", [])
        if item.get("nearest_intrusion_m") is not None
    ]
    return min(distances) if distances else None


def _scene_state_text(alert: dict[str, Any], field_status: str) -> tuple[str, str]:
    if field_status != "VALID":
        return "主动拒绝判断", "当前几何场不可靠，保留 UNKNOWN"
    status = str(alert.get("status", ""))
    if status == "CENTER_RISK":
        return "前方近距侵入", "中心观测区域出现近距离占用"
    if status == "SILENT_NO_NEAR_INTRUSION_OBSERVED":
        return "未观察到近距侵入", "仅表示当前观测，不代表通行安全"
    return "持续观测中", "当前帧不输出通行方向"


def _draw_rgb_showcase(
    row: dict[str, Any],
    field: dict[str, Any],
    alert: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    panel = _fit_image(_asset(row, "rgb_path"), width, height, "真实 RGB  /  主视图")
    status = str(field.get("status", "UNKNOWN"))
    alert_status = str(alert.get("status", ""))
    if status != "VALID":
        corridor_color = (80, 175, 235)
    elif alert_status == "CENTER_RISK":
        corridor_color = OCCUPIED
    else:
        corridor_color = CLEAR
    corridor = np.asarray(
        [
            [round(width * 0.36), height - 22],
            [round(width * 0.64), height - 22],
            [round(width * 0.55), round(height * 0.58)],
            [round(width * 0.45), round(height * 0.58)],
        ],
        dtype=np.int32,
    )
    overlay = panel.copy()
    cv2.fillConvexPoly(overlay, corridor, corridor_color, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.16, panel, 0.84, 0, panel)
    cv2.polylines(panel, [corridor], True, corridor_color, 2, cv2.LINE_AA)
    cv2.arrowedLine(
        panel,
        (width // 2, height - 58),
        (width // 2, round(height * 0.64)),
        corridor_color,
        2,
        cv2.LINE_AA,
        tipLength=0.22,
    )
    _text(panel, "中心观测区", (round(width * 0.45), height - 34), 0.36, TEXT, 1)
    nearest = _nearest_intrusion(field)
    if nearest is not None:
        cv2.rectangle(panel, (width - 185, 50), (width - 18, 112), (16, 20, 26), -1)
        _text(panel, "最近观测侵入", (width - 170, 73), 0.34, MUTED)
        _text(panel, f"{nearest:.2f} m", (width - 170, 103), 0.66, corridor_color, 2)
    return panel


def _draw_status_summary(
    field: dict[str, Any],
    alert: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), CARD, dtype=np.uint8)
    status = str(field.get("status", "UNKNOWN"))
    title, explanation = _scene_state_text(alert, status)
    state_color = _status_color(status)
    if status == "VALID" and str(alert.get("status")) == "CENTER_RISK":
        state_color = OCCUPIED
    cv2.rectangle(panel, (0, 0), (8, height), state_color, -1)
    _text(panel, "当前画面", (28, 29), 0.36, MUTED)
    _text(panel, title, (28, 68), 0.82, state_color, 2)
    _text(panel, explanation, (28, 94), 0.38, TEXT)

    nearest = _nearest_intrusion(field)
    value = f"{nearest:.2f} m" if nearest is not None else "—"
    _text(panel, "最近观测侵入", (505, 31), 0.35, MUTED)
    _text(panel, value, (505, 73), 0.88, TEXT, 2)

    _text(panel, "几何场状态", (820, 31), 0.35, MUTED)
    _text(panel, "可用" if status == "VALID" else "不确定", (820, 70), 0.72, _status_color(status), 2)
    _text(panel, "不是安全或可通行结论", (820, 94), 0.34, (80, 175, 235))
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (66, 75, 87), 1, cv2.LINE_AA)
    return panel


def _draw_depth(row: dict[str, Any], width: int, height: int) -> np.ndarray:
    panel = _fit_image(
        _asset(row, "metric_depth_heatmap_path"),
        width,
        height,
        "注册米制深度  /  传感器教师信号",
    )
    display_range = (row.get("visualization_assets") or {}).get(
        "metric_depth_display_range_m"
    )
    if display_range and len(display_range) == 2:
        bar_top, bar_bottom = 55, height - 24
        gradient = np.linspace(255, 0, bar_bottom - bar_top, dtype=np.uint8)[:, None]
        gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        panel[bar_top:bar_bottom, width - 25 : width - 13] = gradient
        _text(panel, f"{float(display_range[0]):g}m", (width - 65, bar_top + 5), 0.28, TEXT)
        _text(panel, f"{float(display_range[1]):g}m", (width - 65, bar_bottom), 0.28, TEXT)
        _text(panel, "近", (width - 58, bar_top + 23), 0.27, MUTED)
        _text(panel, "远", (width - 58, bar_bottom - 16), 0.27, MUTED)
    return panel


def _draw_bev(field: dict[str, Any], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), CARD, dtype=np.uint8)
    _text(panel, "俯视空间观测  /  身体扫掠范围", (14, 25), 0.50, TEXT, 1)
    origin = (width // 2, height - 28)
    pixels_per_meter = (height - 64) / 4.0
    for horizon in (1.0, 1.5, 2.0, 3.0, 4.0):
        radius = round(horizon * pixels_per_meter)
        cv2.ellipse(panel, origin, (radius, radius), 0, 220, 320, GRID, 1, cv2.LINE_AA)
        _text(panel, f"{horizon:g}m", (origin[0] + 6, origin[1] - radius + 12), 0.31, MUTED)
    half_width_px = round(0.42 * pixels_per_meter)
    cv2.rectangle(
        panel,
        (origin[0] - half_width_px, origin[1] - round(2.0 * pixels_per_meter)),
        (origin[0] + half_width_px, origin[1]),
        (67, 111, 84),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(panel, origin, 8, ACCENT, -1, cv2.LINE_AA)

    if field.get("status") != "VALID":
        _text(panel, "主动拒绝判断", (origin[0] - 83, height // 2), 0.62, (80, 175, 235), 2)
        reasons = ", ".join(field.get("unknown_reasons", []))[:72]
        _text(panel, reasons, (14, height - 9), 0.31, MUTED)
        return panel

    for envelope in field.get("sweep_envelopes", []):
        horizon = float(envelope["horizon_m"])
        for cell in envelope["directions"]:
            angle = math.radians(float(cell["theta_deg"]))
            endpoint = (
                round(origin[0] + math.sin(angle) * horizon * pixels_per_meter),
                round(origin[1] - math.cos(angle) * horizon * pixels_per_meter),
            )
            cv2.circle(panel, endpoint, 3, _state_color(str(cell["state"])), -1, cv2.LINE_AA)
    for item in field.get("clearance_profile", []):
        distance = item.get("nearest_intrusion_m")
        if distance is None:
            continue
        angle = math.radians(float(item["theta_deg"]))
        point = (
            round(origin[0] + math.sin(angle) * float(distance) * pixels_per_meter),
            round(origin[1] - math.cos(angle) * float(distance) * pixels_per_meter),
        )
        cv2.circle(panel, point, 5, OCCUPIED, -1, cv2.LINE_AA)
    for x, color, label in (
        (14, CLEAR, "已观测空域"),
        (120, OCCUPIED, "侵入"),
        (200, UNKNOWN, "未知"),
    ):
        cv2.circle(panel, (x, height - 10), 4, color, -1, cv2.LINE_AA)
        _text(panel, label, (x + 9, height - 6), 0.28, MUTED)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (66, 75, 87), 1, cv2.LINE_AA)
    return panel


def _draw_profile(field: dict[str, Any], alert: dict[str, Any], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), CARD, dtype=np.uint8)
    profile = field.get("clearance_profile", [])
    distances = [float(item["nearest_intrusion_m"]) for item in profile if item.get("nearest_intrusion_m") is not None]
    nearest = min(distances) if distances else None
    _text(panel, "DIRECTIONAL CLEARANCE PROFILE", (14, 25), 0.53, TEXT, 1)
    if nearest is not None:
        _text(panel, f"NEAREST  {nearest:.2f} m", (430, 25), 0.43, _status_color(str(field.get("status"))))
    left, right, top, bottom = 56, width - 18, 42, height - 62
    cv2.rectangle(panel, (left, top), (right, bottom), GRID, 1, cv2.LINE_AA)
    for distance in (1.0, 1.5, 2.0, 3.0, 4.0):
        y = round(bottom - distance / 4.0 * (bottom - top))
        cv2.line(panel, (left, y), (right, y), GRID, 1, cv2.LINE_AA)
        _text(panel, f"{distance:g}m", (12, y + 4), 0.29, MUTED)
    points = []
    for item in profile:
        theta = float(item["theta_deg"])
        x = round(left + (theta + 40.0) / 80.0 * (right - left))
        distance = item.get("nearest_intrusion_m")
        if distance is None:
            cv2.circle(panel, (x, bottom), 3, UNKNOWN, -1, cv2.LINE_AA)
            points.append(None)
        else:
            y = round(bottom - min(float(distance), 4.0) / 4.0 * (bottom - top))
            cv2.circle(panel, (x, y), 4, OCCUPIED, -1, cv2.LINE_AA)
            points.append((x, y))
    previous = None
    for point in points:
        if point is not None and previous is not None:
            cv2.line(panel, previous, point, ACCENT, 2, cv2.LINE_AA)
        previous = point
    if field.get("status") != "VALID":
        reasons = ", ".join(field.get("unknown_reasons", [])) or "FIELD_UNAVAILABLE"
        _text(panel, "JUDGMENT WITHHELD", (width // 2 - 112, top + 45), 0.58, (80, 175, 235), 2)
        _text(panel, reasons[:76], (width // 2 - 150, top + 72), 0.33, MUTED)
    _text(panel, "LEFT  -40 deg", (left - 4, bottom + 17), 0.29, MUTED)
    _text(panel, "CENTER", ((left + right) // 2 - 31, bottom + 17), 0.29, MUTED)
    _text(panel, "+40 deg  RIGHT", (right - 91, bottom + 17), 0.29, MUTED)
    shadow_state = _friendly_shadow_state(alert, str(field.get("status", "UNKNOWN")))
    _text(panel, f"SHADOW STATE  /  {shadow_state}", (14, height - 30), 0.40, TEXT)
    _text(panel, "NO SAFE-DIRECTION OR NAVIGATION AUTHORITY", (14, height - 10), 0.31, (80, 175, 235))
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (66, 75, 87), 1, cv2.LINE_AA)
    return panel


def render_frame(
    row: dict[str, Any],
    *,
    ordinal: int = 0,
    total_frames: int = 1,
    fps: float = 10.0,
) -> np.ndarray:
    field = row.get("metric_traversability_field", row)
    alert = row.get("shadow_demo_alert_projection", {})
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    canvas[76:566, 16:776] = _draw_rgb_showcase(row, field, alert, 760, 490)
    canvas[76:306, 792:1264] = _draw_depth(row, 472, 230)
    canvas[318:566, 792:1264] = _draw_bev(field, 472, 248)
    canvas[578:682, 16:1264] = _draw_status_summary(field, alert, 1248, 104)

    status = str(field.get("status", "UNKNOWN"))
    field_label = "几何场可用" if status == "VALID" else "几何场不确定"
    scene_title, _explanation = _scene_state_text(alert, status)
    _text(canvas, "BlindAssist  /  实时空间观测", (18, 31), 0.67, TEXT, 2)
    _text(canvas, _scene_title(row), (18, 58), 0.39, MUTED)
    _text(canvas, field_label, (790, 30), 0.43, _status_color(status), 2)
    _text(canvas, scene_title, (790, 58), 0.44, TEXT, 1)

    elapsed_s = ordinal / fps
    duration_s = total_frames / fps
    _text(canvas, f"{elapsed_s:04.1f}s / {duration_s:04.1f}s", (1120, 29), 0.34, MUTED)
    progress = min(1.0, (ordinal + 1) / max(total_frames, 1))
    cv2.rectangle(canvas, (16, 694), (1264, 698), GRID, -1)
    cv2.rectangle(canvas, (16, 694), (16 + round(1248 * progress), 698), ACCENT, -1)
    _text(
        canvas,
        "真实 RGB-D 传感器深度教师信号  /  非纯 RGB 模型输出  /  不构成导航或安全能力证明",
        (18, 716),
        0.30,
        MUTED,
    )
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
            frame = render_frame(
                row,
                ordinal=index,
                total_frames=len(records),
                fps=fps,
            )
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
