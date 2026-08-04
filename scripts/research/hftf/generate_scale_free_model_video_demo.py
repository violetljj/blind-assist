#!/usr/bin/env python3
"""Run the latest RGB-only DA V2 + scale-free operator on real video.

The visualization is a recorded real-time-capable model replay.  It does not
read source depth or metric anchors and cannot claim metres, safety, or a
navigable direction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from produce_external_rgb_metric_depth_observations import (
    DepthAnythingV2MetricSource,
)
from render_metric_traversability_field_demo import (
    ACCENT,
    BACKGROUND,
    CARD,
    CLEAR,
    GRID,
    MUTED,
    OCCUPIED,
    TEXT,
    _fit_image,
    _text,
)


WIDTH = 1280
HEIGHT = 720
BANDS = {
    "left": (0.05, 0.35),
    "center": (0.35, 0.65),
    "right": (0.65, 0.95),
}
LABELS_ZH = {
    "RELATIVELY_OPEN_LEFT": "相对更开阔：左侧",
    "RELATIVELY_OPEN_CENTER": "相对更开阔：中间",
    "RELATIVELY_OPEN_RIGHT": "相对更开阔：右侧",
    "AMBIGUOUS": "三方向差异不明确",
    "UNKNOWN": "模型预热中",
}


class ScaleFreeOperator:
    """Frozen operator from SCALE_FREE_TRAVERSABILITY_R0_PROTOCOL."""

    def __init__(self) -> None:
        self.score_history: deque[dict[str, float]] = deque(maxlen=5)
        self.winner_history: deque[str] = deque(maxlen=5)

    @staticmethod
    def raw_scores(depth: np.ndarray) -> dict[str, Any]:
        height, width = depth.shape
        y0, y1 = round(0.30 * height), round(0.90 * height)
        x0, x1 = round(0.05 * width), round(0.95 * width)
        roi = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64)
        valid = np.isfinite(roi) & (roi > 0)
        intrusion = np.full(roi.shape, np.nan, dtype=np.float64)
        row_valid = np.mean(valid, axis=1) >= 0.90
        for row_index in np.flatnonzero(row_valid):
            values = -np.log(roi[row_index, valid[row_index]])
            baseline = float(np.quantile(values, 0.25))
            q = -np.log(roi[row_index])
            intrusion[row_index] = np.where(
                valid[row_index], np.maximum(0.0, q - baseline), np.nan
            )
        scores = {}
        coverage = {}
        for name, (lower, upper) in BANDS.items():
            band_x0 = round((lower - 0.05) / 0.90 * intrusion.shape[1])
            band_x1 = round((upper - 0.05) / 0.90 * intrusion.shape[1])
            values = intrusion[:, band_x0:band_x1]
            finite = np.isfinite(values)
            coverage[name] = float(np.mean(finite))
            scores[name] = (
                float(np.quantile(values[finite], 0.85))
                if coverage[name] >= 0.90 and np.any(finite)
                else None
            )
        status = "VALID" if all(value is not None for value in scores.values()) else "UNKNOWN"
        return {"status": status, "scores": scores, "coverage": coverage}

    def update(self, depth: np.ndarray) -> dict[str, Any]:
        raw = self.raw_scores(depth)
        if raw["status"] != "VALID":
            return {"status": "UNKNOWN", "reason": "INSUFFICIENT_MODEL_DEPTH", "score": raw}
        scores = {key: float(value) for key, value in raw["scores"].items()}
        winner = min(scores, key=scores.get)
        self.score_history.append(scores)
        self.winner_history.append(winner)
        smoothed = {
            name: float(np.median([item[name] for item in self.score_history]))
            for name in BANDS
        }
        if len(self.score_history) < 5:
            return {
                "status": "UNKNOWN",
                "reason": "UNKNOWN_WARMUP",
                "score": raw,
                "smoothed_scores": smoothed,
            }
        ordered = sorted(smoothed, key=smoothed.get)
        margin = smoothed[ordered[1]] - smoothed[ordered[0]]
        wins = sum(item == ordered[0] for item in self.winner_history)
        if margin < 0.08 or wins < 4:
            status = "AMBIGUOUS"
        else:
            status = f"RELATIVELY_OPEN_{ordered[0].upper()}"
        return {
            "status": status,
            "reason": "FROZEN_SCALE_FREE_OPERATOR",
            "score": raw,
            "smoothed_scores": smoothed,
            "margin": margin,
            "winner_count_last_five": wins,
        }


def _sample_video(
    path: Path,
    *,
    start_s: float,
    duration_s: float,
    target_fps: float,
) -> tuple[list[tuple[int, float, np.ndarray]], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError(f"invalid source FPS: {source_fps}")
    first = round(start_s * source_fps)
    count = max(1, math.floor(duration_s * target_fps))
    targets = [round(first + index * source_fps / target_fps) for index in range(count)]
    target_lookup = {value: index for index, value in enumerate(targets)}
    final = targets[-1]
    output: list[tuple[int, float, np.ndarray]] = []
    frame_index = 0
    try:
        while frame_index <= final:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in target_lookup:
                output.append((frame_index, frame_index / source_fps, frame))
            frame_index += 1
    finally:
        capture.release()
    if len(output) != len(targets):
        raise ValueError(f"requested {len(targets)} frames but decoded {len(output)}")
    return output, source_fps


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.full((*depth.shape, 3), 70, dtype=np.uint8)
    low, high = np.quantile(depth[valid], [0.05, 0.95])
    normalized = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    grayscale = np.where(valid, np.round(255 * (1.0 - normalized)), 0).astype(np.uint8)
    preview = cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)
    preview[~valid] = (70, 74, 80)
    return preview


def _decision_color(status: str) -> tuple[int, int, int]:
    if status.startswith("RELATIVELY_OPEN"):
        return CLEAR
    if status == "AMBIGUOUS":
        return (80, 175, 235)
    return MUTED


def _draw_rgb(bgr: np.ndarray, decision: dict[str, Any], width: int, height: int) -> np.ndarray:
    panel = _fit_image(bgr, width, height, "真实户外 RGB  /  模型唯一输入")
    overlay = panel.copy()
    selected = None
    if decision["status"].startswith("RELATIVELY_OPEN_"):
        selected = decision["status"].removeprefix("RELATIVELY_OPEN_").lower()
    for name, (lower, upper) in BANDS.items():
        x0, x1 = round(lower * width), round(upper * width)
        color = CLEAR if name == selected else GRID
        cv2.rectangle(overlay, (x0, 38), (x1, height - 1), color, -1)
        cv2.rectangle(panel, (x0, 38), (x1, height - 1), color, 2, cv2.LINE_AA)
        _text(panel, {"left": "左", "center": "中", "right": "右"}[name], (x0 + 10, 68), 0.50, TEXT, 2)
    cv2.addWeighted(overlay, 0.10, panel, 0.90, 0, panel)
    return panel


def _bev_point(
    origin: tuple[int, int], radius: float, theta_degrees: float
) -> tuple[int, int]:
    """Project a relative polar point while preserving image-left/image-right."""
    theta = math.radians(theta_degrees)
    return (
        round(origin[0] + math.sin(theta) * radius),
        round(origin[1] - math.cos(theta) * radius),
    )


def _sector_polygon(
    origin: tuple[int, int], inner: float, outer: float, lower: float, upper: float
) -> np.ndarray:
    angles = np.linspace(lower, upper, 12)
    points = [_bev_point(origin, inner, angle) for angle in angles]
    points += [_bev_point(origin, outer, angle) for angle in reversed(angles)]
    return np.asarray(points, dtype=np.int32)


def _draw_relative_bev(decision: dict[str, Any], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), CARD, dtype=np.uint8)
    _text(panel, "模型相对俯视观测  /  无米制尺度", (16, 27), 0.45, TEXT, 1)
    scores = decision.get("smoothed_scores") or decision["score"].get("scores", {})
    finite = [float(value) for value in scores.values() if value is not None]
    low = min(finite) if finite else 0.0
    high = max(finite) if finite else 1.0
    spread = max(high - low, 1e-6)
    origin = (width // 2, height - 42)
    outer = min(150, height - 73)

    # These rings are deliberately dimensionless. They organize the display
    # and must never be read as recovered distance.
    for radius in (55, 100, outer):
        arc = np.asarray(
            [_bev_point(origin, radius, angle) for angle in np.linspace(-48, 48, 50)],
            dtype=np.int32,
        )
        cv2.polylines(panel, [arc], False, GRID, 1, cv2.LINE_AA)

    selected = None
    if decision["status"].startswith("RELATIVELY_OPEN_"):
        selected = decision["status"].removeprefix("RELATIVELY_OPEN_").lower()
    sectors = {
        "left": (-45.0, -15.0),
        "center": (-15.0, 15.0),
        "right": (15.0, 45.0),
    }
    overlay = panel.copy()
    for name, (lower_angle, upper_angle) in sectors.items():
        value = scores.get(name)
        normalized = 0.5 if value is None else (float(value) - low) / spread
        # A stronger relative intrusion begins nearer the observer. This is a
        # rank visualization only, not a depth-to-ground projection.
        intrusion_inner = outer - normalized * (outer - 48)
        intrusion_color = (72, 92, 210) if value is not None else MUTED
        cv2.fillPoly(
            overlay,
            [_sector_polygon(origin, intrusion_inner, outer, lower_angle, upper_angle)],
            intrusion_color,
            cv2.LINE_AA,
        )
        if name == selected:
            cv2.fillPoly(
                overlay,
                [_sector_polygon(origin, 28, intrusion_inner, lower_angle, upper_angle)],
                CLEAR,
                cv2.LINE_AA,
            )
        boundary = _sector_polygon(origin, 28, outer, lower_angle, upper_angle)
        cv2.polylines(panel, [boundary], True, GRID, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.58, panel, 0.42, 0, panel)

    cv2.circle(panel, origin, 8, (220, 190, 70), -1, cv2.LINE_AA)
    cv2.line(panel, (origin[0] - 16, origin[1]), (origin[0] + 16, origin[1]), (220, 190, 70), 2)
    cv2.line(panel, origin, (origin[0], origin[1] - 19), (220, 190, 70), 2)
    _text(panel, "画面左", (36, height - 19), 0.30, MUTED)
    _text(panel, "前方", (origin[0] - 25, 54), 0.31, MUTED)
    _text(panel, "画面右", (width - 91, height - 19), 0.30, MUTED)

    score_text = "  ".join(
        f"{label} {float(scores[name]):.3f}" if scores.get(name) is not None else f"{label} --"
        for name, label in (("left", "左"), ("center", "中"), ("right", "右"))
    )
    _text(panel, score_text, (15, height - 5), 0.27, TEXT)
    _text(panel, "红＝相对侵入较强；绿＝相对较弱", (15, 48), 0.27, (80, 175, 235))
    _text(panel, "仅相对分布，不是米制 BEV 或安全方向", (width - 240, 48), 0.25, (80, 175, 235))
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (66, 75, 87), 1, cv2.LINE_AA)
    return panel


def render_frame(
    bgr: np.ndarray,
    depth: np.ndarray,
    decision: dict[str, Any],
    *,
    scene_title: str,
    latency_ms: float,
    ordinal: int,
    total: int,
    fps: float,
) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    canvas[76:612, 16:776] = _draw_rgb(bgr, decision, 760, 536)
    canvas[76:336, 792:1264] = _fit_image(
        _depth_preview(depth), 472, 260, "DA V2 模型深度  /  相对色标"
    )
    canvas[350:612, 792:1264] = _draw_relative_bev(decision, 472, 262)
    status = decision["status"]
    _text(canvas, "BlindAssist  /  最新 RGB 模型实时演示", (18, 31), 0.65, TEXT, 2)
    _text(canvas, scene_title, (18, 58), 0.39, MUTED)
    _text(canvas, LABELS_ZH.get(status, status), (790, 31), 0.51, _decision_color(status), 2)
    _text(canvas, f"DA V2  {latency_ms:.1f} ms  /  {1000.0 / latency_ms:.1f} FPS", (790, 58), 0.34, MUTED)
    _text(canvas, f"{ordinal / fps:04.1f}s / {total / fps:04.1f}s", (1122, 31), 0.32, MUTED)
    progress = (ordinal + 1) / max(total, 1)
    cv2.rectangle(canvas, (16, 632), (1264, 637), GRID, -1)
    cv2.rectangle(canvas, (16, 632), (16 + round(1248 * progress), 637), ACCENT, -1)
    _text(canvas, "模型：Depth Anything V2 Metric Hypersim ViT-S  /  输入：仅 RGB  /  GPU 逐帧真实推理", (18, 670), 0.34, TEXT)
    _text(canvas, "DEVELOPMENT_ONLY：相对方向诊断，不输出米制距离、安全方向、导航或产品能力结论", (18, 706), 0.31, (80, 175, 235))
    return canvas


def generate(
    video_path: Path,
    output_dir: Path,
    *,
    scene_title: str,
    repo: Path,
    checkpoint: Path,
    start_s: float,
    duration_s: float,
    fps: float,
    device: str,
    precision: str,
) -> dict[str, Any]:
    sampled, source_fps = _sample_video(
        video_path, start_s=start_s, duration_s=duration_s, target_fps=fps
    )
    model = DepthAnythingV2MetricSource(
        repo=repo,
        checkpoint=checkpoint,
        device=device,
        input_size=518,
        precision=precision,
    )
    operator = ScaleFreeOperator()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_output = output_dir / "latest_rgb_model_realtime_demo.mp4"
    writer = cv2.VideoWriter(
        str(video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {video_output}")
    rows = []
    latencies = []
    preview_ordinals = {0, len(sampled) // 2, len(sampled) - 1}
    try:
        for ordinal, (source_frame, source_time_s, bgr) in enumerate(sampled):
            started = time.perf_counter()
            depth, metadata = model.infer(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                {"frame_index": source_frame},
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            decision = operator.update(depth)
            frame = render_frame(
                bgr,
                depth,
                decision,
                scene_title=scene_title,
                latency_ms=latency_ms,
                ordinal=ordinal,
                total=len(sampled),
                fps=fps,
            )
            writer.write(frame)
            if ordinal in preview_ordinals:
                cv2.imwrite(str(output_dir / f"preview_{ordinal:06d}.png"), frame)
            rows.append(
                {
                    "schema": "blindassist_latest_rgb_model_realtime_demo_frame_r0",
                    "ordinal": ordinal,
                    "source_frame": source_frame,
                    "source_time_s": source_time_s,
                    "inference_latency_ms": latency_ms,
                    "model_metadata": metadata,
                    "decision": decision,
                    "rgb_only_model_input": True,
                    "source_depth_read": False,
                    "metric_distance_authority": False,
                    "safety_authority": False,
                }
            )
    finally:
        writer.release()
    frames_jsonl = output_dir / "frames.jsonl"
    frames_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema": "blindassist_latest_rgb_model_realtime_demo_summary_r0",
        "status": "RECORDED_REALTIME_CAPABLE_RGB_MODEL_REPLAY_RENDERED",
        "source_video": str(video_path.resolve()),
        "scene_title": scene_title,
        "source_fps": source_fps,
        "output_fps": fps,
        "frames": len(rows),
        "duration_s": len(rows) / fps,
        "model": "Depth Anything V2 Metric Hypersim ViT-S",
        "operator": "frozen Scale-Free Traversability R0",
        "device": device,
        "precision": precision,
        "inference_latency_ms": {
            "median": float(np.median(latencies)),
            "p95": float(np.quantile(latencies, 0.95)),
            "maximum": float(np.max(latencies)),
        },
        "throughput_fps_from_median_inference": 1000.0 / float(np.median(latencies)),
        "rgb_only_model_input": True,
        "source_depth_read": False,
        "metric_distance_authority": False,
        "safety_authority": False,
        "video": str(video_output.resolve()),
        "frames_jsonl": str(frames_jsonl.resolve()),
        "claim_ceiling": "recorded model replay and scale-free diagnostic only; not metric distance, safe direction, navigation, or production evidence",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-title", required=True)
    parser.add_argument("--depth-anything-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.video,
                args.output_dir,
                scene_title=args.scene_title,
                repo=args.depth_anything_repo,
                checkpoint=args.checkpoint,
                start_s=args.start_s,
                duration_s=args.duration_s,
                fps=args.fps,
                device=args.device,
                precision=args.precision,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
