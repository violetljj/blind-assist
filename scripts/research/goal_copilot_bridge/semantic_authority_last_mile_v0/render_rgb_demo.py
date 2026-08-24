"""Render the required SAGE-LM V1 observation and paired-control artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _writer(path: Path, size: tuple[int, int], fps: float = 12.0) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise OSError(f"unable to open MP4 writer: {path}")
    return writer


def _episode_input(cohort: dict, episode_id: str) -> dict:
    return next(row for row in cohort["episodes"] if row["input"]["episode_id"] == episode_id)


def _selected(report: dict) -> dict:
    rescues = [row for row in report["rows"] if row["sage_lm"]["true_arrival"] and not row["baseline"]["true_arrival"]]
    if rescues:
        return min(rescues, key=lambda row: row["sage_lm"]["endpoint_lateral_error_m"])
    return max(report["rows"], key=lambda row: row["sage_lm"]["geometry_confidence"])


def _put(frame: np.ndarray, text: str, y: int, color=(245, 245, 245), scale=0.55) -> None:
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_observation(row: dict, materialized: dict, output: Path) -> None:
    diagnostics = row["sage_lm"]["diagnostics"]
    observation = row["sage_lm"]["observation"]
    anchors = materialized["input"]["exact_anchor_observations"]
    paths = materialized["input"]["rgb_frames"]
    scale = 3
    writer = _writer(output, (256 * scale, 192 * scale))
    try:
        for index, path in enumerate(paths):
            base = cv2.imread(path, cv2.IMREAD_COLOR)
            frame = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            anchor = anchors[index]["bbox_xyxy"]
            if anchor is not None:
                x1, y1, x2, y2 = [int(value * scale) for value in anchor]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 255), 2)
                cv2.putText(frame, "EXACT ANCHOR", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 210, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (30, 30, 180), -1)
                _put(frame, "LOST -> STOP", 24, (255, 255, 255), 0.55)
            if "boundary_x_px" in diagnostics:
                left, right = diagnostics["boundary_x_px"]
                for x in (left, right):
                    cv2.line(frame, (int(x * scale), 50), (int(x * scale), frame.shape[0] - 35), (60, 235, 70), 3)
                centre = int((left + right) * 0.5 * scale)
                cv2.line(frame, (centre, 45), (centre, frame.shape[0] - 30), (255, 90, 40), 2)
            if index == diagnostics.get("frame_indices", [-1, -1])[1]:
                for x1, y1, x2, y2 in diagnostics.get("flow_tracks", []):
                    cv2.arrowedLine(frame, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), (245, 80, 220), 1, tipLength=0.3)
            cv2.rectangle(frame, (0, frame.shape[0] - 70), (frame.shape[1], frame.shape[0]), (22, 22, 22), -1)
            _put(frame, f"center={observation['center_x_m']:.2f}m  range={observation['range_m']:.2f}m  conf={observation['geometry_confidence']:.2f}", frame.shape[0] - 42)
            _put(frame, f"boundary={observation['boundary_confidence']:.2f}  flow={observation['flow_confidence']:.2f}  depth={observation['depth_consistency']:.2f}", frame.shape[0] - 16, (190, 230, 255), 0.48)
            for _ in range(6):
                writer.write(frame)
    finally:
        writer.release()


def _trajectory_inset(canvas: np.ndarray, path: list, truth: dict, upto: int, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x0, y0 = origin
    w, h = 565, 235
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (246, 246, 246), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (120, 120, 120), 1)
    centre = x0 + w // 2
    wall = y0 + h - 20
    scale_x, scale_z = 130.0, 28.0
    aperture_x = truth["aperture_center_x_m"]
    aperture_w = truth["aperture_width_m"]
    left = int(centre + (aperture_x - aperture_w / 2) * scale_x)
    right = int(centre + (aperture_x + aperture_w / 2) * scale_x)
    cv2.line(canvas, (x0 + 10, wall), (left, wall), (70, 70, 70), 5)
    cv2.line(canvas, (right, wall), (x0 + w - 10, wall), (70, 70, 70), 5)
    cv2.line(canvas, (left, wall), (right, wall), (60, 190, 80), 3)
    points = [(int(centre + px * scale_x), int(wall - pz * scale_z)) for px, pz in path[: max(2, upto)]]
    for a, b in zip(points, points[1:]):
        cv2.line(canvas, a, b, color, 3, cv2.LINE_AA)
    cv2.circle(canvas, points[-1], 6, color, -1)


def render_paired(row: dict, materialized: dict, output: Path) -> None:
    writer = _writer(output, (1280, 720))
    frames = materialized["input"]["rgb_frames"]
    total = len(frames) * 6
    try:
        for tick in range(total):
            source = cv2.imread(frames[min(len(frames) - 1, tick // 6)], cv2.IMREAD_COLOR)
            background = cv2.resize(source, (620, 420), interpolation=cv2.INTER_CUBIC)
            canvas = np.full((720, 1280, 3), 24, dtype=np.uint8)
            canvas[55:475, 10:630] = background
            canvas[55:475, 650:1270] = background
            cv2.putText(canvas, "BASELINE: anchor bbox center + scale", (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (75, 100, 245), 2, cv2.LINE_AA)
            cv2.putText(canvas, "SAGE-LM: RGB aperture observation", (668, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (85, 215, 90), 2, cv2.LINE_AA)
            progress = tick / max(1, total - 1)
            base_upto = 2 + int(progress * max(0, len(row["baseline"]["path"]) - 2))
            sage_upto = 2 + int(progress * max(0, len(row["sage_lm"]["path"]) - 2))
            _trajectory_inset(canvas, row["baseline"]["path"], row["truth"], base_upto, (35, 480), (75, 100, 245))
            _trajectory_inset(canvas, row["sage_lm"]["path"], row["truth"], sage_upto, (680, 480), (85, 180, 65))
            cv2.putText(canvas, f"endpoint error {row['baseline']['endpoint_lateral_error_m']:.2f}m | {'ARRIVED' if row['baseline']['true_arrival'] else 'MISSED'}", (45, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (75, 100, 245), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"endpoint error {row['sage_lm']['endpoint_lateral_error_m']:.2f}m | {'ARRIVED' if row['sage_lm']['true_arrival'] else 'MISSED'}", (690, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (85, 215, 90), 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()


def render_trajectory(report: dict, output: Path) -> None:
    canvas = np.full((900, 1740, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "SAGE-LM V1 controlled real-RGB observation: representative trajectories", (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (30, 30, 30), 2, cv2.LINE_AA)
    for column, kind in enumerate(("ROOM_SIGN", "QR_ENTRANCE", "EXACT_SHELF_TARGET")):
        rows = [row for row in report["rows"] if row["kind"] == kind]
        row = max(rows, key=lambda value: value["baseline"]["endpoint_lateral_error_m"] - value["sage_lm"]["endpoint_lateral_error_m"])
        x = 20 + column * 490
        cv2.putText(canvas, f"{kind} | {row['episode_id']}", (x + 20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        _trajectory_inset(canvas, row["baseline"]["path"], row["truth"], len(row["baseline"]["path"]), (x, 110), (75, 100, 245))
        _trajectory_inset(canvas, row["sage_lm"]["path"], row["truth"], len(row["sage_lm"]["path"]), (x, 410), (85, 180, 65))
        cv2.putText(canvas, "baseline", (x + 12, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 100, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, "SAGE-LM", (x + 12, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (85, 180, 65), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Development result: 2/24 SAGE-LM arrivals vs 7/24 baseline; observation adapter did not preserve V0 uplift.", (35, 850), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (35, 35, 170), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"failed to write {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected(report)
    selection_reason = (
        "SAGE_TRUE_ARRIVAL_BASELINE_MISS_LOWEST_SAGE_ERROR"
        if selected["sage_lm"]["true_arrival"] and not selected["baseline"]["true_arrival"]
        else "BEST_GEOMETRY_CONFIDENCE_NO_RESCUE_AVAILABLE"
    )
    materialized = _episode_input(cohort, selected["episode_id"])
    render_observation(selected, materialized, args.output_dir / "observation_overlay.mp4")
    render_paired(selected, materialized, args.output_dir / "baseline_vs_sage_lm.mp4")
    render_trajectory(report, args.output_dir / "trajectory_demo.png")
    (args.output_dir / "demo_selection.json").write_text(
        json.dumps({"episode_id": selected["episode_id"], "selection": selection_reason}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"episode_id": selected["episode_id"], "artifacts": ["observation_overlay.mp4", "baseline_vs_sage_lm.mp4", "trajectory_demo.png"]}, indent=2))


if __name__ == "__main__":
    main()
