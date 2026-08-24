"""Render the SAGE-LM observation ladder and one source-pose frame pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _text(canvas: np.ndarray, value: str, position: tuple[int, int], scale: float, color=(35, 35, 35), thickness: int = 2) -> None:
    cv2.putText(canvas, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _overlay_lines(image: np.ndarray, lines: list[dict], color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    height = output.shape[0]
    for line in lines:
        cv2.line(output, (int(round(line["x_top_px"])), 0), (int(round(line["x_bottom_px"])), height - 1), color, 3, cv2.LINE_AA)
    return output


def render(v1: dict, v1a: dict, v1b: dict, cohort: dict, output: Path) -> None:
    canvas = np.full((1000, 1800, 3), 248, dtype=np.uint8)
    _text(canvas, "SAGE-LM observation ladder", (45, 65), 1.15, thickness=3)
    _text(canvas, "Same frozen 24 episodes and downstream policy", (48, 105), 0.58, (95, 95, 95), 1)
    cv2.rectangle(canvas, (40, 145), (790, 690), (255, 255, 255), -1)
    cv2.rectangle(canvas, (40, 145), (790, 690), (150, 150, 150), 1)
    headers = ((70, "Observation arm"), (470, "Arrival"), (625, "Median error"))
    for x, label in headers:
        _text(canvas, label, (x, 190), 0.55, thickness=2)
    cv2.line(canvas, (60, 215), (770, 215), (150, 150, 150), 1)
    baseline = v1["metrics"]["bbox_center_scale"]
    v1_rgb = v1["metrics"]["sage_lm"]
    v1a_oracle = v1a["metrics"]["sage_lm"]

    def metric_row(label: str, metrics: dict, color: tuple[int, int, int]) -> tuple[str, str, str, tuple[int, int, int]]:
        arrivals = int(round(metrics["target_front_arrival_rate"] * 24))
        return label, f"{arrivals}/24", f"{metrics['median_endpoint_lateral_error_m']:.3f} m", color

    rows = [
        metric_row("bbox baseline", baseline, (75, 100, 235)),
        metric_row("V1 naive RGB", v1_rgb, (40, 40, 190)),
        ("V1-B pose RGB", "N/E", "N/E", (30, 145, 210)),
        metric_row("V1-A all-oracle", v1a_oracle, (65, 175, 70)),
    ]
    for index, (name, arrival, error, color) in enumerate(rows):
        y = 280 + index * 95
        _text(canvas, name, (70, y), 0.68, color, 2)
        _text(canvas, arrival, (485, y), 0.68, color, 2)
        _text(canvas, error, (635, y), 0.68, color, 2)
    _text(canvas, "V1-B N/E: only 2/24 frozen pairs satisfy the intended source-pose motion gate.", (65, 650), 0.48, (35, 35, 180), 1)

    valid_rows = [row for row in v1b["rows"] if row["source_pose_audit"]["intended_active_pair_gate_pass"]]
    selected = valid_rows[0]
    materialized = next(row for row in cohort["episodes"] if row["input"]["episode_id"] == selected["episode_id"])
    active_index = materialized["input"]["active_parallax_frame_index"]
    frame_a = cv2.imread(materialized["input"]["rgb_frames"][0], cv2.IMREAD_COLOR)
    frame_b = cv2.imread(materialized["input"]["rgb_frames"][active_index], cv2.IMREAD_COLOR)
    if frame_a is None or frame_b is None:
        raise ValueError("unable to load ladder frame pair")
    diag = selected["b2"]["diagnostics"]
    frame_a = _overlay_lines(frame_a, diag.get("selected_lines_a", []), (40, 220, 75))
    frame_b = _overlay_lines(frame_b, diag.get("selected_lines_b", []), (40, 220, 75))
    frame_a = cv2.resize(frame_a, (430, 323), interpolation=cv2.INTER_CUBIC)
    frame_b = cv2.resize(frame_b, (430, 323), interpolation=cv2.INTER_CUBIC)
    canvas[180:503, 870:1300] = frame_a
    canvas[180:503, 1330:1760] = frame_b
    _text(canvas, f"Frame A | {selected['episode_id']}", (875, 165), 0.48, thickness=1)
    _text(canvas, "Frame B | detected line candidates", (1335, 165), 0.48, thickness=1)
    _text(canvas, "source-native pose", (1120, 555), 0.62, (120, 80, 20), 2)
    cv2.arrowedLine(canvas, (1050, 575), (1570, 575), (120, 80, 20), 3, tipLength=0.04)
    _text(canvas, "image lines -> interpretation planes -> 3D boundary lines -> aperture", (900, 625), 0.52, (55, 55, 55), 1)
    _text(canvas, "LK: not run", (925, 680), 0.55, (35, 125, 35), 2)
    _text(canvas, "monocular metric depth: not run", (1190, 680), 0.55, (35, 125, 35), 2)

    cv2.rectangle(canvas, (40, 740), (1760, 950), (235, 240, 250), -1)
    _text(canvas, "Adjudication", (70, 790), 0.72, (35, 35, 150), 2)
    _text(canvas, "The original V1 materializer read rotation-vector columns as camera positions.", (70, 840), 0.63, (35, 35, 150), 2)
    _text(canvas, "V1-B is NOT EVALUABLE on the frozen pairs; B1/B2 raw outcomes are not boundary-route negatives.", (70, 890), 0.63, (35, 35, 150), 2)
    _text(canvas, "No VO, policy, threshold, baseline, or cohort selection was changed.", (70, 930), 0.52, (75, 75, 75), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"unable to write {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--v1a-report", type=Path, required=True)
    parser.add_argument("--v1b-report", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(
        json.loads(args.v1_report.read_text(encoding="utf-8")),
        json.loads(args.v1a_report.read_text(encoding="utf-8")),
        json.loads(args.v1b_report.read_text(encoding="utf-8")),
        json.loads(args.cohort.read_text(encoding="utf-8")),
        args.output,
    )
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
