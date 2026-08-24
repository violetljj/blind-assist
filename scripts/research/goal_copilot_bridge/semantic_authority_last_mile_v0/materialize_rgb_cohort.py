"""Materialize 24 curated real-ARKitScenes RGB episodes for SAGE-LM V1.

Depth and trajectory are used only here to choose/evaluate a stable physical
opening proxy.  The emitted algorithm input contains composited RGB, camera
intrinsics, commanded baseline and exact-anchor observations, while truth is
written to a separate evaluator object.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from scripts.research.assistive_geometry.arkitscenes_truth_reader import interpolate_camera_to_world, parse_trajectory

from .experiment import KINDS


def _trajectory(path: Path) -> np.ndarray:
    return parse_trajectory(path)


def _frame_pose(frame: Path, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    timestamp = float(frame.stem.rsplit("_", 1)[1])
    transform, _ = interpolate_camera_to_world(trajectory, timestamp, maximum_gap_seconds=0.25)
    return transform[:3, 3], transform[:3, :3]


def _active_pair(frames: list[Path], trajectory: np.ndarray) -> tuple[int, list[list[float]], float] | None:
    poses = [_frame_pose(frame, trajectory) for frame in frames]
    start_position, start_rotation = poses[0]
    choices = []
    for index, (position, _) in enumerate(poses[2:], start=2):
        delta_camera = start_rotation.T @ (position - start_position)
        lateral = float(abs(delta_camera[0]))
        forward = float(abs(delta_camera[2]))
        if 0.18 <= lateral <= 0.30 and forward <= 0.45:
            choices.append((abs(lateral - 0.24) + 0.2 * forward, index, lateral))
    if not choices:
        return None
    _, index, lateral = min(choices)
    positions = [[float(value) for value in position] for position, _ in poses]
    return index, positions, lateral


def _intrinsics(path: Path) -> dict:
    width, height, fx, fy, cx, cy = (float(value) for value in path.read_text().split())
    return {"width": int(width), "height": int(height), "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def _metric_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"unable to decode source depth: {path}")
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size and float(np.median(finite)) > 50.0:
        depth /= 1000.0
    return depth


def _vertical_lines(bgr: np.ndarray) -> list[tuple[float, float]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 18, minLineLength=42, maxLineGap=14)
    values = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
            if dy >= 42 and dx <= max(7, dy * 0.18):
                values.append(((int(x1) + int(x2)) / 2.0, float(dy)))
    values.sort()
    merged: list[tuple[float, float]] = []
    for x, strength in values:
        if merged and abs(x - merged[-1][0]) <= 5:
            px, ps = merged[-1]
            merged[-1] = ((px * ps + x * strength) / (ps + strength), ps + strength)
        else:
            merged.append((x, strength))
    return merged


def _opening_truth(bgr: np.ndarray, depth: np.ndarray, intr: dict) -> dict | None:
    h, w = depth.shape
    best = None
    lines = _vertical_lines(bgr)
    for index, (left, ls) in enumerate(lines):
        for right, rs in lines[index + 1 :]:
            span = right - left
            if span < w * 0.16 or span > w * 0.58:
                continue
            lx, rx = int(left), int(right)
            y0, y1 = int(h * 0.30), int(h * 0.78)
            inner = depth[y0:y1, lx + 3 : rx - 3]
            inner = inner[np.isfinite(inner) & (inner > 0.25) & (inner < 8.0)]
            outer = np.concatenate(
                [depth[y0:y1, max(0, lx - 9) : max(0, lx - 3)].ravel(), depth[y0:y1, min(w, rx + 3) : min(w, rx + 9)].ravel()]
            )
            outer = outer[np.isfinite(outer) & (outer > 0.25) & (outer < 8.0)]
            if inner.size < 120 or outer.size < 30:
                continue
            range_m = float(np.median(inner))
            width_m = span * range_m / intr["fx"]
            if not 0.50 <= width_m <= 1.65:
                continue
            discontinuity = abs(range_m - float(np.median(outer))) / max(0.25, min(range_m, float(np.median(outer))))
            centre_px = (left + right) / 2.0
            centre_x_m = (centre_px - intr["cx"]) * range_m / intr["fx"]
            edge_score = min(1.0, min(ls, rs) / 130.0)
            score = 0.62 * min(1.0, discontinuity / 0.24) + 0.38 * edge_score
            row = {
                "left_x_px": left,
                "right_x_px": right,
                "center_x_m": centre_x_m,
                "width_m": width_m,
                "range_m": range_m,
                "selection_score": score,
                "source_depth_discontinuity": discontinuity,
            }
            if best is None or score > best[0]:
                best = (score, row)
    return best[1] if best else None


def _draw_anchor(image: np.ndarray, bbox: tuple[int, int, int, int], referent: str, kind: str) -> np.ndarray:
    output = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(output, (x1, y1), (x2, y2), (245, 245, 245), -1)
    cv2.rectangle(output, (x1, y1), (x2, y2), (18, 18, 18), 2)
    cells = 5
    cell = max(2, min((x2 - x1) // cells, (y2 - y1) // cells))
    seed = sum(ord(char) for char in referent)
    for row in range(cells):
        for column in range(cells):
            if ((row * 7 + column * 11 + seed) % 5) < 2:
                xa, ya = x1 + column * cell + 2, y1 + row * cell + 2
                cv2.rectangle(output, (xa, ya), (xa + cell - 1, ya + cell - 1), (15, 15, 15), -1)
    text = "ROOM" if kind == "ROOM_SIGN" else "QR" if kind == "QR_ENTRANCE" else "SKU"
    cv2.putText(output, text, (x1, min(output.shape[0] - 3, y2 + 11)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (20, 190, 245), 1, cv2.LINE_AA)
    return output


def _source_candidates(source_root: Path) -> list[dict]:
    candidates = []
    for rgb_dir in sorted(source_root.glob("*/*/lowres_wide")):
        sequence = rgb_dir.parent
        trajectory_path = sequence / "lowres_wide.traj"
        if not trajectory_path.exists():
            continue
        trajectory = _trajectory(trajectory_path)
        depth_dir = sequence / "lowres_depth"
        intr_dir = sequence / "lowres_wide_intrinsics"
        frames = sorted(rgb_dir.glob("*.png"))
        for start in range(8, max(8, len(frames) - 18), 6):
            frame = frames[start]
            stem = frame.stem
            depth_path = depth_dir / f"{stem}.png"
            intr_path = intr_dir / f"{stem}.pincam"
            if not depth_path.exists() or not intr_path.exists():
                continue
            bgr = cv2.imread(str(frame), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            window = frames[start : start + 16]
            if len(window) < 12:
                continue
            active_pair = _active_pair(window, trajectory)
            if active_pair is None:
                continue
            active_index, camera_positions, actual_lateral = active_pair
            intr = _intrinsics(intr_path)
            truth = _opening_truth(bgr, _metric_depth(depth_path), intr)
            if truth is None or truth["selection_score"] < 0.58:
                continue
            candidates.append(
                {
                    "sequence": sequence,
                    "frames": window,
                    "intrinsics": intr,
                    "truth": truth,
                    "active_parallax_frame_index": active_index,
                    "camera_positions_m": camera_positions,
                    "actual_lateral_baseline_m": actual_lateral,
                }
            )
    candidates.sort(key=lambda row: row["truth"]["selection_score"], reverse=True)
    return candidates


def materialize(source_root: Path, output_dir: Path, seed: int = 240824) -> dict:
    rng = random.Random(seed)
    candidates = _source_candidates(source_root)
    if len(candidates) < 24:
        raise RuntimeError(f"only {len(candidates)} source windows satisfy the real-RGB opening proxy gate")
    selected = []
    per_sequence: dict[str, int] = {}
    for candidate in candidates:
        key = str(candidate["sequence"])
        if per_sequence.get(key, 0) >= 2:
            continue
        selected.append(candidate)
        per_sequence[key] = per_sequence.get(key, 0) + 1
        if len(selected) == 24:
            break
    if len(selected) < 24:
        selected = candidates[:24]

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for index, candidate in enumerate(selected):
        kind = KINDS[index // 8]
        within_kind = index % 8
        control = within_kind >= 6
        truth = candidate["truth"]
        episode_id = f"{kind.lower()}-{within_kind + 1:02d}"
        referent_id = f"SAGE-LM-V1::{episode_id}"
        episode_dir = output_dir / "episodes" / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        anchors = []
        left, right = truth["left_x_px"], truth["right_x_px"]
        span = right - left
        anchor_w, anchor_h = 30, 30
        if control:
            anchor_x = (left + right) / 2.0 + rng.choice((-1, 1)) * span * 0.08
        else:
            anchor_x = left - span * 0.28 if within_kind % 2 == 0 else right + span * 0.28
        anchor_x = int(np.clip(anchor_x, anchor_w // 2 + 2, candidate["intrinsics"]["width"] - anchor_w // 2 - 2))
        anchor_y = int(candidate["intrinsics"]["height"] * (0.36 if kind != "EXACT_SHELF_TARGET" else 0.58))
        bbox = (anchor_x - anchor_w // 2, anchor_y - anchor_h // 2, anchor_x + anchor_w // 2, anchor_y + anchor_h // 2)
        lost_indices: set[int] = set()
        if within_kind in (1, 4):
            lost_indices = next(
                set(pair)
                for pair in ((5, 6), (9, 10), (12, 13))
                if candidate["active_parallax_frame_index"] not in pair
            )
        for frame_index, source_path in enumerate(candidate["frames"]):
            bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            visible = frame_index not in lost_indices
            composited = _draw_anchor(bgr, bbox, referent_id, kind) if visible else bgr
            destination = episode_dir / f"frame_{frame_index:03d}.png"
            if not cv2.imwrite(str(destination), composited):
                raise OSError(f"failed to write {destination}")
            frame_paths.append(str(destination.resolve()))
            anchors.append({"frame_index": frame_index, "referent_id": referent_id, "bbox_xyxy": list(bbox) if visible else None})
        anchor_center_x_m = (anchor_x - candidate["intrinsics"]["cx"]) * truth["range_m"] / candidate["intrinsics"]["fx"]
        episodes.append(
            {
                "input": {
                    "episode_id": episode_id,
                    "kind": kind,
                    "rgb_frames": frame_paths,
                    "intrinsics": candidate["intrinsics"],
                    "commanded_baseline_m": 0.24,
                    "active_parallax_frame_index": candidate["active_parallax_frame_index"],
                    "exact_anchor_observations": anchors,
                },
                "truth": {
                    "episode_id": episode_id,
                    "aperture_center_x_m": truth["center_x_m"],
                    "aperture_width_m": truth["width_m"],
                    "start_range_m": truth["range_m"],
                    "camera_positions_m": candidate["camera_positions_m"],
                    "endpoint_center_x_m": truth["center_x_m"],
                    "source_boundary_x_px": [left, right],
                    "source_depth_discontinuity": truth["source_depth_discontinuity"],
                },
                "baseline_anchor_x_m": anchor_center_x_m,
                "control": control,
                "occlusion_frame_indices": sorted(lost_indices),
                "source": {
                    "dataset": "ARKitScenes",
                    "sequence": candidate["sequence"].name,
                    "first_rgb": str(candidate["frames"][0].resolve()),
                    "selection_score": truth["selection_score"],
                    "actual_lateral_baseline_m": candidate["actual_lateral_baseline_m"],
                },
            }
        )
    manifest = {
        "schema": "sage_lm_v1_controlled_real_rgb_cohort",
        "seed": seed,
        "episode_count": 24,
        "kinds": {kind: 8 for kind in KINDS},
        "controls_per_kind": 2,
        "semantic_anchor_provenance": "CONTROLLED_COMPOSITED_EXACT_ANCHOR",
        "real_capture_scope": "SCENE_TEXTURE_BOUNDARIES_MOTION_AND_DEPTH_PHENOMENA",
        "selection_scope": "CURATED_DEVELOPMENT_SOURCE_DEPTH_SUPPORTED_VERTICAL_OPENING_PROXIES",
        "episodes": episodes,
    }
    (output_dir / "cohort.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=240824)
    args = parser.parse_args()
    manifest = materialize(args.source_root, args.output_dir, args.seed)
    print(json.dumps({"episode_count": manifest["episode_count"], "kinds": manifest["kinds"]}, indent=2))


if __name__ == "__main__":
    main()
