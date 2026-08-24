"""Frozen boundary/flow/monocular-depth adapter for controlled RGB episodes."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .observation import ApertureObservation, RgbEpisodeInput


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DEPTH_REPO = REPO_ROOT / "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main"
DEFAULT_DEPTH_CHECKPOINT = REPO_ROOT / "artifacts.local/models/depth-anything-v2-metric-hypersim-small/depth_anything_v2_metric_hypersim_vits.pth"


class FrozenMetricDepth:
    """Lazy Depth Anything V2 metric-Hypersim support channel."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._model = None

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        if self._model is None:
            import torch

            sys.path.insert(0, str(DEFAULT_DEPTH_REPO / "metric_depth"))
            from depth_anything_v2.dpt import DepthAnythingV2

            model = DepthAnythingV2(
                encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0
            )
            state = torch.load(DEFAULT_DEPTH_CHECKPOINT, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            self._model = model.to(torch.device(self.device)).eval()
        depth = self._model.infer_image(bgr, input_size=518)
        return np.asarray(depth, dtype=np.float32)


class RgbObservationProvider:
    """Estimate an aperture without accepting evaluator truth or identity writes."""

    def __init__(
        self,
        episode_input: RgbEpisodeInput,
        depth_estimator: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.input = episode_input
        self.depth_estimator = depth_estimator or FrozenMetricDepth()
        self.diagnostics: dict = {}

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to decode RGB frame: {path}")
        return image

    @staticmethod
    def _vertical_candidates(gray: np.ndarray, masked_bbox: tuple[int, int, int, int]) -> list[tuple[float, float]]:
        work = gray.copy()
        x1, y1, x2, y2 = masked_bbox
        work[max(0, y1 - 3) : min(work.shape[0], y2 + 4), max(0, x1 - 3) : min(work.shape[1], x2 + 4)] = 0
        edges = cv2.Canny(work, 45, 135, L2gradient=True)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(18, gray.shape[0] // 10),
            minLineLength=max(34, gray.shape[0] // 4),
            maxLineGap=max(10, gray.shape[0] // 12),
        )
        candidates: list[tuple[float, float]] = []
        if lines is not None:
            for raw in lines[:, 0, :]:
                lx1, ly1, lx2, ly2 = (int(v) for v in raw)
                dx, dy = abs(lx2 - lx1), abs(ly2 - ly1)
                if dy < max(28, gray.shape[0] * 0.22) or dx > max(7, dy * 0.22):
                    continue
                candidates.append(((lx1 + lx2) / 2.0, float(dy)))
        candidates.sort()
        merged: list[tuple[float, float]] = []
        for x, strength in candidates:
            if merged and abs(x - merged[-1][0]) <= 5.0:
                px, ps = merged[-1]
                merged[-1] = ((px * ps + x * strength) / (ps + strength), ps + strength)
            else:
                merged.append((x, strength))
        return merged

    @staticmethod
    def _choose_pair(
        candidates: list[tuple[float, float]],
        anchor_bbox: tuple[int, int, int, int],
        width: int,
        depth: np.ndarray,
    ) -> tuple[float, float, float, float, float] | None:
        anchor_x = (anchor_bbox[0] + anchor_bbox[2]) / 2.0
        best = None
        for index, (left_x, left_strength) in enumerate(candidates):
            for right_x, right_strength in candidates[index + 1 :]:
                span = right_x - left_x
                if span < width * 0.14 or span > width * 0.62:
                    continue
                edge_distance = min(abs(anchor_x - left_x), abs(anchor_x - right_x))
                if edge_distance > width * 0.25:
                    continue
                height_score = min(1.0, min(left_strength, right_strength) / 150.0)
                anchor_score = max(0.0, 1.0 - edge_distance / (width * 0.25))
                span_score = max(0.0, 1.0 - abs(span / width - 0.34) / 0.34)
                range_m, depth_score = RgbObservationProvider._depth_support(depth, left_x, right_x)
                score = 0.35 * height_score + 0.20 * anchor_score + 0.10 * span_score + 0.35 * depth_score
                if best is None or score > best[0]:
                    best = (score, left_x, right_x, range_m, depth_score)
        if best is None:
            return None
        return best[1], best[2], float(np.clip(best[0], 0.0, 1.0)), best[3], best[4]

    @staticmethod
    def _flow_support(
        first_gray: np.ndarray,
        last_gray: np.ndarray,
        left_x: float,
        right_x: float,
    ) -> tuple[float, float, list[list[float]]]:
        ys = np.linspace(first_gray.shape[0] * 0.22, first_gray.shape[0] * 0.82, 12, dtype=np.float32)
        points = np.array([[[x, y]] for x in (left_x, right_x) for y in ys], dtype=np.float32)
        forward, status_f, _ = cv2.calcOpticalFlowPyrLK(first_gray, last_gray, points, None)
        if forward is None:
            return 0.0, 0.0, []
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(last_gray, first_gray, forward, None)
        if backward is None:
            return 0.0, 0.0, []
        valid = (status_f[:, 0] > 0) & (status_b[:, 0] > 0)
        reciprocal = np.linalg.norm(backward[:, 0] - points[:, 0], axis=1)
        valid &= reciprocal < 1.8
        if int(valid.sum()) < 8:
            return 0.0, 0.0, []
        displacement = forward[:, 0] - points[:, 0]
        dx = displacement[valid, 0]
        survival = float(valid.mean())
        dispersion = float(np.median(np.abs(dx - np.median(dx))))
        consistency = math.exp(-dispersion / 2.5)
        confidence = float(np.clip(survival * consistency, 0.0, 1.0))
        tracks = [
            [float(a[0]), float(a[1]), float(b[0]), float(b[1])]
            for a, b, ok in zip(points[:, 0], forward[:, 0], valid)
            if ok
        ]
        return confidence, float(abs(np.median(dx))), tracks

    @staticmethod
    def _depth_support(depth: np.ndarray, left_x: float, right_x: float) -> tuple[float, float]:
        h, w = depth.shape
        y0, y1 = int(h * 0.28), int(h * 0.78)
        band = max(2, w // 80)
        lx, rx = int(round(left_x)), int(round(right_x))

        def median_region(a: int, b: int) -> float:
            values = depth[y0:y1, max(0, a) : min(w, b)]
            values = values[np.isfinite(values) & (values > 0.1)]
            return float(np.median(values)) if values.size else math.nan

        inner = median_region(lx + band, rx - band)
        outer_values = [median_region(lx - 3 * band, lx - band), median_region(rx + band, rx + 3 * band)]
        outer_values = [value for value in outer_values if math.isfinite(value)]
        if not math.isfinite(inner) or not outer_values:
            return math.nan, 0.0
        outer = float(np.median(outer_values))
        discontinuity = abs(inner - outer) / max(0.25, min(inner, outer))
        confidence = float(np.clip(discontinuity / 0.28, 0.0, 1.0))
        return inner, confidence

    def observe(self) -> ApertureObservation:
        visible = [obs for obs in self.input.exact_anchor_observations if obs.visible]
        if len(visible) < 2:
            return ApertureObservation(False, None, None, None, 0.0, 0.0, 0.0, 0.0)
        first_anchor, last_anchor = visible[0], visible[-1]
        if first_anchor.referent_id != last_anchor.referent_id:
            raise ValueError("exact semantic authority changed within episode")
        first = self._load(self.input.rgb_frames[first_anchor.frame_index])
        gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        depth = self.depth_estimator(first)
        if depth.shape != gray_first.shape:
            depth = cv2.resize(depth, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_LINEAR)
        candidates = self._vertical_candidates(gray_first, first_anchor.bbox_xyxy)
        pair = self._choose_pair(candidates, first_anchor.bbox_xyxy, first.shape[1], depth)
        if pair is None:
            self.diagnostics = {"referent_id": first_anchor.referent_id, "candidates": candidates}
            return ApertureObservation(True, None, None, None, 0.0, 0.0, 0.0, 0.0)
        left_x, right_x, boundary_confidence, range_m, depth_consistency = pair
        active_anchors = [row for row in visible[1:] if row.frame_index == self.input.active_parallax_frame_index]
        if len(active_anchors) != 1:
            raise ValueError("active parallax endpoint must have one visible exact-anchor observation")
        best_flow = (0.0, 0.0, [], first_anchor.frame_index)
        for later_anchor in active_anchors:
            later = self._load(self.input.rgb_frames[later_anchor.frame_index])
            gray_later = cv2.cvtColor(later, cv2.COLOR_BGR2GRAY)
            confidence, parallax, candidate_tracks = self._flow_support(gray_first, gray_later, left_x, right_x)
            if confidence > best_flow[0] or (confidence == best_flow[0] and parallax > best_flow[1]):
                best_flow = (confidence, parallax, candidate_tracks, later_anchor.frame_index)
        flow_confidence, parallax_px, tracks, flow_frame_index = best_flow
        if not math.isfinite(range_m):
            range_m = (
                self.input.commanded_baseline_m * self.input.intrinsics.fx / parallax_px
                if parallax_px > 0.5
                else math.nan
            )
        geometry_confidence = float(
            np.clip((boundary_confidence * max(flow_confidence, 1e-4) * max(depth_consistency, 1e-4)) ** (1.0 / 3.0), 0.0, 1.0)
        )
        if not math.isfinite(range_m) or range_m <= 0.1:
            center_x_m = width_m = None
            geometry_confidence = 0.0
        else:
            center_px = (left_x + right_x) / 2.0
            center_x_m = (center_px - self.input.intrinsics.cx) * range_m / self.input.intrinsics.fx
            width_m = (right_x - left_x) * range_m / self.input.intrinsics.fx
        self.diagnostics = {
            "referent_id": first_anchor.referent_id,
            "frame_indices": [first_anchor.frame_index, flow_frame_index],
            "anchor_bbox_xyxy": list(first_anchor.bbox_xyxy),
            "boundary_x_px": [left_x, right_x],
            "vertical_candidates": [[x, strength] for x, strength in candidates],
            "flow_tracks": tracks,
            "parallax_px": parallax_px,
        }
        return ApertureObservation(
            True,
            center_x_m,
            width_m,
            float(range_m) if math.isfinite(range_m) else None,
            boundary_confidence,
            flow_confidence,
            depth_consistency,
            geometry_confidence,
        )
