"""Pose-compensated active-parallax boundary candidates for SAGE-LM V1-D."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from .observation import RgbEpisodeInput, RgbEpisodeTruth
from .two_view_observation import (
    ImageLine,
    SourceCameraPose,
    SourcePoseTwoViewBoundaryProvider,
    _intrinsic_matrix,
    _line_distance,
    _normalised_line,
    oracle_pixel_lines,
    triangulate_aperture,
)


TOP_K_PER_SIDE = 8
NMS_RADIUS_PX = 5
FB_ABSOLUTE_TOLERANCE_PX = 1.5
FB_RELATIVE_TOLERANCE = 0.05
CORRIDOR_HALF_WIDTH_FRACTION = 0.34


def rotational_flow(
    height: int,
    width: int,
    intrinsic: np.ndarray,
    source_pose: SourceCameraPose,
    target_pose: SourceCameraPose,
) -> np.ndarray:
    """Return the pixel flow induced by rotation alone (source -> target)."""

    relative_rotation = target_pose.rotation.T @ source_pose.rotation
    homography = intrinsic @ relative_rotation @ np.linalg.inv(intrinsic)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=0).reshape(3, -1)
    projected = homography @ pixels
    projected = projected[:2] / projected[2:3]
    flow = projected - pixels[:2]
    return flow.T.reshape(height, width, 2).astype(np.float32)


def forward_backward_mask(forward: np.ndarray, backward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reject out-of-bounds and forward/backward-inconsistent flow vectors."""

    height, width = forward.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = xx + forward[..., 0]
    map_y = yy + forward[..., 1]
    sampled_backward = np.stack(
        [
            cv2.remap(backward[..., channel], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            for channel in range(2)
        ],
        axis=-1,
    )
    error = np.linalg.norm(forward + sampled_backward, axis=-1)
    scale = np.linalg.norm(forward, axis=-1) + np.linalg.norm(sampled_backward, axis=-1)
    threshold = FB_ABSOLUTE_TOLERANCE_PX + FB_RELATIVE_TOLERANCE * scale
    in_bounds = (map_x >= 0.0) & (map_x <= width - 1.0) & (map_y >= 0.0) & (map_y <= height - 1.0)
    valid = in_bounds & np.isfinite(error) & (error <= threshold)
    return valid, error


def _top_peaks(scores: np.ndarray, valid_x: np.ndarray, count: int) -> list[int]:
    work = np.where(valid_x, scores, -np.inf).astype(np.float64)
    original = work.copy()
    selected: list[int] = []
    for _ in range(count):
        index = int(np.argmax(work))
        if not math.isfinite(float(work[index])):
            break
        selected.append(index)
        work[max(0, index - NMS_RADIUS_PX) : min(len(work), index + NMS_RADIUS_PX + 1)] = -np.inf
    if len(selected) < count:
        for index in np.argsort(original)[::-1]:
            candidate = int(index)
            if not math.isfinite(float(original[candidate])) or candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) == count:
                break
    return selected


def parallax_boundary_roles(
    residual_flow: np.ndarray,
    consistency_mask: np.ndarray,
    anchor_bbox: tuple[int, int, int, int],
) -> tuple[list[ImageLine], list[ImageLine], dict]:
    """Accumulate vertical evidence, then form anchor-compatible LEFT/RIGHT top-8 roles."""

    height, width = residual_flow.shape[:2]
    x1, y1, x2, y2 = anchor_bbox
    anchor_x = (x1 + x2) * 0.5
    residual_x = cv2.GaussianBlur(residual_flow[..., 0], (0, 0), 1.0)
    magnitude = cv2.GaussianBlur(np.linalg.norm(residual_flow, axis=-1), (0, 0), 1.0)
    discontinuity = (
        np.abs(cv2.Sobel(residual_x, cv2.CV_32F, 1, 0, ksize=3))
        + 0.5 * np.abs(cv2.Sobel(magnitude, cv2.CV_32F, 1, 0, ksize=3))
    ) / 8.0

    evidence_mask = consistency_mask.copy()
    evidence_mask[: max(1, int(round(height * 0.05))), :] = False
    evidence_mask[min(height - 1, int(round(height * 0.95))) :, :] = False
    evidence_mask[max(0, y1 - 3) : min(height, y2 + 4), max(0, x1 - 3) : min(width, x2 + 4)] = False
    weighted = np.where(evidence_mask, discontinuity, 0.0)
    support = np.sum(evidence_mask, axis=0)
    column_scores = np.sum(weighted, axis=0) / np.sqrt(np.maximum(support, 1))
    peak_x = _top_peaks(column_scores, np.ones(width, dtype=bool), 24)
    pair_hypotheses = []
    for left_index, left_x in enumerate(sorted(peak_x)):
        for right_x in sorted(peak_x)[left_index + 1 :]:
            span = right_x - left_x
            if span < width * 0.13 or span > width * 0.66:
                continue
            edge_distance = min(abs(anchor_x - left_x), abs(anchor_x - right_x))
            if edge_distance > width * CORRIDOR_HALF_WIDTH_FRACTION:
                continue
            anchor_score = max(0.0, 1.0 - edge_distance / (width * CORRIDOR_HALF_WIDTH_FRACTION))
            pair_hypotheses.append((float(column_scores[left_x] + column_scores[right_x] + 0.25 * anchor_score), left_x, right_x))
    pair_hypotheses.sort(reverse=True)
    left_x: list[int] = []
    right_x: list[int] = []
    for _, left, right in pair_hypotheses:
        if left not in left_x and len(left_x) < TOP_K_PER_SIDE:
            left_x.append(left)
        if right not in right_x and len(right_x) < TOP_K_PER_SIDE:
            right_x.append(right)
        if len(left_x) == TOP_K_PER_SIDE and len(right_x) == TOP_K_PER_SIDE:
            break

    def lines(xs: list[int]) -> list[ImageLine]:
        return [
            ImageLine(
                tuple(_normalised_line(float(x), 0.0, float(x), float(height - 1))),
                float(support[x]),
                1,
            )
            for x in xs
        ]

    selected_x = sorted(set(left_x + right_x))
    return lines(left_x), lines(right_x), {
        "left_candidate_x_px": left_x,
        "right_candidate_x_px": right_x,
        "candidate_scores": {str(x): float(column_scores[x]) for x in selected_x},
        "candidate_support_rows": {str(x): int(support[x]) for x in selected_x},
        "consistent_pixel_fraction": float(np.mean(consistency_mask)),
        "evidence_pixel_fraction": float(np.mean(evidence_mask)),
        "raw_peak_x_px": peak_x,
        "anchor_pair_hypothesis_count": len(pair_hypotheses),
        "anchor_edge_corridor_fraction": CORRIDOR_HALF_WIDTH_FRACTION,
        "residual_parallax_magnitude_median_px": float(np.median(magnitude[consistency_mask])) if np.any(consistency_mask) else None,
        "residual_parallax_magnitude_p90_px": float(np.percentile(magnitude[consistency_mask], 90)) if np.any(consistency_mask) else None,
    }


class RaftSmallFlowExtractor:
    """Frozen torchvision RAFT-Small inference wrapper."""

    def __init__(self) -> None:
        import torch
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        self.torch = torch
        self.weights = Raft_Small_Weights.DEFAULT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = raft_small(weights=self.weights, progress=True).to(self.device).eval()
        checkpoint_name = Path(urlparse(self.weights.url).path).name
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"torchvision RAFT-Small checkpoint not found after model load: {checkpoint}")
        self.checkpoint = checkpoint.resolve()
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest().upper()

    @property
    def identity(self) -> dict:
        return {
            "implementation": "torchvision.models.optical_flow.raft_small",
            "weights": str(self.weights),
            "weights_url": self.weights.url,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self.device),
        }

    def _tensor(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255.0
        return tensor.to(self.device)

    def flow(self, first_bgr: np.ndarray, second_bgr: np.ndarray) -> np.ndarray:
        first, second = self.weights.transforms()(self._tensor(first_bgr), self._tensor(second_bgr))
        with self.torch.inference_mode():
            prediction = self.model(first, second)[-1]
        return prediction[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


class ActiveParallaxBoundaryProvider(SourcePoseTwoViewBoundaryProvider):
    """Keep R2 B1 association/triangulation while replacing RGB-line candidates with parallax boundaries."""

    def __init__(
        self,
        episode_input: RgbEpisodeInput,
        truth: RgbEpisodeTruth,
        pose_a: SourceCameraPose,
        pose_b: SourceCameraPose,
        extractor: RaftSmallFlowExtractor,
    ) -> None:
        super().__init__(episode_input, truth, pose_a, pose_b, "b1")
        self.extractor = extractor
        self.arm_name = "SAGE_LM_V1D_ACTIVE_PARALLAX_BOUNDARY_FIELD_B1"
        self.diagnostics.update({"uses_lk": False, "uses_optical_flow": True, "uses_metric_depth": False})

    def _detect_roles(self) -> tuple[tuple[list[ImageLine], list[ImageLine]], tuple[list[ImageLine], list[ImageLine]]]:
        visible = [row for row in self.input.exact_anchor_observations if row.visible]
        first = next((row for row in visible if row.frame_index == 0), None)
        second = next((row for row in visible if row.frame_index == self.input.active_parallax_frame_index), None)
        if first is None or second is None or first.referent_id != second.referent_id:
            raise ValueError("active source-pose pair requires the same visible exact anchor")
        frame_a = self._load(self.input.rgb_frames[first.frame_index])
        frame_b = self._load(self.input.rgb_frames[second.frame_index])
        forward = self.extractor.flow(frame_a, frame_b)
        backward = self.extractor.flow(frame_b, frame_a)
        valid_a, error_a = forward_backward_mask(forward, backward)
        valid_b, error_b = forward_backward_mask(backward, forward)
        intrinsic = _intrinsic_matrix(self.input)
        residual_a = forward - rotational_flow(frame_a.shape[0], frame_a.shape[1], intrinsic, self.pose_a, self.pose_b)
        residual_b = backward - rotational_flow(frame_b.shape[0], frame_b.shape[1], intrinsic, self.pose_b, self.pose_a)
        left_a, right_a, diagnostics_a = parallax_boundary_roles(residual_a, valid_a, first.bbox_xyxy)
        left_b, right_b, diagnostics_b = parallax_boundary_roles(residual_b, valid_b, second.bbox_xyxy)
        diagnostics_a["forward_backward_error_median_px"] = float(np.median(error_a[valid_a])) if np.any(valid_a) else None
        diagnostics_b["forward_backward_error_median_px"] = float(np.median(error_b[valid_b])) if np.any(valid_b) else None
        self.diagnostics.update({"parallax_field_a": diagnostics_a, "parallax_field_b": diagnostics_b})
        return (left_a, right_a), (left_b, right_b)

    def observe(self):
        roles_a, roles_b = self._detect_roles()
        assert self.truth is not None
        oracle_a, oracle_b = oracle_pixel_lines(self.input, self.truth, self.pose_a, self.pose_b)
        height = self.input.intrinsics.height
        intrinsic = _intrinsic_matrix(self.input)
        pools = (*roles_a, *roles_b)
        oracles = (*oracle_a, *oracle_b)
        direct_hits = [bool(pool) and min(_line_distance(line, oracle, height) for line in pool) <= 9.0 for pool, oracle in zip(pools, oracles)]
        combinations = []
        for left_a in roles_a[0]:
            for right_a in roles_a[1]:
                for left_b in roles_b[0]:
                    for right_b in roles_b[1]:
                        geometry = triangulate_aperture(left_a, right_a, left_b, right_b, self.pose_a, self.pose_b, intrinsic, height * 0.55)
                        if geometry is None:
                            continue
                        lines = (left_a, right_a, left_b, right_b)
                        distances = [_line_distance(line, oracle, height) for line, oracle in zip(lines, oracles)]
                        combinations.append((sum(distances), distances, geometry, lines))
        selected = min(combinations, key=lambda row: row[0], default=None)
        distances = [] if selected is None else selected[1]
        self.diagnostics.update(
            {
                "top_k_per_role": TOP_K_PER_SIDE,
                "role_candidate_counts": [len(pool) for pool in pools],
                "direct_four_boundary_hits": direct_hits,
                "oracle_association_distances_px": distances,
                "valid_geometry_combination_count": len(combinations),
                "oracle_lines_a": [line.as_dict(height) for line in oracle_a],
                "oracle_lines_b": [line.as_dict(height) for line in oracle_b],
            }
        )
        if selected is None or max(distances, default=math.inf) > 9.0:
            self.diagnostics["failure"] = "PARALLAX_BOUNDARY_PAIR_MISSING"
            return self._observation(None)
        self.diagnostics["geometry"] = selected[2].__dict__
        return self._observation(selected[2], math.exp(-float(np.mean(distances)) / 7.0))
