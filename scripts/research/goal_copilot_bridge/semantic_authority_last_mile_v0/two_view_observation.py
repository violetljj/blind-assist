"""Source-pose two-view line geometry for SAGE-LM V1-B.

The core path deliberately contains neither optical flow nor monocular metric
depth.  Each image contributes independently detected line features; known
source poses provide the only metric bridge between the two views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .observation import ApertureObservation, RgbEpisodeInput, RgbEpisodeTruth


@dataclass(frozen=True)
class SourceCameraPose:
    position_world_m: tuple[float, float, float]
    rotation_camera_to_world: tuple[tuple[float, float, float], ...]

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.position_world_m, dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.rotation_camera_to_world, dtype=np.float64)


@dataclass(frozen=True)
class ImageLine:
    coefficients: tuple[float, float, float]
    support_length_px: float
    segment_count: int

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.coefficients, dtype=np.float64)

    def x_at(self, y: float) -> float:
        a, b, c = self.coefficients
        return -(b * y + c) / a

    def as_dict(self, height: int) -> dict:
        return {
            "x_top_px": self.x_at(0.0),
            "x_bottom_px": self.x_at(float(height - 1)),
            "support_length_px": self.support_length_px,
            "segment_count": self.segment_count,
        }


@dataclass(frozen=True)
class BoundaryGeometry:
    center_x_m: float
    width_m: float
    range_m: float
    confidence: float
    left_point_camera_a_m: tuple[float, float, float]
    right_point_camera_a_m: tuple[float, float, float]
    verticality: float
    range_consistency: float
    width_score: float


def _normalised_line(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    line = np.cross(np.asarray([x1, y1, 1.0]), np.asarray([x2, y2, 1.0]))
    norm = float(np.linalg.norm(line[:2]))
    if norm <= 1e-9:
        raise ValueError("degenerate image line")
    line /= norm
    if line[0] < 0:
        line *= -1.0
    return line


def _image_line_from_points(points: list[tuple[float, float]], support_length: float, segment_count: int) -> ImageLine:
    matrix = np.asarray([[x, y, 1.0] for x, y in points], dtype=np.float64)
    # The homogeneous line is the null-space vector of an N x 3 point matrix.
    # For the common two-endpoint case, reduced SVD omits that third vector.
    _, _, vh = np.linalg.svd(matrix, full_matrices=True)
    line = vh[-1]
    norm = float(np.linalg.norm(line[:2]))
    if norm <= 1e-9:
        raise ValueError("degenerate fitted image line")
    line /= norm
    if line[0] < 0:
        line *= -1.0
    return ImageLine(tuple(float(value) for value in line), float(support_length), int(segment_count))


def detect_vertical_lines(
    bgr: np.ndarray,
    masked_bbox: tuple[int, int, int, int] | None,
) -> list[ImageLine]:
    """Detect and merge vertical-ish line segments independently in one frame."""

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    work = gray.copy()
    if masked_bbox is not None:
        x1, y1, x2, y2 = masked_bbox
        work[max(0, y1 - 3) : min(work.shape[0], y2 + 4), max(0, x1 - 3) : min(work.shape[1], x2 + 4)] = 0
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(work)[0]
    segments: list[dict] = []
    if detected is not None:
        for raw in detected[:, 0, :]:
            x1, y1, x2, y2 = (float(value) for value in raw)
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            length = math.hypot(dx, dy)
            if dy < work.shape[0] * 0.16 or dx > max(7.0, dy * 0.24):
                continue
            line = _normalised_line(x1, y1, x2, y2)
            x_mid = float(-(line[1] * (work.shape[0] - 1) * 0.5 + line[2]) / line[0])
            segments.append({"points": [(x1, y1), (x2, y2)], "length": length, "x_mid": x_mid})
    segments.sort(key=lambda row: row["x_mid"])
    groups: list[list[dict]] = []
    for segment in segments:
        if groups:
            weighted_x = sum(row["x_mid"] * row["length"] for row in groups[-1]) / sum(row["length"] for row in groups[-1])
            if abs(segment["x_mid"] - weighted_x) <= 5.5:
                groups[-1].append(segment)
                continue
        groups.append([segment])
    lines = []
    for group in groups:
        points = [point for segment in group for point in segment["points"]]
        support = sum(segment["length"] for segment in group)
        lines.append(_image_line_from_points(points, support, len(group)))
    lines.sort(key=lambda line: line.x_at((work.shape[0] - 1) * 0.5))
    return lines


def _intrinsic_matrix(episode_input: RgbEpisodeInput) -> np.ndarray:
    intr = episode_input.intrinsics
    return np.asarray([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _interpretation_plane(line: ImageLine, pose: SourceCameraPose, intrinsic: np.ndarray) -> tuple[np.ndarray, float]:
    normal_camera = intrinsic.T @ line.vector
    normal_world = pose.rotation @ normal_camera
    normal_world /= np.linalg.norm(normal_world)
    return normal_world, -float(normal_world @ pose.position)


def _triangulate_boundary_line(
    line_a: ImageLine,
    line_b: ImageLine,
    pose_a: SourceCameraPose,
    pose_b: SourceCameraPose,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    normal_a, offset_a = _interpretation_plane(line_a, pose_a, intrinsic)
    normal_b, offset_b = _interpretation_plane(line_b, pose_b, intrinsic)
    direction = np.cross(normal_a, normal_b)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-5:
        return None
    direction /= norm
    system = np.vstack([normal_a, normal_b, direction])
    rhs = np.asarray([-offset_a, -offset_b, 0.0], dtype=np.float64)
    try:
        point = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None
    return point, direction


def _line_point_at_camera_midplane(
    line_world: tuple[np.ndarray, np.ndarray],
    reference_pose: SourceCameraPose,
) -> tuple[np.ndarray, float] | None:
    point_world, direction_world = line_world
    point_camera = reference_pose.rotation.T @ (point_world - reference_pose.position)
    direction_camera = reference_pose.rotation.T @ direction_world
    verticality = float(abs(direction_camera[1]) / max(1e-9, np.linalg.norm(direction_camera)))
    if abs(direction_camera[1]) < 1e-5:
        return None
    point_camera = point_camera + (-point_camera[1] / direction_camera[1]) * direction_camera
    return point_camera, verticality


def triangulate_aperture(
    left_a: ImageLine,
    right_a: ImageLine,
    left_b: ImageLine,
    right_b: ImageLine,
    pose_a: SourceCameraPose,
    pose_b: SourceCameraPose,
    intrinsic: np.ndarray,
    support_normalizer_px: float,
) -> BoundaryGeometry | None:
    left_world = _triangulate_boundary_line(left_a, left_b, pose_a, pose_b, intrinsic)
    right_world = _triangulate_boundary_line(right_a, right_b, pose_a, pose_b, intrinsic)
    if left_world is None or right_world is None:
        return None
    left_result = _line_point_at_camera_midplane(left_world, pose_a)
    right_result = _line_point_at_camera_midplane(right_world, pose_a)
    if left_result is None or right_result is None:
        return None
    left, left_verticality = left_result
    right, right_verticality = right_result
    if not all(math.isfinite(float(value)) for value in (*left, *right)):
        return None
    if left[2] <= 0.35 or right[2] <= 0.35 or left[2] > 8.0 or right[2] > 8.0:
        return None
    if left[0] >= right[0]:
        return None
    width = float(np.linalg.norm(right[[0, 2]] - left[[0, 2]]))
    if not 0.42 <= width <= 1.85:
        return None
    range_m = float((left[2] + right[2]) * 0.5)
    center_x = float((left[0] + right[0]) * 0.5)
    verticality = min(left_verticality, right_verticality)
    range_consistency = math.exp(-abs(float(left[2] - right[2])) / 0.32)
    width_score = math.exp(-abs(width - 0.95) / 0.85)
    support = min(1.0, min(left_a.support_length_px, right_a.support_length_px, left_b.support_length_px, right_b.support_length_px) / support_normalizer_px)
    confidence = float(np.clip(verticality * range_consistency * math.sqrt(width_score * max(support, 1e-6)), 0.0, 1.0))
    return BoundaryGeometry(
        center_x_m=center_x,
        width_m=width,
        range_m=range_m,
        confidence=confidence,
        left_point_camera_a_m=tuple(float(value) for value in left),
        right_point_camera_a_m=tuple(float(value) for value in right),
        verticality=verticality,
        range_consistency=range_consistency,
        width_score=width_score,
    )


def _project_world_line(
    point_world: np.ndarray,
    direction_world: np.ndarray,
    pose: SourceCameraPose,
    intrinsic: np.ndarray,
) -> ImageLine:
    point_camera = pose.rotation.T @ (point_world - pose.position)
    direction_camera = pose.rotation.T @ direction_world
    plane_normal = np.cross(point_camera, direction_camera)
    image_line = np.linalg.inv(intrinsic).T @ plane_normal
    image_line /= np.linalg.norm(image_line[:2])
    if image_line[0] < 0:
        image_line *= -1.0
    return ImageLine(tuple(float(value) for value in image_line), 1e6, 1)


def oracle_pixel_lines(
    episode_input: RgbEpisodeInput,
    truth: RgbEpisodeTruth,
    pose_a: SourceCameraPose,
    pose_b: SourceCameraPose,
) -> tuple[tuple[ImageLine, ImageLine], tuple[ImageLine, ImageLine]]:
    """Create evaluator-only image-line truth without exposing metric geometry to the provider."""

    intrinsic = _intrinsic_matrix(episode_input)
    height = episode_input.intrinsics.height
    if truth.source_boundary_x_px is None:
        raise ValueError("V1-B oracle-pixel diagnostic requires evaluator boundary pixels")
    left_x, right_x = truth.source_boundary_x_px
    first_lines = (
        ImageLine(tuple(_normalised_line(left_x, 0.0, left_x, height - 1)), 1e6, 1),
        ImageLine(tuple(_normalised_line(right_x, 0.0, right_x, height - 1)), 1e6, 1),
    )
    world_lines = []
    for boundary_x in (left_x, right_x):
        x_m = (boundary_x - episode_input.intrinsics.cx) * truth.start_range_m / episode_input.intrinsics.fx
        point_camera = np.asarray([x_m, 0.0, truth.start_range_m], dtype=np.float64)
        direction_camera = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        point_world = pose_a.rotation @ point_camera + pose_a.position
        direction_world = pose_a.rotation @ direction_camera
        world_lines.append((point_world, direction_world))
    second_lines = tuple(_project_world_line(point, direction, pose_b, intrinsic) for point, direction in world_lines)
    return first_lines, second_lines  # type: ignore[return-value]


def _line_distance(first: ImageLine, second: ImageLine, height: int) -> float:
    ys = (height * 0.25, height * 0.5, height * 0.75)
    return float(np.mean([abs(first.x_at(y) - second.x_at(y)) for y in ys]))


def _pair_candidates(lines: list[ImageLine], anchor_bbox: tuple[int, int, int, int], width: int, height: int) -> list[tuple[ImageLine, ImageLine, float]]:
    anchor_x = (anchor_bbox[0] + anchor_bbox[2]) * 0.5
    values = []
    for index, left in enumerate(lines):
        left_x = left.x_at(height * 0.5)
        for right in lines[index + 1 :]:
            right_x = right.x_at(height * 0.5)
            span = right_x - left_x
            if span < width * 0.13 or span > width * 0.66:
                continue
            edge_distance = min(abs(anchor_x - left_x), abs(anchor_x - right_x))
            if edge_distance > width * 0.34:
                continue
            support = min(1.0, min(left.support_length_px, right.support_length_px) / (height * 0.55))
            anchor_score = max(0.0, 1.0 - edge_distance / (width * 0.34))
            values.append((left, right, 0.65 * support + 0.35 * anchor_score))
    return values


class SourcePoseTwoViewBoundaryProvider:
    """B0/B1/B2 provider sharing one source-pose triangulation implementation."""

    ARM_NAMES = {
        "b0": "SAGE_LM_V1B_B0_ORACLE_PIXELS_SOURCE_POSE",
        "b1": "SAGE_LM_V1B_B1_RGB_BOUNDARIES_ORACLE_ASSOCIATION",
        "b2": "SAGE_LM_V1B_B2_RGB_BOUNDARIES_AUTOMATIC_ASSOCIATION",
    }

    def __init__(
        self,
        episode_input: RgbEpisodeInput,
        truth: RgbEpisodeTruth | None,
        pose_a: SourceCameraPose,
        pose_b: SourceCameraPose,
        arm: str,
    ) -> None:
        if arm not in self.ARM_NAMES:
            raise ValueError(f"unsupported V1-B arm: {arm}")
        if arm in {"b0", "b1"} and truth is None:
            raise ValueError(f"{arm} requires evaluator boundary truth")
        if arm == "b2" and truth is not None:
            raise ValueError("b2 automatic association cannot receive evaluator truth")
        if truth is not None and episode_input.episode_id != truth.episode_id:
            raise ValueError("input/truth episode mismatch")
        self.input = episode_input
        self.truth = truth
        self.pose_a = pose_a
        self.pose_b = pose_b
        self.arm = arm
        self.arm_name = self.ARM_NAMES[arm]
        self.diagnostics: dict = {"arm": arm, "uses_lk": False, "uses_metric_depth": False}

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to decode RGB frame: {path}")
        return image

    def _detect(self) -> tuple[list[ImageLine], list[ImageLine], tuple[int, int, int, int]]:
        visible = [row for row in self.input.exact_anchor_observations if row.visible]
        first = next((row for row in visible if row.frame_index == 0), None)
        second = next((row for row in visible if row.frame_index == self.input.active_parallax_frame_index), None)
        if first is None or second is None or first.referent_id != second.referent_id:
            raise ValueError("active source-pose pair requires the same visible exact anchor")
        frame_a = self._load(self.input.rgb_frames[first.frame_index])
        frame_b = self._load(self.input.rgb_frames[second.frame_index])
        return (
            detect_vertical_lines(frame_a, first.bbox_xyxy),
            detect_vertical_lines(frame_b, second.bbox_xyxy),
            first.bbox_xyxy,
        )

    def _observation(self, geometry: BoundaryGeometry | None, confidence_scale: float = 1.0) -> ApertureObservation:
        if geometry is None:
            return ApertureObservation(True, None, None, None, 0.0, 0.0, 0.0, 0.0)
        confidence = float(np.clip(geometry.confidence * confidence_scale, 0.0, 1.0))
        return ApertureObservation(
            True,
            geometry.center_x_m,
            geometry.width_m,
            geometry.range_m,
            confidence,
            1.0,
            1.0,
            confidence,
        )

    def observe(self) -> ApertureObservation:
        intrinsic = _intrinsic_matrix(self.input)
        height = self.input.intrinsics.height
        if self.arm == "b0":
            assert self.truth is not None
            oracle_a, oracle_b = oracle_pixel_lines(self.input, self.truth, self.pose_a, self.pose_b)
            geometry = triangulate_aperture(*oracle_a, *oracle_b, self.pose_a, self.pose_b, intrinsic, height * 0.55)
            self.diagnostics.update(
                {
                    "oracle_lines_a": [line.as_dict(height) for line in oracle_a],
                    "oracle_lines_b": [line.as_dict(height) for line in oracle_b],
                    "geometry": geometry.__dict__ if geometry else None,
                }
            )
            return self._observation(geometry)

        lines_a, lines_b, anchor_bbox = self._detect()
        self.diagnostics.update(
            {
                "detected_lines_a": [line.as_dict(height) for line in lines_a],
                "detected_lines_b": [line.as_dict(height) for line in lines_b],
            }
        )
        if self.arm == "b1":
            assert self.truth is not None
            oracle_a, oracle_b = oracle_pixel_lines(self.input, self.truth, self.pose_a, self.pose_b)
            self.diagnostics.update(
                {
                    "oracle_lines_a": [line.as_dict(height) for line in oracle_a],
                    "oracle_lines_b": [line.as_dict(height) for line in oracle_b],
                }
            )
            selected_a = [min(lines_a, key=lambda candidate: _line_distance(candidate, oracle, height)) for oracle in oracle_a] if lines_a else []
            selected_b = [min(lines_b, key=lambda candidate: _line_distance(candidate, oracle, height)) for oracle in oracle_b] if lines_b else []
            distances = (
                [_line_distance(candidate, oracle, height) for candidate, oracle in zip(selected_a, oracle_a)]
                + [_line_distance(candidate, oracle, height) for candidate, oracle in zip(selected_b, oracle_b)]
            )
            if len(selected_a) != 2 or len(selected_b) != 2 or max(distances, default=math.inf) > 9.0:
                self.diagnostics.update({"oracle_association_distances_px": distances, "failure": "BOUNDARY_CANDIDATE_MISSING"})
                return self._observation(None)
            geometry = triangulate_aperture(*selected_a, *selected_b, self.pose_a, self.pose_b, intrinsic, height * 0.55)
            distance_score = math.exp(-float(np.mean(distances)) / 7.0)
            self.diagnostics.update(
                {
                    "oracle_association_distances_px": distances,
                    "selected_lines_a": [line.as_dict(height) for line in selected_a],
                    "selected_lines_b": [line.as_dict(height) for line in selected_b],
                    "geometry": geometry.__dict__ if geometry else None,
                }
            )
            return self._observation(geometry, distance_score)

        pairs_a = _pair_candidates(lines_a, anchor_bbox, self.input.intrinsics.width, height)
        second_anchor = next(row.bbox_xyxy for row in self.input.exact_anchor_observations if row.frame_index == self.input.active_parallax_frame_index)
        pairs_b = _pair_candidates(lines_b, second_anchor, self.input.intrinsics.width, height)
        best: tuple[float, BoundaryGeometry, tuple[ImageLine, ImageLine], tuple[ImageLine, ImageLine]] | None = None
        for left_a, right_a, pair_score_a in pairs_a:
            for left_b, right_b, pair_score_b in pairs_b:
                geometry = triangulate_aperture(left_a, right_a, left_b, right_b, self.pose_a, self.pose_b, intrinsic, height * 0.55)
                if geometry is None:
                    continue
                pair_support = math.sqrt(pair_score_a * pair_score_b)
                score = geometry.confidence * (0.70 + 0.30 * pair_support)
                if best is None or score > best[0]:
                    best = (score, geometry, (left_a, right_a), (left_b, right_b))
        if best is None:
            self.diagnostics.update({"pair_count_a": len(pairs_a), "pair_count_b": len(pairs_b), "failure": "NO_POSE_CONSISTENT_PAIR"})
            return self._observation(None)
        score, geometry, selected_a, selected_b = best
        self.diagnostics.update(
            {
                "pair_count_a": len(pairs_a),
                "pair_count_b": len(pairs_b),
                "association_score": score,
                "selected_lines_a": [line.as_dict(height) for line in selected_a],
                "selected_lines_b": [line.as_dict(height) for line in selected_b],
                "geometry": geometry.__dict__,
            }
        )
        return self._observation(geometry, score / max(geometry.confidence, 1e-9))
