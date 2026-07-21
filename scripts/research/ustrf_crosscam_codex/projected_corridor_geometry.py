"""Object-agnostic projected-corridor geometry for the cross-camera proxy lane."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


INSIDE = "inside"
OUTSIDE = "outside"
UNCERTAIN = "uncertain_boundary"


@dataclass(frozen=True)
class Classification:
    relation: str
    footpoint_xy_px: tuple[float, float]
    boundary_distance_px: float
    uncertainty_px: float
    nominal_inside: bool


def validate_polygon(polygon: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        raise ValueError("route corridor polygon needs at least three points")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(polygon):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"route corridor point {index} is invalid")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(f"route corridor point {index} is outside normalized image space")
        points.append((x, y))
    twice_area = sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if abs(twice_area) < 1e-6:
        raise ValueError("route corridor polygon has zero area")
    signs: set[int] = set()
    for index in range(len(points)):
        a, b, c = points[index - 1], points[index], points[(index + 1) % len(points)]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-9:
            signs.add(1 if cross > 0 else -1)
    if len(signs) > 1:
        raise ValueError("route corridor polygon must be convex and consistently ordered")
    return points


def point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if (start[1] > y) != (end[1] > y):
            crossing_x = (end[0] - start[0]) * (y - start[1]) / (end[1] - start[1]) + start[0]
            if x < crossing_x:
                inside = not inside
    return inside


def point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared))
    return math.hypot(point[0] - (start[0] + ratio * dx), point[1] - (start[1] + ratio * dy))


def classify_bottom_center(
    box_xyxy_px: Sequence[float], *, frame_width: int, frame_height: int,
    polygon_xy_norm: Sequence[Sequence[float]], uncertainty_frame_ratio: float,
) -> Classification:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if len(box_xyxy_px) != 4:
        raise ValueError("detection box must contain xyxy")
    left, top, right, bottom = map(float, box_xyxy_px)
    if not all(math.isfinite(value) for value in (left, top, right, bottom)) or not (0 <= left < right <= frame_width and 0 <= top < bottom <= frame_height):
        raise ValueError("detection box is invalid")
    if not math.isfinite(uncertainty_frame_ratio) or not 0.0 <= uncertainty_frame_ratio <= 0.25:
        raise ValueError("projection uncertainty ratio is invalid")
    footpoint = ((left + right) / 2.0, bottom)
    return classify_contact_point(
        footpoint,
        frame_width=frame_width,
        frame_height=frame_height,
        polygon_xy_norm=polygon_xy_norm,
        uncertainty_frame_ratio=uncertainty_frame_ratio,
    )


def classify_contact_point(
    contact_xy_px: Sequence[float], *, frame_width: int, frame_height: int,
    polygon_xy_norm: Sequence[Sequence[float]], uncertainty_frame_ratio: float,
) -> Classification:
    """Classify a frozen target ground-contact point independently of a detector bbox."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if len(contact_xy_px) != 2:
        raise ValueError("contact point must contain xy")
    x, y = map(float, contact_xy_px)
    if not all(math.isfinite(value) for value in (x, y)) or not (0 <= x <= frame_width and 0 <= y <= frame_height):
        raise ValueError("contact point is invalid")
    if not math.isfinite(uncertainty_frame_ratio) or not 0.0 <= uncertainty_frame_ratio <= 0.25:
        raise ValueError("projection uncertainty ratio is invalid")
    polygon_norm = validate_polygon(polygon_xy_norm)
    polygon_px = [(px * frame_width, py * frame_height) for px, py in polygon_norm]
    footpoint = (x, y)
    inside = point_in_polygon(footpoint, polygon_px)
    distance = min(
        point_segment_distance(footpoint, polygon_px[index], polygon_px[(index + 1) % len(polygon_px)])
        for index in range(len(polygon_px))
    )
    uncertainty = uncertainty_frame_ratio * frame_width
    relation = UNCERTAIN if distance <= uncertainty else (INSIDE if inside else OUTSIDE)
    return Classification(relation, footpoint, distance, uncertainty, inside)


def robust_relation(relations: Sequence[str]) -> str:
    if not relations or any(value not in (INSIDE, OUTSIDE, UNCERTAIN) for value in relations):
        raise ValueError("robust relation needs admitted classifications")
    if all(value == INSIDE for value in relations):
        return INSIDE
    if all(value == OUTSIDE for value in relations):
        return OUTSIDE
    return UNCERTAIN
