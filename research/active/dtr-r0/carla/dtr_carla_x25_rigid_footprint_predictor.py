"""Freeze and predict X25 rigid-footprint route risk from C2/C3 RGB-D.

X25 changes one representation boundary: a metric track carries a robust 2D
occupancy footprint instead of collapsing every instance mask to one point.
The detector, issued-plan adherence gate, causal history, 0.60 s HOLD, route
horizon, tube radius, and 0.10 s confirmation remain inherited from X24.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_rgbd_model_adapter as adapter  # noqa: E402
import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x24_plan_route_core as route  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X25_RIGID_FOOTPRINT_TRACK"
FREEZE_SCHEMA = "blindassist-dtr-carla-x25-rigid-footprint-freeze-v1"
PREDICTION_SCHEMA = "blindassist-dtr-carla-x25-rigid-footprint-predictions-v1"
ARM_X25 = "X25_ISSUED_PLAN_RIGID_FOOTPRINT"

FOOTPRINT_LOW_QUANTILE = 0.02
FOOTPRINT_HIGH_QUANTILE = 0.98
MAXIMUM_REGISTRATION_POINTS = 96
REGISTRATION_ITERATIONS = 3
REGISTRATION_INLIER_M = 0.80
MINIMUM_REGISTRATION_PAIRS = 6
ROUTE_TIME_STEP_S = 0.05
BISECTION_ITERATIONS = 10
EPSILON = 1e-9


@dataclass(frozen=True)
class FootprintMeasurement:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    position_xy: np.ndarray
    footprint_xy: np.ndarray
    surface_points_xy: np.ndarray
    depth_support: int


@dataclass
class FootprintTrack:
    track_id: str
    class_id: int
    class_name: str
    history: list[tuple[float, np.ndarray]] = field(default_factory=list)
    last_seen_s: float = -math.inf
    last_position_xy: np.ndarray | None = None
    last_rigid_center_xy: np.ndarray | None = None
    last_surface_points_xy: np.ndarray | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    footprint_offsets_xy: np.ndarray | None = None
    position_xy: np.ndarray | None = None
    velocity_xy: np.ndarray | None = None
    state_time_s: float = -math.inf
    depth_support: int | None = None
    registration_residual_m: float | None = None


def fixed_constants() -> dict[str, Any]:
    return {
        **x24.fixed_constants(),
        "representation": "RIGID_2D_QUANTILE_OBB_CARRIED_BY_TRACK",
        "footprint_quantiles": [FOOTPRINT_LOW_QUANTILE, FOOTPRINT_HIGH_QUANTILE],
        "maximum_registration_points": MAXIMUM_REGISTRATION_POINTS,
        "translation_registration_iterations": REGISTRATION_ITERATIONS,
        "translation_registration_inlier_m": REGISTRATION_INLIER_M,
        "minimum_registration_pairs": MINIMUM_REGISTRATION_PAIRS,
        "route_time_step_seconds": ROUTE_TIME_STEP_S,
        "route_entry_bisection_iterations": BISECTION_ITERATIONS,
    }


def paths(run_root: Path) -> dict[str, Path]:
    return {
        "x24_freeze": run_root / "freeze-x24.json",
        "x24_predictions": run_root / "predictions-x24.json",
        "freeze": run_root / "freeze-x25.json",
        "predictions": run_root / "predictions-x25.json",
    }


def robust_obb(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    x24.require(len(points) >= adapter.MINIMUM_MASK_DEPTH_POINTS, "x25_footprint_support")
    center = np.median(points, axis=0)
    centered = points - center[None, :]
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    if np.linalg.det(axes) < 0.0:
        axes[:, 1] *= -1.0
    projected = centered @ axes
    low = np.quantile(projected, FOOTPRINT_LOW_QUANTILE, axis=0)
    high = np.quantile(projected, FOOTPRINT_HIGH_QUANTILE, axis=0)
    corners_local = np.asarray(
        [
            [low[0], low[1]],
            [high[0], low[1]],
            [high[0], high[1]],
            [low[0], high[1]],
        ],
        dtype=np.float64,
    )
    return center, center[None, :] + corners_local @ axes.T


def deterministic_subsample(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) <= MAXIMUM_REGISTRATION_POINTS:
        return points.copy()
    order = np.lexsort((points[:, 1], points[:, 0]))
    indices = np.linspace(0, len(points) - 1, MAXIMUM_REGISTRATION_POINTS, dtype=np.int64)
    return points[order[indices]].copy()


def mask_footprint_measurement(
    mask: np.ndarray,
    depth_m: np.ndarray,
    calibration: adapter.CameraCalibration,
    camera_transform: Mapping[str, Any],
    route_frame: adapter.AnchorFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    mask_value = np.asarray(mask, dtype=bool)
    depth_value = np.asarray(depth_m, dtype=np.float64)
    expected = (calibration.height, calibration.width)
    x24.require(mask_value.shape == expected and depth_value.shape == expected, "x25_rgbd_shape")
    uu, vv = adapter.angular_grid_pixels(calibration.width, calibration.height)
    selected_depth = depth_value[vv, uu]
    valid = (
        mask_value[vv, uu]
        & np.isfinite(selected_depth)
        & (selected_depth > adapter.MINIMUM_DEPTH_M)
        & (selected_depth < calibration.depth_max_m)
    )
    if int(np.count_nonzero(valid)) < adapter.MINIMUM_MASK_DEPTH_POINTS:
        return None
    candidate_depth = selected_depth[valid]
    q15 = float(np.quantile(candidate_depth, adapter.NEAR_DEPTH_QUANTILE))
    slab_m = max(
        adapter.MINIMUM_NEAR_SLAB_M,
        min(adapter.MAXIMUM_NEAR_SLAB_M, adapter.NEAR_SLAB_DEPTH_RATIO * q15),
    )
    foreground = valid & (selected_depth <= q15 + slab_m)
    support = int(np.count_nonzero(foreground))
    if support < adapter.MINIMUM_MASK_DEPTH_POINTS:
        return None
    pixels = np.column_stack((uu[foreground], vv[foreground])).astype(np.float64)
    camera_points = adapter.unproject_pixels_camera_flu(
        pixels,
        selected_depth[foreground],
        calibration.intrinsic,
    )
    world_points = adapter.camera_flu_to_world(camera_points, camera_transform)
    anchor_points = adapter.world_to_anchor_fru(world_points, route_frame)[:, :2]
    center, footprint = robust_obb(anchor_points)
    return center, footprint, deterministic_subsample(anchor_points), support


def candidate_measurements(
    observation: adapter.FrameObservation,
    candidate_value: Mapping[str, Any],
    calibration: adapter.CameraCalibration,
    route_frame: adapter.AnchorFrame,
) -> list[FootprintMeasurement]:
    if not candidate_value["candidates"]:
        return []
    depth_m = adapter.load_depth_m(observation, calibration)
    output: list[FootprintMeasurement] = []
    for candidate in candidate_value["candidates"]:
        measured = mask_footprint_measurement(
            x24.normalized_polygon_mask(candidate, calibration.width, calibration.height),
            depth_m,
            calibration,
            observation.camera_transform,
            route_frame,
        )
        if measured is None:
            continue
        center, footprint, points, support = measured
        output.append(
            FootprintMeasurement(
                class_id=int(candidate["class_id"]),
                class_name=str(candidate["class_name"]),
                confidence=float(candidate["confidence"]),
                bbox=tuple(float(value) for value in candidate["bbox_xyxy_normalized"]),
                position_xy=center,
                footprint_xy=footprint,
                surface_points_xy=points,
                depth_support=support,
            )
        )
    return output


def nearest_residuals(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    delta = target_xy[None, :, :] - source_xy[:, None, :]
    distance_squared = np.sum(delta * delta, axis=2)
    return delta[np.arange(len(source_xy)), np.argmin(distance_squared, axis=1)]


def register_translation(
    previous_xy: np.ndarray,
    current_xy: np.ndarray,
    initial_delta_xy: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    previous = np.asarray(previous_xy, dtype=np.float64).reshape(-1, 2)
    current = np.asarray(current_xy, dtype=np.float64).reshape(-1, 2)
    translation = np.asarray(initial_delta_xy, dtype=np.float64).reshape(2).copy()
    accepted: np.ndarray | None = None
    for _ in range(REGISTRATION_ITERATIONS):
        moved = previous + translation[None, :]
        forward = nearest_residuals(moved, current)
        reverse = -nearest_residuals(current, moved)
        residuals = np.vstack((forward, reverse))
        inliers = residuals[np.linalg.norm(residuals, axis=1) <= REGISTRATION_INLIER_M]
        if len(inliers) < MINIMUM_REGISTRATION_PAIRS:
            return None
        correction = np.median(inliers, axis=0)
        translation += correction
        accepted = inliers - correction[None, :]
    assert accepted is not None
    residual = float(np.median(np.linalg.norm(accepted, axis=1)))
    return translation, residual


class RigidFootprintTracker:
    def __init__(self) -> None:
        self.tracks: dict[str, FootprintTrack] = {}
        self.next_id = 1

    @staticmethod
    def predicted_position(track: FootprintTrack, now_s: float) -> np.ndarray | None:
        if track.position_xy is not None and track.velocity_xy is not None:
            return track.position_xy + track.velocity_xy * max(0.0, now_s - track.state_time_s)
        return None if track.last_position_xy is None else track.last_position_xy

    def update(self, measurements: Sequence[FootprintMeasurement], now_s: float, *,
               fit_window_s: float | None = None) -> set[str]:
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if now_s - value.last_seen_s <= x24.HOLD_WINDOW_S + EPSILON
        }
        costs: list[tuple[float, float, str, int]] = []
        for track_id, track in self.tracks.items():
            predicted = self.predicted_position(track, now_s)
            if predicted is None:
                continue
            for index, measurement in enumerate(measurements):
                if measurement.class_id != track.class_id:
                    continue
                distance = float(np.linalg.norm(predicted - measurement.position_xy))
                if distance <= x24.ASSOCIATION_DISTANCE_M + EPSILON:
                    overlap = 0.0 if track.last_bbox is None else x24.bbox_iou(track.last_bbox, measurement.bbox)
                    costs.append((distance, -overlap, track_id, index))
        costs.sort()
        assigned_tracks: set[str] = set()
        assigned_measurements: dict[int, str] = {}
        for _distance, _overlap, track_id, index in costs:
            if track_id not in assigned_tracks and index not in assigned_measurements:
                assigned_tracks.add(track_id)
                assigned_measurements[index] = track_id

        measured_ids: set[str] = set()
        for index, measurement in enumerate(measurements):
            track_id = assigned_measurements.get(index)
            if track_id is None:
                track_id = f"footprint-{self.next_id:06d}"
                self.next_id += 1
                self.tracks[track_id] = FootprintTrack(
                    track_id=track_id,
                    class_id=measurement.class_id,
                    class_name=measurement.class_name,
                )
            track = self.tracks[track_id]
            rigid_center = measurement.position_xy.copy()
            registration_residual: float | None = None
            if track.last_rigid_center_xy is not None and track.last_surface_points_xy is not None:
                predicted = self.predicted_position(track, now_s)
                initial = (
                    np.zeros(2, dtype=np.float64)
                    if predicted is None
                    else predicted - track.last_rigid_center_xy
                )
                registered = register_translation(
                    track.last_surface_points_xy,
                    measurement.surface_points_xy,
                    initial,
                )
                if registered is not None:
                    translation, registration_residual = registered
                    rigid_center = track.last_rigid_center_xy + translation

            track.history.append((now_s, rigid_center.copy()))
            track.history = [
                row for row in track.history if now_s - row[0] <= x24.TRACK_HISTORY_S + EPSILON
            ]
            track.last_seen_s = now_s
            track.last_position_xy = rigid_center.copy()
            track.last_rigid_center_xy = rigid_center.copy()
            track.last_surface_points_xy = measurement.surface_points_xy.copy()
            track.last_bbox = measurement.bbox
            track.footprint_offsets_xy = measurement.footprint_xy - rigid_center[None, :]
            track.depth_support = measurement.depth_support
            track.registration_residual_m = registration_residual
            motion = x24.robust_motion(track.history, now_s, window_s=fit_window_s)
            if motion is not None:
                track.position_xy, track.velocity_xy = motion
                track.state_time_s = now_s
            measured_ids.add(track_id)
        return measured_ids

    def emitted(self, now_s: float, measured_ids: set[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for track_id, track in sorted(self.tracks.items()):
            if (
                track.position_xy is None
                or track.velocity_xy is None
                or track.footprint_offsets_xy is None
            ):
                continue
            age_s = now_s - track.last_seen_s
            if age_s > x24.HOLD_WINDOW_S + EPSILON:
                continue
            position = track.position_xy + track.velocity_xy * max(0.0, now_s - track.state_time_s)
            footprint = position[None, :] + track.footprint_offsets_xy
            output.append(
                {
                    "track_id": track_id,
                    "class_id": track.class_id,
                    "class_name": track.class_name,
                    "disposition": "MEASURED" if track_id in measured_ids else "HOLD",
                    "evidence_age_s": max(0.0, age_s),
                    "position_forward_m": float(position[0]),
                    "position_right_m": float(position[1]),
                    "velocity_forward_mps": float(track.velocity_xy[0]),
                    "velocity_right_mps": float(track.velocity_xy[1]),
                    "footprint_xy": [[float(value) for value in row] for row in footprint],
                    "footprint_area_m2": polygon_area(footprint),
                    "depth_grid_support": track.depth_support if track_id in measured_ids else None,
                    "registration_residual_m": (
                        track.registration_residual_m if track_id in measured_ids else None
                    ),
                }
            )
        return output


def polygon_area(polygon_xy: np.ndarray) -> float:
    polygon = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2)
    return float(
        0.5
        * abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )


def point_segment_distance(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    edge = second - first
    denominator = float(np.dot(edge, edge))
    if denominator <= EPSILON:
        return float(np.linalg.norm(point - first))
    ratio = float(np.dot(point - first, edge) / denominator)
    projection = first + min(1.0, max(0.0, ratio)) * edge
    return float(np.linalg.norm(point - projection))


def point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    cross = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        relative = point - first
        cross.append(float(edge[0] * relative[1] - edge[1] * relative[0]))
    return all(value >= -EPSILON for value in cross) or all(value <= EPSILON for value in cross)


def point_polygon_distance(point_xy: np.ndarray, polygon_xy: np.ndarray) -> float:
    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    polygon = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2)
    if point_in_convex_polygon(point, polygon):
        return 0.0
    return min(
        point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def segment_distance_at(
    footprint_xy: np.ndarray,
    target_velocity_xy: np.ndarray,
    segment: route.RouteSegment,
    offset_s: float,
) -> float:
    wearer = np.asarray(segment.start_position_xy, dtype=np.float64) + np.asarray(
        segment.velocity_xy, dtype=np.float64
    ) * (offset_s - segment.start_offset_s)
    footprint = footprint_xy + target_velocity_xy[None, :] * offset_s
    return point_polygon_distance(wearer, footprint)


def closing_speed_at(
    footprint_xy: np.ndarray,
    target_velocity_xy: np.ndarray,
    segment: route.RouteSegment,
    offset_s: float,
) -> float:
    left = max(segment.start_offset_s, offset_s - ROUTE_TIME_STEP_S * 0.5)
    right = min(segment.end_offset_s, offset_s + ROUTE_TIME_STEP_S * 0.5)
    if right <= left + EPSILON:
        return 0.0
    return (
        segment_distance_at(footprint_xy, target_velocity_xy, segment, left)
        - segment_distance_at(footprint_xy, target_velocity_xy, segment, right)
    ) / (right - left)


def first_footprint_route_entry_s(
    footprint_xy: Sequence[Sequence[float]],
    target_velocity_xy: Sequence[float],
    route_segments: Sequence[route.RouteSegment],
) -> float | None:
    footprint = np.asarray(footprint_xy, dtype=np.float64).reshape(-1, 2)
    velocity = np.asarray(target_velocity_xy, dtype=np.float64).reshape(2)
    radius = route.DEFAULT_TUBE_RADIUS_M
    minimum_closing = route.DEFAULT_MIN_CLOSING_SPEED_MPS
    for segment in route_segments:
        start = float(segment.start_offset_s)
        end = float(segment.end_offset_s)
        duration = max(0.0, end - start)
        samples = max(1, int(math.ceil(duration / ROUTE_TIME_STEP_S)))
        times = np.linspace(start, end, samples + 1)
        previous_t = float(times[0])
        previous_d = segment_distance_at(footprint, velocity, segment, previous_t)
        if previous_d <= radius + EPSILON:
            if closing_speed_at(footprint, velocity, segment, previous_t) + EPSILON >= minimum_closing:
                return previous_t
        for raw_t in times[1:]:
            current_t = float(raw_t)
            current_d = segment_distance_at(footprint, velocity, segment, current_t)
            if previous_d > radius + EPSILON and current_d <= radius + EPSILON:
                low, high = previous_t, current_t
                for _ in range(BISECTION_ITERATIONS):
                    middle = (low + high) * 0.5
                    if segment_distance_at(footprint, velocity, segment, middle) <= radius:
                        high = middle
                    else:
                        low = middle
                if closing_speed_at(footprint, velocity, segment, high) + EPSILON >= minimum_closing:
                    return high
            previous_t, previous_d = current_t, current_d
    return None


def arm_frame(
    selection: route.RouteSelection,
    *,
    receipt: Mapping[str, Any] | None,
    observation: adapter.FrameObservation,
    wearer_position: tuple[float, float],
    wearer_velocity: tuple[float, float],
    tracks: Sequence[Mapping[str, Any]],
    confirmation: x24.RiskConfirmation,
    sample_period_s: float,
) -> dict[str, Any]:
    if selection.mode_changed:
        confirmation.reset()
    segments = route.build_route_segments(
        selection,
        receipt=receipt,
        now_s=observation.time_s,
        wearer_position_xy=wearer_position,
        wearer_velocity_xy=wearer_velocity,
    )
    entries = {
        str(track["track_id"]): first_footprint_route_entry_s(
            track["footprint_xy"],
            (track["velocity_forward_mps"], track["velocity_right_mps"]),
            segments,
        )
        for track in tracks
    }
    confirmed = confirmation.update(entries, now_s=observation.time_s, sample_period_s=sample_period_s)
    confirmed_entries = [float(entries[key]) for key in confirmed if entries[key] is not None]
    return {
        "route_mode": selection.mode,
        "authority": selection.authority,
        "plan_receipt_sha256": selection.receipt_sha256,
        "plan_position_residual_m": selection.plan_position_residual_m,
        "plan_velocity_direction_error_degrees": selection.plan_velocity_direction_error_deg,
        "fallback_reason": selection.fallback_reason,
        "route_mode_changed": selection.mode_changed,
        "route_risk": bool(confirmed_entries),
        "minimum_entry_s": min(confirmed_entries) if confirmed_entries else None,
        "candidate_risk_track_ids": sorted(key for key, value in entries.items() if value is not None),
        "confirmed_risk_track_ids": sorted(confirmed),
    }


def predict_episode(
    episode: adapter.Episode,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: adapter.CameraCalibration,
) -> dict[str, Any]:
    x24.require(len(candidate_values) == len(episode.observations), f"x25_candidate_count:{episode.episode_id}")
    tracker = RigidFootprintTracker()
    receipt_cache: dict[Path, dict[str, Any]] = {}
    confirmation = x24.RiskConfirmation()
    previous_mode: str | None = None
    frames: list[dict[str, Any]] = []
    route_mode_counts: Counter[str] = Counter()
    measured_frames = emitted_frames = hold_frames = 0
    for ordinal, (observation, candidate_value) in enumerate(zip(episode.observations, candidate_values)):
        neighbour = (
            episode.observations[ordinal - 1]
            if ordinal
            else episode.observations[ordinal + 1]
        )
        sample_period_s = abs(observation.time_s - neighbour.time_s)
        x24.require(sample_period_s > 0.0, f"x25_sample_period:{episode.episode_id}:{ordinal}")
        measurements = candidate_measurements(
            observation,
            candidate_value,
            calibration,
            episode.route_frame,
        )
        measured_frames += int(bool(measurements))
        measured_ids = tracker.update(measurements, observation.time_s)
        tracks = tracker.emitted(observation.time_s, measured_ids)
        emitted_frames += int(bool(tracks))
        hold_frames += int(any(track["disposition"] == "HOLD" for track in tracks))
        wearer_position, wearer_velocity = x24.wearer_anchor_state(observation, episode.route_frame)
        receipt = x24.load_receipt(observation, receipt_cache)
        selection = route.select_route(
            receipt,
            session_id=observation.navigation_session_id,
            now_s=observation.time_s,
            wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity,
            previous_mode=previous_mode,
        )
        previous_mode = selection.mode
        route_mode_counts[selection.mode] += 1
        frames.append(
            {
                "sample_index": observation.sample_index,
                "time_s": observation.time_s,
                "world_frame": observation.world_frame,
                "raw_candidates": len(candidate_value["candidates"]),
                "metric_footprint_measurements": len(measurements),
                "tracks": tracks,
                "arms": {
                    ARM_X25: arm_frame(
                        selection,
                        receipt=receipt,
                        observation=observation,
                        wearer_position=wearer_position,
                        wearer_velocity=wearer_velocity,
                        tracks=tracks,
                        confirmation=confirmation,
                        sample_period_s=sample_period_s,
                    )
                },
            }
        )
    return {
        "episode_id": episode.episode_id,
        "frames": frames,
        "diagnostics": {
            "frame_count": len(frames),
            "metric_footprint_measurement_frames": measured_frames,
            "emitted_track_frames": emitted_frames,
            "track_coverage": emitted_frames / max(1, len(frames)),
            "hold_frames": hold_frames,
            "x25_route_mode_counts": dict(sorted(route_mode_counts.items())),
        },
        "arms": {
            ARM_X25: {
                "route_risk_frames": sum(bool(frame["arms"][ARM_X25]["route_risk"]) for frame in frames),
                "first_route_risk_time_s": next(
                    (frame["time_s"] for frame in frames if frame["arms"][ARM_X25]["route_risk"]),
                    None,
                ),
            }
        },
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(not output["freeze"].exists(), f"x25_freeze_exists:{output['freeze']}")
    frozen_x24, contract, _candidates = x24.require_freeze(run_root)
    x24.require(output["x24_predictions"].is_file(), "x25_x24_baseline_missing")
    value = {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN_TRUTH_BLIND_PENDING_PREDICTION",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "source": {
            "x24_freeze_sha256": x24.sha256_file(output["x24_freeze"]),
            "x24_predictions_sha256": x24.sha256_file(output["x24_predictions"]),
            "model_manifest_sha256": contract.manifest_sha256,
            "candidate_aggregate_sha256": frozen_x24["candidates"]["aggregate_sha256"],
        },
        "algorithm": {
            "path": str(Path(__file__).resolve()),
            "sha256": x24.sha256_file(Path(__file__).resolve()),
        },
        "episodes": len(contract.episodes),
        "frames": len(x24.flatten_observations(contract)),
        "fixed_constants": fixed_constants(),
        "arm": ARM_X25,
    }
    x24.write_json_exclusive(output["freeze"], value)
    return {**value, "freeze_sha256": x24.sha256_file(output["freeze"])}


def require_freeze(run_root: Path) -> tuple[dict[str, Any], adapter.SanitizedModelContract, list[dict[str, Any]]]:
    output = paths(run_root)
    frozen = x24.read_json(output["freeze"].resolve(strict=True), "x25_freeze")
    x24.require(frozen.get("schema") == FREEZE_SCHEMA, "x25_freeze_schema")
    x24.require(frozen.get("fixed_constants") == fixed_constants(), "x25_constants_drift")
    x24.require(
        frozen["algorithm"]["sha256"] == x24.sha256_file(Path(__file__).resolve()),
        "x25_algorithm_drift",
    )
    frozen_x24, contract, candidates = x24.require_freeze(run_root)
    x24.require(
        frozen["source"]["x24_freeze_sha256"] == x24.sha256_file(output["x24_freeze"]),
        "x25_x24_freeze_drift",
    )
    x24.require(
        frozen["source"]["x24_predictions_sha256"] == x24.sha256_file(output["x24_predictions"]),
        "x25_x24_prediction_drift",
    )
    x24.require(
        frozen["source"]["candidate_aggregate_sha256"] == frozen_x24["candidates"]["aggregate_sha256"],
        "x25_candidate_drift",
    )
    return frozen, contract, candidates


def predict(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    output = paths(run_root)
    x24.require(not output["predictions"].exists(), f"x25_predictions_exist:{output['predictions']}")
    frozen, contract, candidate_values = require_freeze(run_root)
    cursor = 0
    episodes: dict[str, Any] = {}
    for episode in contract.episodes:
        count = len(episode.observations)
        episodes[episode.episode_id] = predict_episode(
            episode,
            candidate_values[cursor : cursor + count],
            contract.calibration,
        )
        cursor += count
    value = {
        "schema": PREDICTION_SCHEMA,
        "status": "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "arms": [ARM_X25],
        "episodes": episodes,
        "fixed_constants": fixed_constants(),
        "source": {
            "freeze_sha256": x24.sha256_file(output["freeze"]),
            **frozen["source"],
        },
        "claim_boundary": {
            "synthetic_development": True,
            "evaluator_opened": False,
            "current_actor_oracle_used": False,
        },
    }
    x24.write_json_exclusive(output["predictions"], value)
    return {**value, "predictions_sha256": x24.sha256_file(output["predictions"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "predict"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    value = freeze(args) if args.command == "freeze" else predict(args)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
