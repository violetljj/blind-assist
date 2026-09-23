"""UE cadence extension of the frozen X24/X25 trackers.

Only motion fit window selection differs. The update method is retained here
from the September 5 extension because the frozen tracker has no fit hook;
changing its source or patching module globals would invalidate CARLA evidence.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import dtr_carla_x24_plan_adherent_predictor as x24
import dtr_carla_x25_rigid_footprint_predictor as x25


def robust_motion(history: Sequence[tuple[float, np.ndarray]], now_s: float, *,
                  window_s: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit real measurements within the caller's declared sampling contract.

    Omission retains the original window. A source adapter may explicitly supply
    a satisfiable window without changing the sample, span or slope requirements.
    """
    window_s = x24.VELOCITY_WINDOW_S if window_s is None else float(window_s)
    x24.require(math.isfinite(window_s) and window_s > 0.0, "x24_motion_fit_window")
    window = [row for row in history if now_s - row[0] <= window_s + x24.EPSILON]
    if len(window) < x24.MINIMUM_FIT_SAMPLES or window[-1][0] - window[0][0] < x24.MINIMUM_FIT_SPAN_S - x24.EPSILON:
        return None
    times = np.asarray([row[0] for row in window], dtype=np.float64)
    positions = np.stack([row[1] for row in window]).astype(np.float64)
    slopes: list[np.ndarray] = []
    for left in range(len(window)):
        for right_index in range(left + 1, len(window)):
            delta_s = times[right_index] - times[left]
            if delta_s >= x24.MINIMUM_SLOPE_SPAN_S - x24.EPSILON:
                slopes.append((positions[right_index] - positions[left]) / delta_s)
    if not slopes:
        return None
    velocity = np.median(np.stack(slopes), axis=0)
    position = np.median(positions - (times - now_s)[:, None] * velocity[None, :], axis=0)
    return position, velocity


class CadenceRigidFootprintTracker(x25.RigidFootprintTracker):
    def update(self, measurements: Sequence[x25.FootprintMeasurement], now_s: float, *,
               fit_window_s: float | None = None) -> set[str]:
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if now_s - value.last_seen_s <= x24.HOLD_WINDOW_S + x25.EPSILON
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
                if distance <= x24.ASSOCIATION_DISTANCE_M + x25.EPSILON:
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
                self.tracks[track_id] = x25.FootprintTrack(
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
                registered = x25.register_translation(
                    track.last_surface_points_xy,
                    measurement.surface_points_xy,
                    initial,
                )
                if registered is not None:
                    translation, registration_residual = registered
                    rigid_center = track.last_rigid_center_xy + translation

            track.history.append((now_s, rigid_center.copy()))
            track.history = [
                row for row in track.history if now_s - row[0] <= x24.TRACK_HISTORY_S + x25.EPSILON
            ]
            track.last_seen_s = now_s
            track.last_position_xy = rigid_center.copy()
            track.last_rigid_center_xy = rigid_center.copy()
            track.last_surface_points_xy = measurement.surface_points_xy.copy()
            track.last_bbox = measurement.bbox
            track.footprint_offsets_xy = measurement.footprint_xy - rigid_center[None, :]
            track.depth_support = measurement.depth_support
            track.registration_residual_m = registration_residual
            motion = robust_motion(track.history, now_s, window_s=fit_window_s)
            if motion is not None:
                track.position_xy, track.velocity_xy = motion
                track.state_time_s = now_s
            measured_ids.add(track_id)
        return measured_ids
