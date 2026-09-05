"""Pending-inference adapter for byte-frozen classical R1 components.

Only detector/depth measurements and current issued-plan observations enter.
Tiny features are emitted without a fitted model, probability, or risk claim.
No X24/X73/X94-derived tracks are consumed and no frozen constants are changed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import dtr_carla_rgbd_model_adapter as adapter
import dtr_carla_x24_plan_adherent_predictor as x24
import dtr_carla_x24_plan_route_core as route
import dtr_carla_raw_kalman_baseline as kalman
import dtr_carla_classic_motion_baselines as classic

ARMS = (classic.ARM_FINITE_DIFFERENCE_CV, classic.ARM_CAUSAL_CTRV)


def _position(track):
    return track['position_forward_m'], track['position_right_m']


def _velocity(track):
    return track['velocity_forward_mps'], track['velocity_right_mps']


def _minimum(entries):
    return min((v for v in entries.values() if v is not None), default=None)


def predict_episode(episode: adapter.Episode,
                    candidate_values: Sequence[Mapping[str, Any]],
                    calibration: adapter.CameraCalibration) -> dict[str, Any]:
    x24.require(len(candidate_values) == len(episode.observations), 'classic_adapter_candidate_count')
    fd = classic.CausalFiniteDifferenceTracker()
    tracker = kalman.RawKalmanTracker()
    turn_history = classic.CausalTurnHistory()
    receipt_cache: dict[Path, dict[str, Any]] = {}
    previous_mode = None
    previous_time = -math.inf
    frames = []
    for observation, candidate in zip(episode.observations, candidate_values):
        now = observation.time_s
        x24.require(math.isfinite(now) and now > previous_time, 'classic_adapter_time_order')
        previous_time = now
        measurements = x24.candidate_measurements(observation, candidate, calibration, episode.route_frame)
        fd_tracks = fd.update(measurements, now)
        measured_ids = tracker.update(measurements, now)
        tracks = tracker.emitted(now, measured_ids)
        yaw_rates = turn_history.update(tracks, now)
        wearer_position, wearer_velocity = x24.wearer_anchor_state(observation, episode.route_frame)
        receipt = x24.load_receipt(observation, receipt_cache)
        selection = route.select_route(receipt, session_id=observation.navigation_session_id,
            now_s=now, wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity, previous_mode=previous_mode)
        previous_mode = selection.mode
        segments = route.build_route_segments(selection, receipt=receipt, now_s=now,
            wearer_position_xy=wearer_position, wearer_velocity_xy=wearer_velocity)
        radial_segments = (route.RouteSegment(0., route.DEFAULT_ROUTE_HORIZON_S,
                                             wearer_position, wearer_velocity),)
        fd_entries = {str(t['track_id']): route.first_metric_tube_entry_s(
            _position(t), _velocity(t), segments) for t in fd_tracks}
        cv_entries = {str(t['track_id']): route.first_metric_tube_entry_s(
            _position(t), _velocity(t), segments) for t in tracks}
        radial_entries = {str(t['track_id']): route.first_metric_tube_entry_s(
            _position(t), _velocity(t), radial_segments) for t in tracks}
        ctrv_entries = {str(t['track_id']): classic.first_ctrv_route_entry_s(
            target_position_xy=_position(t), target_velocity_xy=_velocity(t),
            yaw_rate_rad_s=yaw_rates[str(t['track_id'])], route_segments=segments) for t in tracks}
        arms = {}
        for arm, entries in ((ARMS[0], fd_entries), (ARMS[1], ctrv_entries)):
            hits = {k: v for k, v in entries.items() if v is not None}
            arms[arm] = kalman._prediction_row(selection, risk=bool(hits),
                entry_s=_minimum(hits), candidate_ids=tuple(hits))
        features = classic.tiny_feature_vector(radial_entry_s=_minimum(radial_entries),
            cv_entry_s=_minimum(cv_entries), ctrv_entry_s=_minimum(ctrv_entries),
            tracks=tracks, current_measurement_count=sum(t['disposition'] == 'MEASURED' for t in tracks),
            issued_plan_mode=selection.mode == route.ROUTE_MODE_ISSUED_PLAN)
        frames.append({'sample_index': observation.sample_index, 'time_s': now,
            'world_frame': observation.world_frame, 'raw_candidates': len(candidate['candidates']),
            'metric_measurements': len(measurements), 'tracks': tracks, 'fd_tracks': fd_tracks,
            'causal_yaw_rates_rad_s': yaw_rates, 'arms': arms,
            'tiny_features': features.tolist(), 'tiny_status': 'FEATURES_ONLY_NOT_FITTED'})
    return {'episode_id': episode.episode_id, 'frames': frames,
        'tiny_feature_names': list(classic.TINY_FEATURE_NAMES),
        'diagnostics': {'frame_count': len(frames),
            'measurement_frames': sum(f['metric_measurements'] > 0 for f in frames),
            'emitted_track_frames': sum(bool(f['tracks']) for f in frames)},
        'arms': {arm: {'route_risk_frames': sum(f['arms'][arm]['route_risk'] for f in frames)}
                 for arm in ARMS}}
