"""Action-only X25 geometry with an explicit causal source-cadence contract.

The retained X24/X25 defaults are unchanged. Four real measurements must fit
within a window fixed after the first two source observations; missing detections
never enlarge it. This is sensor geometry, not an issued-plan credential.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

class ActionFootprints:
    def __init__(self, episode_id, route_frame, calibration):
        # Construction-time import avoids a server/replay import cycle.
        import ue_dtr_replay as replay

        self.replay = replay
        self.episode_id = episode_id
        self.route_frame = route_frame
        self.calibration = calibration
        self.tracker = replay.x25.RigidFootprintTracker()
        self.update_count = 0
        self.first_time_s = self.last_time_s = None
        self.source_interval_s = self.fit_window_s = None
        self.failed = False
        original = replay.x24
        self.fit_contract = {
            "original_window_s": original.VELOCITY_WINDOW_S,
            "minimum_samples": original.MINIMUM_FIT_SAMPLES,
            "minimum_span_s": original.MINIMUM_FIT_SPAN_S,
            "minimum_slope_span_s": original.MINIMUM_SLOPE_SPAN_S,
            "history_limit_s": original.TRACK_HISTORY_S,
            "epsilon": original.EPSILON,
        }

    def _validate(self, observation, candidate):
        if self.failed:
            raise RuntimeError("Action footprint update failed; reconstruct from verified input")
        if observation.episode_id != self.episode_id or observation.sample_index != self.update_count:
            raise ValueError("Action footprints require this episode's next source sample from zero")
        now = float(observation.time_s)
        if not math.isfinite(now) or self.last_time_s is not None and now <= self.last_time_s:
            raise ValueError("Action footprint source timestamps must strictly increase")
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("candidates"), list):
            raise ValueError("Current detector candidates must be a list")
        source = candidate.get("source")
        if source is not None:
            if not isinstance(source, Mapping):
                raise ValueError("Detector source identity must be an object")
            for key, expected in (("episode_id", self.episode_id),
                                  ("sample_index", observation.sample_index),
                                  ("time_s", observation.time_s),
                                  ("world_frame", observation.world_frame)):
                if source.get(key) != expected:
                    raise ValueError(f"Action detector/source join mismatch: {key}")
            if str(source.get("image_sha256", "")).lower() != observation.rgb.sha256.lower():
                raise ValueError("Action detector/source image identity mismatch")
        return now

    def _measurements(self, observation, candidate):
        with self.replay.native_depth_loader():
            return self.replay.x25.candidate_measurements(
                observation, candidate, self.calibration, self.route_frame)

    def update(self, observation, candidate):
        now = self._validate(observation, candidate)
        try:
            measurements = self._measurements(observation, candidate)
            if self.update_count == 1:
                self.source_interval_s = now - self.first_time_s
                self.fit_window_s = max(self.fit_contract["original_window_s"],
                    (self.fit_contract["minimum_samples"] - 1) * self.source_interval_s)
            # Reuse original measurement association, rigid registration, history
            # pruning and HOLD. Only measured tracks can contribute a fit sample.
            measured_ids = self.tracker.update(measurements, now, fit_window_s=self.fit_window_s)
            if self.fit_window_s is None:
                state = "AWAITING_SECOND_SOURCE_OBSERVATION"
            elif self.fit_window_s > self.fit_contract["history_limit_s"] + self.fit_contract["epsilon"]:
                state = "UNSUPPORTED_SOURCE_CADENCE_EXCEEDS_HISTORY"
            else:
                state = "FROZEN_SOURCE_CADENCE"
            tracks = self.tracker.emitted(now, measured_ids) if state == "FROZEN_SOURCE_CADENCE" else []
        except Exception:
            self.failed = True
            raise
        if self.first_time_s is None:
            self.first_time_s = now
        self.last_time_s = now
        self.update_count += 1
        return {
            "schema": "ue-action-footprints-v1",
            "authority": "ACTION_ONLY_SENSOR_FOOTPRINTS",
            "coordinate_frame": "ANCHOR_FORWARD_RIGHT",
            "episode_id": self.episode_id,
            "sample_index": observation.sample_index,
            "time_s": now,
            "world_frame": observation.world_frame,
            "raw_candidates": len(candidate["candidates"]),
            "metric_footprint_measurements": len(measurements),
            "tracks": tracks,
            "fit_contract": {
                "state": state,
                "authority": "ACTION_ONLY_CAUSAL_SOURCE_CADENCE",
                "rule": "MAX_ORIGINAL_WINDOW_AND_MINIMUM_SAMPLE_SPAN_AT_FIRST_SOURCE_INTERVAL",
                "first_observed_time_s": self.first_time_s,
                "source_cadence_s": self.source_interval_s,
                "effective_window_s": self.fit_window_s,
                "history_s": self.fit_contract["history_limit_s"],
                "frozen_after_sample_index": 1 if self.source_interval_s is not None else None,
                **self.fit_contract,
            },
        }
