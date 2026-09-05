"""Source-cadence mechanism checks; no model inference or simulator outcomes."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import ue_dtr_replay as replay
from ue_action_footprints import ActionFootprints


class ActionFootprintsTests(unittest.TestCase):
    def setUp(self):
        self.anchor = replay.adapter.AnchorFrame((0., 0.), 0., (1., 0.), (0., 1.))
        self.calibration = replay.adapter.CameraCalibration(160, 90, 90., 100.)

    def action(self):
        return ActionFootprints("episode", self.anchor, self.calibration)

    def observation(self, index, t):
        return SimpleNamespace(episode_id="episode", sample_index=index, time_s=t,
            world_frame=index, rgb=SimpleNamespace(sha256="0" * 64))

    def measurement(self, t):
        position = np.array([2. + .25*t, -.2 + .4*t])
        corners = position + np.array([[-.2, -.1], [.2, -.1], [.2, .1], [-.2, .1]])
        return replay.x25.FootprintMeasurement(0, "person", .9, (.4, .3, .6, .7),
            position, corners, corners.copy(), 32)

    def advance(self, action, index, t, measurements=None, candidate=None):
        value = {"candidates": [{}]} if candidate is None else candidate
        with patch.object(action, "_measurements", return_value=(
                [self.measurement(t)] if measurements is None else measurements)):
            return action.update(self.observation(index, t), value)

    def test_twenty_hz_matches_original_tracker_every_frame(self):
        original, action = replay.x25.RigidFootprintTracker(), self.action()
        for index in range(30):
            t = index * .05
            measurements = [self.measurement(t)] if index not in (12, 13, 14) else []
            measured = original.update(measurements, t)
            expected = original.emitted(t, measured)
            actual = self.advance(action, index, t, measurements)
            self.assertEqual(expected, actual["tracks"])
        self.assertEqual(.5, action.fit_window_s)

    def test_five_hz_four_real_measurements_make_motion_reachable(self):
        original, action = replay.x25.RigidFootprintTracker(), self.action()
        for index in range(4):
            t = index * .2
            measurements = [self.measurement(t)]
            measured = original.update(measurements, t)
            self.assertEqual([], original.emitted(t, measured))
            frame = self.advance(action, index, t, measurements)
            self.assertEqual(1 if index == 3 else 0, len(frame["tracks"]))
        self.assertAlmostEqual(.6, frame["fit_contract"]["effective_window_s"])
        self.assertEqual(4, frame["fit_contract"]["minimum_samples"])
        self.assertEqual(1, frame["fit_contract"]["frozen_after_sample_index"])
        self.assertAlmostEqual(.25, frame["tracks"][0]["velocity_forward_mps"])
        self.assertAlmostEqual(.4, frame["tracks"][0]["velocity_right_mps"])

    def test_missing_detections_never_expand_the_frozen_window(self):
        action = self.action()
        for index in range(9):
            t = index * .2
            measurements = [self.measurement(t)] if index % 2 == 0 else []
            frame = self.advance(action, index, t, measurements)
            self.assertEqual([], frame["tracks"])
        self.assertAlmostEqual(.6, action.fit_window_s)
        self.assertTrue(all(len(track.history) <= 3 for track in action.tracker.tracks.values()))

    def test_hold_does_not_add_history_or_refresh_measurement_age(self):
        action = self.action()
        for index in range(4):
            self.advance(action, index, index * .2)
        track = next(iter(action.tracker.tracks.values()))
        history_count, fit_time = len(track.history), track.state_time_s
        frame = self.advance(action, 4, .8, [])
        self.assertEqual("HOLD", frame["tracks"][0]["disposition"])
        self.assertAlmostEqual(.2, frame["tracks"][0]["evidence_age_s"])
        self.assertEqual(history_count, len(track.history))
        self.assertEqual(fit_time, track.state_time_s)
        for index in (5, 6, 7):
            frame = self.advance(action, index, index * .2, [])
        self.assertEqual([], frame["tracks"])

    def test_duplicate_skipped_and_foreign_samples_cannot_advance(self):
        action = self.action()
        self.advance(action, 0, 0.)
        for index, t in ((0, 0.), (2, .4), (1, 0.)):
            with self.assertRaises(ValueError):
                self.advance(action, index, t)
        bad = {"candidates": [], "source": {"episode_id": "other"}}
        with self.assertRaisesRegex(ValueError, "source join"):
            self.advance(action, 1, .2, [], bad)
        self.assertEqual(1, action.update_count)
        self.assertEqual(1, len(next(iter(action.tracker.tracks.values())).history))
        self.assertIsNone(action.fit_window_s)
        self.advance(action, 1, .2)
        self.assertAlmostEqual(.6, action.fit_window_s)

    def test_duplicate_detections_do_not_supply_multiple_times_to_one_track(self):
        action = self.action()
        frame = self.advance(action, 0, 0., [self.measurement(0.)] * 4)
        self.assertEqual([], frame["tracks"])
        self.assertEqual([1, 1, 1, 1], [len(t.history) for t in action.tracker.tracks.values()])

    def test_window_uses_first_source_interval_not_detection_or_later_intervals(self):
        action = self.action()
        self.advance(action, 0, 10., [])
        frame = self.advance(action, 1, 10.2, [])
        frozen = frame["fit_contract"]["effective_window_s"]
        frame = self.advance(action, 2, 10.7)
        self.assertAlmostEqual(.6, frozen)
        self.assertEqual(frozen, frame["fit_contract"]["effective_window_s"])
        self.assertEqual([], frame["tracks"])

    def test_slow_source_reports_unsupported_without_extending_history(self):
        action = self.action()
        for index in range(4):
            frame = self.advance(action, index, index * .4)
        self.assertEqual("UNSUPPORTED_SOURCE_CADENCE_EXCEEDS_HISTORY", frame["fit_contract"]["state"])
        self.assertAlmostEqual(1.2, frame["fit_contract"]["effective_window_s"])
        self.assertEqual(1., frame["fit_contract"]["history_s"])
        self.assertEqual([], frame["tracks"])

    def test_native_measurement_codec_and_original_algorithms_are_restored(self):
        original_fit = replay.x24.robust_motion
        original_depth = replay.adapter.load_depth_m
        original_constants = replay.x24.fixed_constants()
        temporary = replay.REPO / "artifacts.local" / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as directory:
            path = Path(directory) / "depth.npy"
            np.save(path, np.full((90, 160), 4., dtype=np.float32))
            observation = self.observation(0, 0.)
            observation.depth = replay.adapter.ImageReference(path,
                hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, 160, 90)
            observation.camera_transform = dict(x=0., y=0., z=1.6, pitch=0., yaw=0., roll=0.)
            candidate = {"candidates": [{"class_id": 0, "class_name": "person", "confidence": .9,
                "bbox_xyxy_normalized": [.3, .3, .7, .7],
                "polygon_xy_normalized": [[.3, .3], [.7, .3], [.7, .7], [.3, .7]]}]}
            frame = self.action().update(observation, candidate)
            self.assertEqual(1, frame["metric_footprint_measurements"])
        self.assertIs(original_fit, replay.x24.robust_motion)
        self.assertIs(original_depth, replay.adapter.load_depth_m)
        self.assertEqual(original_constants, replay.x24.fixed_constants())


if __name__ == "__main__":
    unittest.main()
