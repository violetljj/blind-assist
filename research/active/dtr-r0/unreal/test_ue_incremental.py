import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ue_dtr_replay as replay
from ue_incremental import IncrementalX73, _Program, _Recent, source_paths, _source, SOURCE_CONTRACT


def prediction(frames, arm, counter):
    return {"frames": copy.deepcopy(frames), "arms": {arm: {}}, "diagnostics": {counter: {}}}


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.anchor = replay.adapter.AnchorFrame((0., 0.), 0., (1., 0.), (0., 1.))
        self.calibration = replay.adapter.CameraCalibration(64, 48, 90., 100.)

    def observation(self, index, *, t=None, plan=None, x=0., y=0., vx=1., vy=0.):
        transform = {"x": x, "y": y, "z": 0., "pitch": 0., "yaw": 0., "roll": 0.}
        image = SimpleNamespace(path=Path("unused"), sha256="0"*64)
        return replay.adapter.FrameObservation(
            "episode_0000", index, index*.2 if t is None else t, index, "episode_0000",
            dict(transform, z=1.6), image, image,
            {"transform": transform, "command_velocity": {"x": vx, "y": vy, "z": 0.}},
            plan or {"path": None, "receipt_sha256": None, "authority": "NO_PLAN"})

    def test_warmup_is_deferred_and_bad_source_cannot_advance_state(self):
        engine = IncrementalX73("episode_0000", self.anchor, self.calibration)
        with patch.object(replay, "predict_episode", side_effect=AssertionError("batch called")):
            first = self.observation(0)
            self.assertEqual("WARMUP", engine.update(first, {"candidates": []})["event"])
            self.assertEqual(0, engine.processed_count)
            self.assertIsNone(engine.last_frame)
            for observation in (first, self.observation(2), self.observation(1, t=0.)):
                with self.assertRaises(ValueError):
                    engine.update(observation, {"candidates": []})
            with self.assertRaisesRegex(ValueError, "source join"):
                engine.update(self.observation(1), {"candidates": [], "source": {"episode_id": "other"}})
            engine.update(self.observation(1), {"candidates": []})
        self.assertEqual(2, engine.update_count)
        self.assertEqual(2, engine.processed_count)
        self.assertTrue(all(stage.processed_count == 2 for stage in engine.stages))

    def test_history_is_bounded_and_cannot_be_replayed(self):
        history = _Recent()
        for value in range(7):
            history.append(value)
        self.assertEqual({5: 5, 6: 6}, history.items)
        self.assertEqual(6, history[-1])
        with self.assertRaises(IndexError):
            _ = history[0]
        with self.assertRaises(IndexError):
            _ = history[7]
        with self.assertRaises(TypeError):
            list(history)

    def test_program_sources_are_explicit_and_local(self):
        with self.assertRaisesRegex(ValueError, "allowlist"):
            _Program(prediction)
        paths = source_paths()
        self.assertIn(Path(replay.x73.__file__).resolve(), paths)
        self.assertIn(SOURCE_CONTRACT.resolve(), paths)
        self.assertTrue(all(path.is_relative_to(replay.REPO) for path in paths))

    def test_unreviewed_pre_or_post_loop_change_is_rejected(self):
        import ue_incremental as incremental
        function = replay.x73.apply_credentialed_parent_hull_reconstruction_episode
        original = _source(function)
        for changed in (original.replace("    value = copy.deepcopy(core)", "    added_semantics = True\n    value = copy.deepcopy(core)"),
                        original.replace("    return value", "    value['new_result'] = True\n    return value")):
            with patch.object(incremental.inspect, "getsource", return_value=changed):
                with self.assertRaisesRegex(ValueError, "Unreviewed.*body"):
                    _Program(function)

    def test_plan_change_expiry_and_adherence_match_every_batch_prefix(self):
        temporary = replay.REPO / "artifacts.local" / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as directory:
            def plan(name, begin, end, p0, p1):
                value = {"schema_version": "dtr-c1-plan-receipt-v1", "coordinate_frame": "ANCHOR_FORWARD_RIGHT",
                         "plan_id": name, "session_id": "episode_0000", "issued_at_s": begin,
                         "valid_from_s": begin, "expires_at_s": end,
                         "time_parameterized_waypoints": [
                             {"time_s": begin, "forward_m": p0[0], "right_m": p0[1]},
                             {"time_s": end, "forward_m": p1[0], "right_m": p1[1]}]}
                value["receipt_sha256"] = replay.x24.route.compute_receipt_sha256(value)
                path = Path(directory) / (name+".json")
                path.write_text(json.dumps(value), encoding="utf-8")
                return {"path": str(path), "receipt_sha256": value["receipt_sha256"], "authority": "VALID"}
            straight = plan("straight", 0., .7, (0., 0.), (.7, 0.))
            lateral = plan("lateral", .6, 4., (.2, .2), (.2, 3.6))
            observations = (
                self.observation(0, x=0., plan=straight), self.observation(1, x=.2, plan=straight),
                self.observation(2, x=.2, y=0., vx=0., vy=1., plan=straight),
                self.observation(3, t=.6, x=.2, y=.2, vx=0., vy=1., plan=lateral),
                self.observation(4, x=.2, y=.4, vx=0., vy=1.),
                self.observation(5, x=1., plan=dict(straight, authority="EXPIRED")),
            )
            episode = replay.adapter.Episode("episode_0000", self.anchor, observations)
            engine = IncrementalX73(episode.episode_id, self.anchor, self.calibration)
            candidates, modes, metric_ids = [], [], []
            for observation in observations:
                candidates.append({"candidates": []})
                row = engine.update(observation, candidates[-1])
                if len(candidates) > 1:
                    prefix = replace(episode, observations=observations[:len(candidates)])
                    batch = replay.predict_episode(prefix, candidates, self.calibration)
                    self.assertEqual(batch["frames"][-1], engine.last_frame)
                    self.assertEqual(replay.compact_rows(episode.episode_id, batch)[-1], row)
                    modes.append(row["route_mode"])
                    metric_ids.append(id(engine.metric.scope["tracker"]))
            self.assertIn("ISSUED_PLAN", modes)
            self.assertIn(replay.x24.route.ROUTE_MODE_OBSERVED_CV, modes)
            self.assertEqual(1, len(set(metric_ids)))
            self.assertTrue(all(stage.processed_count == len(observations) for stage in engine.stages))

    def test_parent_continuation_keeps_state_and_rejects_changed_parent(self):
        import dtr_carla_x41_metric_credentialed_parent_continuation as x41
        x40 = x41.x40
        frames, metrics = [], []
        for index, (parent, positive) in enumerate((("parent-A", True), ("parent-A", False),
                                                   ("parent-B", False), ("parent-A", False))):
            track = f"fragment-{index}"
            arm = {"route_risk": positive, "confirmed_risk_track_ids": [track] if positive else [],
                   "cross_representation_adjudication_suppressed": not positive,
                   "cross_representation_suppressed_track_ids": [track],
                   "cross_representation_suppressed_minimum_entry_s": .5}
            frames.append({"sample_index": index, "time_s": index*.2,
                           "tracks": [{"track_id": track, "parent_track_id": parent}], "arms": {x40.ARM_X40: arm}})
            metrics.append({"sample_index": index, "time_s": index*.2,
                            "arms": {replay.x24.ARM_X24: {"candidate_risk_track_ids": ["metric"]}}})
        stage = _Program(x41.predict_episode)
        emitted = []
        for index, frame in enumerate(frames):
            episode = SimpleNamespace(observations=(self.observation(index),), route_frame=self.anchor)
            actual = stage.update(episode=episode, observation=episode.observations[0], candidate={},
                                  calibration=self.calibration, ordinal=index, frame=frame, metric=metrics[index])
            with patch.object(x40, "predict_episode", return_value=prediction(frames[:index+1], x40.ARM_X40, "x40_route_mode_counts")), \
                 patch.object(replay.x24, "predict_episode", return_value={"frames": metrics[:index+1]}):
                expected = x41.predict_episode(episode, [], self.calibration)["frames"][-1]
            self.assertEqual(expected, actual)
            emitted.append(actual["arms"][x41.ARM_X41]["metric_credentialed_parent_continuation_used"])
        self.assertEqual([False, True, False, False], emitted)
        self.assertEqual({"parent-A"}, stage.scope["credentialed_parents"])

    def test_x73_birth_hull_new_route_and_release_preserve_order(self):
        x73, x72 = replay.x73, replay.x72
        observations, frames, rigid = [], [], []
        for index in range(5):
            observations.append(self.observation(index, plan={"name": "side" if index == 2 else "forward"}))
            row = {"track_id": "fragment", "parent_track_id": "surface-cone-A", "class_name": x73.SURFACE_CLASS,
                   "footprint_xy": [[2., -.2], [2.4, -.2], [2.4, .2], [2., .2]],
                   "footprint_area_m2": .16, "disposition": "MEASURED", "risk_eligible": True,
                   "velocity_forward_mps": -.5, "velocity_right_mps": 0.}
            arm = {"route_risk": index == 0, "confirmed_risk_parent_track_ids": ["surface-cone-A"] if index == 0 else [],
                   "x69_mature_cross_route_rigid_contradiction_release_used": index == 3}
            frames.append({"sample_index": index, "time_s": index*.2, "tracks": [row], "risk_eligible_tracks": 1,
                           "arms": {x72.ARM_X72: arm}})
            rigid.append({"sample_index": index, "time_s": index*.2, "tracks": []})
        episode = replay.adapter.Episode("episode_0000", self.anchor, tuple(observations))
        stage = _Program(x73.apply_credentialed_parent_hull_reconstruction_episode)
        flags, risks = [], []
        with patch.object(replay.x24, "wearer_anchor_state", return_value=((0., 0.), (1., 0.))), \
             patch.object(replay.x24, "load_receipt", side_effect=lambda o,c: o.issued_plan), \
             patch.object(replay.x24.route, "select_route", return_value=SimpleNamespace(mode="ISSUED_PLAN")), \
             patch.object(replay.x24.route, "build_route_segments", side_effect=lambda s, **kw: [kw["receipt"]["name"]]), \
             patch.object(replay.x25, "first_footprint_route_entry_s", side_effect=lambda h,v,s: .4 if s == ["forward"] else None):
            for index, observation in enumerate(observations):
                prefix = replace(episode, observations=episode.observations[:index+1])
                actual = stage.update(episode=prefix, observation=observation, candidate={}, calibration=self.calibration,
                                      ordinal=index, frame=frames[index], rigid=rigid[index])
                batch = x73.apply_credentialed_parent_hull_reconstruction_episode(
                    prediction(frames[:index+1], x72.ARM_X72, "x72_route_mode_counts"),
                    {"frames": rigid[:index+1]}, prefix)
                self.assertEqual(batch["frames"][-1], actual)
                flags.append(actual["arms"][x73.ARM_X73]["x73_credentialed_parent_hull_reconstruction_used"])
                risks.append(actual["arms"][x73.ARM_X73]["route_risk"])
        self.assertEqual([False, True, False, False, False], flags)
        self.assertEqual([True, True, False, False, False], risks)
        self.assertEqual(set(), stage.scope["credentialed_surface_parents"])

    def test_dropout_continuation_uses_previous_stage_frame_and_cannot_reseed(self):
        x54, x53 = replay.x54, replay.x54.x53
        source = {"track_id": "metric", "parent_track_id": "metric", "disposition": "MEASURED",
                  "support_footprint_mode": x54.SOURCE_MODE, "motion_authority": x54.x27.RIGID_DYNAMIC,
                  "risk_eligible": True, "depth_grid_support": 10, "position_forward_m": 4.,
                  "position_right_m": 0., "velocity_forward_mps": -1., "velocity_right_mps": 0.,
                  "evidence_age_s": 0.}
        observations = tuple(self.observation(i) for i in range(3))
        episode = replay.adapter.Episode("episode_0000", self.anchor, observations)
        frames = [{"sample_index": i, "time_s": o.time_s, "tracks": [source] if i == 0 else [],
                   "risk_eligible_tracks": int(i == 0), "arms": {x53.ARM_X53: {
                       "route_risk": i == 0, "confirmed_risk_track_ids": ["metric"] if i == 0 else []}}}
                  for i, o in enumerate(observations)]
        stage = _Program(x54.apply_metric_bootstrap_dropout_episode)
        outcomes = []
        with patch.object(replay.x24, "load_receipt", return_value=None), \
             patch.object(replay.x24.route, "select_route", return_value=SimpleNamespace(mode="ISSUED_PLAN")), \
             patch.object(replay.x24.route, "first_selected_route_entry_s", return_value=.5), \
             patch.object(x54.x47, "visible_free_space", return_value=False):
            for index, observation in enumerate(observations):
                prefix = replace(episode, observations=observations[:index+1])
                actual = stage.update(episode=prefix, observation=observation, candidate={}, calibration=self.calibration,
                                      ordinal=index, frame=frames[index])
                expected = x54.apply_metric_bootstrap_dropout_episode(
                    prefix, prediction(frames[:index+1], x53.ARM_X53, "x53_route_mode_counts"), self.calibration)
                self.assertEqual(expected["frames"][-1], actual)
                outcomes.append(actual["arms"][x54.ARM_X54]["x54_metric_bootstrap_dropout_used"])
                # A caller editing its current display/query copy cannot rewrite
                # the stage-owned previous carrier and manufacture another birth.
                actual["tracks"].clear()
        self.assertEqual([False, True, False], outcomes)


if __name__ == "__main__":
    unittest.main()
