from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from dtr_r0 import (
    Arm,
    CausalFrame,
    DTRConfig,
    DTRR0Arm,
    EgoPose,
    EventLifecycle,
    Observation,
    Prediction,
    Signal,
    Vec2,
    compute_event_metrics,
    run_arm,
)
from evaluate import read_jsonl
from dtr_r1 import run_r1_arm
from dtr_r3 import (
    DTRR3Arm,
    R3Arm,
    RouteSegment,
    WorldTargetObservation,
    _trajectory_evidence,
)
from coda_static_ceiling import segment_to_box_entry_fraction
from jrdb_native_ceiling import (
    AlertSegment,
    TruthEvent,
    average_precision,
    maximum_event_alert_matching,
)


def frame_from_world(
    time_s: float,
    ego: Vec2,
    target: Vec2 | None,
    *,
    body_yaw_rad: float = 0.0,
    sensor_yaw_rad: float = 0.0,
    pose_present: bool = True,
) -> CausalFrame:
    pose = EgoPose(
        x_m=ego.x,
        y_m=ego.y,
        body_yaw_rad=body_yaw_rad,
        sensor_yaw_rad=sensor_yaw_rad,
    )
    if target is None:
        observations: tuple[Observation, ...] = ()
    else:
        forward_m, left_m = pose.world_to_local(target)
        observations = (
            Observation(
                track_id="target",
                forward_m=forward_m,
                left_m=left_m,
                radius_m=0.20,
            ),
        )
    return CausalFrame(
        time_s=time_s,
        ego_pose=pose if pose_present else None,
        observations=observations,
    )


def timeline(
    target_at,
    *,
    duration_s: float = 7.0,
    sensor_yaw_at=lambda _time_s: 0.0,
) -> list[CausalFrame]:
    frames: list[CausalFrame] = []
    for index in range(int(duration_s * 2) + 1):
        time_s = index * 0.5
        frames.append(
            frame_from_world(
                time_s,
                Vec2(time_s, 0.0),
                target_at(time_s),
                sensor_yaw_rad=sensor_yaw_at(time_s),
            )
        )
    return frames


class RouteIntersectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DTRConfig(route_half_width_m=0.45)

    def assert_triggers(self, frames: list[CausalFrame]) -> None:
        predictions = run_arm(frames, Arm.C_ROUTE_INTERSECTION, self.config)
        self.assertIn(Signal.ONSET, [item.signal for item in predictions])

    def assert_never_triggers(self, frames: list[CausalFrame]) -> None:
        predictions = run_arm(frames, Arm.C_ROUTE_INTERSECTION, self.config)
        self.assertFalse(
            any(item.signal in (Signal.ONSET, Signal.HOLD) for item in predictions)
        )

    def test_crossing_and_oncoming_trigger(self) -> None:
        self.assert_triggers(
            timeline(lambda time_s: Vec2(4.0, 0.80 * (4.0 - time_s)))
        )
        self.assert_triggers(timeline(lambda time_s: Vec2(7.0 - time_s, 0.0)))

    def test_imminent_intersection_inside_one_point_five_seconds_still_triggers(self) -> None:
        frames = [
            frame_from_world(0.0, Vec2(0.0, 0.0), Vec2(2.0, 0.0)),
            frame_from_world(0.5, Vec2(0.5, 0.0), Vec2(1.5, 0.0)),
        ]
        self.assert_triggers(frames)

    def test_parallel_and_static_outside_route_do_not_trigger(self) -> None:
        self.assert_never_triggers(
            timeline(lambda time_s: Vec2(time_s + 0.5, 2.0))
        )
        self.assert_never_triggers(timeline(lambda _time_s: Vec2(4.0, 2.0)))

    def test_head_turn_does_not_create_target_motion(self) -> None:
        frames = timeline(
            lambda _time_s: Vec2(4.0, 2.0),
            sensor_yaw_at=lambda time_s: 1.05 * math.sin(time_s),
        )
        self.assert_never_triggers(frames)

    def test_target_exit_produces_stable_clear(self) -> None:
        predictions = run_arm(
            timeline(lambda time_s: Vec2(4.0, 1.05 * (4.0 - time_s))),
            Arm.C_ROUTE_INTERSECTION,
            self.config,
        )
        onset_index = next(
            index for index, item in enumerate(predictions) if item.signal is Signal.ONSET
        )
        self.assertGreater(len(predictions) - onset_index, 3)
        self.assertTrue(all(item.signal is Signal.CLEAR for item in predictions[-3:]))

    def test_unknown_never_clears_an_active_event(self) -> None:
        runner = DTRR0Arm(Arm.C_ROUTE_INTERSECTION, self.config)
        first = runner.step(frame_from_world(0.0, Vec2(0.0, 0.0), Vec2(4.0, 0.0)))
        onset = runner.step(
            frame_from_world(0.5, Vec2(0.5, 0.0), Vec2(3.5, 0.0))
        )
        missing_pose = runner.step(
            frame_from_world(
                1.0,
                Vec2(1.0, 0.0),
                Vec2(3.0, 0.0),
                pose_present=False,
            )
        )
        resumed = runner.step(
            frame_from_world(1.5, Vec2(1.5, 0.0), Vec2(2.5, 0.0))
        )
        self.assertIs(first.signal, Signal.UNKNOWN)
        self.assertIs(onset.signal, Signal.ONSET)
        self.assertIs(missing_pose.signal, Signal.UNKNOWN)
        self.assertIs(resumed.signal, Signal.HOLD)

    def test_single_negative_frame_does_not_fragment_active_event(self) -> None:
        lifecycle = EventLifecycle(clear_grace_s=0.5)
        self.assertIs(lifecycle.update(0.0, True), Signal.ONSET)
        self.assertIs(lifecycle.update(0.2, False), Signal.HOLD)
        self.assertIs(lifecycle.update(0.4, True), Signal.HOLD)

    def test_continuous_negative_clears_only_after_grace(self) -> None:
        lifecycle = EventLifecycle(clear_grace_s=0.5)
        self.assertIs(lifecycle.update(0.0, True), Signal.ONSET)
        self.assertIs(lifecycle.update(0.1, False), Signal.HOLD)
        self.assertIs(lifecycle.update(0.59, False), Signal.HOLD)
        self.assertIs(lifecycle.update(0.6, False), Signal.CLEAR)
        self.assertIs(lifecycle.update(0.7, True), Signal.ONSET)
        self.assertIs(lifecycle.update(0.8, False), Signal.HOLD)
        self.assertIs(lifecycle.update(1.0, None), Signal.UNKNOWN)
        self.assertIs(lifecycle.update(1.5, False), Signal.HOLD)
        self.assertIs(lifecycle.update(2.0, False), Signal.CLEAR)


class RobustOccupancyR1Tests(unittest.TestCase):
    def test_crossing_escalates_once_while_parallel_motion_stays_clear(self) -> None:
        crossing = run_r1_arm(
            timeline(lambda time_s: Vec2(4.0, 0.80 * (4.0 - time_s)))
        )
        parallel = run_r1_arm(
            timeline(lambda time_s: Vec2(time_s + 0.5, 2.0))
        )
        self.assertEqual(
            [item.signal for item in crossing].count(Signal.ESCALATE), 1
        )
        self.assertIn(Signal.ONSET, [item.signal for item in crossing])
        self.assertFalse(
            any(
                item.signal in (Signal.ONSET, Signal.HOLD, Signal.ESCALATE)
                for item in parallel
            )
        )


class BaselineAndMetricTests(unittest.TestCase):
    def test_static_continuous_collision_handles_crossing_grazing_and_rotation(self) -> None:
        crossing = segment_to_box_entry_fraction(
            -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.10, 0.05
        )
        grazing = segment_to_box_entry_fraction(
            -1.0, 0.10, 1.0, 0.10, 0.0, 0.0, 0.0, 0.10, 0.10, 0.05
        )
        miss = segment_to_box_entry_fraction(
            -1.0, 0.101, 1.0, 0.101, 0.0, 0.0, 0.0, 0.10, 0.10, 0.05
        )
        diagonal = math.sqrt(0.5)
        rotated = segment_to_box_entry_fraction(
            -diagonal,
            -diagonal,
            diagonal,
            diagonal,
            0.0,
            0.0,
            math.pi / 4.0,
            0.10,
            0.10,
            0.05,
        )
        self.assertAlmostEqual(crossing, 0.45)
        self.assertAlmostEqual(grazing, 0.475)
        self.assertIsNone(miss)
        self.assertAlmostEqual(rotated, 0.45)

    def test_r3_continuous_toi_catches_between_sample_crossing(self) -> None:
        start = Vec2(0.0, -1.0)
        velocity = Vec2(0.0, 20.0)
        route = (
            RouteSegment(0.0, 0.1, Vec2(0.0, 0.0), Vec2(0.0, 0.0)),
        )
        self.assertGreater(start.norm(), 0.20)
        self.assertGreater((start + velocity * 0.1).norm(), 0.20)
        evidence = _trajectory_evidence(start, velocity, route, 0.20, 0.05)
        self.assertIsNotNone(evidence.entry_time_s)
        self.assertGreater(evidence.entry_time_s, 0.0)
        self.assertLess(evidence.entry_time_s, 0.1)

    def test_r3_distributional_tie_is_not_a_majority(self) -> None:
        runner = DTRR3Arm(R3Arm.B_STRAIGHT_DISTRIBUTIONAL)
        decision = runner._evaluate_track(
            WorldTargetObservation(0.0, "target", Vec2(2.0, 2.0), 0.30),
            (Vec2(-2.0, -2.0), Vec2(0.0, 0.0)),
            (
                RouteSegment(0.0, 3.0, Vec2(0.0, 0.0), Vec2(0.0, 0.0)),
            ),
            distributional=True,
        )
        self.assertEqual(decision.entry_support, 0.5)
        self.assertFalse(decision.raw_alert)

    def test_event_matching_and_tie_aware_average_precision(self) -> None:
        events = [
            TruthEvent(1, 2, 2, "crossing"),
            TruthEvent(3, 4, 4, "oncoming"),
        ]
        self.assertEqual(
            maximum_event_alert_matching(
                [AlertSegment(1, 5), AlertSegment(6, 7)],
                events,
            ),
            1,
        )
        self.assertAlmostEqual(
            average_precision([0.9, 0.8, 0.2], [True, False, True]),
            5.0 / 6.0,
        )

    def test_smoke_evaluator_rejects_unsealed_scientific_input(self) -> None:
        episode = {
            "schema_version": "dtr-r0-episode-v1",
            "episode_id": "not-smoke",
            "mechanism_smoke_only": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.jsonl"
            path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed to mechanism-smoke"):
                read_jsonl(path)

    def test_baseline_rules_and_missing_pose_contract(self) -> None:
        b0 = DTRR0Arm(Arm.B0_DETECTION)
        self.assertIs(
            b0.step(frame_from_world(0.0, Vec2(0.0, 0.0), Vec2(6.0, 0.0))).signal,
            Signal.ONSET,
        )
        self.assertIs(
            b0.step(
                frame_from_world(
                    0.5,
                    Vec2(0.5, 0.0),
                    Vec2(5.5, 0.0),
                    pose_present=False,
                )
            ).signal,
            Signal.HOLD,
        )

        b1_predictions = run_arm(
            [
                frame_from_world(0.0, Vec2(0.0, 0.0), Vec2(4.0, 0.0)),
                frame_from_world(0.5, Vec2(0.5, 0.0), Vec2(2.5, 0.0)),
            ],
            Arm.B1_DISTANCE,
        )
        self.assertIs(b1_predictions[0].signal, Signal.CLEAR)
        self.assertIs(b1_predictions[1].signal, Signal.ONSET)

        b2_predictions = run_arm(
            [
                frame_from_world(0.0, Vec2(0.0, 0.0), Vec2(6.0, 0.0)),
                frame_from_world(0.5, Vec2(0.5, 0.0), Vec2(5.5, 0.0)),
            ],
            Arm.B2_RADIAL_TTC,
        )
        self.assertIs(b2_predictions[0].signal, Signal.UNKNOWN)
        self.assertIs(b2_predictions[1].signal, Signal.ONSET)

    def test_event_metric_counting(self) -> None:
        frames = [{"time_s": value} for value in (0.0, 1.0, 2.0, 3.0, 4.0)]
        episodes = [
            {
                "episode_id": "critical",
                "frames": frames,
                "truth": {
                    "critical_event": True,
                    "event_start_s": 3.0,
                    "event_end_s": 3.5,
                    "warning_start_s": 0.0,
                    "warning_end_s": 3.0,
                    "exit_time_s": 3.5,
                },
            },
            {
                "episode_id": "noncritical",
                "frames": frames,
                "truth": {"critical_event": False},
            },
        ]

        def prediction(time_s: float, signal: Signal) -> Prediction:
            raw = None if signal is Signal.UNKNOWN else signal in (Signal.ONSET, Signal.HOLD)
            return Prediction(time_s, signal, raw, "test")

        predictions = {
            "critical": [
                prediction(0.0, Signal.UNKNOWN),
                prediction(1.0, Signal.ONSET),
                prediction(2.0, Signal.HOLD),
                prediction(3.0, Signal.HOLD),
                prediction(4.0, Signal.CLEAR),
            ],
            "noncritical": [
                prediction(0.0, Signal.CLEAR),
                prediction(1.0, Signal.ONSET),
                prediction(2.0, Signal.HOLD),
                prediction(3.0, Signal.CLEAR),
                prediction(4.0, Signal.CLEAR),
            ],
        }
        metrics = compute_event_metrics(episodes, predictions, clear_grace_s=0.5)
        self.assertEqual(metrics["critical_event_recall"], 1.0)
        self.assertEqual(metrics["irrelevant_alert_segments"], 1)
        self.assertAlmostEqual(metrics["false_alerts_per_minute"], 7.5)
        self.assertEqual(metrics["median_first_alert_lead_s"], 2.0)
        self.assertEqual(metrics["mean_alert_segments_per_critical_event"], 1.0)
        self.assertEqual(metrics["stable_clear_rate"], 1.0)
        self.assertEqual(metrics["median_clear_delay_s"], 0.5)
        self.assertEqual(metrics["known_frame_coverage"], 0.9)


if __name__ == "__main__":
    unittest.main()
