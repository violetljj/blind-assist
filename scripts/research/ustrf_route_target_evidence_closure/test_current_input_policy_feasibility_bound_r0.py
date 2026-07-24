#!/usr/bin/env python3
from __future__ import annotations

import unittest

import current_input_policy_feasibility_bound_r0 as subject


def frame(
    frame_id: int,
    timestamp_ns: int,
    *,
    track_id: int = 7,
    active: bool = True,
    route_known: bool = True,
    reset: bool = False,
) -> dict:
    observed = [track_id] if active else []
    return {
        "source_id": "source",
        "sequence_id": "sequence",
        "frame_id": frame_id,
        "source_capture_timestamp_ns": timestamp_ns,
        "state_reset_before_frame": reset,
        "reset_segment": 0,
        "route_known": route_known,
        "observed_track_ids": observed,
        "active_relation_track_ids": observed if active and route_known else [],
    }


class CurrentInputPolicyFeasibilityBoundTest(unittest.TestCase):
    def test_track_id_is_scope_only_and_alpha_renaming_is_invariant(self) -> None:
        first = {
            ("source", "sequence"): [
                frame(0, 100, reset=True),
                frame(1, 200),
                frame(2, 400),
            ]
        }
        renamed = {
            ("source", "sequence"): [
                frame(0, 100, track_id=99, reset=True),
                frame(1, 200, track_id=99),
                frame(2, 400, track_id=99),
            ]
        }
        left = subject.build_track_scopes(first)
        right = subject.build_track_scopes(renamed)
        left_runs = next(iter(left.values()))
        right_runs = next(iter(right.values()))
        self.assertEqual(left_runs, right_runs)

    def test_timestamp_translation_preserves_support_durations(self) -> None:
        base = {
            ("source", "sequence"): [
                frame(0, 100, reset=True),
                frame(1, 250),
                frame(2, 500),
            ]
        }
        shifted = {
            ("source", "sequence"): [
                frame(0, 10_100, reset=True),
                frame(1, 10_250),
                frame(2, 10_500),
            ]
        }
        left = next(iter(subject.build_track_scopes(base).values()))
        right = next(iter(subject.build_track_scopes(shifted).values()))
        self.assertEqual(
            [
                row["support_duration_ns"]
                for row in left[0]
            ],
            [
                row["support_duration_ns"]
                for row in right[0]
            ],
        )

    def test_route_unknown_is_fail_closed_and_splits_runs(self) -> None:
        rows = {
            ("source", "sequence"): [
                frame(0, 0, reset=True),
                frame(1, 100),
                frame(2, 200, active=False, route_known=False),
                frame(3, 300),
                frame(4, 400),
            ]
        }
        runs = next(iter(subject.build_track_scopes(rows).values()))
        self.assertEqual([2, 2], [len(run) for run in runs])

    def test_reset_separates_track_scope(self) -> None:
        rows = {
            ("source", "sequence"): [
                frame(0, 0, reset=True),
                frame(1, 100),
                {
                    **frame(2, 200, reset=True),
                    "reset_segment": 1,
                },
                {
                    **frame(3, 300),
                    "reset_segment": 1,
                },
            ]
        }
        scopes = subject.build_track_scopes(rows)
        self.assertEqual({0, 1}, {key[2] for key in scopes})

    def test_half_open_negative_interval(self) -> None:
        intervals = [{"start_ns": 100, "end_ns": 200}]
        self.assertTrue(subject._timestamp_is_negative(100, intervals))
        self.assertTrue(subject._timestamp_is_negative(199, intervals))
        self.assertFalse(subject._timestamp_is_negative(200, intervals))

    def test_activation_ranges_encode_two_frame_guard_and_no_renewal(self) -> None:
        scopes = {
            ("source", "sequence", 0, 7): [
                [
                    {"frame_id": 0, "timestamp_ns": 0, "support_duration_ns": 0},
                    {
                        "frame_id": 1,
                        "timestamp_ns": 100,
                        "support_duration_ns": 100,
                    },
                ],
                [
                    {
                        "frame_id": 3,
                        "timestamp_ns": 300,
                        "support_duration_ns": 0,
                    },
                    {
                        "frame_id": 4,
                        "timestamp_ns": 500,
                        "support_duration_ns": 200,
                    },
                ],
            ]
        }
        events = {
            ("source", "sequence", "event"): [
                {
                    "track_id": 7,
                    "reset_segment": 0,
                    "oracle_frame_id": 4,
                    "oracle_timestamp_ns": 500,
                }
            ]
        }
        intervals, keys = subject.build_activation_intervals(scopes, events, {})
        self.assertEqual(1, len(keys))
        self.assertEqual(
            [
                (1, 100, 0),
                (101, 200, 1),
            ],
            [
                (
                    row["lower_duration_ns"],
                    row["upper_duration_ns"],
                    row["coverage_mask"],
                )
                for row in intervals
            ],
        )

    def test_frontier_feasible_toy(self) -> None:
        rows = [
            {
                "lower_duration_ns": 1,
                "upper_duration_ns": 10,
                "negative_activation": False,
                "coverage_mask": 0b11,
            }
        ]
        result = subject.sweep_empirical_frontier(rows, 2, 0)
        self.assertTrue(result["simultaneous_empirical_gate_attainable"])
        self.assertEqual(2, result["maximum_supported_unique_event_coverage"])

    def test_frontier_not_feasible_toy(self) -> None:
        rows = [
            {
                "lower_duration_ns": 1,
                "upper_duration_ns": 10,
                "negative_activation": True,
                "coverage_mask": 0b01,
            },
            {
                "lower_duration_ns": 11,
                "upper_duration_ns": 20,
                "negative_activation": False,
                "coverage_mask": 0b10,
            },
        ]
        result = subject.sweep_empirical_frontier(rows, 2, 0)
        self.assertFalse(result["simultaneous_empirical_gate_attainable"])
        self.assertEqual(1, result["maximum_supported_unique_event_coverage"])

    def test_frontier_summary_does_not_return_threshold_or_policy(self) -> None:
        result = subject.sweep_empirical_frontier(
            [
                {
                    "lower_duration_ns": 1,
                    "upper_duration_ns": 2,
                    "negative_activation": False,
                    "coverage_mask": 1,
                }
            ],
            1,
            0,
        )
        serialized = repr(result).lower()
        self.assertNotIn("duration_breakpoint", serialized)
        self.assertNotIn("policy", serialized)
        self.assertNotIn("witness", serialized)

    def test_only_two_legal_terminal_states_are_named(self) -> None:
        self.assertEqual(
            (
                "CURRENT_INPUT_POLICY_FAMILY_FEASIBLE",
                "CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE",
            ),
            subject.LEGAL_TERMINAL_STATES,
        )


if __name__ == "__main__":
    unittest.main()
