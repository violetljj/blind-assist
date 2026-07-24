#!/usr/bin/env python3
from __future__ import annotations

import unittest

from route_invalid_reset_lifecycle_diagnostic import (
    DiagnosticContractError,
    apply_guard,
)


def frame(
    frame_id: int,
    *,
    route_known: bool = True,
    reset: bool = False,
    deliveries: list[int] | None = None,
    delivery_track_ids: list[int] | None = None,
    closures: list[int] | None = None,
    active: bool = False,
) -> dict:
    return {
        "source_id": "source",
        "sequence_id": "sequence",
        "frame_id": frame_id,
        "source_capture_timestamp_ns": frame_id * 100_000_000,
        "state_reset_before_frame": reset,
        "route_known": route_known,
        "candidate_active": active,
        "deliveries": deliveries or [],
        "delivery_track_ids": delivery_track_ids or [],
        "closures": closures or [],
    }


class RouteInvalidResetLifecycleDiagnosticTest(unittest.TestCase):
    def test_route_invalid_closes_active_on_same_frame(self) -> None:
        guarded, counters = apply_guard(
            [
                frame(
                    0,
                    reset=True,
                    deliveries=[1],
                    delivery_track_ids=[9],
                    active=True,
                ),
                frame(1, route_known=False, active=True),
            ]
        )
        self.assertTrue(guarded[0]["guarded_active"])
        self.assertFalse(guarded[1]["guarded_active"])
        self.assertEqual(
            ["route_invalid"],
            [
                event["reason"]
                for event in guarded[1]["lifecycle_events"]
                if event["kind"] == "closure"
            ],
        )
        self.assertEqual(0, counters["guarded_route_invalid_active_frames"])

    def test_reset_closes_prior_scope_and_reuses_no_key(self) -> None:
        guarded, counters = apply_guard(
            [
                frame(
                    0,
                    reset=True,
                    deliveries=[1],
                    delivery_track_ids=[9],
                    active=True,
                ),
                frame(
                    5,
                    reset=True,
                    deliveries=[1],
                    delivery_track_ids=[10],
                    active=True,
                ),
            ]
        )
        first_key = guarded[0]["lifecycle_events"][0][
            "scoped_episode_key"
        ]
        reset_events = guarded[1]["lifecycle_events"]
        self.assertEqual("reset_scope_end", reset_events[0]["reason"])
        second_key = reset_events[1]["scoped_episode_key"]
        self.assertNotEqual(first_key, second_key)
        self.assertEqual(0, counters.get("active_key_cross_reset", 0))

    def test_new_activation_gets_new_ordinal_in_same_reset_scope(self) -> None:
        guarded, _ = apply_guard(
            [
                frame(
                    0,
                    reset=True,
                    deliveries=[1],
                    delivery_track_ids=[9],
                    active=True,
                ),
                frame(1, closures=[1]),
                frame(
                    2,
                    deliveries=[1],
                    delivery_track_ids=[9],
                    active=True,
                ),
            ]
        )
        first = guarded[0]["lifecycle_events"][0][
            "scoped_episode_key"
        ]
        second = guarded[2]["lifecycle_events"][0][
            "scoped_episode_key"
        ]
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith("activation-1"))
        self.assertTrue(second.endswith("activation-2"))

    def test_route_invalid_frame_cannot_deliver(self) -> None:
        with self.assertRaisesRegex(
            DiagnosticContractError,
            "baseline_delivery_on_route_invalid_frame",
        ):
            apply_guard(
                [
                    frame(
                        0,
                        reset=True,
                        route_known=False,
                        deliveries=[1],
                        delivery_track_ids=[9],
                    )
                ]
            )

    def test_truth_fields_are_rejected_before_guard_state_update(self) -> None:
        with self.assertRaisesRegex(
            DiagnosticContractError,
            "guard_input_forbidden_field",
        ):
            apply_guard(
                [
                    {
                        **frame(0, reset=True),
                        "truth_terminal_clear_frame": 0,
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
