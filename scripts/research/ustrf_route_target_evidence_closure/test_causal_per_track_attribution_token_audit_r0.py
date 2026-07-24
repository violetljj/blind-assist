from __future__ import annotations

import unittest

from causal_per_track_attribution_token_audit_r0 import (
    CausalTokenContractError,
    collapse_candidate_projections,
    produce_sequence_ledger,
    project_allowed_frame,
)


def allowed_frame(
    frame_id: int,
    *,
    reset: bool = False,
    segment: int = 0,
    known: bool = True,
    observed: tuple[int, ...] = (7,),
    active: tuple[int, ...] = (),
) -> dict:
    return {
        "source_id": "source",
        "sequence_id": "sequence",
        "frame_id": frame_id,
        "source_capture_timestamp_ns": 1_000 + frame_id,
        "reset_segment": segment,
        "state_reset_before_frame": reset,
        "route_known": known,
        "observed_track_ids": list(observed),
        "active_relation_track_ids": list(active),
    }


def raw_frame(frame_id: int, *, active: tuple[int, ...] = ()) -> dict:
    return {
        "source_id": "source",
        "sequence_id": "sequence",
        "frame_id": frame_id,
        "source_capture_timestamp_ns": 1_000 + frame_id,
        "state_reset_before_frame": frame_id == 0,
        "reset_segment": 0,
        "route_known": True,
        "observed_tracks": [{"track_id": 7, "box": [0, 0, 1, 1]}],
        "active_relation_track_ids": list(active),
    }


class CausalPerTrackTokenTests(unittest.TestCase):
    def test_emits_once_after_two_consecutive_active_frames(self) -> None:
        ledger = produce_sequence_ledger(
            "source",
            "sequence",
            [
                allowed_frame(0, reset=True, active=(7,)),
                allowed_frame(1, active=(7,)),
                allowed_frame(2, active=(7,)),
            ],
            min_consecutive_relation_frames=2,
        )
        self.assertEqual(ledger["token_count"], 1)
        self.assertEqual(
            ledger["frames"][1]["token_activations"][0][
                "qualification_frame_id"
            ],
            1,
        )

    def test_repeat_activation_is_reported_but_not_reemitted(self) -> None:
        ledger = produce_sequence_ledger(
            "source",
            "sequence",
            [
                allowed_frame(0, reset=True, active=(7,)),
                allowed_frame(1, active=(7,)),
                allowed_frame(2, active=()),
                allowed_frame(3, active=(7,)),
                allowed_frame(4, active=(7,)),
            ],
            min_consecutive_relation_frames=2,
        )
        self.assertEqual(ledger["token_count"], 1)
        self.assertEqual(ledger["repeat_activation_count"], 1)
        self.assertEqual(
            ledger["frames"][4]["repeat_activations_suppressed"][0][
                "original_qualification_frame_id"
            ],
            1,
        )

    def test_reset_allows_new_reset_scoped_token(self) -> None:
        ledger = produce_sequence_ledger(
            "source",
            "sequence",
            [
                allowed_frame(0, reset=True, segment=0, active=(7,)),
                allowed_frame(1, segment=0, active=(7,)),
                allowed_frame(10, reset=True, segment=1, active=(7,)),
                allowed_frame(11, segment=1, active=(7,)),
            ],
            min_consecutive_relation_frames=2,
        )
        self.assertEqual(ledger["token_count"], 2)
        ids = [
            row["token_id"]
            for frame in ledger["frames"]
            for row in frame["token_activations"]
        ]
        self.assertEqual(len(ids), len(set(ids)))

    def test_unknown_route_clears_streak_and_cannot_emit(self) -> None:
        ledger = produce_sequence_ledger(
            "source",
            "sequence",
            [
                allowed_frame(0, reset=True, active=(7,)),
                allowed_frame(1, known=False, active=()),
                allowed_frame(2, active=(7,)),
                allowed_frame(3, active=(7,)),
            ],
            min_consecutive_relation_frames=2,
        )
        self.assertEqual(ledger["token_count"], 1)
        self.assertEqual(
            ledger["frames"][3]["token_activations"][0][
                "qualification_frame_id"
            ],
            3,
        )

    def test_active_relation_on_unknown_route_is_rejected(self) -> None:
        frames = [allowed_frame(0, reset=True, known=False, active=(7,))]
        with self.assertRaisesRegex(
            CausalTokenContractError, "active_relation_on_unknown_route"
        ):
            produce_sequence_ledger(
                "source",
                "sequence",
                frames,
                min_consecutive_relation_frames=2,
            )

    def test_unobserved_active_track_is_rejected(self) -> None:
        frames = [
            allowed_frame(
                0, reset=True, observed=(7,), active=(8,)
            )
        ]
        with self.assertRaisesRegex(
            CausalTokenContractError, "active_relation_track_unobserved"
        ):
            produce_sequence_ledger(
                "source",
                "sequence",
                frames,
                min_consecutive_relation_frames=2,
            )

    def test_forbidden_truth_or_event_fields_are_rejected(self) -> None:
        for forbidden in (
            "event_id",
            "truth_box",
            "alertable_window",
            "future_frames",
            "clearance",
            "oracle_token",
            "candidate_id",
        ):
            frame = allowed_frame(0, reset=True)
            frame[forbidden] = "forbidden"
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(
                    CausalTokenContractError, "producer_forbidden_input"
                ):
                    produce_sequence_ledger(
                        "source",
                        "sequence",
                        [frame],
                        min_consecutive_relation_frames=2,
                    )

    def test_three_candidate_projections_collapse_only_when_identical(
        self,
    ) -> None:
        traces = {}
        for candidate in (
            "C1_CAUSAL_ROUTE_RELATION_FSM",
            "C2_ROUTE_OCCUPANCY_EPISODE_FSM",
            "C3_DUAL_KEY_CLEARANCE_FSM",
        ):
            traces[(candidate, "source", "sequence")] = {
                "frames": [raw_frame(0), raw_frame(1, active=(7,))]
            }
        collapsed = collapse_candidate_projections(traces)
        self.assertEqual(len(collapsed), 1)
        traces[
            ("C3_DUAL_KEY_CLEARANCE_FSM", "source", "sequence")
        ]["frames"][1]["active_relation_track_ids"] = []
        with self.assertRaisesRegex(
            CausalTokenContractError,
            "producer_candidate_projection_mismatch",
        ):
            collapse_candidate_projections(traces)

    def test_projected_frame_contains_only_allowed_runtime_facts(self) -> None:
        projected = project_allowed_frame(raw_frame(0, active=(7,)))
        self.assertEqual(
            set(projected),
            {
                "source_id",
                "sequence_id",
                "frame_id",
                "source_capture_timestamp_ns",
                "reset_segment",
                "state_reset_before_frame",
                "route_known",
                "observed_track_ids",
                "active_relation_track_ids",
            },
        )


if __name__ == "__main__":
    unittest.main()
