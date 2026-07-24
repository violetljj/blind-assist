from __future__ import annotations

import unittest

from causal_token_policy_risk_gate_r1 import (
    clustered_rate_upper_bound,
    poisson_one_sided_upper_rate,
    produce_policy_ledger,
    stratified_worst_source_cluster_upper_bound,
    validated_negative_interval_index,
)


def frame(
    frame_id: int,
    timestamp_ns: int,
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
        "source_capture_timestamp_ns": timestamp_ns,
        "reset_segment": segment,
        "state_reset_before_frame": reset,
        "route_known": known,
        "observed_track_ids": list(observed),
        "active_relation_track_ids": list(active),
    }


def produce(rows: list[dict]) -> dict:
    return produce_policy_ledger(
        "source",
        "sequence",
        rows,
        minimum_consecutive_active_frames=2,
        minimum_active_relation_duration_ns=500_000_000,
        maximum_token_ttl_ns=500_000_000,
    )


class CausalTokenPolicyRiskGateTests(unittest.TestCase):
    def test_duration_and_frame_floor_are_both_required(self) -> None:
        ledger = produce(
            [
                frame(0, 0, reset=True, active=(7,)),
                frame(1, 100_000_000, active=(7,)),
                frame(2, 500_000_000, active=(7,)),
            ]
        )
        self.assertEqual(ledger["token_count"], 1)
        token = ledger["frames"][2]["token_activations"][0]
        self.assertEqual(token["support_frame_count"], 3)
        self.assertEqual(token["support_duration_ns"], 500_000_000)

    def test_relation_gap_invalidates_and_requalification_is_suppressed(self) -> None:
        ledger = produce(
            [
                frame(0, 0, reset=True, active=(7,)),
                frame(1, 500_000_000, active=(7,)),
                frame(2, 600_000_000, active=()),
                frame(3, 700_000_000, active=(7,)),
                frame(4, 1_200_000_000, active=(7,)),
            ]
        )
        self.assertEqual(ledger["token_count"], 1)
        self.assertEqual(ledger["requalification_suppressed_count"], 1)
        repeat = ledger["frames"][4]["requalifications_suppressed"][0]
        self.assertNotIn("token_id", repeat)
        self.assertNotEqual(
            repeat["requalification_attempt_id"],
            repeat["original_token_id"],
        )
        self.assertEqual(
            ledger["frames"][2]["token_invalidations"][0][
                "invalidation_reason"
            ],
            "active_relation_gap",
        )

    def test_ttl_is_finite_and_terminalized(self) -> None:
        ledger = produce(
            [
                frame(0, 0, reset=True, active=(7,)),
                frame(1, 500_000_000, active=(7,)),
                frame(2, 1_000_000_000, active=(7,)),
            ]
        )
        token = ledger["frames"][1]["token_activations"][0]
        self.assertEqual(token["invalidation_reason"], "ttl_elapsed")
        self.assertEqual(
            token["effective_valid_until_timestamp_ns"], 1_000_000_000
        )

    def test_reset_allows_new_scope(self) -> None:
        ledger = produce(
            [
                frame(0, 0, reset=True, segment=0, active=(7,)),
                frame(1, 500_000_000, segment=0, active=(7,)),
                frame(10, 1_000_000_000, reset=True, segment=1, active=(7,)),
                frame(11, 1_500_000_000, segment=1, active=(7,)),
            ]
        )
        self.assertEqual(ledger["token_count"], 2)

    def test_multiple_fresh_support_requalifications_have_unique_attempt_ids(
        self,
    ) -> None:
        ledger = produce(
            [
                frame(0, 0, reset=True, active=(7,)),
                frame(1, 500_000_000, active=(7,)),
                frame(2, 600_000_000, active=()),
                frame(3, 700_000_000, active=(7,)),
                frame(4, 1_200_000_000, active=(7,)),
                frame(5, 1_300_000_000, active=()),
                frame(6, 1_400_000_000, active=(7,)),
                frame(7, 1_900_000_000, active=(7,)),
            ]
        )
        attempts = [
            row["requalification_attempt_id"]
            for item in ledger["frames"]
            for row in item["requalifications_suppressed"]
        ]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(attempts), len(set(attempts)))

    def test_unknown_route_clears_support_and_never_activates(self) -> None:
        ledger = produce(
            [
                frame(0, 0, reset=True, active=(7,)),
                frame(1, 500_000_000, known=False),
                frame(2, 600_000_000, active=(7,)),
                frame(3, 1_100_000_000, active=(7,)),
            ]
        )
        self.assertEqual(ledger["token_count"], 1)
        self.assertEqual(
            ledger["frames"][3]["token_activations"][0][
                "qualification_frame_id"
            ],
            3,
        )

    def test_forbidden_truth_field_is_rejected(self) -> None:
        row = frame(0, 0, reset=True)
        row["event_id"] = "forbidden"
        with self.assertRaisesRegex(Exception, "producer_forbidden_input"):
            produce([row])

    def test_zero_event_poisson_floor_matches_frozen_contract(self) -> None:
        upper = poisson_one_sided_upper_rate(0, 5.9915, 0.95)
        self.assertIsNotNone(upper)
        self.assertLess(abs(float(upper) - 0.5), 0.0001)

    def test_positive_event_poisson_upper_exceeds_point_rate(self) -> None:
        upper = poisson_one_sided_upper_rate(3, 10.0, 0.95)
        self.assertGreater(float(upper), 0.3)

    def test_cluster_bootstrap_is_deterministic_and_session_scoped(self) -> None:
        sessions = [(0, 2.0), (1, 2.0), (0, 2.0)]
        first = clustered_rate_upper_bound(
            sessions, 0.95, iterations=1_000, seed=20260724
        )
        second = clustered_rate_upper_bound(
            sessions, 0.95, iterations=1_000, seed=20260724
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(float(first), 1 / 6)

    def test_negative_exposure_rejects_overlap_and_duplicate_id(self) -> None:
        mask = {
            "negative_exposure_intervals": [
                {
                    "unit_id": "a",
                    "source_id": "source",
                    "sequence_id": "sequence",
                    "start_ns": 0,
                    "end_ns": 10,
                    "duration_ns": 10,
                },
                {
                    "unit_id": "b",
                    "source_id": "source",
                    "sequence_id": "sequence",
                    "start_ns": 9,
                    "end_ns": 20,
                    "duration_ns": 11,
                },
            ]
        }
        with self.assertRaisesRegex(
            Exception, "negative_interval_overlap"
        ):
            validated_negative_interval_index(mask)
        mask["negative_exposure_intervals"][1]["start_ns"] = 10
        mask["negative_exposure_intervals"][1]["duration_ns"] = 10
        mask["negative_exposure_intervals"][1]["unit_id"] = "a"
        with self.assertRaisesRegex(
            Exception, "negative_interval_unit_id_duplicate"
        ):
            validated_negative_interval_index(mask)

    def test_worst_source_bootstrap_resamples_then_takes_max(self) -> None:
        result = stratified_worst_source_cluster_upper_bound(
            {
                "a": [(0, 2.0), (0, 2.0), (0, 2.0)],
                "b": [(1, 2.0), (0, 2.0), (0, 2.0)],
            },
            0.95,
            iterations=1_000,
            seed=20260724,
            minimum_sessions_per_source=3,
        )
        self.assertIsNotNone(result)
        self.assertGreaterEqual(float(result), 1 / 6)
        self.assertGreater(
            float(poisson_one_sided_upper_rate(0, 1.0, 0.9875)),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
