#!/usr/bin/env python3
from __future__ import annotations

import unittest

import candidate_independent_policy_failure_attribution_r1 as subject


class PolicyFailureAttributionTest(unittest.TestCase):
    @staticmethod
    def token(
        *,
        qualification: int = 100,
        effective_until: int = 600,
        reason: str = "ttl_elapsed",
    ) -> dict:
        return {
            "token_id": "token",
            "qualification_timestamp_ns": qualification,
            "effective_valid_until_timestamp_ns": effective_until,
            "invalidation_reason": reason,
        }

    def test_qualification_insufficient_covers_both_mechanisms(self) -> None:
        for evidence in (
            "NO_POLICY_TOKEN_FOR_TRACK_RESET",
            "ORACLE_BEFORE_POLICY_QUALIFICATION",
        ):
            self.assertEqual(
                "QUALIFICATION_INSUFFICIENT",
                subject.classify_opportunity(evidence),
            )

    def test_ttl_is_distinct_from_early_invalidation(self) -> None:
        self.assertEqual(
            "ORACLE_AFTER_TTL",
            subject.classify_opportunity("ORACLE_AFTER_TTL_ELAPSED"),
        )
        self.assertEqual(
            "RELATION_GAP_BEFORE_ORACLE",
            subject.classify_opportunity("ORACLE_AFTER_ACTIVE_RELATION_GAP"),
        )

    def test_route_track_relation_remain_distinct(self) -> None:
        self.assertEqual(
            "ROUTE_UNKNOWN_BEFORE_ORACLE",
            subject.classify_opportunity("ORACLE_AFTER_ROUTE_UNKNOWN"),
        )
        self.assertEqual(
            "TRACK_UNOBSERVED_BEFORE_ORACLE",
            subject.classify_opportunity("ORACLE_AFTER_TRACK_UNOBSERVED"),
        )

    def test_reset_and_sequence_end_are_not_silently_folded(self) -> None:
        self.assertEqual(
            "RESET_BEFORE_ORACLE",
            subject.classify_opportunity("ORACLE_AFTER_RESET_BEFORE_FRAME"),
        )
        self.assertEqual(
            "SEQUENCE_END_BEFORE_ORACLE",
            subject.classify_opportunity("ORACLE_AFTER_SEQUENCE_END"),
        )

    def test_unrecognized_evidence_fails_to_unexplained(self) -> None:
        self.assertEqual(
            "UNEXPLAINED",
            subject.classify_opportunity("UNKNOWN_EVIDENCE"),
        )

    def test_covered_cell_cannot_be_classified_as_miss(self) -> None:
        with self.assertRaisesRegex(
            subject.parent.PolicyGateContractError, "covered_cell"
        ):
            subject.classify_opportunity("COVERED_WITHIN_VALIDITY")

    def test_timestamp_validity_is_half_open(self) -> None:
        token = self.token()
        self.assertEqual(
            "COVERED_WITHIN_VALIDITY",
            subject.opportunity_evidence(token, 100)[0],
        )
        self.assertEqual(
            "COVERED_WITHIN_VALIDITY",
            subject.opportunity_evidence(token, 599)[0],
        )
        self.assertEqual(
            "ORACLE_AFTER_TTL_ELAPSED",
            subject.opportunity_evidence(token, 600)[0],
        )

    def test_oracle_before_qualification_is_not_backfilled(self) -> None:
        self.assertEqual(
            "ORACLE_BEFORE_POLICY_QUALIFICATION",
            subject.opportunity_evidence(self.token(), 99)[0],
        )
        self.assertEqual(
            "NO_POLICY_TOKEN_FOR_TRACK_RESET",
            subject.opportunity_evidence(None, 99)[0],
        )

    def test_early_invalidation_reason_is_preserved(self) -> None:
        token = self.token(effective_until=300, reason="track_unobserved")
        self.assertEqual(
            "ORACLE_AFTER_TRACK_UNOBSERVED",
            subject.opportunity_evidence(token, 300)[0],
        )

    def test_negative_token_join_groups_source_sequence_reason(self) -> None:
        risk = {
            "token_invalidations": [
                {
                    "token_id": "token",
                    "invalidation_reason": "active_relation_gap",
                    "effective_valid_until_timestamp_ns": 10,
                    "last_valid_frame_id": 1,
                }
            ],
            "negative_exposure_tokens": [
                {
                    "token_id": "token",
                    "source_id": "source",
                    "sequence_id": "sequence",
                    "qualification_timestamp_ns": 1,
                    "negative_exposure_unit_ids": ["negative"],
                }
            ],
        }
        result = subject._negative_token_attribution({}, risk)
        self.assertEqual(0, result["unattributed_count"])
        self.assertEqual(
            {
                "source_id": "source",
                "sequence_id": "sequence",
                "invalidation_reason": "active_relation_gap",
                "negative_token_count": 1,
            },
            result["groups"][0],
        )


if __name__ == "__main__":
    unittest.main()
