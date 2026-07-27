from __future__ import annotations

import unittest

from scripts.research.egomotion_compensated_looming.rcle_unseen_external_confirmation_r0.metrics import (
    BELOW_ROLE,
    FAIL,
    NOT_EVALUABLE,
    PASS,
    POSITIVE_ROLE,
    derive_confirmation_rows,
    evaluate_confirmation,
)


def window(
    source: str,
    role: str,
    values: list[float | None],
    *,
    window_id: str,
    sequence_id: str = "seq",
    start_s: float = 0.0,
    dt_s: float = 0.1,
) -> list[dict]:
    rows = []
    for pair_index, value in enumerate(values):
        previous = start_s + pair_index * dt_s
        rows.append(
            {
                "source_id": source,
                "sequence_id": sequence_id,
                "window_id": window_id,
                "role": role,
                "pair_index": pair_index,
                "previous_timestamp_s": previous,
                "current_timestamp_s": previous + dt_s,
                "evaluable": value is not None,
                "compensated_expansion_median_per_s": value,
            }
        )
    return rows


def passing_cohort() -> list[dict]:
    rows: list[dict] = []
    for source in ("source_a", "source_b"):
        rows += window(
            source,
            POSITIVE_ROLE,
            [0.02] * 20,
            window_id="positive",
        )
        rows += window(
            source,
            BELOW_ROLE,
            [0.02, 0.0] * 5,
            window_id="below",
        )
    return rows


def by_identity(result: dict, source: str, role: str) -> dict:
    return next(
        item
        for item in result["window_summaries"]
        if item["source_id"] == source and item["role"] == role
    )


class TriggerDerivationTest(unittest.TestCase):
    def test_strict_threshold_and_three_pair_boundary(self) -> None:
        rows = passing_cohort()
        rows[:5] = window(
            "source_a",
            POSITIVE_ROLE,
            [0.01, 0.0100001, 0.02, 0.03, 0.01],
            window_id="positive",
        )
        derived = derive_confirmation_rows(rows)
        first = derived[:5]
        self.assertEqual(
            [item["old_trigger"] for item in first],
            [False, True, True, True, False],
        )
        self.assertEqual(
            [item["r1_trigger"] for item in first],
            [False, False, False, True, False],
        )
        self.assertEqual(
            [item["reset_reason"] for item in first],
            [
                "WINDOW_BOUNDARY",
                None,
                None,
                None,
                "AT_OR_BELOW_THRESHOLD",
            ],
        )

    def test_abstention_resets_streak(self) -> None:
        rows = passing_cohort()
        rows[:6] = window(
            "source_a",
            POSITIVE_ROLE,
            [0.02, 0.02, None, 0.02, 0.02, 0.02],
            window_id="positive",
        )
        first = derive_confirmation_rows(rows)[:6]
        self.assertEqual(
            [item["r1_trigger"] for item in first],
            [False, False, False, False, False, True],
        )
        self.assertEqual(first[2]["reset_reason"], "ABSTENTION")

    def test_same_window_names_across_sources_are_isolated(self) -> None:
        rows: list[dict] = []
        for source in ("source_a", "source_b"):
            rows += window(
                source,
                POSITIVE_ROLE,
                [0.02, 0.02, 0.02],
                window_id="shared_positive",
            )
            rows += window(
                source,
                BELOW_ROLE,
                [0.02, 0.0, 0.02],
                window_id="shared_below",
            )
        derived = derive_confirmation_rows(rows)
        starts = [
            item
            for item in derived
            if item["window_id"] == "shared_positive" and item["pair_index"] == 0
        ]
        self.assertEqual(len(starts), 2)
        self.assertTrue(all(item["r1_trigger"] is False for item in starts))
        self.assertTrue(
            all(item["reset_reason"] == "WINDOW_BOUNDARY" for item in starts)
        )


class LocalGateTest(unittest.TestCase):
    def test_exact_gate_boundaries_pass(self) -> None:
        rows = passing_cohort()
        rows[:20] = window(
            "source_a",
            POSITIVE_ROLE,
            [0.02] * 20,
            window_id="positive",
        )
        result = evaluate_confirmation(rows)
        self.assertEqual(result["cohort_status"], PASS)
        positive = by_identity(result, "source_a", POSITIVE_ROLE)
        self.assertAlmostEqual(
            positive["gates"]["positive_retention"]["value"], 0.9
        )
        self.assertAlmostEqual(
            positive["gates"]["positive_first_trigger_delay_s"]["value"], 0.2
        )
        self.assertTrue(result["all_local_gates_and"])

    def test_exact_below_reduction_boundary_passes(self) -> None:
        rows = passing_cohort()
        rows[20:30] = window(
            "source_a",
            BELOW_ROLE,
            [0.02] * 9 + [0.0, 0.02],
            window_id="below",
        )
        result = evaluate_confirmation(rows)
        below = by_identity(result, "source_a", BELOW_ROLE)
        self.assertAlmostEqual(
            below["gates"]["below_relative_reduction"]["value"], 0.30
        )
        self.assertEqual(below["status"], PASS)

    def test_old_below_zero_is_not_evaluable(self) -> None:
        rows = passing_cohort()
        rows[20:30] = window(
            "source_a",
            BELOW_ROLE,
            [0.0] * 10,
            window_id="below",
        )
        result = evaluate_confirmation(rows)
        below = by_identity(result, "source_a", BELOW_ROLE)
        self.assertEqual(below["status"], NOT_EVALUABLE)
        self.assertEqual(
            below["status_reason"], "NOT_EVALUABLE_OLD_BELOW_ZERO"
        )
        self.assertEqual(result["cohort_status"], NOT_EVALUABLE)

    def test_old_positive_zero_is_fail(self) -> None:
        rows = passing_cohort()
        rows[:20] = window(
            "source_a",
            POSITIVE_ROLE,
            [0.0] * 20,
            window_id="positive",
        )
        result = evaluate_confirmation(rows)
        positive = by_identity(result, "source_a", POSITIVE_ROLE)
        self.assertEqual(positive["status"], FAIL)
        self.assertEqual(positive["status_reason"], "FAIL_OLD_POSITIVE_ZERO")
        self.assertEqual(result["cohort_status"], FAIL)

    def test_r1_without_first_trigger_is_fail(self) -> None:
        rows = passing_cohort()
        rows[:20] = window(
            "source_a",
            POSITIVE_ROLE,
            [0.02, 0.0] * 10,
            window_id="positive",
        )
        result = evaluate_confirmation(rows)
        positive = by_identity(result, "source_a", POSITIVE_ROLE)
        self.assertEqual(positive["status"], FAIL)
        self.assertEqual(positive["status_reason"], "FAIL_R1_NO_FIRST_TRIGGER")

    def test_local_failure_cannot_be_rescued_by_pooled_pass(self) -> None:
        rows = passing_cohort()
        rows[20:30] = window(
            "source_a",
            BELOW_ROLE,
            [0.02] * 10,
            window_id="below",
        )
        result = evaluate_confirmation(rows)
        local = by_identity(result, "source_a", BELOW_ROLE)
        pooled = result["pooled_diagnostics"]["pooled_gates"][
            "below_relative_reduction"
        ]
        self.assertEqual(local["status"], FAIL)
        self.assertGreaterEqual(pooled["value"], pooled["threshold"])
        self.assertEqual(result["cohort_status"], FAIL)
        self.assertFalse(result["all_local_gates_and"])

    def test_fail_precedes_not_evaluable(self) -> None:
        rows = passing_cohort()
        rows[20:30] = window(
            "source_a",
            BELOW_ROLE,
            [0.0] * 10,
            window_id="below",
        )
        rows[50:60] = window(
            "source_b",
            BELOW_ROLE,
            [0.02] * 10,
            window_id="below",
        )
        result = evaluate_confirmation(rows)
        self.assertEqual(result["cohort_status"], FAIL)
        self.assertEqual(
            result["scientific_outcome"], "CONFIRMATION_FAIL_STOP_AT_R1"
        )


class InputValidationTest(unittest.TestCase):
    def test_pair_index_gap_is_rejected(self) -> None:
        rows = passing_cohort()
        rows[1]["pair_index"] = 2
        with self.assertRaisesRegex(ValueError, "PAIR_INDEX_NOT_CONTIGUOUS"):
            derive_confirmation_rows(rows)

    def test_timestamp_chain_break_is_rejected(self) -> None:
        rows = passing_cohort()
        rows[1]["previous_timestamp_s"] += 0.001
        with self.assertRaisesRegex(ValueError, "PAIR_TIMESTAMPS_NOT_CONTIGUOUS"):
            derive_confirmation_rows(rows)

    def test_pair_dt_above_geometry_limit_is_rejected(self) -> None:
        rows = passing_cohort()
        rows[0]["current_timestamp_s"] += 0.0001
        with self.assertRaisesRegex(ValueError, "PAIR_DT_EXCEEDS_MAX"):
            derive_confirmation_rows(rows)

    def test_composite_window_block_cannot_reappear(self) -> None:
        rows = passing_cohort()
        positive = rows[:20]
        rows = positive[:2] + rows[20:] + positive[2:]
        with self.assertRaisesRegex(ValueError, "WINDOW_ROWS_NOT_CONTIGUOUS"):
            derive_confirmation_rows(rows)

    def test_exactly_two_sources_are_required(self) -> None:
        rows = [
            row
            for row in passing_cohort()
            if row["source_id"] == "source_a"
        ]
        with self.assertRaisesRegex(ValueError, "SOURCE_COUNT"):
            derive_confirmation_rows(rows)

    def test_one_window_per_role_per_source_is_required(self) -> None:
        rows = passing_cohort()
        rows += window(
            "source_a",
            BELOW_ROLE,
            [0.02, 0.0, 0.02],
            window_id="extra_below",
        )
        with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_WINDOW_COUNT"):
            derive_confirmation_rows(rows)

    def test_abstention_cannot_carry_expansion(self) -> None:
        rows = passing_cohort()
        rows[0]["evaluable"] = False
        with self.assertRaisesRegex(ValueError, "ABSTENTION_EXPANSION_PRESENT"):
            derive_confirmation_rows(rows)


if __name__ == "__main__":
    unittest.main()
