#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

import route_conditioned_scale_growth_separability_r0 as core


class ScaleGrowthMathTest(unittest.TestCase):
    def test_normalized_area_scale(self) -> None:
        actual = core.normalized_scale([0.0, 0.0, 320.0, 240.0], 640, 480)
        self.assertAlmostEqual(actual, 0.5 * math.log(0.25))

    def test_boundary_touch_is_invalid(self) -> None:
        self.assertTrue(
            core.bbox_touches_boundary([0.0, 5.0, 10.0, 20.0], 640, 480)
        )
        self.assertFalse(
            core.bbox_touches_boundary([0.1, 5.0, 639.9, 479.9], 640, 480)
        )

    def test_theil_sen_uses_real_irregular_timestamps(self) -> None:
        rows = [
            {"timestamp_s": 0.0, "scale": 1.0},
            {"timestamp_s": 0.1, "scale": 1.2},
            {"timestamp_s": 0.4, "scale": 1.8},
            {"timestamp_s": 0.55, "scale": 2.1},
            {"timestamp_s": 0.6, "scale": 2.2},
        ]
        self.assertAlmostEqual(core.theil_sen_slope(rows), 2.0)

    def test_window_includes_exact_600ms_boundary(self) -> None:
        rows = [
            {"timestamp_s": value, "scale": value}
            for value in (0.0, 0.15, 0.30, 0.45, 0.60)
        ]
        window, reason = core.eligible_window(rows)
        self.assertIsNone(reason)
        self.assertEqual(len(window), 5)

    def test_window_rejects_gap_above_150ms(self) -> None:
        rows = [
            {"timestamp_s": value, "scale": value}
            for value in (0.0, 0.15, 0.301, 0.45, 0.60)
        ]
        _, reason = core.eligible_window(rows)
        self.assertEqual(reason, "maximum_adjacent_gap_exceeded")

    def test_window_rejects_fewer_than_five(self) -> None:
        rows = [
            {"timestamp_s": value, "scale": value}
            for value in (0.0, 0.1, 0.2, 0.3)
        ]
        _, reason = core.eligible_window(rows)
        self.assertEqual(reason, "minimum_observations_not_met")

    def test_frontier_is_complete_unique_descending(self) -> None:
        self.assertEqual(
            core.complete_threshold_breakpoints([0.1, -0.2, 0.1, 0.0]),
            [0.1, 0.0, -0.2],
        )

    def test_nonfinite_frontier_fails_closed(self) -> None:
        with self.assertRaises(core.ScaleGrowthContractError):
            core.complete_threshold_breakpoints([0.0, float("nan")])


class ContractReceiptTest(unittest.TestCase):
    def test_process_id_is_only_nondeterministic_contract_field(self) -> None:
        left = {"process_id": 1, "violations": [{"code": "x"}]}
        right = {"process_id": 2, "violations": [{"code": "x"}]}
        self.assertTrue(core._same_except_process_id(left, right))

    def test_legal_terminal_roster_is_closed(self) -> None:
        self.assertEqual(
            core.LEGAL_TERMINALS,
            (
                "PURE_SCALE_GROWTH_NOT_SUFFICIENT_FOR_STANDALONE_TOKEN_QUALIFICATION",
                "SCALE_GROWTH_DISCOVERY_CANDIDATE_FROZEN",
                "FAIL_CLOSED_INPUT_OR_CONTRACT_BLOCKED",
            ),
        )


if __name__ == "__main__":
    unittest.main()
