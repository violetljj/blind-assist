#!/usr/bin/env python3
"""Focused tests for the post-hoc R6 factor-split canary."""

from __future__ import annotations

import copy
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import factor_split_canary
from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime.test_r5_confirmation import _positive_query_record


class FactorSplitCanaryTests(unittest.TestCase):
    def test_fixed_baseline_query_block_removes_knownness_regret_without_claiming_pass(self) -> None:
        parent, video = r5.R5_ROSTER[0]
        rows = []
        for grid in range(9):
            row = _positive_query_record(parent, video, 0, grid)
            direct = copy.deepcopy(row["direct_apple_support"])
            direct["query_point_clearance"] = {
                "evaluable": False,
                "reason_codes": ["SYNTHETIC_DIRECT_QUERY_UNKNOWN"],
                "value_m": None,
                "truth_value_m": None,
                "abs_error_m": None,
                "query_support_points": None,
                "observed_forward_m": None,
                "local_valid_fraction": None,
            }
            row["direct_apple_support"] = direct
            row["selected_hybrid"] = copy.deepcopy(direct)
            row["effects"] = r5._effects(row["baseline"], direct)
            row.pop("content_sha256")
            rows.append(r5._seal(row))
        result = factor_split_canary.evaluate_factor_split_landscape(rows, expected_parent_frame_counts={parent: 1})
        self.assertEqual(9, result["baseline_query_known_count"])
        self.assertEqual(9, result["composite_query_known_count"])
        self.assertTrue(result["all_gate_landscape_would_pass"])
        self.assertTrue(result["post_hoc"])
        self.assertFalse(result["promotion_allowed"])
        self.assertNotIn("passed", result)
        self.assertNotIn("terminal", result)


if __name__ == "__main__":
    unittest.main()
