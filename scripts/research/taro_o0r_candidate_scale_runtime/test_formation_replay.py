#!/usr/bin/env python3
"""Focused aggregation and fail-closed tests for formation replay."""

from __future__ import annotations

import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import formation_replay


PARENT_COUNTS = {
    "470974": 25, "469216": 16, "423614": 11, "467370": 24, "469460": 23, "438794": 26, "467346": 43, "472473": 43,
    "466965": 8, "470808": 10, "482587": 6, "468410": 11, "482858": 8, "482984": 2, "469607": 42, "421593": 18,
    "423474": 23, "470876": 17, "464981": 6, "478016": 10, "422217": 31, "466437": 13, "484003": 9, "421655": 25,
}
FIT = set(list(PARENT_COUNTS)[:8])


def _records() -> list[dict]:
    rows = []
    for parent, count in PARENT_COUNTS.items():
        role = "ADAPTER_FIT" if parent in FIT else "O0R_EVAL_CANDIDATE"
        for frame_index in range(count):
            for query_index in range(9):
                prospective = {
                    "support": {"evaluable": True, "reason_codes": [], "normal_angular_error_rad": 0.1, "height_abs_error_m": 0.2},
                    "boundary": {"evaluable": True, "reason_codes": [], "point_id_jaccard": 0.7, "xyz_median_error_m": 0.3},
                    "query_clearance": {"evaluable": False, "reason_codes": ["QUERY_UNKNOWN"]},
                }
                baseline = {
                    "support": {"evaluable": True, "reason_codes": [], "normal_angular_error_rad": 0.2, "height_abs_error_m": 0.4},
                    "boundary": {"evaluable": True, "reason_codes": [], "point_id_jaccard": 0.5, "xyz_median_error_m": 0.6},
                    "query_clearance": {"evaluable": False, "reason_codes": ["QUERY_UNKNOWN"]},
                }
                rows.append(
                    {
                        "source_role": role,
                        "parent_id": parent,
                        "physical_frame_id": f"{parent}:{frame_index}",
                        "prospective": prospective,
                        "baseline": baseline,
                        "effects": {
                            "support_normal_error_reduction_rad": 0.1,
                            "support_height_error_reduction_m": 0.2,
                            "boundary_jaccard_increase": 0.2,
                            "boundary_xyz_error_reduction_m": 0.3,
                            "query_clearance_error_reduction_m": None,
                        },
                    }
                )
    return rows


class FormationReplaySummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _records()

    def test_fixed_denominator_parent_first_summary(self) -> None:
        summary = formation_replay.summarize(self.rows)
        self.assertEqual("TARO_O0R_R6_FORMATION_REPLAY_COMPLETE_NON_PROMOTABLE", summary["terminal"])
        self.assertEqual((24, 450, 4050), (summary["parent_count"], summary["frame_count"], summary["query_slot_count"]))
        self.assertAlmostEqual(0.2, summary["paired_effects"]["support_height_error_reduction_m"]["parent_macro_median"])
        self.assertEqual(24, summary["paired_effects"]["support_height_error_reduction_m"]["positive_parent_count"])
        self.assertFalse(summary["scientific_pass_fail_assigned"])
        self.assertFalse(summary["promotion_authorized"])

    def test_unknown_is_not_coerced_to_negative(self) -> None:
        summary = formation_replay.summarize(self.rows)
        query = summary["modes"]["prospective"]["query_clearance"]["abs_error_m"]
        self.assertEqual(0, query["query_evaluable_count"])
        self.assertIsNone(query["parent_macro_median"])
        self.assertEqual(8100, summary["unknown_reason_counts"]["QUERY_UNKNOWN"])

    def test_incomplete_query_denominator_is_rejected(self) -> None:
        with self.assertRaisesRegex(formation_replay.FormationReplayError, "4050"):
            formation_replay.summarize(self.rows[:-1])


if __name__ == "__main__":
    unittest.main()
