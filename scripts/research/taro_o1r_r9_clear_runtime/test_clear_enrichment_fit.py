from __future__ import annotations

import unittest

from scripts.research.taro_o1r_r9_clear_runtime import clear_enrichment_fit as fit


def feature(query_id: str, anchors: int) -> dict:
    return {"query_id": query_id, "query_receipt": {}, "r6_state": "UNKNOWN", "positive_obstacle_veto": False, "occupied_hits": [[ [False] ]], "far_valid_anchor_count": anchors, "far_fractions": [0.0, 0.0, 0.0], "observed_support_points": 300000}


class ClearEnrichmentFitTests(unittest.TestCase):
    def test_rule_grid_is_unique_and_nonempty(self) -> None:
        rules = fit.candidate_rules()
        self.assertGreater(len(rules), 1000)
        self.assertEqual(len(rules), len({row["rule_id"] for row in rules}))

    def test_eligibility_is_source_only_and_fail_closed(self) -> None:
        rule = next(row for row in fit.candidate_rules() if row["minimum_far_valid_anchor_count"] == 18 and row["maximum_far_valid_anchor_count"] == 24 and row["minimum_observed_support_points"] == 200000 and row["maximum_far_fraction"] == 0.0)
        self.assertTrue(fit.eligible(feature("q", 20), rule))
        missing = feature("q", 20)
        missing["query_receipt"] = None
        self.assertFalse(fit.eligible(missing, rule))
        occupied = feature("q", 20)
        occupied["occupied_hits"] = [[[True]]]
        self.assertFalse(fit.eligible(occupied, rule))


if __name__ == "__main__":
    unittest.main()
