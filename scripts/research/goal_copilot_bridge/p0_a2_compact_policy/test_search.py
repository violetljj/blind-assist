from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p0_a2_compact_policy import search


def predicate(name: str, mask: int) -> dict:
    return search._predicate(name, "min", 0.5, mask)


def policy(macro: float) -> dict:
    return {
        "metrics": {
            "ambiguous_false_commit_rate_venue_parent_macro": {"value": macro},
        }
    }


class SearchTest(unittest.TestCase):
    def test_boolean_forms_use_expected_masks(self) -> None:
        left = predicate("a", 0b1100)
        right = predicate("b", 0b1010)
        self.assertEqual(0b1000, search._combine("and", [left, right])["mask"])
        self.assertEqual(0b1110, search._combine("or", [left, right])["mask"])

    def test_three_predicate_forms_are_canonical_and_complete(self) -> None:
        forms = search._forms_for_three(predicate("a", 1), predicate("b", 2), predicate("c", 4))
        self.assertEqual(8, len(forms))
        self.assertEqual(8, len({search._expression_key(row["expression"]) for row in forms}))
        self.assertTrue(all(row["complexity"] == 3 for row in forms))

    def test_terminal_requires_five_point_hard_feasible_gain(self) -> None:
        incumbent = policy(0.196)
        self.assertEqual(
            "CLEAR_COMPACT_POLICY_IMPROVEMENT",
            search._decide_terminal(incumbent, policy(0.146), None, 0.05),
        )
        self.assertEqual(
            "COMPLEXITY_ONLY_BUYS_ABSTENTION",
            search._decide_terminal(incumbent, policy(0.160), policy(0.140), 0.05),
        )
        self.assertEqual(
            "A1_COMPACT_RULE_RETAINED_NO_MEANINGFUL_COMPLEXITY_GAIN",
            search._decide_terminal(incumbent, policy(0.160), policy(0.150), 0.05),
        )

    def test_worst_parent_bins_keep_parent_denominators(self) -> None:
        rows = [
            {"episode_id": "a1", "venue_parent_id": "a", "truth": "AMBIGUOUS"},
            {"episode_id": "a2", "venue_parent_id": "a", "truth": "AMBIGUOUS"},
            {"episode_id": "b1", "venue_parent_id": "b", "truth": "AMBIGUOUS"},
            {"episode_id": "r", "venue_parent_id": "r", "truth": "RESOLVABLE"},
        ]
        report = search._parent_behavior(rows, {"a1"})
        self.assertEqual({"ZERO": 1, "GT_0_LE_0_25": 0, "GT_0_25_LE_0_50": 1, "GT_0_50": 0}, report["bins"])
        self.assertEqual(0.5, report["worst_rate"])
        self.assertEqual(["a"], report["worst_parent_ids"])


if __name__ == "__main__":
    unittest.main()
