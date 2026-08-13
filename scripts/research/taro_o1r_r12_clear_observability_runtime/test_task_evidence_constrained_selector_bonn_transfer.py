import unittest

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_constrained_selector_bonn_transfer as subject


class ConstraintFirstSelectorTest(unittest.TestCase):
    @staticmethod
    def _candidate(name: str, macro: float, strict: int, passive: float = 10.0, generic: float = 10.0) -> dict:
        return {
            "family": "ANALYTIC",
            "name": name,
            "fit_parent_macro": {"ranker": macro, "passive": passive, "generic": generic},
            "strict_win_parent_count": strict,
        }

    def test_constraint_precedes_macro_optimization(self) -> None:
        inadmissible_high_mean = self._candidate("high", 14.0, 3)
        admissible = self._candidate("eligible", 11.0, 5)
        selected = subject.select_admissible_candidate([inadmissible_high_mean, admissible])
        self.assertIsNotNone(selected)
        self.assertEqual("eligible", selected["name"])

    def test_no_candidate_returns_none(self) -> None:
        self.assertIsNone(subject.select_admissible_candidate([self._candidate("fail", 9.0, 6)]))


if __name__ == "__main__":
    unittest.main()
