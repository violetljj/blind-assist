import unittest

from functional_part_binding import (
    FunctionalBindingState,
    FunctionalPartCandidate,
    TaskRelationalFunctionalSelector,
)


def part(candidate_id: str, parent: str, x: float, z: float) -> FunctionalPartCandidate:
    return FunctionalPartCandidate(candidate_id, parent, (x, 0.0, z))


class FunctionalPartBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = TaskRelationalFunctionalSelector()
        self.drawer_parts = [
            part("top-left", "cabinet", -0.2, 1.0),
            part("top-right", "cabinet", 0.2, 1.01),
            part("middle-left", "cabinet", -0.2, 0.7),
            part("middle-right", "cabinet", 0.2, 0.71),
            part("bottom-left", "cabinet", -0.2, 0.4),
            part("bottom-right", "cabinet", 0.2, 0.41),
            part("other-parent", "nightstand", 0.0, 0.5),
        ]

    def test_ordinal_selects_a_set_without_crossing_parent_identity(self) -> None:
        decision = self.selector.select(
            "Open the second drawer of the dresser", "cabinet", self.drawer_parts
        )
        self.assertEqual(FunctionalBindingState.SET_VALUED, decision.state)
        self.assertEqual(("middle-left", "middle-right"), decision.selected_candidate_ids)
        self.assertNotIn("other-parent", decision.selected_candidate_ids)

    def test_bottom_relation_selects_lowest_row(self) -> None:
        decision = self.selector.select(
            "Open the bottom drawer", "cabinet", self.drawer_parts
        )
        self.assertEqual(("bottom-left", "bottom-right"), decision.selected_candidate_ids)

    def test_missing_relation_abstains_across_multiple_rows(self) -> None:
        decision = self.selector.select("Open the dresser", "cabinet", self.drawer_parts)
        self.assertEqual(FunctionalBindingState.AMBIGUOUS, decision.state)
        self.assertEqual("REQUEST_TASK_RELATION_OR_VIEW", decision.action)


if __name__ == "__main__":
    unittest.main()
