import unittest

from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d3_phase_a import (
    expected_selected_rows,
)


class D3PhaseAValidatorTest(unittest.TestCase):
    def test_expected_selection_uses_all_rows_and_first_eligible(self) -> None:
        processed = [
            {"pool_order": index + 1, "eligible": index % 2 == 0}
            for index in range(48)
        ]
        selected = expected_selected_rows(processed, 8)
        self.assertEqual([1, 3, 5, 7, 9, 11, 13, 15], [row["pool_order"] for row in selected])
        self.assertEqual(48, len(processed))

    def test_fewer_than_target_returns_only_available_without_roles(self) -> None:
        processed = [
            {"pool_order": 1, "eligible": True},
            {"pool_order": 2, "eligible": False},
        ]
        self.assertEqual(1, len(expected_selected_rows(processed, 32)))


if __name__ == "__main__":
    unittest.main()
