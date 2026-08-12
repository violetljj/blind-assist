import unittest

from scripts.research.hftf.deployment.depthart.audit_depthart_task_preserving_d3_phase_a_terminal import (
    FAIL_TERMINAL,
    PASS_TERMINAL,
    expected_candidate_summary,
    expected_terminal_selection,
)


class D3PhaseATerminalAuditTest(unittest.TestCase):
    def test_fail_terminal_publishes_no_partial_selection(self) -> None:
        processed = [
            {
                "pool_order": index + 1,
                "visit_id": f"visit-{index + 1}",
                "video_id": f"video-{index + 1}",
                "fold": "Training",
                "eligible": index < 21,
            }
            for index in range(48)
        ]
        terminal, selected = expected_terminal_selection(processed, 32)
        self.assertEqual(FAIL_TERMINAL, terminal)
        self.assertEqual([], selected)
        self.assertEqual(21, len(expected_candidate_summary(processed)))

    def test_pass_terminal_locks_first_qualified_in_pool_order(self) -> None:
        processed = [
            {
                "pool_order": index + 1,
                "visit_id": f"visit-{index + 1}",
                "video_id": f"video-{index + 1}",
                "fold": "Training",
                "eligible": index != 1,
            }
            for index in range(48)
        ]
        terminal, selected = expected_terminal_selection(processed, 32)
        self.assertEqual(PASS_TERMINAL, terminal)
        self.assertEqual([1] + list(range(3, 34)), [row["pool_order"] for row in selected])


if __name__ == "__main__":
    unittest.main()
