from __future__ import annotations

import unittest

from scripts.research.egomotion_compensated_looming.rcle_low_reference_false_trigger_r1.temporal_confirmation import (
    apply_confirmation,
)


def row(
    pair_index: int,
    value: float,
    *,
    window: str = "w0",
    evaluable: bool = True,
) -> dict:
    return {
        "window_id": window,
        "role": "POSITIVE_APPROACH_WINDOW",
        "pair_index": pair_index,
        "previous_timestamp_s": float(pair_index),
        "current_timestamp_s": float(pair_index + 1),
        "dt_s": 1.0,
        "evaluable": evaluable,
        "reason": None if evaluable else "ABSTAIN",
        "compensated_expansion_median_per_s": value if evaluable else None,
        "trigger": bool(evaluable and value > 0.01),
    }


class TemporalConfirmationTest(unittest.TestCase):
    def test_requires_three_consecutive_pairs(self) -> None:
        revised = apply_confirmation([row(0, 0.02), row(1, 0.03), row(2, 0.04)])
        self.assertEqual([item["revised_trigger"] for item in revised], [False, False, True])

    def test_threshold_is_strict(self) -> None:
        revised = apply_confirmation(
            [row(0, 0.02), row(1, 0.01), row(2, 0.03), row(3, 0.04), row(4, 0.05)]
        )
        self.assertEqual(
            [item["revised_trigger"] for item in revised],
            [False, False, False, False, True],
        )

    def test_abstention_resets(self) -> None:
        revised = apply_confirmation(
            [
                row(0, 0.02),
                row(1, 0.03),
                row(2, 0.0, evaluable=False),
                row(3, 0.04),
                row(4, 0.05),
                row(5, 0.06),
            ]
        )
        self.assertEqual(
            [item["revised_trigger"] for item in revised],
            [False, False, False, False, False, True],
        )

    def test_window_boundary_resets(self) -> None:
        revised = apply_confirmation(
            [row(0, 0.02), row(1, 0.03), row(0, 0.04, window="w1")]
        )
        self.assertEqual([item["revised_trigger"] for item in revised], [False, False, False])


if __name__ == "__main__":
    unittest.main()
