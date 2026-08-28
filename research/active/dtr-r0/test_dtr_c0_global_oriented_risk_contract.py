from __future__ import annotations

import unittest

from dtr_c0_global_oriented_risk_contract import (
    CLEAR,
    CONTACT,
    PROXIMITY,
    UNKNOWN,
    Interval,
    _contract_metrics_evaluable,
    _match_intervals,
    classify_future_contract,
)


class DTRC0GlobalOrientedRiskContractTest(unittest.TestCase):
    def test_truth_precedence_keeps_circle_only_as_proximity(self) -> None:
        self.assertEqual(
            classify_future_contract(obb_hit=False, circle_hit=True, full_future=True),
            PROXIMITY,
        )
        self.assertEqual(
            classify_future_contract(obb_hit=True, circle_hit=True, full_future=True),
            CONTACT,
        )
        self.assertEqual(
            classify_future_contract(obb_hit=False, circle_hit=False, full_future=True),
            CLEAR,
        )
        self.assertEqual(
            classify_future_contract(obb_hit=False, circle_hit=False, full_future=False),
            UNKNOWN,
        )

    def test_global_event_matching_is_one_to_one(self) -> None:
        predictions = [Interval(10, 30), Interval(31, 40)]
        truths = [Interval(20, 35), Interval(37, 45)]
        self.assertEqual(_match_intervals(predictions, truths), [(0, 0), (1, 1)])

    def test_component_identity_does_not_enter_interval_correctness(self) -> None:
        prediction = Interval(10, 12)
        truth = Interval(11, 13)
        self.assertTrue(prediction.overlaps(truth))

    def test_always_contact_window_is_not_evaluable(self) -> None:
        saturated = {
            frame: {"label": CONTACT} for frame in range(115, 258)
        }
        self.assertFalse(_contract_metrics_evaluable(saturated))

        bounded = {
            frame: {"label": CONTACT if 130 <= frame <= 140 else CLEAR}
            for frame in range(115, 258)
        }
        self.assertTrue(_contract_metrics_evaluable(bounded))


if __name__ == "__main__":
    unittest.main()
