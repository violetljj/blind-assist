from __future__ import annotations

import unittest

from dtr_m3_realized_future_contract_decomposition import (
    _box_clearance,
    classify_false_contract,
    classify_positive_contract,
)


class DTRM3RealizedFutureContractDecompositionTest(unittest.TestCase):
    def test_evaluator_circle_can_hit_while_narrow_obb_side_misses(self) -> None:
        # The evaluator radius is 0.5*max(0.94, 0.45)=0.47: a center at
        # lateral 1.10 m hits the 0.65+0.47 circle, while a yaw=0 OBB exposes
        # only its 0.225 m narrow half-width and misses the 0.65 route body.
        eval_circle_clearance = 1.10 - (0.65 + 0.47)
        obb_clearance = _box_clearance(0.0, 1.10, 0.0, 0.94, 0.45)
        self.assertLessEqual(eval_circle_clearance, 0.0)
        self.assertGreater(obb_clearance, 0.0)

    def test_positive_contract_separates_center_dynamics(self) -> None:
        self.assertEqual(
            classify_positive_contract(
                eval_circle_hit=True,
                realized_obb_hit=True,
                realized_center_current_obb_hit=True,
                cv_center_realized_shape_hit=False,
                cv_center_current_obb_hit=False,
            ),
            "REALIZED_CENTER_DYNAMICS_SUFFICIENT",
        )

    def test_false_contract_keeps_attribution_separate(self) -> None:
        self.assertEqual(
            classify_false_contract(
                source_is_target=False,
                eval_circle_hit=True,
                realized_obb_hit=True,
            ),
            "ATTRIBUTED_OTHER_NATIVE_COMPONENT",
        )


if __name__ == "__main__":
    unittest.main()
