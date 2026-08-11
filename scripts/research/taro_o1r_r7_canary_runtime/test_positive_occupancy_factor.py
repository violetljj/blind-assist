from __future__ import annotations

import copy
import inspect
import unittest

from scripts.research.taro_o1r_r7_canary_runtime import positive_occupancy_factor as factor
from scripts.research.taro_o1r_r7_canary_runtime.test_r7_canary import _feature, _unavailable_source


class PositiveOccupancyFactorTests(unittest.TestCase):
    def test_public_builder_has_no_truth_side(self) -> None:
        names = inspect.signature(factor.build_positive_occupancy_factor).parameters
        self.assertFalse(any(token in name.lower() for name in names for token in ("faro", "truth", "label", "outcome")))

    def test_frozen_positive_grid_emits_occupied(self) -> None:
        feature = _feature()
        feature["occupied_hits"][0][0][2] = True
        self.assertEqual(factor.state_from_feature(feature)[0], "OCCUPIED_OBSERVED")

    def test_absence_and_prior_clear_map_to_unknown(self) -> None:
        feature = _feature()
        self.assertEqual(factor.state_from_feature(feature)[0], "UNKNOWN")
        feature["r6_state"] = "CLEAR_OBSERVED"
        self.assertEqual(factor.state_from_feature(feature)[0], "UNKNOWN")

    def test_prior_occupied_is_preserved(self) -> None:
        feature = _feature()
        feature["query_receipt"] = None
        feature["r6_state"] = "OCCUPIED_OBSERVED"
        self.assertEqual(factor.state_from_feature(feature)[0], "OCCUPIED_OBSERVED")

    def test_unavailable_source_retains_nine_unknowns_and_no_clear(self) -> None:
        bundle = factor.build_positive_occupancy_factor(_unavailable_source())
        self.assertEqual(bundle["state_counts"], {"CLEAR_OBSERVED": 0, "OCCUPIED_OBSERVED": 0, "UNKNOWN": 9})
        self.assertFalse(bundle["unknown_is_negative"])

    def test_tamper_to_clear_is_rejected(self) -> None:
        bundle = factor.build_positive_occupancy_factor(_unavailable_source())
        tampered = copy.deepcopy(bundle)
        tampered["query_results"][0]["state"] = "CLEAR_OBSERVED"
        with self.assertRaises(factor.PositiveOccupancyFactorError):
            factor.validate_positive_occupancy_factor(tampered)


if __name__ == "__main__":
    unittest.main()
