#!/usr/bin/env python3

import math
import unittest

from scripts.research.hftf.evaluate_dav2_model_variant_gate_r2 import (
    finite_number,
    json_safe,
    safe_ge,
    safe_le,
)


class FailClosedComparisonTest(unittest.TestCase):
    def test_finite_comparisons_preserve_direction(self) -> None:
        self.assertTrue(safe_le(0.1, 0.2))
        self.assertFalse(safe_le(0.3, 0.2))
        self.assertTrue(safe_ge(0.3, 0.2))
        self.assertFalse(safe_ge(0.1, 0.2))

    def test_undefined_and_nonfinite_fail(self) -> None:
        for value in (None, math.inf, -math.inf, math.nan, True):
            self.assertFalse(finite_number(value))
            self.assertFalse(safe_le(value, 1.0))
            self.assertFalse(safe_ge(value, 0.0))

    def test_json_safe_replaces_nonfinite(self) -> None:
        self.assertEqual(json_safe({"x": math.inf, "y": [1.0, math.nan]}), {"x": None, "y": [1.0, None]})


if __name__ == "__main__":
    unittest.main()
