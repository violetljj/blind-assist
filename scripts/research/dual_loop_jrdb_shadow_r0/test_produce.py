import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("produce.py")
SPEC = importlib.util.spec_from_file_location("dual_loop_jrdb_shadow_produce", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PRODUCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRODUCE)


class ProduceTest(unittest.TestCase):
    def test_signed_approach_rate_is_positive_when_range_decreases(self) -> None:
        previous = {"sensor_centroid_logical_rgb360_m": [3.0, 4.0, 0.0]}
        current = {"sensor_centroid_logical_rgb360_m": [0.0, 4.0, 0.0]}

        rate = PRODUCE.signed_approach_rate(previous, current, 0, 1_000_000_000)

        self.assertAlmostEqual(1.0, rate)

    def test_signed_approach_rate_rejects_non_monotonic_time(self) -> None:
        row = {"sensor_centroid_logical_rgb360_m": [1.0, 0.0, 0.0]}
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PRODUCE.signed_approach_rate(row, row, 5, 5)

    def test_centroid_range_rejects_nonfinite_value(self) -> None:
        row = {"sensor_centroid_logical_rgb360_m": [math.nan, 0.0, 0.0]}
        with self.assertRaisesRegex(ValueError, "not finite"):
            PRODUCE.centroid_range(row)


if __name__ == "__main__":
    unittest.main()
