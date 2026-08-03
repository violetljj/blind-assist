import unittest

from benchmark_motion_occupancy_a0_runtime import distribution


class BenchmarkMotionOccupancyA0RuntimeTest(unittest.TestCase):
    def test_distribution(self) -> None:
        result = distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["mean_ms"], 2.5)
        self.assertEqual(result["median_ms"], 2.5)
        self.assertEqual(result["maximum_ms"], 4.0)


if __name__ == "__main__":
    unittest.main()
