import unittest

from evaluate_frozen_motion_occupancy_bonn_cross_dataset import admission


class FrozenMotionOccupancyBonnCrossDatasetTest(unittest.TestCase):
    def test_admission_requires_source_and_pooled_coverage(self) -> None:
        reports = [{"paired_valid_fraction": 0.95}, {"paired_valid_fraction": 0.91}]
        per_source = [{"opportunities": 1000}, {"opportunities": 950}]
        pooled = {"gates": {"a": True, "b": True}}
        self.assertTrue(all(admission(reports, per_source, pooled).values()))

    def test_admission_fails_low_source_coverage(self) -> None:
        reports = [{"paired_valid_fraction": 0.95}, {"paired_valid_fraction": 0.89}]
        per_source = [{"opportunities": 1000}, {"opportunities": 950}]
        pooled = {"gates": {"a": True}}
        self.assertFalse(all(admission(reports, per_source, pooled).values()))


if __name__ == "__main__":
    unittest.main()
