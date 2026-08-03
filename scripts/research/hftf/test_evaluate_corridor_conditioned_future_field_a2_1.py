import unittest

from evaluate_corridor_conditioned_future_field_a2_1 import corridor_extra_features


class CorridorConditionedFutureFieldA21Test(unittest.TestCase):
    def test_corridor_extra_features(self) -> None:
        values = corridor_extra_features(1.5, 1.5)
        self.assertEqual(values.tolist()[:2], [1.5, 0.0])
        self.assertAlmostEqual(values[2], 0.5)


if __name__ == "__main__":
    unittest.main()
