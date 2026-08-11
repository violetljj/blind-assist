import unittest

from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d2_train_only import (
    checkpoint_parameter_count,
)


class D2TrainOnlyValidatorTest(unittest.TestCase):
    def test_parameter_count(self):
        checkpoint = {"state_dict": {
            "layers.0.weight": [[0.0] * 11 for _ in range(16)],
            "layers.0.bias": [0.0] * 16,
            "layers.2.weight": [[0.0] * 16 for _ in range(5)],
            "layers.2.bias": [0.0] * 5,
        }}
        self.assertEqual(277, checkpoint_parameter_count(checkpoint))


if __name__ == "__main__":
    unittest.main()
