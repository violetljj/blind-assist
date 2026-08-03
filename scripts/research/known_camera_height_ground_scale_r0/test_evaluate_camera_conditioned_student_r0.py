import unittest

import numpy as np

from evaluate_camera_conditioned_student_r0 import fit_ridge, predict_ridge


class CameraConditionedStudentR0Test(unittest.TestCase):
    def test_ridge_recovers_affine_training_relation(self):
        first = np.linspace(-2.0, 2.0, 15)
        x = np.column_stack((first, first**2))
        y = 0.4 + 0.2 * x[:, 0] - 0.1 * x[:, 1]
        model = fit_ridge(x, y, alpha=0.0)
        predicted = np.asarray([predict_ridge(model, row) for row in x])
        np.testing.assert_allclose(predicted, y, atol=1e-8)

    def test_ridge_rejects_too_few_rows(self):
        with self.assertRaises(ValueError):
            fit_ridge(np.ones((3, 4)), np.ones(3))


if __name__ == "__main__":
    unittest.main()
