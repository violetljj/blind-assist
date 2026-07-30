import importlib.util
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).with_name("produce.py")
SPEC = importlib.util.spec_from_file_location("dual_loop_gmc_produce", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProduceTest(unittest.TestCase):
    def test_inside_roi_accepts_center_and_rejects_outside(self) -> None:
        points = np.asarray([[50.0, 50.0], [90.0, 90.0]])
        result = MODULE.inside_roi(points, (50.0, 50.0, 20.0, 20.0), (100, 100), 0.0)
        self.assertEqual([True, False], result.tolist())

    def test_rectangle_mask_clamps_to_image(self) -> None:
        mask = MODULE.rectangle_mask((10, 10), (1.0, 1.0, 8.0, 8.0))
        self.assertGreater(int(mask.sum()), 0)
        self.assertEqual((10, 10), mask.shape)


if __name__ == "__main__":
    unittest.main()
