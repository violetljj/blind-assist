import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("audit_tartanair_temporal_reprojection.py")
SPEC = importlib.util.spec_from_file_location("temporal_reprojection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TemporalReprojectionTest(unittest.TestCase):
    def test_source_native_identity_pair_has_zero_residual_when_cuda_available(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required by source-native audit")
        depth = np.full((4, 6), 3.0, dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        intrinsics = np.array([[4.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        result = MODULE._pair_metrics(depth, depth, pose, pose, intrinsics, torch)
        self.assertEqual(24, result["valid_projection_count"])
        self.assertAlmostEqual(0.0, result["median_abs_depth_residual_m"], places=6)
        self.assertAlmostEqual(0.0, result["p95_abs_depth_residual_m"], places=6)
        self.assertAlmostEqual(0.0, result["relative_outlier_fraction_gt_10pct"], places=6)


if __name__ == "__main__":
    unittest.main()
