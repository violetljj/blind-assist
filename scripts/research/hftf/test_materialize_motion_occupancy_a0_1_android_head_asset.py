import gzip
import unittest

import numpy as np

from evaluate_motion_conditioned_occupancy_a0 import FEATURE_NAMES
from materialize_motion_occupancy_a0_1_android_head_asset import serialize_rows


class MaterializeMotionOccupancyAndroidHeadAssetTest(unittest.TestCase):
    def test_serializes_exact_feature_order_and_probability(self) -> None:
        model = {
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": [0.0] * len(FEATURE_NAMES),
            "feature_scale": [1.0] * len(FEATURE_NAMES),
            "weights_intercept_then_features": [0.0] + [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
        }
        features = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
        features[1, 0] = 1.0
        raw = serialize_rows(features, np.asarray(["a", "b"]), model)
        lines = raw.decode("utf-8").splitlines()
        self.assertEqual(lines[0].split("\t")[1:-1], list(FEATURE_NAMES))
        self.assertEqual(float(lines[1].split("\t")[-1]), 0.5)
        self.assertGreater(float(lines[2].split("\t")[-1]), 0.5)
        self.assertEqual(gzip.decompress(gzip.compress(raw, mtime=0)), raw)


if __name__ == "__main__":
    unittest.main()
