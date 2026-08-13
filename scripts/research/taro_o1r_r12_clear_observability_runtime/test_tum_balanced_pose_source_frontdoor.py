import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as subject


class TumBalancedPoseSourceFrontdoorTest(unittest.TestCase):
    def test_depth_decode_uses_frozen_tum_scale(self) -> None:
        from io import BytesIO
        from PIL import Image

        raw = np.full((480, 640), 5000, dtype=np.uint16)
        buffer = BytesIO()
        Image.fromarray(raw).save(buffer, format="PNG")
        depth = subject._decode_depth(buffer.getvalue())
        self.assertAlmostEqual(1.0, float(depth[0, 0]))

    def test_low_observation_never_uses_invalid_depth_as_support(self) -> None:
        depth = np.zeros((480, 640), dtype=np.float64)
        intrinsics = np.asarray([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]])
        _low, _points, valid = subject._low_observation(depth, intrinsics)
        self.assertFalse(np.any(valid))


if __name__ == "__main__":
    unittest.main()
