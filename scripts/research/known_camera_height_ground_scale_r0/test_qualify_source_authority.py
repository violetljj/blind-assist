import unittest

import numpy as np

from qualify_source_authority import fit_gravity_plane_height_proxy


class QualifySourceAuthorityTest(unittest.TestCase):
    def test_recovers_gravity_constrained_height_proxy(self) -> None:
        height, width = 120, 160
        intrinsics = np.asarray(
            [[120.0, 0.0, 79.5], [0.0, 120.0, 59.5], [0.0, 0.0, 1.0]]
        )
        rows = np.arange(height, dtype=np.float64)[:, None]
        denominator = (rows - intrinsics[1, 2]) / intrinsics[1, 1]
        depth = np.zeros((height, width), dtype=np.float64)
        valid = denominator[:, 0] > 0.0
        depth[valid, :] = 1.2 / denominator[valid]
        confidence = np.full((height, width), 2, dtype=np.uint8)
        result = fit_gravity_plane_height_proxy(
            depth, confidence, intrinsics, np.asarray([0.0, -1.0, 0.0])
        )
        self.assertEqual("VALID", result["status"])
        self.assertAlmostEqual(1.2, result["height_proxy_m"], places=6)

    def test_does_not_flip_wrong_gravity_sign(self) -> None:
        height, width = 120, 160
        intrinsics = np.asarray(
            [[120.0, 0.0, 79.5], [0.0, 120.0, 59.5], [0.0, 0.0, 1.0]]
        )
        depth = np.full((height, width), 2.0)
        confidence = np.full((height, width), 2, dtype=np.uint8)
        result = fit_gravity_plane_height_proxy(
            depth, confidence, intrinsics, np.asarray([0.0, 1.0, 0.0])
        )
        self.assertEqual("UNKNOWN", result["status"])


if __name__ == "__main__":
    unittest.main()
