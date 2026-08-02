import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d5_tartanground_development_pilot import (
    HEIGHT_BANDS_M,
    anchor_basis,
    decode_depth,
    depth_points_world,
    known_field,
    member_frame_id,
    reprojection_consistency,
)


class TartanGroundDevelopmentPilotTest(unittest.TestCase):
    def test_depth_round_trip_and_ned_projection(self):
        depth = np.full((640, 640), 2.0, dtype="<f4")
        encoded_input = depth.view(np.uint8).reshape(640, 640, 4)
        ok, encoded = cv2.imencode(".png", encoded_input)
        self.assertTrue(ok)

        decoded = decode_depth(encoded.tobytes())
        points = depth_points_world(
            decoded,
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )

        self.assertEqual(decoded.shape, (640, 640))
        self.assertTrue(np.allclose(decoded, 2.0))
        self.assertTrue(np.allclose(points[0], 2.0))

    def test_anchor_basis_places_ground_below_camera(self):
        basis = anchor_basis(
            np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]),
            1.2,
        )
        origin, forward, right, up = basis

        self.assertTrue(np.allclose(origin, [1.0, 2.0, 4.2]))
        self.assertTrue(np.allclose(forward, [1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(right, [0.0, 1.0, 0.0]))
        self.assertTrue(np.allclose(up, [0.0, 0.0, -1.0]))

    def test_known_field_accepts_visible_probes(self):
        pose = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        probes = np.zeros((6 * 6 * len(HEIGHT_BANDS_M), 3, 9))
        probes[:, 0, :] = 1.0
        depth = np.full((640, 640), 2.0, dtype=np.float64)

        known = known_field(probes, depth, pose)

        self.assertEqual(known.shape, (6, 6, 3))
        self.assertTrue(np.all(known))

    def test_identical_pose_depth_reprojects_exactly(self):
        pose = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        depth = np.full((640, 640), 2.0, dtype=np.float64)

        result = reprojection_consistency(depth, pose, depth, pose)

        self.assertGreater(result["comparable_points"], 0)
        self.assertAlmostEqual(result["median_relative_depth_error"], 0.0)
        self.assertAlmostEqual(result["fraction_within_5_percent"], 1.0)

    def test_member_frame_id(self):
        self.assertEqual(
            member_frame_id("depth_lcam_front/000304_lcam_front_depth.png"),
            304,
        )


if __name__ == "__main__":
    unittest.main()
