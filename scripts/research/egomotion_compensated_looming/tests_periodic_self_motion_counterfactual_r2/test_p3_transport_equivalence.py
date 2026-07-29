from __future__ import annotations

import unittest

import cv2
import numpy as np

from ..periodic_self_motion_counterfactual_r2 import generator_geometry as geometry
from ..periodic_self_motion_counterfactual_r2 import p3_transport_r0 as transport


class P3TransportEquivalenceTests(unittest.TestCase):
    def test_world_from_camera_homography_order_and_native_k(self) -> None:
        angle = np.deg2rad(7.0)
        previous = np.eye(3)
        current = np.array(
            (
                (np.cos(angle), 0.0, np.sin(angle)),
                (0.0, 1.0, 0.0),
                (-np.sin(angle), 0.0, np.cos(angle)),
            ),
            dtype=np.float64,
        )
        expected = geometry.K @ (current.T @ previous) @ np.linalg.inv(geometry.K)
        actual = transport.rotation_homography(previous, current, geometry.K)
        np.testing.assert_array_equal(actual, expected)
        reversed_order = geometry.K @ (previous.T @ current) @ np.linalg.inv(geometry.K)
        self.assertFalse(np.array_equal(actual, reversed_order))
        advio_k = np.array(
            ((1082.4, 0, 364.6778), (0, 1084.4, 643.308), (0, 0, 1)),
            dtype=np.float64,
        )
        self.assertFalse(
            np.array_equal(
                actual,
                advio_k @ (current.T @ previous) @ np.linalg.inv(advio_k),
            )
        )

    def test_rgb_not_bgr_and_valid_mask_is_not_filled(self) -> None:
        rgb = np.array([[[255, 0, 31], [0, 255, 7]]], dtype=np.uint8)
        actual = transport.rgb_to_gray(rgb)
        expected = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bgr_mutation = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(actual, bgr_mutation))
        mask = np.array([[True, False]])
        np.testing.assert_array_equal(
            transport.valid_mask(mask, (1, 2)),
            np.array([[255, 0]], dtype=np.uint8),
        )

    def test_invalid_transport_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(transport.InvalidTransport, "RGB"):
            transport.rgb_to_gray(np.zeros((2, 2, 3), dtype=np.float32))
        with self.assertRaisesRegex(transport.InvalidTransport, "VALID_MASK"):
            transport.valid_mask(np.ones((2, 2), dtype=np.uint8), (2, 2))
        with self.assertRaisesRegex(transport.InvalidTransport, "ROTATION"):
            transport.rotation_homography(np.eye(2), np.eye(3), np.eye(3))

    def test_file_reference_and_adapter_pair_core_are_exact(self) -> None:
        result = transport.run_equivalence()
        self.assertEqual(
            result["terminal"],
            "TRANSPORT_EQUIVALENCE_PASS / VALID / PREFLIGHT_ONLY",
        )
        self.assertTrue(result["fixture"]["rows_equal"])
        self.assertTrue(result["fixture"]["state_equal"])
        self.assertTrue(result["fixture"]["partial_valid_mask"])
        self.assertFalse(result["formal_execution_authorized"])
        self.assertFalse(result["p4_activated"])


if __name__ == "__main__":
    unittest.main()
