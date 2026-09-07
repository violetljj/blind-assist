"""Bounded analytic support controls; no replay cohort or model execution."""
import unittest

import numpy as np
import torch

from near_field import Camera, NearFieldEncoder
from probe_source import CAMERA, generate_cases


class SurfaceSupportTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.camera = Camera(40, 30, 90., 1.6)
        self.encoder = NearFieldEncoder(self.camera)
        self.plane = (0., 0., -1.6)

    def sparse_triple(self, *, horizontal_surface):
        depth = np.full((30, 40), np.nan, dtype=np.float32)
        # Three downward rays hit the same 0.9 m elevation at distinct depths.
        # A single column prevents horizontal depth triples from rescuing it.
        if horizontal_surface:
            depth[20:23, 20] = -.7 / self.encoder.ray_z[20:23, 20].numpy()
        else:
            depth[20:23, 20] = 2.
        return depth

    def test_default_matches_explicit_depth_for_all_evidence_fields(self):
        depth = self.sparse_triple(horizontal_surface=False)
        for plane in (None, self.plane):
            with self.subTest(plane=plane):
                default = self.encoder.encode(depth, ground_plane=plane)
                explicit = self.encoder.encode(depth, ground_plane=plane, support_mode="depth")
                for field in ("distance_m", "pixel_index", "support_count", "weak_count",
                              "valid_fraction", "region_distance_m"):
                    np.testing.assert_array_equal(getattr(default, field), getattr(explicit, field))
                self.assertEqual(default.summary(), explicit.summary())

    def test_horizontal_surface_on_sloped_rays_gains_support(self):
        depth = self.sparse_triple(horizontal_surface=True)
        before = depth.copy()
        baseline = self.encoder.encode(depth, ground_plane=self.plane, support_mode="depth")
        self.assertFalse(baseline.alerts().any())
        self.assertEqual(int(baseline.weak_count.sum()), 3)
        for mode in ("elevation", "dual"):
            with self.subTest(mode=mode):
                evidence = self.encoder.encode(depth, ground_plane=self.plane, support_mode=mode)
                self.assertTrue(evidence.alerts()[1, 1])
                self.assertEqual(int(evidence.support_count.sum()), 3)
                self.assertEqual(int(evidence.weak_count.sum()), 0)
        np.testing.assert_array_equal(depth, before)

    def test_dual_preserves_vertical_depth_support(self):
        depth = self.sparse_triple(horizontal_surface=False)
        baseline = self.encoder.encode(depth, ground_plane=self.plane, support_mode="depth")
        elevation = self.encoder.encode(depth, ground_plane=self.plane, support_mode="elevation")
        dual = self.encoder.encode(depth, ground_plane=self.plane, support_mode="dual")
        self.assertTrue(baseline.alerts()[1, 1])
        self.assertFalse(elevation.alerts().any())
        np.testing.assert_array_equal(dual.support_count, baseline.support_count)
        np.testing.assert_array_equal(dual.region_distance_m, baseline.region_distance_m)

    def test_coherent_artifact_remains_false_alert_negative_control(self):
        case = next(case for case in generate_cases()
                    if case["name"] == "one_frame_depth_artifact_3x3")
        self.assertFalse(case["expected"].any())  # Provenance remains negative truth.
        encoder = NearFieldEncoder(Camera(**CAMERA))
        plane = (0., 0., -CAMERA["camera_height_m"])
        for mode in ("depth", "elevation", "dual"):
            with self.subTest(mode=mode):
                evidence = encoder.encode(case["depth"], ground_plane=plane, support_mode=mode)
                self.assertTrue(evidence.alerts().any())
                self.assertTrue((evidence.alerts() & ~case["expected"]).any())

    def test_surface_support_does_not_rescue_invalid_or_ground_pixels(self):
        for mode in ("depth", "elevation", "dual"):
            for middle in (np.nan, 0.):
                with self.subTest(mode=mode, middle=middle):
                    depth = self.sparse_triple(horizontal_surface=True)
                    depth[21, 20] = middle
                    evidence = self.encoder.encode(depth, ground_plane=self.plane, support_mode=mode)
                    self.assertEqual(int(evidence.support_count.sum()), 0)
            depth = np.full((30, 40), np.nan, dtype=np.float32)
            depth[25:28, 20] = -1.6 / self.encoder.ray_z[25:28, 20].numpy()
            evidence = self.encoder.encode(depth, ground_plane=self.plane, support_mode=mode)
            self.assertFalse(evidence.alerts().any())
            self.assertEqual(int(evidence.support_count.sum()), 0)
            self.assertEqual(int(evidence.weak_count.sum()), 0)

    def test_invalid_mode_and_missing_ground_rejected(self):
        depth = self.sparse_triple(horizontal_surface=True)
        for mode in ("", "unknown", "Elevation"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.encoder.encode(depth, ground_plane=self.plane, support_mode=mode)
        for mode in ("elevation", "dual"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.encoder.encode(depth, support_mode=mode)


if __name__ == "__main__":
    unittest.main()
