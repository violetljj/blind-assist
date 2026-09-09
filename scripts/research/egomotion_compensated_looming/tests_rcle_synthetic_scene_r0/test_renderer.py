"""Tests for the isolated RCLE synthetic-scene renderer."""

import json
from pathlib import Path
import unittest

import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.renderer import (
    SURFACE_PANEL_BASE,
    camera_intrinsics,
    render_frame,
)


MODULE = Path(__file__).resolve().parents[1] / "rcle_synthetic_scene_r0"


def motion(spec, motion_id):
    return next(item for item in spec["motions"] if item["motion_id"] == motion_id)


class RendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((MODULE / "dataset_spec.json").read_text(encoding="utf-8"))
        cls.scene = cls.spec["splits"]["development"]["scene_families"][0]

    def test_intrinsics_are_frozen_float64(self):
        actual = camera_intrinsics(self.spec)
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(
            actual,
            np.array(
                [[280.0, 0.0, 159.5], [0.0, -280.0, 119.5], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        )

    def test_rendered_frame_contract_and_geometry(self):
        frame = render_frame(
            self.spec, self.scene, motion(self.spec, "below_static"), 0
        )
        self.assertEqual(frame.rgb.shape, (240, 320, 3))
        self.assertEqual(frame.rgb.dtype, np.uint8)
        self.assertEqual(frame.depth_m.shape, (240, 320))
        self.assertEqual(frame.depth_m.dtype, np.float64)
        self.assertEqual(frame.surface_id.shape, (240, 320))
        self.assertEqual(frame.surface_id.dtype, np.int16)
        self.assertTrue(np.all(np.isfinite(frame.depth_m)))
        self.assertTrue(np.all(frame.depth_m > 0.0))
        self.assertTrue(np.all(frame.surface_id > 0))
        self.assertEqual(frame.t_world_camera.shape, (4, 4))
        self.assertEqual(frame.t_camera_world.shape, (4, 4))
        np.testing.assert_allclose(
            frame.t_world_camera @ frame.t_camera_world,
            np.eye(4),
            atol=1e-14,
            rtol=0.0,
        )
        np.testing.assert_array_equal(frame.t_world_camera[:3, 3], [0.0, 1.5, 1.0])

    def test_center_hits_nearest_frontoparallel_panel_at_camera_z_depth(self):
        frame = render_frame(
            self.spec, self.scene, motion(self.spec, "below_static"), 0
        )
        # Principal point lies between four pixels, all within the central panel.
        center_ids = frame.surface_id[119:121, 159:161]
        center_depth = frame.depth_m[119:121, 159:161]
        np.testing.assert_array_equal(
            center_ids, np.full((2, 2), SURFACE_PANEL_BASE, np.int16)
        )
        np.testing.assert_allclose(center_depth, 4.5, atol=1e-14, rtol=0.0)

    def test_repeat_render_is_byte_and_float_deterministic(self):
        selected_motion = motion(self.spec, "positive_rotation_approach")
        first = render_frame(self.spec, self.scene, selected_motion, 37)
        second = render_frame(self.spec, self.scene, selected_motion, 37)
        np.testing.assert_array_equal(first.rgb, second.rgb)
        np.testing.assert_array_equal(first.depth_m, second.depth_m)
        np.testing.assert_array_equal(first.surface_id, second.surface_id)
        np.testing.assert_array_equal(first.t_world_camera, second.t_world_camera)
        np.testing.assert_array_equal(first.t_camera_world, second.t_camera_world)

    def test_translation_and_yaw_are_encoded_in_pose(self):
        frame = render_frame(
            self.spec,
            self.scene,
            motion(self.spec, "positive_rotation_approach"),
            25,
        )
        expected_position = np.array([0.0, 1.5, 1.0 + 0.35 * 2.5])
        np.testing.assert_allclose(
            frame.t_world_camera[:3, 3], expected_position, atol=1e-15, rtol=0.0
        )
        expected_yaw = np.deg2rad(15.0)
        expected_rotation = np.array(
            [
                [np.cos(expected_yaw), 0.0, np.sin(expected_yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(expected_yaw), 0.0, np.cos(expected_yaw)],
            ]
        )
        np.testing.assert_allclose(
            frame.t_world_camera[:3, :3], expected_rotation, atol=1e-15, rtol=0.0
        )
        self.assertAlmostEqual(
            float(np.linalg.det(frame.t_world_camera[:3, :3])), 1.0, delta=1e-15
        )

    def test_motion_and_scene_seed_have_bounded_effects(self):
        static = render_frame(
            self.spec, self.scene, motion(self.spec, "below_static"), 20
        )
        yawed = render_frame(
            self.spec, self.scene, motion(self.spec, "below_pure_yaw"), 20
        )
        self.assertFalse(np.array_equal(static.depth_m, yawed.depth_m))
        self.assertFalse(np.array_equal(static.rgb, yawed.rgb))

        retextured_scene = dict(self.scene, seed=self.scene["seed"] + 1)
        retextured = render_frame(
            self.spec, retextured_scene, motion(self.spec, "below_static"), 20
        )
        np.testing.assert_array_equal(static.depth_m, retextured.depth_m)
        np.testing.assert_array_equal(static.surface_id, retextured.surface_id)
        self.assertFalse(np.array_equal(static.rgb, retextured.rgb))

    def test_invalid_frame_or_panel_variant_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            render_frame(
                self.spec, self.scene, motion(self.spec, "below_static"), -1
            )
        with self.assertRaisesRegex(TypeError, "integer"):
            render_frame(
                self.spec, self.scene, motion(self.spec, "below_static"), 1.5
            )
        with self.assertRaisesRegex(ValueError, "unsupported panel_variant"):
            render_frame(
                self.spec,
                dict(self.scene, panel_variant=999),
                motion(self.spec, "below_static"),
                0,
            )


if __name__ == "__main__":
    unittest.main()
