import json
import unittest

import numpy as np
from run_offline_stress_r0 import DEFAULT_PROTOCOL
from run_rgb_offline_stress_r0 import (
    array_sha256,
    build_rgb_scenarios,
    crop_with_intrinsics,
    inverse_roll_depth,
    postprocess_depth,
    roll_image,
    select_parent_balanced,
    transform_rgb,
)


class RGBOfflineStressProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))

    def test_protocol_routes_cached_and_rgb_to_separate_runners(self) -> None:
        layer = self.protocol["rgb_second_layer"]
        self.assertFalse(layer["cached_runner_implemented"])
        self.assertTrue(layer["rgb_runner_implemented"])
        self.assertEqual(50, layer["subset"]["record_count"])
        self.assertEqual(5, layer["subset"]["records_per_parent"])
        self.assertIn("rerun the exact frozen DA checkpoint", layer["required_operation"])

    def test_fixed_scenarios_are_complete_and_truth_scoped(self) -> None:
        scenarios = build_rgb_scenarios(self.protocol)
        self.assertEqual(25, len(scenarios))
        self.assertEqual(25, len({row["id"] for row in scenarios}))
        families = {row["family"] for row in scenarios}
        self.assertTrue(
            {
                "rgb_clean",
                "rgb_center_crop",
                "rgb_roll",
                "rgb_pitch_homography_canary",
                "rgb_gaussian_blur",
                "rgb_motion_blur",
                "rgb_gamma_darkening",
                "rgb_exposure",
            }.issubset(families)
        )
        comparable = {row["family"] for row in scenarios if row["truth_comparable"]}
        self.assertEqual(
            {
                "rgb_clean",
                "rgb_gaussian_blur",
                "rgb_motion_blur",
                "rgb_gamma_darkening",
                "rgb_exposure",
            },
            comparable,
        )
        self.assertTrue(
            all(
                not row["truth_comparable"]
                for row in scenarios
                if row["family"] in {"rgb_center_crop", "rgb_roll", "rgb_pitch_homography_canary"}
            )
        )

    def test_parent_balanced_selection_ignores_outcome_fields(self) -> None:
        records = []
        for parent in ("b", "a"):
            for anchor in range(10):
                records.append(
                    {
                        "parent_id": parent,
                        "anchor_frame_id": anchor,
                        "truth": {"outcome_must_not_select": 1000 - anchor},
                    }
                )
        selected = select_parent_balanced(records, 5)
        self.assertEqual(10, len(selected))
        by_parent = {
            parent: [row["anchor_frame_id"] for row in selected if row["parent_id"] == parent]
            for parent in ("a", "b")
        }
        self.assertEqual([0, 2, 4, 7, 9], by_parent["a"])
        self.assertEqual([0, 2, 4, 7, 9], by_parent["b"])
        mutated = [dict(row, truth={"changed": -999}) for row in records]
        self.assertEqual(
            [(row["parent_id"], row["anchor_frame_id"]) for row in selected],
            [
                (row["parent_id"], row["anchor_frame_id"])
                for row in select_parent_balanced(mutated, 5)
            ],
        )


class RGBOfflineStressTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.mgrid[0:80, 0:100]
        self.bgr = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
        self.intrinsics = np.asarray(
            [[70.0, 0.0, 49.5], [0.0, 72.0, 39.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def test_crop_updates_intrinsics_exactly_and_mismatch_keeps_old(self) -> None:
        transformed, corrected = crop_with_intrinsics(self.bgr, self.intrinsics, 0.8)
        self.assertEqual(self.bgr.shape, transformed.shape)
        self.assertAlmostEqual(87.5, corrected[0, 0])
        self.assertAlmostEqual(90.0, corrected[1, 1])
        self.assertAlmostEqual((49.5 - 10) * 1.25, corrected[0, 2])
        self.assertAlmostEqual((39.5 - 8) * 1.25, corrected[1, 2])

        updated, updated_k, _ = transform_rgb(
            self.bgr,
            self.intrinsics,
            {
                "family": "rgb_center_crop",
                "retained_fraction": 0.8,
                "coordinate_mode": "updated_intrinsics",
            },
        )
        stale, stale_k, _ = transform_rgb(
            self.bgr,
            self.intrinsics,
            {
                "family": "rgb_center_crop",
                "retained_fraction": 0.8,
                "coordinate_mode": "stale_intrinsics_mismatch",
            },
        )
        np.testing.assert_array_equal(updated, stale)
        np.testing.assert_array_equal(updated_k, corrected)
        np.testing.assert_array_equal(stale_k, self.intrinsics)

    def test_roll_corrected_coordinate_is_separate_from_mismatch(self) -> None:
        transformed, affine = roll_image(self.bgr, 5.0)
        self.assertEqual(self.bgr.shape, transformed.shape)
        depth = np.arange(80 * 100, dtype=np.float64).reshape(80, 100)
        corrected = postprocess_depth(
            depth,
            {
                "family": "rgb_roll",
                "coordinate_mode": "inverse_warp_depth_to_original_coordinates",
            },
            {"roll_affine": affine},
        )
        mismatch = postprocess_depth(
            depth,
            {
                "family": "rgb_roll",
                "coordinate_mode": "uncompensated_coordinate_mismatch",
            },
            {"roll_affine": affine},
        )
        self.assertEqual(depth.shape, corrected.shape)
        self.assertFalse(np.array_equal(corrected, mismatch, equal_nan=True))
        np.testing.assert_array_equal(mismatch, depth)
        np.testing.assert_array_equal(corrected, inverse_roll_depth(depth, affine))

    def test_blur_and_low_light_are_real_rgb_transforms_and_deterministic(self) -> None:
        scenarios = (
            {"family": "rgb_gaussian_blur", "sigma": 1.5},
            {"family": "rgb_motion_blur", "length": 9},
            {"family": "rgb_gamma_darkening", "gamma": 1.8},
            {"family": "rgb_exposure", "multiplier": 0.5},
        )
        for scenario in scenarios:
            first, first_k, _ = transform_rgb(self.bgr, self.intrinsics, scenario)
            second, second_k, _ = transform_rgb(self.bgr, self.intrinsics, scenario)
            self.assertEqual(self.bgr.shape, first.shape)
            self.assertEqual(np.uint8, first.dtype)
            self.assertNotEqual(array_sha256(self.bgr), array_sha256(first))
            self.assertEqual(array_sha256(first), array_sha256(second))
            np.testing.assert_array_equal(first_k, self.intrinsics)
            np.testing.assert_array_equal(first_k, second_k)

    def test_pitch_is_explicitly_only_a_non_identity_canary(self) -> None:
        transformed, matrix, _ = transform_rgb(
            self.bgr,
            self.intrinsics,
            {"family": "rgb_pitch_homography_canary", "degrees": 5.0},
        )
        self.assertEqual(self.bgr.shape, transformed.shape)
        self.assertNotEqual(array_sha256(self.bgr), array_sha256(transformed))
        np.testing.assert_array_equal(matrix, self.intrinsics)


if __name__ == "__main__":
    unittest.main()
