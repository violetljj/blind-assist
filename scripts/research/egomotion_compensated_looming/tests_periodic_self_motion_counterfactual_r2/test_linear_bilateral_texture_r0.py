from __future__ import annotations

import hashlib
from unittest import mock
import unittest

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    linear_bilateral_texture_r0 as operator,
)


def _fixture(height: int = 96, width: int = 128) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    base = np.where(x < width // 2, 64, 192).astype(np.int16)
    checker = np.where((x + y) % 2 == 0, -4, 4).astype(np.int16)
    gray = np.clip(base + checker, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _transition_width(profile: np.ndarray) -> int:
    low = float(np.mean(profile[: profile.size // 4]))
    high = float(np.mean(profile[-profile.size // 4 :]))
    normalized = (profile - low) / (high - low)
    ten = int(np.flatnonzero(normalized >= 0.10)[0])
    ninety = int(np.flatnonzero(normalized >= 0.90)[0])
    return ninety - ten


class LinearBilateralTextureR0Test(unittest.TestCase):
    def test_identity_declares_nonlinear_filter_and_not_psf_none(self) -> None:
        identity = operator.frozen_operator_identity()
        self.assertEqual(
            identity["operator_class"],
            "NONLINEAR_EDGE_PRESERVING_SPATIAL_FILTER",
        )
        self.assertEqual(identity["color_space"], "LINEAR_RGB_FLOAT32")
        self.assertIs(identity["psf_none"], False)
        operator.validate_operator_identity(identity)

    def test_repeat_and_thread_count_sha_are_identical(self) -> None:
        rgb = _fixture(160, 192)
        original_threads = cv2.getNumThreads()
        try:
            hashes: list[str] = []
            for threads in (1, 1, 18, 18):
                cv2.setNumThreads(threads)
                hashes.append(
                    _sha256(operator.apply_linear_bilateral_texture(rgb))
                )
        finally:
            cv2.setNumThreads(original_threads)
        self.assertEqual(len(set(hashes)), 1)

    def test_rgb_and_geometry_inputs_are_not_mutated(self) -> None:
        rgb = _fixture()
        rgb_before = rgb.copy()
        valid_mask = np.ones(rgb.shape[:2], dtype=bool)
        object_id = np.where(
            np.indices(rgb.shape[:2])[1] < rgb.shape[1] // 2,
            11,
            12,
        ).astype(np.int32)
        valid_before = valid_mask.copy()
        object_before = object_id.copy()

        degraded = operator.apply_linear_bilateral_texture(rgb)

        np.testing.assert_array_equal(rgb, rgb_before)
        np.testing.assert_array_equal(valid_mask, valid_before)
        np.testing.assert_array_equal(object_id, object_before)
        self.assertEqual(degraded.dtype, np.uint8)
        self.assertEqual(degraded.shape, rgb.shape)
        self.assertFalse(np.shares_memory(degraded, rgb))

    def test_analytic_edge_fixture_preserves_step_spread(self) -> None:
        rgb = _fixture()
        degraded = operator.apply_linear_bilateral_texture(rgb)
        clean_profile = rgb[:, :, 0].astype(np.float64).mean(axis=0)
        degraded_profile = (
            degraded[:, :, 0].astype(np.float64).mean(axis=0)
        )

        self.assertEqual(_transition_width(clean_profile), 0)
        self.assertEqual(_transition_width(degraded_profile), 0)
        clean_contrast = float(
            clean_profile[-16:].mean() - clean_profile[:16].mean()
        )
        degraded_contrast = float(
            degraded_profile[-16:].mean()
            - degraded_profile[:16].mean()
        )
        self.assertGreaterEqual(degraded_contrast / clean_contrast, 0.99)
        self.assertLessEqual(degraded_contrast / clean_contrast, 1.01)
        self.assertFalse(np.array_equal(degraded, rgb))

    def test_every_parameter_and_semantic_mutation_fails_closed(self) -> None:
        mutations = {
            "OPERATOR_ID": "DRIFT",
            "OPERATOR_CLASS": "LINEAR_FILTER",
            "COLOR_SPACE": "SRGB_UINT8",
            "DIAMETER": 5,
            "SIGMA_COLOR": 0.09,
            "SIGMA_SPACE": 4.0,
            "BORDER_TYPE": cv2.BORDER_REPLICATE,
            "BORDER_NAME": "BORDER_REPLICATE",
            "PSF_NONE": True,
        }
        for name, value in mutations.items():
            with self.subTest(name=name):
                with mock.patch.object(operator, name, value):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "QMS_R0_OPERATOR_IDENTITY_DRIFT",
                    ):
                        operator.apply_linear_bilateral_texture(_fixture())

    def test_external_identity_parameter_mutation_is_rejected(self) -> None:
        identity = operator.frozen_operator_identity()
        for key, value in (
            ("diameter", 9),
            ("sigma_color", 0.081),
            ("sigma_space", 2.0),
            ("border_name", "BORDER_CONSTANT"),
            ("psf_none", True),
        ):
            with self.subTest(key=key):
                mutation = dict(identity)
                mutation[key] = value
                with self.assertRaisesRegex(
                    ValueError,
                    "QMS_R0_OPERATOR_IDENTITY_INVALID",
                ):
                    operator.validate_operator_identity(mutation)

    def test_conversion_is_float32_and_uint8_round_trip_is_exact(self) -> None:
        values = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
        rgb = np.repeat(values, 3, axis=2)
        linear = operator.srgb_u8_to_linear_rgb_float32(rgb)
        self.assertEqual(linear.dtype, np.float32)
        rebuilt = operator.linear_rgb_float32_to_srgb_u8(linear)
        np.testing.assert_array_equal(rebuilt, rgb)


if __name__ == "__main__":
    unittest.main()
