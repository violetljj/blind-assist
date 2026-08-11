#!/usr/bin/env python3

import unittest

import numpy as np

from materialize_ag_st_uncertainty_factor_labels import (
    pixel_radius_to_angular_uncertainty,
    support_uncertainty_proxy,
    validate_payload,
)


class UncertaintyFactorLabelsTest(unittest.TestCase):
    def test_angular_uncertainty_is_resize_invariant(self) -> None:
        radius = np.full((9, 11), 2.0, dtype=np.float32)
        valid = np.ones_like(radius, dtype=np.bool_)
        k = np.asarray([[10.0, 0.0, 5.0], [0.0, 12.0, 4.0], [0.0, 0.0, 1.0]])
        base = pixel_radius_to_angular_uncertainty(radius, valid, k)
        scale = 3
        high_radius = np.repeat(np.repeat(radius * scale, scale, axis=0), scale, axis=1)
        high_valid = np.ones_like(high_radius, dtype=np.bool_)
        high_k = k.copy()
        high_k[:2] *= scale
        high = pixel_radius_to_angular_uncertainty(high_radius, high_valid, high_k)
        np.testing.assert_allclose(high[::scale, ::scale], base, atol=1e-6)

    def test_angular_uncertainty_preserves_unknown(self) -> None:
        radius = np.ones((3, 4), dtype=np.float32)
        valid = np.ones_like(radius, dtype=np.bool_)
        valid[0] = False
        k = np.asarray([[8.0, 0.0, 1.5], [0.0, 8.0, 1.0], [0.0, 0.0, 1.0]])
        output = pixel_radius_to_angular_uncertainty(radius, valid, k)
        self.assertTrue(np.isnan(output[0]).all())
        self.assertTrue(np.isfinite(output[1:]).all())

    def test_support_uncertainty_rises_with_decision_ambiguity(self) -> None:
        probability = np.asarray([[0.0, 0.2, 0.5, 0.7, 1.0]], dtype=np.float32)
        valid = np.ones_like(probability, dtype=np.bool_)
        depth = np.full_like(probability, 2.0)
        metric_uncertainty = np.zeros_like(probability)
        output = support_uncertainty_proxy(
            probability,
            valid,
            depth,
            metric_uncertainty,
            0.0,
        )
        self.assertAlmostEqual(float(output[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(output[0, 2]), 1.0, places=6)
        self.assertAlmostEqual(float(output[0, 4]), 0.0, places=6)
        self.assertGreater(float(output[0, 1]), float(output[0, 0]))
        self.assertGreater(float(output[0, 3]), float(output[0, 4]))

    def test_support_uncertainty_without_valid_pixels_is_unknown(self) -> None:
        shape = (2, 3)
        output = support_uncertainty_proxy(
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.bool_),
            np.ones(shape, dtype=np.float32),
            np.ones(shape, dtype=np.float32),
            float("nan"),
        )
        self.assertTrue(np.isnan(output).all())

    def test_payload_validator_accepts_factor_specific_unknown_closure(self) -> None:
        shape = (2, 3)
        payload = {}
        for prefix, value_key in (
            ("depth", "depth_uncertainty_m_hw"),
            ("support", "support_uncertainty_probability_hw"),
            ("boundary", "boundary_angular_uncertainty_rad_hw"),
        ):
            valid = np.asarray([[True, False, True], [False, True, False]])
            payload[value_key] = np.where(valid, 0.1, np.nan).astype(np.float32)
            payload[f"{prefix}_uncertainty_valid_hw"] = valid.astype(np.uint8)
            payload[f"{prefix}_uncertainty_unknown_hw"] = (~valid).astype(np.uint8)
            payload[f"{prefix}_uncertainty_quality_tier_hw"] = np.where(valid, 3, 0).astype(np.uint8)
            payload[f"{prefix}_uncertainty_provenance_hw"] = np.where(valid, 1, 0).astype(np.uint8)
        validate_payload(payload)


if __name__ == "__main__":
    unittest.main()
