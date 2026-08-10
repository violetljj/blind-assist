#!/usr/bin/env python3

import unittest

import numpy as np

from materialize_ag_st_source_native_boundary_corpus import (
    build_payload,
    conservative_source_boundary,
)


class SourceNativeBoundaryCorpusTest(unittest.TestCase):
    def test_conservative_boundary_rejects_plane_and_keeps_step(self) -> None:
        intrinsics = np.asarray(
            [[120.0, 0.0, 11.5], [0.0, 120.0, 11.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        depth = np.full((24, 24), 2.0, dtype=np.float32)
        valid = np.ones_like(depth, dtype=np.bool_)
        probability, factor_valid = conservative_source_boundary(depth, valid, intrinsics)
        self.assertEqual(0, int(np.sum(probability >= 0.5)))
        depth[:, 12:] = 2.8
        probability, factor_valid = conservative_source_boundary(depth, valid, intrinsics)
        self.assertGreater(int(np.sum(factor_valid & (probability >= 0.5))), 0)

    def test_payload_keeps_invalid_pixels_unknown(self) -> None:
        probability = np.asarray([[0.0, 1.0]], dtype=np.float32)
        valid = np.asarray([[False, True]])
        payload = build_payload(probability, valid, np.ones_like(probability))
        np.testing.assert_array_equal(payload["boundary_unknown_hw"], [[1, 0]])
        self.assertTrue(np.isinf(payload["boundary_uncertainty_px_hw"][0, 0]))
        self.assertEqual(0, int(payload["boundary_quality_tier_hw"][0, 0]))


if __name__ == "__main__":
    unittest.main()
