#!/usr/bin/env python3

import unittest

import numpy as np

from build_ag_st_factor_labels import PROVENANCE_SOURCE_NATIVE, TIER_A_SOURCE
from materialize_ag_st_sequence_identity_labels import horizontal_source_world_heights


class SequenceIdentityLabelTest(unittest.TestCase):
    def test_horizontal_source_height_recovery(self) -> None:
        height = width = 16
        depth = np.full((height, width), 1.5, dtype=np.float32)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
        pose[2, 3] = 1.5
        label = {
            "metric_depth_m_hw": depth,
            "metric_depth_valid_hw": np.ones_like(depth, dtype=np.bool_),
            "provenance_code_hw": np.full_like(depth, PROVENANCE_SOURCE_NATIVE, dtype=np.uint8),
            "quality_tier_hw": np.full_like(depth, TIER_A_SOURCE, dtype=np.uint8),
            "intrinsics_output": np.asarray(
                [[100.0, 0.0, 7.5], [0.0, 100.0, 7.5], [0.0, 0.0, 1.0]]
            ),
            "camera_to_world_output": pose,
        }
        values = horizontal_source_world_heights(label)
        self.assertGreater(len(values), 20)
        np.testing.assert_allclose(values, 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
