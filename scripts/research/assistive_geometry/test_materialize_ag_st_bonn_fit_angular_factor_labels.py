from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_ag_st_factor_labels import PROVENANCE_SOURCE_NATIVE, TIER_A_SOURCE  # noqa: E402
from materialize_ag_st_bonn_fit_angular_factor_labels import (  # noqa: E402
    build_factor_payload,
    frame_id,
)


class BonnFitAngularFactorLabelsTest(unittest.TestCase):
    def test_frame_id_is_stable(self) -> None:
        self.assertEqual(
            "bonn_rgbd_fit__rgbd_bonn_static__rgb_000402",
            frame_id("rgbd_bonn_static", 402),
        )

    def test_payload_keeps_unsupported_factors_unknown(self) -> None:
        height, width = 24, 32
        depth = np.full((height, width), 2.0, dtype=np.float32)
        depth[:, width // 2 :] = 3.0
        valid = np.ones((height, width), dtype=np.bool_)
        valid[0, :] = False
        depth[~valid] = 0.0
        intrinsics = np.asarray(
            [[80.0, 0.0, 15.5], [0.0, 80.0, 11.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        payload = build_factor_payload(depth, valid, intrinsics)

        self.assertTrue(np.array_equal(payload["metric_depth_valid_hw"], valid))
        self.assertTrue(np.isnan(payload["metric_depth_m_hw"][~valid]).all())
        self.assertTrue(
            np.all(payload["quality_tier_hw"][valid] == TIER_A_SOURCE)
        )
        self.assertTrue(
            np.all(
                payload["provenance_code_hw"][valid]
                == PROVENANCE_SOURCE_NATIVE
            )
        )
        self.assertFalse(payload["support_truth_valid_hw"].any())
        self.assertFalse(payload["evidence_truth_valid_hw"].any())
        self.assertTrue(payload["support_unknown_hw"].all())
        self.assertTrue(payload["evidence_unknown_hw"].all())

    def test_boundary_angular_fields_fail_closed(self) -> None:
        depth = np.full((20, 28), 1.5, dtype=np.float32)
        depth[:, 14:] = 2.7
        valid = np.ones(depth.shape, dtype=np.bool_)
        valid[:2, :] = False
        depth[~valid] = 0.0
        intrinsics = np.asarray(
            [[70.0, 0.0, 13.5], [0.0, 70.0, 9.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        payload = build_factor_payload(depth, valid, intrinsics)
        boundary_valid = payload["boundary_truth_valid_hw"].astype(np.bool_)
        angle = payload["boundary_angular_distance_rad_hw"]
        soft = payload["boundary_angular_soft_probability_hw"]

        self.assertTrue(boundary_valid.any())
        self.assertTrue(np.isnan(angle[~boundary_valid]).all())
        self.assertTrue(np.all(soft[~boundary_valid] == 0.0))
        self.assertTrue(np.isfinite(angle[boundary_valid]).all())
        self.assertTrue(
            np.all((soft[boundary_valid] >= 0.0) & (soft[boundary_valid] <= 1.0))
        )


if __name__ == "__main__":
    unittest.main()
