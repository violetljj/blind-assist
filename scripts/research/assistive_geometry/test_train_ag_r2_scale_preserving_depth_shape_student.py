#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from scripts.research.assistive_geometry.train_ag_r2_scale_preserving_depth_shape_student import (
    ScalePreservingDepthShapeHead,
)


class ScalePreservingDepthShapeStudentTest(unittest.TestCase):
    def test_forward_preserves_all_pixel_log_scale(self) -> None:
        torch.manual_seed(7)
        model = ScalePreservingDepthShapeHead(hidden=16)
        feature = torch.randn(2, 192, 8, 6)
        base = torch.rand(2, 1, 8, 6) * 4.0 + 0.5
        output = model(feature, base)
        base_log = base.log()
        torch.testing.assert_close(
            output["predicted_log_depth"].mean(dim=(-2, -1)),
            base_log.mean(dim=(-2, -1)),
            atol=1.0e-6,
            rtol=0.0,
        )
        self.assertEqual(tuple(output["predicted_log_depth"].shape), (2, 1, 8, 6))


if __name__ == "__main__":
    unittest.main()
