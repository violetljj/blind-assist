from __future__ import annotations

import unittest
import math

import torch

from scripts.research.assistive_geometry.train_ag_r2_f1_factor_learnability import (
    DEPTHART_PYRAMID_CHANNELS,
)
from scripts.research.assistive_geometry.train_ag_r2_metric_depth_student import (
    GLOBAL_LOG_CORRECTION_RANGE,
    MetricDepthStudentHead,
    metric_depth_loss,
)


class MetricDepthStudentTest(unittest.TestCase):
    def test_zero_initialized_head_is_depthart_identity(self) -> None:
        model = MetricDepthStudentHead(hidden=16, global_hidden=16).eval()
        feature = torch.randn(2, DEPTHART_PYRAMID_CHANNELS, 12, 16)
        base = torch.rand(2, 1, 12, 16) * 4.0 + 0.2
        with torch.no_grad():
            output = model(feature, base)
        torch.testing.assert_close(output["predicted_log_depth"], base.log())
        torch.testing.assert_close(
            output["global_log_scale_correction"], torch.zeros(2)
        )

    def test_global_metric_scale_is_bounded(self) -> None:
        model = MetricDepthStudentHead(hidden=16, global_hidden=16).eval()
        with torch.no_grad():
            model.global_scale[-1].bias.fill_(100.0)
        feature = torch.randn(1, DEPTHART_PYRAMID_CHANNELS, 8, 10)
        base = torch.ones(1, 1, 8, 10)
        with torch.no_grad():
            output = model(feature, base)
        self.assertLessEqual(
            abs(float(output["global_log_scale_correction"][0])),
            GLOBAL_LOG_CORRECTION_RANGE,
        )

    def test_loss_rewards_correct_global_scale(self) -> None:
        model = MetricDepthStudentHead(hidden=16, global_hidden=16).eval()
        feature = torch.zeros(1, DEPTHART_PYRAMID_CHANNELS, 8, 10)
        base = torch.ones(1, 1, 8, 10)
        target = {
            "metric_depth_m": torch.full((1, 1, 8, 10), 2.0),
            "metric_valid": torch.ones(1, 1, 8, 10, dtype=torch.bool),
        }
        with torch.no_grad():
            identity = model(feature, base)
            model.global_scale[-1].bias.fill_(
                math.atanh(math.log(2.0) / GLOBAL_LOG_CORRECTION_RANGE)
            )
            corrected = model(feature, base)
        self.assertLess(
            float(metric_depth_loss(corrected, target)["total"]),
            float(metric_depth_loss(identity, target)["total"]),
        )


if __name__ == "__main__":
    unittest.main()
