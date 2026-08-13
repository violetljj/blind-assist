from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ag_st_frame_advantage_lcb_router import (  # noqa: E402
    FrameAdvantageObservation,
    calibrate_frame_gate,
    gated_selector_observations,
    knn_support_lower_bounds,
    pinball_loss,
)
from train_ag_st_no_regret_selector import SelectorObservation  # noqa: E402


def row(parent: str, domain: str, *, beneficial: bool) -> FrameAdvantageObservation:
    truth = np.ones((1, 4), dtype=np.float32)
    base = np.full((1, 4), 1.3, dtype=np.float32)
    expert = np.full((1, 4), 1.0 if beneficial else 1.5, dtype=np.float32)
    return FrameAdvantageObservation(
        parent_id=parent,
        domain=domain,
        observable=np.zeros(3, dtype=np.float32),
        selector=SelectorObservation(
            parent_id=parent,
            domain=domain,
            truth_depth_m=truth,
            valid=np.ones_like(truth, dtype=bool),
            base_depth_m=base,
            expert_depth_m=expert,
            selector_probability=np.full((1, 4), 0.9, dtype=np.float32),
        ),
        mae_advantage_m=0.3 if beneficial else -0.2,
        bad_rate_advantage=1.0 if beneficial else 0.0,
    )


class FrameAdvantageLcbRouterTest(unittest.TestCase):
    def test_pinball_penalizes_unsafe_overestimate_more_at_lower_quantile(self) -> None:
        target = torch.zeros(1, 2)
        underestimate = pinball_loss(torch.full((1, 2), -1.0), target)
        overestimate = pinball_loss(torch.full((1, 2), 1.0), target)
        self.assertGreater(float(overestimate), float(underestimate))

    def test_frame_gate_can_veto_harmful_candidate_without_opening_pixels(self) -> None:
        rows = [row("safe", "A", beneficial=True), row("harm", "B", beneficial=False)]
        gated = gated_selector_observations(rows, np.asarray([1.0, -1.0]), 0.0)
        self.assertTrue(np.all(gated[0].selector_probability == 0.9))
        self.assertTrue(np.all(gated[1].selector_probability < 0.0))

    def test_calibration_freezes_nontrivial_parent_safe_gate(self) -> None:
        rows = [row("safe", "A", beneficial=True), row("harm", "B", beneficial=False)]
        calibrated = calibrate_frame_gate(
            rows,
            np.asarray([1.0, -1.0]),
            0.2,
            candidates=(0.0, 1.5),
        )
        self.assertEqual(
            "FRAME_ADVANTAGE_LCB_NONTRIVIAL_GATE_FROZEN",
            calibrated["decision"],
        )
        self.assertEqual(0.0, calibrated["frame_score_threshold"])
        self.assertGreater(
            calibrated["selected_summary"]["parent_macro"][
                "selected_coverage_fraction"
            ],
            0.0,
        )

    def test_knn_support_excludes_query_parent(self) -> None:
        fit = [
            row("a", "A", beneficial=False),
            row("b", "A", beneficial=True),
            row("c", "B", beneficial=True),
            row("d", "B", beneficial=True),
        ]
        fit[0] = FrameAdvantageObservation(
            **{**fit[0].__dict__, "observable": np.asarray([0.0, 0.0, 0.0], dtype=np.float32)}
        )
        query = FrameAdvantageObservation(
            **{**fit[0].__dict__, "observable": np.asarray([0.0, 0.0, 0.0], dtype=np.float32)}
        )
        lower, _ = knn_support_lower_bounds(
            fit,
            [query],
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            neighbors=1,
        )
        self.assertGreater(float(lower[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
