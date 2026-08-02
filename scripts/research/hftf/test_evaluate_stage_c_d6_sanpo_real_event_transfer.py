import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    aggregate,
    hold_sampled_values,
    reference_comparison,
    score_event,
    single_frame_logits,
)
from train_stage_c_d5_tartanground_development_student import (
    TemporalStudent,
)


class SanpoRealEventTransferTest(unittest.TestCase):
    def test_hold_sampled_values_is_causal_zero_order_hold(self):
        self.assertEqual(
            [False, False, True, True, False],
            hold_sampled_values(
                [False, True, False],
                [0, 2, 4],
                5,
            ),
        )

    def test_event_scoring_matches_riskseg_event_semantics(self):
        positive = {
            "parent_event_id": "positive",
            "source_session_id": "session-positive",
            "bucket": "blocking_obstacle_positive",
            "alertable_interval_frames": [1, 3],
            "passed_interval_frames": [4, 5],
        }
        negative = {
            "parent_event_id": "negative",
            "source_session_id": "session-negative",
            "bucket": "normal_walkable_negative",
            "alertable_interval_frames": None,
            "passed_interval_frames": None,
        }
        positive_score = score_event(
            positive,
            [False, False, True, False, False, False],
        )
        negative_score = score_event(
            negative,
            [False, True, False],
        )
        self.assertTrue(positive_score["event_hit"])
        self.assertTrue(positive_score["passed_cleared"])
        self.assertEqual(1, positive_score["response_delay_frames"])
        self.assertTrue(negative_score["false_alert_event"])

    def test_aggregate_and_pareto_comparison(self):
        rows = []
        for index in range(16):
            rows.append(
                {
                    "positive": True,
                    "event_hit": index < 14,
                    "critical_miss": index >= 14,
                    "passed_cleared": index < 6,
                    "response_delay_frames": 0 if index < 14 else None,
                    "bucket": (
                        "blocking_obstacle_positive"
                        if index < 8
                        else "boundary_level_change_positive"
                    ),
                }
            )
        for index in range(14):
            rows.append(
                {
                    "positive": False,
                    "false_alert_event": index < 5,
                    "bucket": (
                        "normal_walkable_negative"
                        if index < 7
                        else "parallel_curb_negative"
                    ),
                }
            )
        metrics = aggregate(rows)
        comparison = reference_comparison(
            metrics,
            {
                "hit_event_count": 13,
                "critical_miss_count": 3,
                "false_alert_event_count": 6,
                "cleared_event_count": 5,
            },
        )
        self.assertEqual(14, metrics["hit_event_count"])
        self.assertEqual(5, metrics["false_alert_event_count"])
        self.assertEqual(6, metrics["cleared_event_count"])
        self.assertTrue(
            comparison["development_pareto_dominates_current_yolo"]
        )

    def test_single_frame_fast_path_matches_repeated_current(self):
        torch.manual_seed(7)
        model = TemporalStudent.__new__(TemporalStudent)
        torch.nn.Module.__init__(model)
        model.architecture = "directional"
        model.temporal_mode = "joint"
        model.encoder = torch.nn.Conv2d(3, 6, kernel_size=3, padding=1)
        model.temporal_depthwise = torch.nn.Conv3d(
            6,
            6,
            kernel_size=(5, 1, 1),
            groups=6,
            bias=False,
        )
        model.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(6, 4, kernel_size=1, bias=False),
            torch.nn.GroupNorm(2, 4),
            torch.nn.Hardswish(),
        )
        model.dropout = torch.nn.Dropout(0.0)
        model.pool = torch.nn.AdaptiveAvgPool2d((1, 6))
        model.head = torch.nn.Conv1d(
            4,
            2 * 3 * 3 * 6,
            kernel_size=1,
        )
        model.eval()
        frames = torch.randn(2, 3, 8, 12)
        repeated = frames[:, None].repeat(1, 5, 1, 1, 1)
        expected = model(repeated)
        actual = single_frame_logits(model, frames)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
