import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_source_centered_relation_encoder_canary import (
    RelationEncoder,
    build_relation_rows,
    collect_sanpo_support_episodes,
    summarize_predictions,
)


class SourceCenteredRelationEncoderCanaryTest(unittest.TestCase):
    def test_sanpo_support_separates_phase_labels(self):
        manifest = {
            "events": [
                {
                    "parent_event_id": "event-1",
                    "source_session_id": "source-1",
                    "bucket": "blocking_obstacle_positive",
                    "alertable_interval_frames": [0, 2],
                    "passed_interval_frames": [3, 4],
                    "frames": [
                        {"timestamp_ms": value}
                        for value in (0, 100, 200, 300, 400)
                    ],
                }
            ]
        }
        matrix = np.arange(10).reshape(5, 2)
        episodes, matrices = collect_sanpo_support_episodes(
            manifest,
            [matrix],
        )
        self.assertEqual([0, 1], [row["label"] for row in episodes])
        self.assertEqual((1, 2), matrices[0].shape)
        self.assertEqual((2, 2), matrices[1].shape)

    def test_encoder_shape(self):
        output = RelationEncoder()(torch.zeros(4, 256, 3, 6))
        self.assertEqual((4,), tuple(output.shape))

    def test_relation_rows_use_episode_balanced_source_baseline(self):
        episodes = [
            {
                "episode_id": "clear-long",
                "source_id": "source-a",
                "label": 0,
            },
            {
                "episode_id": "clear-short",
                "source_id": "source-a",
                "label": 0,
            },
            {
                "episode_id": "alert",
                "source_id": "source-a",
                "label": 1,
            },
        ]
        width = 128 * 3 * 6
        matrices = [
            np.stack(
                (
                    np.full(width, 1.0),
                    np.full(width, 3.0),
                )
            ),
            np.full((1, width), 6.0),
            np.full((1, width), 9.0),
        ]
        features, labels, sources, episode_ids, baselines = (
            build_relation_rows(episodes, matrices)
        )
        self.assertEqual((4, 256, 3, 6), features.shape)
        np.testing.assert_allclose(features[-1, :128], 5.0)
        np.testing.assert_allclose(features[-1, 128:], 5.0)
        self.assertEqual([0, 0, 0, 1], labels.tolist())
        self.assertEqual(["source-a"] * 4, sources.tolist())
        self.assertEqual("alert", episode_ids[-1])
        self.assertEqual(2, baselines[0]["no_alert_episode_count"])

    def test_prediction_summary_is_episode_balanced(self):
        result = summarize_predictions(
            np.asarray([0.9, 0.1, 0.8, 0.7]),
            np.asarray([1, 1, 0, 0]),
            np.asarray(["a", "a", "b", "b"]),
            np.asarray(["positive", "positive", "negative", "negative"]),
        )
        self.assertEqual(0.5, result["frame_alert_recall"])
        self.assertEqual(0.0, result["frame_no_alert_recall"])
        self.assertEqual(1.0, result["episode_alert_recall"])
        self.assertEqual(0.0, result["episode_no_alert_recall"])


if __name__ == "__main__":
    unittest.main()
