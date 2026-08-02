import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_sanpo_paired_pretraining_public_transfer_canary import (
    inventory,
    select_intervention_bearing_public_episodes,
    summarize_by_source,
)


def episode(source_id, label, frame_count):
    return {
        "source_id": source_id,
        "label": label,
        "frames": [{} for _ in range(frame_count)],
    }


class SanpoPairedPretrainingPublicTransferCanaryTest(
    unittest.TestCase
):
    def test_keeps_both_classes_for_intervention_sources_only(self):
        episodes = [
            episode("positive-a", 0, 2),
            episode("positive-a", 1, 3),
            episode("clear-only", 0, 5),
            episode("positive-b", 1, 7),
        ]
        selected, sources = (
            select_intervention_bearing_public_episodes(episodes)
        )
        self.assertEqual(
            ["positive-a", "positive-b"],
            sources,
        )
        self.assertEqual(
            [("positive-a", 0), ("positive-a", 1), ("positive-b", 1)],
            [
                (row["source_id"], row["label"])
                for row in selected
            ],
        )
        self.assertEqual(
            {
                "episode_count": 3,
                "source_count": 2,
                "frame_count": 12,
            },
            inventory(selected),
        )

    def test_source_macro_does_not_collapse_to_pooled_metric(self):
        summaries, macro = summarize_by_source(
            np.asarray([0.9, 0.1, 0.1, 0.9]),
            np.asarray([1, 0, 1, 0]),
            np.asarray(["a", "a", "b", "b"]),
            np.asarray(["a-1", "a-0", "b-1", "b-0"]),
        )
        self.assertEqual({"a", "b"}, set(summaries))
        self.assertEqual(0.5, macro["frame_alert_recall"])
        self.assertEqual(0.5, macro["frame_no_alert_recall"])
        self.assertEqual(0.5, macro["frame_balanced_accuracy"])


if __name__ == "__main__":
    unittest.main()
