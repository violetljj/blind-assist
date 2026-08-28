from __future__ import annotations

import unittest

from dtr_c2_fresh_global_obb_replay import score_sequence


class FreshGlobalReplayScoreTest(unittest.TestCase):
    def test_bounded_contact_match_and_clear_false_segment(self) -> None:
        labels = ["CLEAR", "CLEAR", "CONTACT", "CLEAR", "CLEAR"]
        timeline = [
            {
                "frame": index,
                "time_s": float(index),
                "label": label,
                "first_hit_delta_s": 0.0 if label == "CONTACT" else None,
                "responsible_components": ["p1"] if label == "CONTACT" else [],
            }
            for index, label in enumerate(labels)
        ]
        predictions = {
            index: {
                "active": ({"p1"} if index == 2 else {"noise"} if index == 4 else set()),
                "raw": set(),
            }
            for index in range(len(timeline))
        }
        score = score_sequence(
            sequence="synthetic",
            timeline=timeline,
            prediction_frames=predictions,
        )
        self.assertEqual(score["bounded_contact_events"], 1)
        self.assertEqual(score["bounded_contact_events_recalled"], 1)
        self.assertEqual(score["false_alert_segments"], 1)
        self.assertEqual(score["median_first_alert_lead_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
