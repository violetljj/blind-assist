import unittest

from .evaluate import Track, associate, simulate_feedback, track_decision


def detection(height: float, left: float = 40.0) -> dict:
    return {
        "class_id": 0,
        "label": "person",
        "confidence": 0.9,
        "left": left,
        "top": 10.0,
        "right": left + 20.0,
        "bottom": 10.0 + height,
        "frame_width": 100,
        "frame_height": 100,
        "source": "OBJECT_DETECTOR",
        "temporal_promotion_eligible": True,
    }


class EvaluateTest(unittest.TestCase):
    def test_multitrack_preserves_unselected_history(self) -> None:
        tracks: list[Track] = []
        next_epoch = 0
        match = {}
        for index in range(7):
            tracks, match, _, next_epoch = associate(
                tracks,
                [detection(20.0 + index * 3.0), detection(15.0, left=70.0)],
                1_000_000_000 + index * 100_000_000,
                next_epoch,
            )
        self.assertEqual("CONFIRM_APPROACH", track_decision(match[0].history))

    def test_scene_rates_require_unique_association(self) -> None:
        tracks, _, _, next_epoch = associate(
            [], [detection(20.0), detection(15.0, left=70.0)], 1_000_000_000, 0
        )
        _, _, rates, _ = associate(
            tracks,
            [detection(18.0), detection(14.0, left=70.0)],
            1_100_000_000,
            next_epoch,
        )
        self.assertEqual(2, len(rates))
        self.assertTrue(all(rate < 0 for rate in rates))

    def test_feedback_veto_does_not_consume_cooldown(self) -> None:
        keys = [("s", "0"), ("s", "1")]
        risk = {
            "level": "HIGH",
            "direction": "CENTER",
            "proximity": "NEAR",
            "source_detection": detection(30.0),
        }
        baseline = {
            key: {"stable_risk": risk, "feedback_triggered": index == 0}
            for index, key in enumerate(keys)
        }
        timestamps = {keys[0]: 1_000_000_000, keys[1]: 1_100_000_000}
        result = simulate_feedback(
            keys, baseline, timestamps, vetoes={keys[0]: True}
        )
        self.assertFalse(result[keys[0]])
        self.assertTrue(result[keys[1]])


if __name__ == "__main__":
    unittest.main()
