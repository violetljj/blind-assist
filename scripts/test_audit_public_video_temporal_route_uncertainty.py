import unittest

import numpy as np

import audit_public_video_temporal_route_uncertainty as subject


class TemporalRouteUncertaintyAuditTest(unittest.TestCase):
    def test_top_k_can_recover_non_argmax_obstacle(self):
        predicted = np.zeros((3, 4, 4), dtype=np.float32)
        predicted[:, 0, 0] = 2.0
        predicted[:, 0, 1] = 1.5
        obstacle = np.zeros((4, 4), dtype=bool)
        obstacle[0, 1] = True
        values = subject.frame_readouts(predicted, obstacle)
        self.assertEqual(0.0, values["argmax_hit"])
        self.assertEqual(1.0, values["top3_any_hit"])

    def test_strict_separation_rejects_tie(self):
        events = [{"label": 1, "score": 0.2}, {"label": 0, "score": 0.2}]
        self.assertFalse(subject.strict_separation(events, "score"))


if __name__ == "__main__":
    unittest.main()
