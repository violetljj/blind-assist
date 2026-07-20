import unittest

import run_public_video_future_ego_trace_multisource_probe as subject


class FutureEgoTraceMultisourceGateTest(unittest.TestCase):
    def test_accepts_only_strict_complete_separation(self):
        rows = [
            {"label": 1, "score": 0.4, "valid_frame_fraction": 1.0},
            {"label": 1, "score": 0.3, "valid_frame_fraction": 0.8},
            {"label": 0, "score": 0.2, "valid_frame_fraction": 0.9},
        ]
        self.assertTrue(all(subject.separation_checks(rows, 0.5).values()))

    def test_rejects_tie_and_low_coverage(self):
        rows = [
            {"label": 1, "score": 0.2, "valid_frame_fraction": 1.0},
            {"label": 0, "score": 0.2, "valid_frame_fraction": 0.4},
        ]
        checks = subject.separation_checks(rows, 0.5)
        self.assertFalse(checks["strict_complete_separation"])
        self.assertFalse(checks["all_events_have_sufficient_valid_frames"])


if __name__ == "__main__":
    unittest.main()
