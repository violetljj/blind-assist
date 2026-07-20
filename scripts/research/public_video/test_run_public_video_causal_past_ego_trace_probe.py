import unittest

import run_public_video_causal_past_ego_trace_probe as subject


class CausalPastEgoTraceTest(unittest.TestCase):
    def test_all_requested_frames_precede_current_time(self):
        self.assertEqual([9000, 8000, 7000], subject.past_timestamps(10000, [1000, 2000, 3000]))
        self.assertTrue(all(value < 10000 for value in subject.past_timestamps(10000, [1000, 2000, 3000])))

    def test_horizon_order_is_preserved(self):
        self.assertEqual([4000, 2000], subject.past_timestamps(5000, [1000, 3000]))


if __name__ == "__main__":
    unittest.main()
