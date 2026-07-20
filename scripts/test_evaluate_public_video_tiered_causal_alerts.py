import unittest

import evaluate_public_video_tiered_causal_alerts as subject


class TieredCausalAlertsTest(unittest.TestCase):
    def test_upgrade_requires_consecutive_seconds(self) -> None:
        frames = [
            {"timestamp_ms": 0, "trace_intrusion_score": 1 / 3},
            {"timestamp_ms": 1000, "trace_intrusion_score": 1 / 3},
        ]
        self.assertEqual(1000, subject.first_upgrade(frames, 1 / 3, 2))

    def test_gap_resets_upgrade_run(self) -> None:
        frames = [
            {"timestamp_ms": 0, "trace_intrusion_score": 1.0},
            {"timestamp_ms": 2000, "trace_intrusion_score": 1.0},
        ]
        self.assertIsNone(subject.first_upgrade(frames, 1 / 3, 2))

    def test_below_threshold_never_upgrades(self) -> None:
        frames = [
            {"timestamp_ms": 0, "trace_intrusion_score": 0.0},
            {"timestamp_ms": 1000, "trace_intrusion_score": 0.0},
        ]
        self.assertIsNone(subject.first_upgrade(frames, 1 / 3, 2))

    def test_reads_confirmed_lifecycle_clear_timestamp(self) -> None:
        intervals = [{"confirmed_clear_timestamp_ms": 123000}]
        self.assertEqual(123000, subject.confirmed_clear_timestamp(intervals))

    def test_multiple_intervals_do_not_count_as_one_clear_lifecycle(self) -> None:
        intervals = [
            {"confirmed_clear_timestamp_ms": 123000},
            {"confirmed_clear_timestamp_ms": 456000},
        ]
        self.assertIsNone(subject.confirmed_clear_timestamp(intervals))


if __name__ == "__main__":
    unittest.main()
