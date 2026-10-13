import unittest

from report_causal_pairing import causal, nearest, quantiles, session_frames, summarize, wait_join


def image(index, time_ms):
    return dict(index=index, received_elapsed_ns=int(time_ms * 1e6), boot_id='boot', source_frame_id=index)


def tof(index, time_ms, reference=None):
    return dict(index=index, sequence=index, valid_frame=True, received_elapsed_ns=int(time_ms * 1e6),
                imu_alignment={'imu_record_index': reference})


class PairingTests(unittest.TestCase):
    def test_future_nearest_is_not_causal_and_exact_boundaries_are_included(self):
        rows = [image(1, 900), image(2, 1010)]
        self.assertEqual(nearest(rows, 1_000_000_000)['index'], 2)
        self.assertEqual(causal(rows, 1_000_000_000)['index'], 1)
        self.assertIsNone(causal(rows, 1_001_000_000))
        self.assertIsNone(causal(rows, 899_000_000))
        self.assertEqual(causal(rows, 1_010_000_000)['index'], 2)

    def test_nearest_first_tie_and_buffer_last_insertion_are_preserved(self):
        rows = [image(1, 900), image(2, 900)]
        self.assertEqual(nearest(rows, 1_000_000_000)['index'], 1)
        self.assertEqual(causal(rows, 1_000_000_000)['index'], 2)
        self.assertEqual(wait_join(rows, 1_000_000_000, 0)['index'], 2)

    def test_bounded_wait_cannot_read_past_deadline_or_expand_age_gate(self):
        rows = [image(1, 1000), image(2, 1120)]
        receipt = 1_100_000_000
        self.assertEqual(wait_join(rows, receipt, 0)['index'], 1)
        self.assertEqual(wait_join(rows, receipt, 19)['index'], 1)
        self.assertEqual(wait_join(rows, receipt, 20)['index'], 2)
        self.assertIsNone(wait_join([image(1, 1201)], receipt, 100))

    def test_frozen_imu_is_saved_reference_not_offline_nearest(self):
        imu = [dict(image(7, 990), valid_frame=True, gravity_unit=[0, -1, 0]),
               dict(image(8, 1001), valid_frame=True, gravity_unit=[0, -1, 0])]
        frames, _ = session_frames('s', [tof(1, 1000, 7)], [image(1, 990)], imu)
        row = frames[0]
        self.assertEqual(row['imu_frozen_saved']['record_index'], 7)
        self.assertEqual(row['imu_causal']['record_index'], 7)
        self.assertEqual(row['imu_nearest_diagnostic']['record_index'], 8)

    def test_changed_identity_is_separate_from_coverage_loss(self):
        frames, _ = session_frames('s', [tof(1, 1000), tof(2, 1200), tof(3, 1400)],
                                   [image(1, 990), image(2, 1001), image(3, 1190), image(4, 1420)], [])
        summary = summarize(frames)
        t = summary['rgb_nearest_to_causal']
        self.assertEqual((t['both_same_record'], t['both_different_record'], t['left_only']), (1, 1, 1))
        self.assertEqual(summary['bounded_wait_rgb']['20']['from_zero_wait']['right_only'], 1)
        self.assertEqual(summary['bounded_wait_rgb']['20']['from_zero_wait']['left_only'], 0)

    def test_missing_frames_remain_in_denominator_and_windows_never_cross_sessions(self):
        first, _ = session_frames('a', [tof(i, i * 200) for i in range(1, 4)], [], [])
        second, _ = session_frames('b', [tof(i, i * 200) for i in range(1, 4)], [], [])
        summary = summarize(first + second)
        self.assertEqual(summary['denominator_cnh'], 6)
        self.assertEqual(summary['modes']['rgb_causal']['missing'], 6)
        windows = summary['potential_three_cnh_windows']['rgb_causal']
        self.assertEqual(windows['three_adjacent_valid_cnh_windows'], 2)
        self.assertEqual(windows['distinct_rgb_ids']['0'], 2)

    def test_repeated_frame_windows_and_pooled_age_quantiles(self):
        frames, _ = session_frames('s', [tof(1, 1000), tof(2, 1050), tof(3, 1200)], [image(1, 990)], [])
        summary = summarize(frames)
        self.assertEqual(summary['modes']['rgb_causal']['extra_uses_beyond_first'], 1)
        windows = summary['potential_three_cnh_windows']['rgb_causal']
        self.assertEqual(windows['windows_reusing_rgb'], 1)
        self.assertEqual(windows['at_least_two_matched_but_one_rgb_id'], 1)
        self.assertEqual(quantiles([0, 10, 20])['p95'], 19)


if __name__ == '__main__':
    unittest.main()
