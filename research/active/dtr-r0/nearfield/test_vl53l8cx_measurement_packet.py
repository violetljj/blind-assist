"""Input integrity and causality checks; no sensor or model performance claims."""
import unittest
import numpy as np
from vl53l8cx_measurement_packet import MeasurementPacket, align_to_rgb, from_legacy, from_uld


class PacketTests(unittest.TestCase):
    def test_legacy_bytes_and_absent_quality(self):
        ranges = np.full((64, 2), .12345679, np.float32)
        ranges[0] = [0, np.nan]
        valid = np.ones((64, 2), bool); valid[0] = False
        p = from_legacy(ranges, valid, measurement_id='static-case')
        self.assertEqual(p.range_m.tobytes(), ranges.tobytes())
        self.assertEqual(p.valid.tobytes(), valid.tobytes())
        self.assertIsNone(p.acquisition_end_ns)
        for name in ('signal_per_spad', 'ambient_per_spad', 'target_status', 'nb_target_detected'):
            value, present = p.observed_field(name)
            self.assertFalse(present.any()); self.assertFalse(value.any())

    def test_uld_target_status_and_count_are_independent(self):
        d = np.full((64, 2), 720, np.int16)
        s = np.full((64, 2), 5, np.uint8);s[0] = [9, 3]
        count = np.full(64, 2, np.uint8);count[1] = 0
        p = from_uld(d, s, count, measurement_id='device-1')
        self.assertFalse(p.valid[:2].any())
        self.assertTrue(p.valid[2:].all())
        np.testing.assert_array_equal(p.target_status, s)
        q = from_uld(d, s, count, measurement_id='device-1', accepted_statuses=(5, 6, 9))
        self.assertTrue(q.valid[0, 0]);self.assertFalse(q.valid[0, 1])
        self.assertFalse(q.valid[1].any())

    def test_zero_quality_available_is_not_missing(self):
        signal = np.zeros((64, 1), np.float32);signal[1] = np.nan
        p = from_uld(np.full((64, 1), 700), np.full((64, 1), 5), np.ones(64, int),
                     measurement_id='d', quality={'signal_per_spad': signal})
        values, present = p.observed_field('signal_per_spad')
        self.assertEqual(values[0, 0], 0);self.assertTrue(present[0, 0])
        self.assertEqual(values[1, 0], 0);self.assertFalse(present[1, 0])

    def test_no_truth_field_or_wrong_quality_shape(self):
        args=(np.ones((64, 1))*700, np.ones((64, 1), int)*5, np.ones(64, int))
        for quality in ({'native_depth': np.ones((64, 1))}, {'ambient_per_spad': np.ones((64, 1))}):
            with self.assertRaises(ValueError):from_uld(*args, measurement_id='d', quality=quality)

    def test_multitarget_order_and_shape_are_preserved(self):
        d=np.tile([1800, 700, 2500, 1200], (64, 1))
        p=from_uld(d, np.full((64, 4), 5), np.full(64, 4), measurement_id='four', target_order='strongest')
        np.testing.assert_allclose(p.range_m[0], [1.8, .7, 2.5, 1.2])
        self.assertEqual(p.target_order, 'strongest');self.assertEqual(p.accepted_statuses, (5,))
        with self.assertRaises(ValueError):
            from_uld(d, np.full((64, 4), 5), np.full(64, 5), measurement_id='invalid-count')

    def packet(self, name, end, received, sequence='clip-a'):
        return from_uld(np.full((64, 1), 700), np.full((64, 1), 5), np.ones(64, int),
                        measurement_id=name, sequence_id=sequence, clock_id='mapped-clock',
                        acquisition_start_ns=end-5, acquisition_end_ns=end, received_ns=received,
                        configured_hz=15)

    def align(self, packets, rgb=100, decision=105, **kwargs):
        return align_to_rgb(packets, sequence_id='clip-a', clock_id='mapped-clock',
                            rgb_timestamp_ns=rgb, decision_timestamp_ns=decision, max_age_ns=40, **kwargs)

    def test_future_or_unreceived_packet_never_used(self):
        old=self.packet('old', 80, 85)
        result=self.align([old, self.packet('future', 102, 103), self.packet('late', 95, 110), self.packet('other-clip', 99, 100, 'clip-b')])
        self.assertIs(result['packet'], old);self.assertEqual(result['age_ns'], 20)

    def test_same_sample_reuse_and_explicit_stale(self):
        p=self.packet('tof-0', 80, 85)
        self.assertTrue(self.align([p], previous_measurement_id='tof-0')['reused'])
        self.assertEqual(self.align([p], rgb=121, decision=121)['state'], 'STALE')
        self.assertEqual(self.align([])['state'], 'NO_ALIGNED_TOF')
        static=from_legacy(np.ones((64, 2), np.float32), np.ones((64, 2), bool), measurement_id='case')
        self.assertEqual(self.align([static])['state'], 'NO_ALIGNED_TOF')

    def test_impossible_configuration_rejected(self):
        with self.assertRaises(ValueError):self.packet('d', 100, 90)
        with self.assertRaisesRegex(ValueError, 'rate exceeds'):
            from_uld(np.full((64, 1), 700), np.full((64, 1), 5), np.ones(64, int),
                     measurement_id='d', configured_hz=60)


if __name__ == '__main__':
    unittest.main()
