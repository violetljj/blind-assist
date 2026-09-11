"""Focused validity, provenance and original-return preservation checks."""
import unittest
import numpy as np

from tof_return_sensitivity import MODES, as_measurement_packet, constrain


class ReturnSensitivityTests(unittest.TestCase):
    def fixture(self):
        r = np.zeros((1, 64, 2), np.float32)
        v = np.zeros_like(r, dtype=bool)
        r[0, :5] = [[.75, 1.], [.5, 1.5], [.9, 0], [0, 1.1], [.8, .8]]
        v[0, :5] = [[1, 1], [1, 1], [1, 0], [0, 1], [1, 1]]
        return r, v

    def test_both_endpoints_keep_original_distance_in_original_slot(self):
        r, v = self.fixture()
        before = (r.tobytes(), v.tobytes())
        for mode, keep in zip(MODES, (0, 1)):
            out, affected = constrain(r, v, mode)
            np.testing.assert_array_equal(np.flatnonzero(affected[0]), [0, 4])
            self.assertEqual(out['ranges'][affected, keep].tobytes(), r[affected, keep].tobytes())
            self.assertTrue(out['valid'][affected, keep].all())
            self.assertFalse(out['valid'][affected, 1-keep].any())
            self.assertFalse(out['ranges'][affected, 1-keep].any())
            self.assertEqual(out['ranges'][~affected].tobytes(), r[~affected].tobytes())
            self.assertEqual(out['valid'][~affected].tobytes(), v[~affected].tobytes())
            np.testing.assert_array_equal(out['valid'].any(2), v.any(2))
        self.assertEqual(before, (r.tobytes(), v.tobytes()))

    def test_missing_stays_missing_without_rewriting_invalid_bytes(self):
        r = np.full((2, 64, 2), np.nan, np.float32)
        v = np.zeros_like(r, dtype=bool)
        for mode in MODES:
            out, affected = constrain(r, v, mode)
            self.assertFalse(affected.any())
            self.assertFalse(out['valid'].any())
            self.assertEqual(out['ranges'].tobytes(), r.tobytes())

    def test_strict_threshold_boundary(self):
        r, v = self.fixture()
        # Exactly representable .25m interval: equality is not unresolved.
        out, affected = constrain(r, v, MODES[0], separation_m=.25)
        self.assertFalse(affected[0, 0])
        self.assertEqual(out['ranges'][0, 0].tobytes(), r[0, 0].tobytes())

    def test_misordered_or_nonfinite_usable_returns_are_rejected(self):
        for values in ([1., .75], [np.nan, 1.], [0., 1.]):
            r, v = self.fixture(); r[0, 0] = values
            with self.assertRaises(ValueError):
                constrain(r, v, MODES[0])

    def test_packet_never_fabricates_status_quality_or_time(self):
        r, v = self.fixture()
        p = as_measurement_packet(r[0], v[0], MODES[1], measurement_id='case')
        self.assertEqual(p.origin, 'SIMULATION_PROXY')
        self.assertEqual(p.target_order, 'legacy_first_last')
        self.assertIsNone(p.acquisition_end_ns)
        self.assertIsNone(p.nb_target_detected)
        for name in ('target_status', 'signal_per_spad', 'range_sigma_mm', 'ambient_per_spad'):
            self.assertFalse(p.observed_field(name)[1].any())
        self.assertEqual(p.range_m[0, 1], r[0, 0, 1])

    def test_device_multitarget_arrays_are_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            constrain(np.ones((1, 64, 4), np.float32), np.ones((1, 64, 4), bool), MODES[0])


if __name__ == '__main__':
    unittest.main()
