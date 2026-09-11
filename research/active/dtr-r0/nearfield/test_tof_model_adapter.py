import unittest
import numpy as np
from tof_model_adapter import adapt_packet, legacy_batch_inputs, legacy_packet_inputs
from vl53l8cx_measurement_packet import from_legacy, from_uld


class AdapterTests(unittest.TestCase):
    def test_lone_second_return_and_original_packet_preserved(self):
        r = np.zeros((64, 2), np.float32);v = np.zeros_like(r, bool)
        r[0] = [np.nan, 1.8];v[0, 1] = True
        p = from_legacy(r, v, measurement_id='second')
        before = p.range_m.tobytes()
        a = adapt_packet(p);old, indices = legacy_packet_inputs(p)
        np.testing.assert_array_equal(a.range_m[0], np.array([r[0, 1], 0], np.float32))
        np.testing.assert_array_equal(a.valid[0], [True, False])
        np.testing.assert_array_equal(indices[0], [1, 0])
        np.testing.assert_array_equal(old['ranges'], a.range_m)
        self.assertEqual(before, p.range_m.tobytes())
        self.assertFalse(a.target_fields['signal_per_spad'][1].any())

    def test_quality_status_follow_permutation_even_for_invalid_target(self):
        d = np.tile([1800, 700, 2500, 1200], (64, 1))
        status = np.tile([5, 5, 3, 5], (64, 1))
        signal = np.tile([11., 0., 33., np.nan], (64, 1))
        p = from_uld(d, status, np.full(64, 4), measurement_id='quality',
                     quality={'signal_per_spad': signal, 'ambient_per_spad': np.zeros(64)})
        a = adapt_packet(p)
        np.testing.assert_array_equal(a.original_slot[0], [1, 3, 0, 2])
        np.testing.assert_array_equal(a.target_fields['signal_per_spad'][0][0], [0, 0, 11, 33])
        np.testing.assert_array_equal(a.target_fields['signal_per_spad'][1][0], [True, False, True, True])
        np.testing.assert_array_equal(a.target_fields['target_status'][0][0], [5, 5, 5, 3])
        self.assertTrue(a.zone_fields['ambient_per_spad'][1].all())
        self.assertFalse(a.zone_fields['nb_spads_enabled'][1].any())
        self.assertEqual(a.raw_target_order, 'strongest')
        np.testing.assert_array_equal(p.target_status, status)
        # Restore original per-target fields exactly, including unavailable mask.
        inverse = np.argsort(a.original_slot, axis=1)
        for name, (value, mask) in a.target_fields.items():
            original, present = p.observed_field(name)
            np.testing.assert_array_equal(np.take_along_axis(value, inverse, 1), original)
            np.testing.assert_array_equal(np.take_along_axis(mask, inverse, 1), present)

    def test_stable_ties_empty_zones_and_padding(self):
        r = np.full((1, 64, 2), 1., np.float32);v = np.ones_like(r, bool)
        v[0, 0] = False
        a, indices = legacy_batch_inputs(r, v)
        np.testing.assert_array_equal(indices[0], np.tile([0, 1], (64, 1)))
        self.assertFalse(a['ranges'][0, 0].any())
        b, ii = legacy_batch_inputs(r[:, :, :1], v[:, :, :1])
        self.assertFalse(b['valid'][:, :, 1].any());self.assertTrue((ii[:, :, 1] == -1).all())
        again, _ = legacy_batch_inputs(a['ranges'], a['valid'])
        np.testing.assert_array_equal(a['ranges'], again['ranges'])
        np.testing.assert_array_equal(a['valid'], again['valid'])

    def test_domain_rejected_instead_of_fabricated_resize_or_silent_truncation(self):
        for shape in [(1, 16, 2), (1, 64, 4)]:
            with self.assertRaises(ValueError):legacy_batch_inputs(np.ones(shape, np.float32), np.ones(shape, bool))
        r = np.full((1, 64, 2), 4.01, np.float32)
        with self.assertRaisesRegex(ValueError, '4m'):legacy_batch_inputs(r, np.ones_like(r, bool))
        for value in [np.nan, np.inf, 0., -1.]:
            with self.assertRaises(ValueError):legacy_batch_inputs(np.full_like(r, value), np.ones_like(r, bool))
        with self.assertRaisesRegex(ValueError, 'float32'):
            legacy_batch_inputs(np.full(r.shape, 1e-80, np.float64), np.ones_like(r, bool))

    def test_permuted_valid_slots_same_model_input(self):
        r = np.tile(np.array([1.8, .7], np.float32), (3, 64, 1));v = np.ones_like(r, bool)
        v[0, :, 0] = False;v[1] = False
        a, _ = legacy_batch_inputs(r, v)
        b, _ = legacy_batch_inputs(r[:, :, ::-1], v[:, :, ::-1])
        np.testing.assert_array_equal(a['ranges'], b['ranges'])
        np.testing.assert_array_equal(a['valid'], b['valid'])


if __name__ == '__main__':unittest.main()
