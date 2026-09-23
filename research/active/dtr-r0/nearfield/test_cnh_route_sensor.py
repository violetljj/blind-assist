"""Focused physical invariants and attribution integrity for the CNH proxy."""
import unittest
from dataclasses import replace
import numpy as np
from cnh_route_sensor import (H1, H2, H3, RAW_BIN_M, SensorConfig, SensorParameters,
    angular_rays, axial_to_radial, decode_cnh, derive_readout, ideal_distribution,
    scalar_projection, synthesize_response)


class SensorTests(unittest.TestCase):
    def setUp(self):
        self.p = SensorParameters(noise_scale=0, pulse_sigma_bins=0, tail_mass=0,
                                  crosstalk_fraction=0, neighbour_leak=0)

    def response(self, distances, weights=1, **kwargs):
        d = np.broadcast_to(np.asarray(distances), (8, 8, len(distances)))
        return synthesize_response(d, .5, 1, weights, params=kwargs.pop('params', self.p), **kwargs)

    def test_buffer_window_and_integration_constraints(self):
        self.assertEqual(H2.validate().buffer_bytes, 2028)
        self.assertEqual(H3.validate().buffer_bytes, 5468)
        H1.validate()
        with self.assertRaises(ValueError):
            SensorConfig('bad', 8, 8, 24, 4, 5).validate()
        with self.assertRaises(ValueError):
            SensorConfig('bad', 8, 8, 18, 8, 5).validate()
        with self.assertRaises(ValueError):
            replace(H3, hz=15).validate()

    def test_signed_scalers_not_photon_clipping(self):
        np.testing.assert_array_equal(decode_cnh([-4, 4, -1], [1, -1, 0]), [-2, 8, -1])
        with self.assertRaises(ValueError):
            decode_cnh([2**31], [0])

    def test_radial_plane_conversion(self):
        rays, weights = angular_rays(4)
        radial = axial_to_radial(2, rays)
        np.testing.assert_allclose(radial*rays[..., 2], 2)
        self.assertGreater(radial[0, 0].mean(), radial[3, 3].mean())
        self.assertTrue((weights > 0).all())

    def test_sampling_density_and_missing_area(self):
        a, b = self.response([2]), self.response([2]*10)
        np.testing.assert_allclose(a['histogram'], b['histogram'])
        miss = self.response([2, np.nan])
        np.testing.assert_allclose(miss['histogram'], a['histogram']/2)

    def test_two_visible_layers_and_no_600mm_erasure(self):
        response = self.response([1.6, 1.9], [.2, .8])
        hist = derive_readout(response, H2)['histogram'][0, 0]
        self.assertEqual(np.count_nonzero(hist), 2)
        self.assertGreater(hist.sum(), 0)

    def test_window_loss_not_false_last_bin(self):
        response = self.response([7])
        self.assertEqual(response['histogram'].sum(), 0)
        self.assertGreater(response['diagnostics']['out_of_window_energy'].sum(), 0)
        readout = derive_readout(response, H3)
        self.assertFalse(readout['valid'].any())
        self.assertTrue(np.isnan(readout['distance_m']).all())

    def test_signed_noise_reproducible_attribution(self):
        p = replace(self.p, noise_scale=1, ambient_counts=20, signal_counts=0)
        response = self.response([2], params=p, seed=87)
        np.testing.assert_equal(response['histogram'], self.response([2], params=p, seed=87)['histogram'])
        self.assertTrue((response['histogram'] < 0).any())
        full = derive_readout(response, H2)
        scalar = scalar_projection(full)
        self.assertNotIn('histogram', scalar)
        for key in ('distance_m', 'status', 'ambient', 'valid'):
            self.assertIs(full[key], scalar[key])

    def test_h2_aggregation_preserves_total_mass(self):
        response = self.response([2])
        full = derive_readout(response, H2)
        np.testing.assert_allclose(full['histogram'].sum(), response['histogram'].sum())
        self.assertEqual(full['histogram'].shape, (4, 4, 24))
        self.assertEqual(derive_readout(response, H3)['histogram'].shape, (8, 8, 16))

    def test_pulse_and_leak_no_wrap_or_energy_creation(self):
        p = replace(self.p, pulse_sigma_bins=2, tail_mass=.2, neighbour_leak=.1)
        response = self.response([RAW_BIN_M/2], params=p)
        self.assertLess(response['histogram'].sum(), response['diagnostics']['raw_signal_energy'].sum())
        self.assertLess(response['histogram'][..., -1].max(), 1e-5)

    def test_batch_and_ideal_not_nir_weighted(self):
        distance = np.full((2, 8, 8, 3), 2.)
        response = synthesize_response(distance, .5, 1, 1, params=self.p)
        self.assertEqual(derive_readout(response, H2)['histogram'].shape, (2, 4, 4, 24))
        ideal = ideal_distribution(distance, 1)
        np.testing.assert_allclose(ideal['histogram'].sum(-1), 1)


if __name__ == '__main__':
    unittest.main()
