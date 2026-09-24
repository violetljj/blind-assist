import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from cnh_street_development_baseline import sample_depth, fixed_peaks, load_depth
from cnh_route_sensor import angular_rays, SensorParameters
from dataclasses import asdict


class DevelopmentBaselineTests(unittest.TestCase):
    def test_canonical_depth_uses_validity_without_transport(self):
        class FakeExr:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def channels(self): return {'Z':SimpleNamespace(pixels=np.array([[1., 2.]], dtype=np.float32))}
        with patch.object(Path, 'exists', return_value=True), patch.dict('sys.modules', {'OpenEXR':SimpleNamespace(File=lambda *a,**k:FakeExr())}), patch('numpy.load', return_value=np.array([[True, False]])):
            depth, path = load_depth(Path('unused-fixture'))
        self.assertEqual(path.name, 'depth_left.exr')
        self.assertEqual(depth[0, 0], 1.)
        self.assertTrue(np.isnan(depth[0, 1]))

    def test_axial_conversion_and_missing(self):
        camera = dict(width=640, height=360, K=[[400, 0, 320], [0, 400, 180], [0, 0, 1]])
        radial, weights = sample_depth(np.full((360, 640), 2.), camera)
        rays, expected_weights = angular_rays(16)
        x = np.rint(400*rays[..., 0]/rays[..., 2]+320)
        y = np.rint(400*rays[..., 1]/rays[..., 2]+180)
        np.testing.assert_allclose(radial, 2*np.sqrt(1+((x-320)/400)**2+((y-180)/400)**2))
        np.testing.assert_allclose(weights, expected_weights)
        missing, _ = sample_depth(np.zeros((360, 640)), camera)
        self.assertTrue(np.isnan(missing).all())

    def test_local_peaks_not_adjacent_plateau_or_noise(self):
        h = np.zeros((8, 8, 16)); h[..., 4:6] = 100; h[..., 10] = 80; h[..., 14] = 1
        readout = dict(histogram=h, ambient=np.zeros((8, 8)), bin_centers_m=np.arange(16)*.3)
        peaks, valid = fixed_peaks(readout, asdict(SensorParameters()))
        np.testing.assert_allclose(peaks[..., 0], 1.2)
        np.testing.assert_allclose(peaks[..., 1], 3.)
        self.assertTrue(valid.all())
        readout['histogram'][:] = 0
        peaks, valid = fixed_peaks(readout, asdict(SensorParameters()))
        self.assertFalse(valid.any()); self.assertTrue(np.isnan(peaks).all())


if __name__ == '__main__':
    unittest.main()
