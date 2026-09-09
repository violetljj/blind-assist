"""Engineering-only artificial geometry checks, not sensor validation."""
import unittest
import numpy as np
import torch
from body_query_tof_coverage import footprints
from tof_simulated_packets import LAWS, disturbances, measure, packets, range_histogram


class SimulatedPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA engineering test device unavailable')
        cls.device = 'cuda'
        cls.footprints = footprints(cls.device)

    def test_frontal_plane_returns_radial_not_axial(self):
        scene = torch.full((1,360,640), 2.09, device=self.device)
        for fov in (15,27):
            mask, weights, norms = self.footprints[fov]
            expected, _ = np.histogram(
                float(scene[0,0,0]) * norms.cpu().numpy().astype(np.float64),
                bins=np.linspace(.1,4.,40), weights=weights.cpu().numpy().astype(np.float64))
            expected /= weights.double().sum().item()
            supported = np.flatnonzero(expected >= .02)
            output = measure(scene, self.footprints[fov])
            for law, index in zip(LAWS, (supported[0], expected.argmax(), supported[-1])):
                distance, valid = output[law]
                self.assertTrue(valid.item())
                self.assertAlmostEqual(distance.item(), .15 + .1*index, places=10)
            # Radial projection crosses a bin edge even on a frontal plane.
            mass = range_histogram(scene, self.footprints[fov])
            np.testing.assert_allclose(mass.cpu().numpy()[0], expected, atol=1e-12)
            self.assertAlmostEqual(mass.sum().item(), 1., places=10)
            self.assertGreater(len(supported), 1)

    def test_two_depth_laws_and_full_footprint_support(self):
        mask, weights, norm = self.footprints[27]
        scene = torch.full((1,360,640), float('nan'), device=self.device)
        radial = torch.full_like(norm, 2.55)
        radial[:len(norm)//4] = 1.25
        scene[:,mask] = radial/norm
        result = measure(scene, self.footprints[27])
        self.assertAlmostEqual(result[LAWS[0]][0].item(), 1.25, places=10)
        self.assertAlmostEqual(result[LAWS[1]][0].item(), 2.55, places=10)
        self.assertAlmostEqual(result[LAWS[2]][0].item(), 2.55, places=10)
        # One finite pixel must not become100% after renormalizing known pixels.
        scene[:,mask] = float('nan')
        iy, ix = mask.nonzero()[0].tolist()
        scene[0,iy,ix] = 1.25/norm[0]
        self.assertFalse(measure(scene,self.footprints[27])[LAWS[0]][1].item())

    def test_no_supported_input_invalid(self):
        scenes = torch.stack([torch.full((360,640), value, device=self.device)
                              for value in (float('nan'),0.,-1.,float('inf'),5.)])
        for value, valid in measure(scenes,self.footprints[15]).values():
            self.assertFalse(valid.any().item())
            self.assertTrue(torch.isnan(value).all().item())

    def test_common_deterministic_disturbance_and_packet_isolation(self):
        noise, dropout = disturbances(200)
        noise2, dropout2 = disturbances(200)
        np.testing.assert_array_equal(noise,noise2)
        np.testing.assert_array_equal(dropout,dropout2)
        self.assertTrue(dropout.any() and (~dropout).any())
        self.assertTrue(np.all(np.abs(noise)<=.05))
        supported=np.ones(200,dtype=bool);supported[0]=False
        ids=[f'opaque{i}' for i in range(200)]
        a=packets(ids,np.full(200,1.25),supported,noise,dropout,15)
        b=packets(ids,np.full(200,2.55),supported,noise,dropout,27)
        self.assertIsNone(a[0]['range_m'])
        for x,y in zip(a,b):
            self.assertEqual(x['valid'],y['valid'])
            self.assertEqual(set(x),{'sample_id','range_m','valid','diagonal_fov_deg','range_uncertainty_m','delta_ms','source'})
            if x['valid']:self.assertAlmostEqual(y['range_m']-x['range_m'],1.3)
        # Opaque ID does not enter the range/noise calculation.
        c=packets(['other']*200,np.full(200,1.25),supported,noise,dropout,15)
        self.assertEqual([x['range_m'] for x in a],[x['range_m'] for x in c])


if __name__=='__main__':
    unittest.main()
