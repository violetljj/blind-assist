"""Continuous projection enclosure and evidence fallback, without scene outcomes."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import mz107_rgb_association as geometry
import mz108_competitive_association as previous
import mz109_interval_extent as current
from test_mz107_rgb_association import fixture


def row_fixture():
    row, image = fixture()
    row['time_s'] = 0.
    return row, image


class IntervalExtent(unittest.TestCase):
    def test_envelope_contains_sampled_continuous_projection(self):
        row, _ = row_fixture(); intr = row['rgb_intrinsics']
        rng = np.random.default_rng(109)
        for index in range(12):
            u = tuple(sorted(rng.uniform(30,610,2)))
            v = tuple(sorted(rng.uniform(20,340,2)))
            ranges = tuple(sorted(rng.uniform(.4,4.,2)))
            pitches = tuple(sorted(rng.uniform(-10,10,2)))
            yaws = tuple(sorted(rng.uniform(-190,190,2)))
            bounds = current.projection_envelope(u,v,intr,ranges,pitches,yaws,1.7)
            self.assertIsNotNone(bounds)
            for _ in range(50):
                px,py,r,p,y = [rng.uniform(*interval) for interval in (u,v,ranges,pitches,yaws)]
                direction = geometry.pixel_ray(px,py,intr,p,y)
                point = r*direction/math.hypot(direction[0],direction[1])+[0.,0.,1.7]
                for value,(lo,hi) in zip(point,bounds):
                    self.assertGreaterEqual(value,lo-1e-12)
                    self.assertLessEqual(value,hi+1e-12)

    def test_yaw_interior_extremum_prevents_vertex_only_inside_claim(self):
        row, _ = row_fixture(); intr = row['rgb_intrinsics']; r = 3.605
        for yaw in (-4.,4.):
            self.assertTrue(geometry.point_inside(r*geometry.ray(0,0,0,yaw)+[0,0,1.7]))
        self.assertFalse(geometry.point_inside(np.array([r,0,1.7])))
        bounds = current.projection_envelope((320,320),(180,180),intr,(r,r),(0,0),(-4,4),1.7)
        self.assertAlmostEqual(bounds[0][1],r)
        self.assertFalse(current.enclosed(bounds))

    def test_inside_outside_and_corridor_corner_ambiguity(self):
        row, _ = row_fixture(); row['camera_pitch_deg'] = 0.
        inside = current.classify([300,150,340,210],2.,row,0.)
        self.assertEqual(inside['state'],'IN')
        self.assertTrue(current.enclosed(inside['witness_xyz']))
        u,v = inside['witness_pixel']
        self.assertTrue(302 <= u <= 338 and 152 <= v <= 208)
        outside = current.classify([450,140,500,220],2.5,row,0.)
        self.assertEqual(outside['state'],'OUT')
        self.assertTrue(current.disjoint(outside['outer_xyz']))
        self.assertEqual(current.classify([353,170,363,190],3.61,row,0.)['state'],'AMBIGUOUS')

    def test_thin_or_invalid_and_nonforward_are_ambiguous(self):
        row, _ = row_fixture()
        for box in ([320,140,324,200],[324,140,320,200],[300,150,340,154]):
            self.assertEqual(current.classify(box,2.5,row,0.)['state'],'AMBIGUOUS')
        row['camera_pitch_deg'] = 90.
        self.assertEqual(current.classify([300,150,340,210],2.5,row,0.)['state'],'AMBIGUOUS')
        self.assertIsNone(current.projection_envelope((320,320),(180,180),row['rgb_intrinsics'],
            (0,.2),(0,0),(0,0),1.7))

    def test_ambiguous_falls_back_to_three_sensor_not_nominal_and_does_not_mutate(self):
        row, image = row_fixture()
        nominal = previous.predict_frame(row,image,0.,use_regions=False,competitive=False)
        self.assertTrue(nominal['baseline']); self.assertFalse(nominal['candidate'])
        before = copy.deepcopy(nominal)
        with patch.object(current,'classify',return_value=dict(state='AMBIGUOUS')):
            refined = current.refine(row,nominal)
        self.assertTrue(refined['candidate'])
        self.assertEqual(nominal,before)
        inverse = copy.deepcopy(nominal)
        inverse['associations'][0].update(baseline_support=False,refined_support=True)
        inverse.update(baseline=False,candidate=True)
        with patch.object(current,'classify',return_value=dict(state='AMBIGUOUS')):
            self.assertFalse(current.refine(row,inverse)['candidate'])

    def test_independent_tof_survives_out_and_rgb_disabled_is_baseline(self):
        row, image = row_fixture()
        for key,value in [('tof64_range_m',2.),('tof64_status',5),('tof64_theta_deg',0.),('tof64_phi_deg',0.)]:
            row[key].append(value)
        nominal = previous.predict_frame(row,image,0.,use_regions=False,competitive=False)
        with patch.object(current,'classify',return_value=dict(state='OUT')):
            refined = current.refine(row,nominal)
        self.assertTrue(refined['tof_support'] and refined['candidate'])
        disabled = previous.predict_frame(row,None,0.)
        refined = current.refine(row,disabled)
        self.assertEqual(refined['candidate'],disabled['baseline'])
        self.assertTrue(all(a['geometry']['state']=='UNAVAILABLE' for a in refined['associations']))


if __name__ == '__main__':
    unittest.main()
