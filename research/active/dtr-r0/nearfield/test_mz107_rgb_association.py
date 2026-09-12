"""Focused geometry and evidence-preservation checks, no simulator outcomes."""
import unittest
import math
import numpy as np
import mz107_rgb_association as m
from mz107_sensors import basis


def fixture():
    intr = dict(width=640,height=360,fx=457.,fy=457.,cx=320.,cy=180.)
    row = dict(rgb_intrinsics=intr,camera_pitch_deg=-3.,camera_in_body_m=[0,0,1.7],
        tof_packet_received=True,tof64_range_m=[2.5],tof64_status=[5],
        tof64_theta_deg=[11.25],tof64_phi_deg=[2.8],
        radar_packet_received=True,radar_range_m=[2.5],radar_angle=[0.],radar_valid=[True])
    image = np.zeros((360,640,3),np.uint8)
    image[110:230,400:420] = 220
    return row, image


class Geometry(unittest.TestCase):
    def test_source_pinhole_rays(self):
        for yaw in (-12,0,12):
            f,r,u = map(np.asarray,basis(dict(pitch=-3,yaw=yaw)))
            for a in np.arange(8)*5.625-19.6875:
                for e in np.arange(8)*5.625-19.6875:
                    source = f+math.tan(math.radians(a))*r+math.tan(math.radians(e))*u
                    np.testing.assert_allclose(m.ray(a,e,-3,yaw),source/np.linalg.norm(source),atol=1e-14)

    def test_real_pixels_can_refine_offroute(self):
        row,image=fixture();p=m.predict_frame(row,image,0)
        self.assertTrue(p['baseline']); self.assertFalse(p['candidate'])
        self.assertEqual(p['associations'][0]['state'],'ASSOCIATED')

    def test_missing_rgb_is_fallback(self):
        row,_=fixture();p=m.predict_frame(row,None,0)
        self.assertEqual(p['candidate'],p['baseline'])
        self.assertEqual(p['associations'][0]['state'],'UNKNOWN')

    def test_conflicting_range_is_fallback(self):
        row,image=fixture();row['tof64_range_m']=[3.8]
        p=m.predict_frame(row,image,0)
        self.assertTrue(p['candidate']);self.assertEqual(p['associations'][0]['state'],'UNKNOWN')

    def test_independent_tof_survives(self):
        row,image=fixture()
        for key,value in [('tof64_range_m',2.),('tof64_status',5),('tof64_theta_deg',0.),('tof64_phi_deg',0.)]:
            row[key].append(value)
        p=m.predict_frame(row,image,0)
        self.assertTrue(p['tof_support']);self.assertTrue(p['candidate'])

    def test_empty_sensors_unknown(self):
        row,image=fixture();row['tof_packet_received']=False;row['radar_packet_received']=False
        p=m.predict_frame(row,image,0)
        self.assertFalse(p['candidate']);self.assertEqual(p['candidate_state'],'UNKNOWN')


if __name__=='__main__': unittest.main()
