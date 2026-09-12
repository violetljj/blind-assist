import unittest
import numpy as np
import mz102_stereo_witness as w

POSE=dict(yaw=0,pitch=0,roll=0,camera_in_body_m=[0,0,1.7])


class WitnessTests(unittest.TestCase):
    def test_thin_vertical_support_is_kept(self):
        d=np.full((360,640),np.nan,np.float32);d[130:330,319:321]=2.
        counts,_,_=w.qualify(d,POSE)
        self.assertTrue((counts>0).all())

    def test_tiny_near_ghost_rejected(self):
        d=np.full((360,640),np.nan,np.float32);d[170:178,320]=1.3
        c,_,_=w.qualify(d,POSE);self.assertFalse(c.any())

    def test_outside_surface_cannot_sponsor_query(self):
        d=np.full((360,640),np.nan,np.float32);d[146,280:640]=2.
        # At z1.849m HEAD top, inside query width .36 permits a real wide patch;
        # move component until only2pixels remain inside its right boundary.
        d[:]=np.nan;d[180,360:640]=2.
        c,_,_=w.qualify(d,POSE);self.assertEqual(c[1],0)

    def test_depth_oscillation_not_length(self):
        d=np.full((360,640),np.nan,np.float32);d[180:187,320]=np.linspace(2.,2.4,7)
        c,_,_=w.qualify(d,POSE);self.assertFalse(c.any())

    def test_perturbation_separately_removes_boundary_surface(self):
        d=np.full((360,640),np.nan,np.float32);d[170:185,300:340]=3.10
        c,_,_=w.qualify(d,POSE);p,_,_=w.qualify(d,POSE,perturbation=True)
        self.assertGreater(c[1],0);self.assertEqual(p[1],0)

    def test_empty_is_unknown(self):
        c,_,mask=w.qualify(np.full((360,640),np.nan,np.float32),POSE,perturbation=True)
        self.assertFalse(c.any());self.assertFalse(mask.any())


if __name__=='__main__':unittest.main()
