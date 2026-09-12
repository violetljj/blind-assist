"""Independent projection and correspondence checks without task outcomes."""
import unittest
import cv2
import numpy as np
from mz106_temporal_geometry import FOCAL, current_to_past, geometric_contradiction, track


class TemporalGeometryTests(unittest.TestCase):
    def test_camera_axes_and_inverse(self):
        a=dict(x=1.,y=0.,z=1.7,yaw=0.,pitch=0.,roll=0.)
        b={**a,'x':0.}
        r,t=current_to_past(a,b)
        np.testing.assert_allclose(r,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(t,[0,0,1],atol=1e-12)
        a.update(y=.2,yaw=15.,pitch=-4.)
        r,t=current_to_past(a,b);ri,ti=current_to_past(b,a)
        np.testing.assert_allclose(ri@r,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(ri@t+ti,0,atol=1e-12)

    def test_near_vs_far_and_no_motion(self):
        xy=np.array([[500.,180.]])
        z=1.5;tz=.175
        # Physical forward translation: past image radius scales by z/(z+tz).
        near=np.array([[320+180*z/(z+tz),180.]])
        far=np.array([[320+180*10/(10+tz),180.]])
        for observed,expected in [(near,False),(far,True)]:
            rejected,usable,_=geometric_contradiction(xy,[z],observed,np.array([True]),np.eye(3),[0,0,tz])
            self.assertTrue(usable[0]);self.assertEqual(bool(rejected[0]),expected)
        rejected,usable,_=geometric_contradiction(xy,[z],far,np.array([True]),np.eye(3),[0,0,0])
        self.assertFalse(usable.any());self.assertFalse(rejected.any())
        rejected,usable,_=geometric_contradiction(xy,[z],far,np.array([False]),np.eye(3),[0,0,tz])
        self.assertFalse(usable.any());self.assertFalse(rejected.any())

    def test_optical_flow_direction_and_cycle(self):
        rng=np.random.default_rng(106)
        current=rng.integers(0,256,(160,240),dtype=np.uint8)
        past=cv2.warpAffine(current,np.float32([[1,0,4],[0,1,0]]),(240,160))
        xy=np.array([[100.,80.],[150.,90.]],dtype='float32')
        observed,valid,cycle,_=track(current,past,xy)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(observed,xy+[4,0],atol=.05)
        self.assertLess(cycle.max(),.05)


if __name__=='__main__':
    cv2.setNumThreads(2)
    unittest.main()
