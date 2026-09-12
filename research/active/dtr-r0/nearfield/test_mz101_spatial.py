import unittest
import numpy as np
import mz101_spatial as m


class SpatialTests(unittest.TestCase):
    def test_stereo_known_disparity(self):
        rng=np.random.default_rng(11)
        left=rng.integers(0,256,(360,640,3),dtype=np.uint8)
        right=np.zeros_like(left);right[:,:-16]=left[:,16:]
        depth,_=m.stereo_depth(left,right)
        expected=640/(2*np.tan(np.radians(35)))*.1/16
        crop=depth[20:-20,140:480]
        self.assertGreater(np.isfinite(crop).mean(),.95)
        self.assertLess(abs(np.nanmedian(crop)-expected),.02)

    def test_no_depth_for_featureless_pair(self):
        im=np.zeros((360,640,3),np.uint8)
        d,_=m.stereo_depth(im,im)
        self.assertEqual(np.isfinite(d).sum(),0)

    def test_body_head_and_unknown(self):
        pose=dict(yaw=0,pitch=0,camera_in_body_m=[0,0,1.7])
        counts,_=m.readout(np.array([[2,0,0],[2,0,-.6],[2,.6,0],[2,0,.5]]),pose)
        np.testing.assert_array_equal(counts,[1,1])
        empty,_=m.readout(np.zeros((0,3)),pose)
        np.testing.assert_array_equal(empty,[0,0])

    def test_rotation_rig(self):
        r=m.rotation(12,-7,3)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(m.rotation(90,0)@np.array([1,0,0]),[0,1,0],atol=1e-12)
        self.assertGreater((m.rotation(0,10)@np.array([1,0,0]))[2],0)

    def test_hysteresis_and_episode_reset(self):
        q=np.array([[1],[1],[0],[0],[1],[1]])
        pred=m.hysteresis(q,['a','a','a','a','a','b'])
        np.testing.assert_array_equal(pred[:,0],[0,1,1,0,0,0])

    def test_tof_packet_and_uninflated_boundary(self):
        p=m.tof_points(np.ones(64)*2,np.ones(64,bool))
        self.assertEqual(p.shape,(64,3))
        np.testing.assert_allclose(np.linalg.norm(p,axis=1),2)
        pose=dict(yaw=0,pitch=0,camera_in_body_m=[0,0,1.7])
        # Point remains above HEAD; voxel bin floor must not inflate support.
        c,_=m.readout([[2,0,.16]],pose)
        np.testing.assert_array_equal(c,[0,0])


if __name__=='__main__':unittest.main()
