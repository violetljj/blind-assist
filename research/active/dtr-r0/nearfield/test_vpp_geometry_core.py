import hashlib
import math
from pathlib import Path
import unittest
import numpy as np
from vpp_geometry_core import Projector, ray_hints, frame_seed

RIG = dict(width=640,height=360,hfov_deg=70.,baseline_m=.1,tof_hfov_deg=45.)
SOURCE = Path(__file__).resolve().parents[4]/'artifacts.local/work/vpp-geometry-20260923/vppstereo'


class HintTests(unittest.TestCase):
    def test_radial_depth_conversion(self):
        hints,seeds=ray_hints(np.full(64,2.),np.ones(64,bool),RIG)
        self.assertEqual(len(seeds),64);self.assertEqual(np.count_nonzero(hints),64)
        for s in seeds:
            x,y,z=s['xyz_camera'];self.assertAlmostEqual(math.sqrt(x*x+y*y+z*z),2.)
            self.assertLess(x,2.)
            self.assertAlmostEqual(s['disparity'],320/math.tan(math.radians(35))*.1/x,places=5)
            self.assertLessEqual(abs(s['u']-s['u_float']),.5)
            self.assertLessEqual(abs(s['v']-s['v_float']),.5)

    def test_missing_and_out_of_range_not_hints(self):
        r=np.array([np.nan,np.inf,-1,.49,4.01,1.]+[0.]*58)
        valid=np.ones(64,bool);valid[5]=False
        hints,seeds=ray_hints(r,valid,RIG)
        self.assertEqual(seeds,[]);self.assertEqual(np.count_nonzero(hints),0)

    def test_frame_seed_reproducible(self):
        self.assertEqual(frame_seed('mz101','x'),frame_seed('mz101','x'))
        self.assertNotEqual(frame_seed('mz101','x'),frame_seed('mz102','x'))


@unittest.skipUnless((SOURCE/'vpp_standalone.py').exists(),'Frozen external source required')
class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projector=Projector(SOURCE,hashlib.sha256((SOURCE/'vpp_standalone.py').read_bytes()).hexdigest())

    def test_zero_hints_byte_identity(self):
        rng=np.random.default_rng(13);a=rng.integers(0,256,(360,640,3),dtype=np.uint8);b=a[:,::-1].copy()
        left,right=self.projector.apply(a,b,np.zeros((360,640),np.float32),7)
        self.assertEqual(left.tobytes(),a.tobytes());self.assertEqual(right.tobytes(),b.tobytes())

    def test_seeded_projection_repeat_and_integer_correspondence(self):
        a=np.full((360,640,3),128,np.uint8);h=np.zeros((360,640),np.float32);h[180,320]=10
        l,r=self.projector.apply(a,a,h,42);l2,r2=self.projector.apply(a,a,h,42)
        np.testing.assert_array_equal(l,l2);np.testing.assert_array_equal(r,r2)
        np.testing.assert_array_equal(l[179:182,319:322],r[179:182,309:312])
        self.assertGreater(np.count_nonzero(l!=a),0)
        self.assertLessEqual(np.any(l!=a,axis=2).sum(),9)
        self.assertTrue((a==128).all())


if __name__=='__main__':unittest.main()
