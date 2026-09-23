import unittest
import numpy as np
from stereo_adapt_loss import targets,loss_weights

class LossTests(unittest.TestCase):
    def test_far_is_training_target(self):
        z=np.full((360,640),12.,np.float32);d,v,r=targets(z)
        self.assertTrue(v[180,320]);self.assertEqual(r[180,320],3)
        self.assertGreater(d[180,320],0)
    def test_missing_and_out_of_right_image_excluded(self):
        z=np.full((360,640),2.,np.float32);z[180,320]=np.nan
        d,v,r=targets(z);self.assertFalse(v[180,320]);self.assertFalse(v[180,0])
        self.assertEqual(r[180,320],-1)
    def test_boundary_precedence_and_balancing(self):
        z=np.full((360,640),12.,np.float32);z[190:195,318:323]=2
        d,v,r=targets(z);self.assertEqual(r[190,320],0);self.assertEqual(r[192,320],1)
        a=loss_weights(v,r,'ordinary');b=loss_weights(v,r,'balanced')
        self.assertAlmostEqual(float(a.sum()),1,places=5);self.assertAlmostEqual(float(b.sum()),1,places=5)
        self.assertGreater(float(b[r==0].sum()),float(a[r==0].sum()))
        self.assertGreater(float(b[r==3].sum()),.49)

if __name__=='__main__':unittest.main()
