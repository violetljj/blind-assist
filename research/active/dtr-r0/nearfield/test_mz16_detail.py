import unittest
import numpy as np
import torch
from mz15_shared_support import LocalSupportReadout
from mz16_detail_readout import make_model
from mz16_local_metrics import average_precision,threshold_at_fpr,summarize_local


class DetailTests(unittest.TestCase):
    def test_roi_projection_and_physical_geometry(self):
        original=LocalSupportReadout(False);a=make_model(original.state_dict());b=make_model(original.state_dict())
        torch.testing.assert_close(a.grid,b.grid,rtol=0,atol=0)
        torch.testing.assert_close(a.rays,original.rays,rtol=0,atol=0)
        native=(original.grid+1)*torch.tensor([320.,180.])-.5
        reconstructed=(a.grid+1)*112-.5+torch.tensor([208.,68.])
        torch.testing.assert_close(native,reconstructed)
        r=torch.full((2,64,2),2.);v=torch.ones_like(r,dtype=torch.bool)
        self.assertTrue(torch.equal(a.eligibility(r,v)[2],original.eligibility(r,v)[2]))
        out=a.inspect(torch.randn(2,64,28,28),r,v)
        self.assertEqual(tuple(out['field'].shape),(2,64,2,49,4))
    def test_tied_average_precision_and_fpr_boundary(self):
        self.assertEqual(average_precision(np.array([1.,1.]),np.array([True,False])),.5)
        self.assertEqual(average_precision(np.array([2.,1.]),np.array([True,False])),1.)
        s=np.arange(100,dtype=np.float32);y=np.zeros(100,bool)
        threshold=threshold_at_fpr(s,y)
        self.assertEqual(int((s.astype(np.float64)>=threshold).sum()),1)
        self.assertEqual(summarize_local([s],[y],[threshold])['per_query'][0]['fp'],1)
        self.assertIsNone(average_precision(s,y))


if __name__=='__main__':unittest.main()
