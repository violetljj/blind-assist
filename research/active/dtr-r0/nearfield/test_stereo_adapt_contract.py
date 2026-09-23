"""CPU-only target quantization, partition and loss-gradient contract fixtures."""
import unittest
import warnings

import numpy as np
import torch

from stereo_adapt_loss import targets,loss_weights,sequence_loss


class StereoAdaptContractTests(unittest.TestCase):
    def test_float32_discontinuity_threshold_is_not_rounded_cm(self):
        z=np.full((360,640),2.,np.float32)
        below=np.float32(2.05)
        self.assertLess(float(below-np.float32(2)),.05)
        z[180,321]=below
        self.assertEqual(targets(z)[2][180,320],1)
        z[180,321]=np.nextafter(below,np.float32(np.inf))
        self.assertGreater(float(z[180,321]-np.float32(2)),.05)
        self.assertEqual(targets(z)[2][180,320],0)

    def test_four_metre_edge_and_far_precedence(self):
        z=np.full((360,640),4.,np.float32)
        d,v,r=targets(z)
        self.assertTrue(v[180,320]);self.assertNotEqual(r[180,320],3)
        z[180,320]=np.nextafter(np.float32(4),np.float32(np.inf))
        # All >4m valid pixels stay far even adjacent to near/missing values.
        z[179,320]=np.nan
        d,v,r=targets(z)
        self.assertTrue(v[180,320]);self.assertEqual(r[180,320],3)

    def test_projection_mask_uses_unrounded_disparity(self):
        z=np.full((360,640),2.,np.float32)
        d,v,r=targets(z)
        # Actual disparity lies between22 and23; integer rounding would admit22.
        self.assertTrue(22<float(d[180,22])<23)
        self.assertFalse(v[180,22]);self.assertTrue(v[180,23])
        self.assertEqual(r[180,22],-1)

    def test_invalid_centres_never_become_supervision(self):
        z=np.full((360,640),2.,np.float32)
        z[180,315:320]=[np.nan,np.inf,0.,-1.,1e-8]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            d,v,r=targets(z)
        self.assertFalse(v[180,315:320].any())
        self.assertTrue(np.all(r[180,315:320]==-1))
        np.testing.assert_array_equal(r>=0,v)
        for mode in ('ordinary','balanced'):
            w=loss_weights(v,r,mode)
            self.assertTrue(np.all(w[~v]==0))
            self.assertTrue(np.isfinite(w).all())

    def test_nonempty_region_masses_match_half_uniform_half_equal(self):
        r=np.array([[0,1,2,2,3,3,3,3,-1]],np.int8);v=r>=0
        ordinary=loss_weights(v,r,'ordinary');balanced=loss_weights(v,r,'balanced')
        for k,n in enumerate((1,1,2,4)):
            self.assertAlmostEqual(float(ordinary[r==k].sum()),n/8)
            self.assertAlmostEqual(float(balanced[r==k].sum()),.5*n/8+.5/4)
        single=np.where(v,3,-1)
        np.testing.assert_allclose(loss_weights(v,single,'balanced'),ordinary)

    def test_sequence_gradient_is_nonzero_on_valid_and_zero_on_missing(self):
        target=torch.zeros((2,3),device='cpu')
        weights=torch.tensor([[.2,.3,0.],[.1,.4,0.]],device='cpu')
        first=torch.ones((2,3),device='cpu',requires_grad=True)
        last=torch.full((2,3),2.,device='cpu',requires_grad=True)
        loss=sequence_loss([first,last],target,weights)
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()),(.9+2)/1.9,places=6)
        torch.testing.assert_close(first.grad,weights*(.9/1.9))
        torch.testing.assert_close(last.grad,weights*(1/1.9))
        self.assertGreater(float(first.grad.abs().sum()),0.)
        self.assertGreater(float(last.grad.abs().sum()),0.)
        self.assertEqual(float(last.grad[:,2].sum()),0.)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
