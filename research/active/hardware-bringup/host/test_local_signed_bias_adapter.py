"""Focused v4 contract tests; synthetic arrays, no protected or pilot data."""
import unittest
import numpy as np
from local_signed_bias_adapter import real_adapter,sim_adapter,native_evidence,history_support


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.b=np.zeros((8,8,16));self.b[...,0]=10;self.b[...,1:4]=-4
        self.a=np.ones((16,8,8))*4
        self.noise=dict(adapter_coefficients_a_b_c=[.14581365856,.08966464374,.0812905902],numerical_variance_floor=1e-9)

    def adapt(self,h,b=None):return real_adapter(h,self.a,self.b if b is None else b,self.noise)

    def test_zero_physical_and_signed_background(self):
        for h in (np.zeros((16,8,8,16)),np.broadcast_to(np.maximum(self.b,0),(16,8,8,16)),np.broadcast_to(self.b,(16,8,8,16))):
            z,positive=native_evidence(self.adapt(h));self.assertFalse(positive.any());self.assertEqual(np.count_nonzero(z),0)

    def test_no_positive_from_negative_bias(self):
        z=self.adapt(np.zeros((16,8,8,16)))
        self.assertEqual(np.count_nonzero(z),0)

    def test_cap_and_true_surface(self):
        h=np.zeros((16,8,8,16));h[...,0]=5;h[...,4]=30;h[...,5]=-2
        z=self.adapt(h)
        self.assertEqual(np.count_nonzero(z[...,0]),0)
        self.assertTrue((z[...,4]>0).all());self.assertTrue((z[...,5]<0).all())
        z2=self.adapt(h,np.minimum(self.b,0));self.assertTrue((z2[...,0]>0).all())

    def test_inherited_mask_and_negative_only(self):
        h=np.zeros((16,8,8,16));h[...,0]=1e4
        z,positive=native_evidence(h,(0,));self.assertFalse(positive.any());self.assertEqual(np.count_nonzero(z),0)
        z,positive=native_evidence(-np.ones_like(h));self.assertFalse(positive.any());self.assertEqual(np.count_nonzero(z),0)

    def test_sim_signed_noise_is_not_relu(self):
        b=np.maximum(self.b,0);h=np.arange(16*1024,dtype=np.float32).reshape(16,8,8,16)%17-8
        expected=((h-b)/np.sqrt(16*self.a[...,None]+b)).astype(np.float16)
        np.testing.assert_array_equal(sim_adapter(h,self.a,b),expected)
        for raw in (np.zeros_like(h),np.broadcast_to(b,h.shape)):
            self.assertEqual(np.count_nonzero(sim_adapter(raw,self.a,b)),0)

    def test_causal_history_and_reset(self):
        p=np.zeros(17,bool);p[0]=True
        lengths=np.minimum(np.arange(17)+1,8);expected=np.arange(17)<8
        np.testing.assert_array_equal(history_support(p,lengths),expected)
        lengths[5:]=np.minimum(np.arange(12)+1,8)
        self.assertFalse(history_support(p,lengths)[5:].any())

    def test_invalid_is_unknown(self):
        with self.assertRaises(ValueError):native_evidence(np.full((16,8,8,16),np.nan))
        with self.assertRaises(ValueError):native_evidence(np.zeros((16,8,8,16)),(1,))


if __name__=='__main__':unittest.main()
