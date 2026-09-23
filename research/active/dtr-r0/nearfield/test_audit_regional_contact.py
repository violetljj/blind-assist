"""Regression fixtures for independent pooling and stride-sensitive reductions."""
import unittest
import numpy as np
from audit_regional_contact import independent_pool


class AuditRegionalTests(unittest.TestCase):
    def test_bilinear_pool_on_affine_feature_grid(self):
        y,x=np.mgrid[:8,:14]
        visual=np.stack([x+2*y+c for c in range(40)]).astype(np.float32)
        raw=np.zeros((1,4864),np.float32);raw[0,:4480]=visual.ravel()
        sensor=raw[:,4480:].reshape(1,64,6);sensor[:,:,2:]=[0,0,1,1]
        sensor[:,1,2:]=[0,.25,.5,.75]
        pooled=independent_pool(raw)
        np.testing.assert_allclose(pooled[0,0],np.arange(40)+13.5,atol=1e-12)
        np.testing.assert_allclose(pooled[0,1],np.arange(40)+9.5,atol=1e-12)

    def test_layout_reduction_roundoff_does_not_change_moments(self):
        rgb=(np.random.default_rng(816).normal(size=(128,64,40))+.2).astype(np.float32)
        strided=rgb.transpose(0,2,1).copy().transpose(0,2,1)
        np.testing.assert_array_equal(rgb,strided)
        self.assertNotEqual(rgb.strides,strided.strides)
        cmean=rgb.mean(axis=(0,1));original_mean=strided.mean(axis=(0,1))
        self.assertFalse(np.array_equal(cmean,original_mean))
        reference=rgb.astype(float).mean(axis=(0,1))
        np.testing.assert_allclose(reference,original_mean,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(reference,cmean,atol=2e-6,rtol=2e-6)
        restored=np.ascontiguousarray(strided).transpose(0,2,1).copy().transpose(0,2,1)
        np.testing.assert_array_equal(restored.mean(axis=(0,1)),original_mean)
        np.testing.assert_array_equal(restored.std(axis=(0,1)),strided.std(axis=(0,1)))


if __name__=='__main__':unittest.main()
