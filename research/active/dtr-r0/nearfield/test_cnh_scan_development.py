"""Synthetic scanner mathematics tests, never dataset access."""
import unittest
import numpy as np
from cnh_scan_development import (shift_matrix,accumulate,aggregate_h3,box_sum,scan,fit_bias,
                                  paired_bootstrap,select_taus,TAUS)


class ScannerTests(unittest.TestCase):
    def test_interpolation_variance_covariance(self):
        a=shift_matrix(4,.25);variance=np.array([2.,3.,5.,7.])
        full=a@np.diag(variance)@a.T
        np.testing.assert_allclose((a*a)@variance,np.diag(full))
        np.testing.assert_allclose((a[:-1]*a[1:])@variance,np.diag(full,1))
        self.assertNotEqual(float((a@variance)[0]),float(full[0,0]))
        coeff=a[:2].sum(0)
        self.assertAlmostEqual(float(coeff**2@variance),float(full[:2,:2].sum()))

    def test_raw_aggregation_covariance(self):
        rng=np.random.default_rng(5);v=rng.uniform(1,3,128)
        a=shift_matrix(128,.4);full=a@np.diag(v)@a.T
        diagonal=np.diag(full)[None,None];cov=np.diag(full,1)[None,None]
        _,h3v,h3c=aggregate_h3(np.zeros((1,1,128)),diagonal,cov)
        b=np.eye(16).repeat(8,axis=1)
        expected=b@full@b.T
        np.testing.assert_allclose(h3v[0,0],np.diag(expected))
        np.testing.assert_allclose(h3c[0,0],np.diag(expected,1))

    def test_box_sum(self):
        x=np.arange(2*4*5*6).reshape(2,4,5,6)
        y=box_sum(x,(2,3,2))
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    np.testing.assert_array_equal(y[:,i,j,k],x[:,i:i+2,j:j+3,k:k+2].sum((1,2,3)))

    def test_causal_and_clips(self):
        rng=np.random.default_rng(1);r=rng.normal(size=(8,64,16));v=np.ones_like(r)
        data=dict(clip_ids=np.array(['a']*4+['b']*4),steps=np.tile(np.arange(4),2))
        a=accumulate(r,v,data,corrected=True)
        changed=r.copy();changed[3:]*=100
        b=accumulate(changed,v,data,corrected=True)
        np.testing.assert_array_equal(a[0][:3],b[0][:3])
        np.testing.assert_array_equal(a[0][4],r[4])

    def test_train_bias_and_selection(self):
        hist=np.arange(6*64*16).reshape(6,64,16);tr=np.array([True]*3+[False]*3)
        bias=fit_bias(hist,tr,'train_median');hist[~tr]*=100
        np.testing.assert_array_equal(bias,fit_bias(hist,tr,'train_median'))
        y=np.tile([0,1,0,1,0,1],(6,1))
        scores={f'{fam}_tau{tau}':y.astype(float) for fam in ('S1_K1','S2_K4_LITERAL','S2_K4_CORRECTED') for tau in TAUS}
        for item in select_taus(y,scores,tr).values():self.assertEqual(item['tau'],.25)

    def test_fixed_window_null_not_max_normal(self):
        rng=np.random.default_rng(20260925)
        z=(rng.poisson(128,30000)-rng.poisson(128,30000))/16
        self.assertLess(abs(z.mean()),.03)
        self.assertLess(abs(z.std()-1),.03)
        # Deliberately no N(0,1) assertion for maximization over windows.

    def test_bootstrap_identity(self):
        y=np.tile([0,1,0,1,0,1],(12,1));s=y*.2+.1
        report=paired_bootstrap(y,s,s,np.repeat(['a','b','c'],4),draws=20)
        self.assertEqual(report['delta_ap'],0)
        self.assertEqual(report['ci95'],[0,0])


if __name__=='__main__':unittest.main()
