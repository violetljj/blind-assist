"""Analytic fixture checks for frozen-output contact decomposition."""
import unittest
import warnings

import numpy as np

from contact_decomposition import quantile, crossing, focus, decompose


class ContactDecompositionTests(unittest.TestCase):
    def test_quantile_inverts_normalized_cdf(self):
        mu=np.array([1.,1.5,2.]);scale=np.array([.2,.3,.4]);p=np.array([.2,.5,.8])
        z=quantile(mu,scale,p)
        sigmoid=lambda x:1/(1+np.exp(-x))
        recovered=sigmoid((z-mu)/scale)/sigmoid((3-mu)/scale)
        np.testing.assert_allclose(recovered,p,atol=1e-12)

    def test_unit_quantile_and_left_clamp(self):
        np.testing.assert_array_equal(quantile([.3,1.,2.99],[.01,.3,100.],1.),[3.,3.,3.])
        self.assertEqual(float(quantile(.3,.2,.001)),.3)

    def test_crossing_cutoff_and_censoring(self):
        z=crossing(np.array([.5,.4,0.,1.]),1.5,.2,.5)
        self.assertEqual(z[0],3.)
        self.assertTrue(np.isinf(z[1:3]).all())
        np.testing.assert_allclose(z[3],quantile(1.5,.2,.5))
        self.assertTrue(np.isinf(crossing(np.array([0.,.5,1.]),1.5,.2,1.000001)).all())

    def test_partition_and_q_unity_false_crossings(self):
        q=np.array([.2,.2,.9,.9,.9]);mu=np.ones(5);scale=np.full(5,.1)
        truth=np.array([1.,2.,1.,2.,np.inf])
        result=focus(q,mu,scale,truth,.5)
        self.assertEqual(result['partition'],dict(blocked_median_accurate=1,blocked_median_inaccurate=1,
            admitted_median_accurate=1,admitted_median_inaccurate=1))
        self.assertEqual(sum(result['partition'].values()),4)
        self.assertEqual(result['actual']['missing'],2)
        self.assertEqual(result['conditional_median']['within5cm'],2)
        self.assertEqual(result['false_admissions'],1)
        self.assertEqual(result['q_unity_false_crossings'],1)
        self.assertEqual(result['q_unity_same_cutoff_diagnostic']['resolved'],4)

    def test_all_negative_cutoff_diagnostic(self):
        result=focus(np.array([.2,1.]),np.ones(2),np.full(2,.1),np.array([1.,np.inf]),1.000001)
        self.assertEqual(result['actual']['missing'],1)
        self.assertEqual(result['false_admissions'],0)
        self.assertEqual(result['q_unity_false_crossings'],0)
        self.assertEqual(result['q_unity_same_cutoff_diagnostic']['resolved'],0)

    def test_width_violations_and_truth_checks(self):
        widths=np.array([.4,.6,.8])
        q=np.array([[[.8,.3,.9],[.3,.4,.5]]])
        mu=np.array([[[1.,1.3,1.],[1.2,1.1,1.]]]);scale=np.full_like(q,.05)
        truth=np.array([[[1.5,1.4,1.3],[np.inf,1.1,1.]]])
        shape=decompose(q,mu,scale,truth,widths,.5)['width_curve']
        self.assertEqual(shape['adjacent_pairs'],4)
        self.assertEqual(shape['both_truth_finite_pairs'],3)
        self.assertEqual(shape['q_decrease_pairs'],1)
        self.assertEqual(shape['q_decrease_curves'],1)
        self.assertAlmostEqual(shape['maximum_q_drop'],.5)
        self.assertEqual(shape['conditional_median_increases_over1cm'],1)
        self.assertEqual(shape['conditional_median_wrong_direction_curves'],1)
        self.assertEqual(shape['admitted_contact_disappears'],1)
        self.assertEqual(shape['truth_wrong_direction_pairs'],0)
        self.assertEqual(shape['truth_finite_disappears'],0)
        broken=truth.copy();broken[0,0,1]=1.8;broken[0,0,2]=np.inf
        bad=decompose(q,mu,scale,broken,widths,.5)['width_curve']
        self.assertEqual(bad['truth_wrong_direction_pairs'],1)
        self.assertEqual(bad['truth_finite_disappears'],1)

    def test_infinite_truth_no_invalid_arithmetic(self):
        q=np.full((1,2,3),.2);mu=np.ones_like(q);scale=np.full_like(q,.1);truth=np.full_like(q,np.inf)
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            result=decompose(q,mu,scale,truth,np.array([.4,.6,.8]),.5)
        self.assertEqual(result['all_widths']['actual']['truth_count'],0)
        self.assertIsNone(result['all_widths']['actual']['conditional_MAE_m'])
        self.assertEqual(result['all_widths']['q_unity_false_crossings'],6)

    def test_invalid_quantiles_and_anchor(self):
        for p in (0.,1.1,np.nan):
            with self.assertRaises(ValueError):quantile(1.,.2,p)
        with self.assertRaises(ValueError):quantile(1.,0.,.5)
        x=np.ones((1,2,3))
        with self.assertRaises(ValueError):decompose(x,x,x,x,np.array([.3,.4,.5]),.5)


if __name__=='__main__':unittest.main()
