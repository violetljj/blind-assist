"""Expanded-observation checks of grouped weighted ranking and recalibration."""
import itertools
import unittest
import numpy as np
import cnh_readout_uncertainty_dev as M
import cnh_direction_information_dev as I


class WeightedEquivalence(unittest.TestCase):
    def setUp(self):
        self.w=np.array(list(itertools.product(range(3),repeat=3)),int)
        self.positive=np.array([1.1,.9,.9,.2]);self.pu=np.array([0,1,1,2])
        self.negative=np.array([.1,.3,.3,.9,.8,.2]);self.nu=np.array([0,0,1,1,2,2])

    def test_step_area_matches_expanded_whole_ties_exhaustively(self):
        groups=M.ranking_groups(self.positive,self.pu,self.negative,self.nu,3)
        for cap in (.05,.5):
            area,pm,nm=M.weighted_step_area(groups,self.w,cap)
            for j,w in enumerate(self.w):
                pos=np.repeat(self.positive,w[self.pu]);neg=np.repeat(self.negative,w[self.nu])
                expected=M.K._rank_summary(pos,neg,cap)['partial_step_area_normalized']
                self.assertEqual(pm[j],len(pos));self.assertEqual(nm[j],len(neg))
                if expected is None:self.assertTrue(np.isnan(area[j]))
                else:self.assertAlmostEqual(area[j],expected,places=13)

    def test_threshold_and_independent_weighted_evaluation_match_expanded(self):
        groups=M.score_groups(self.negative,self.nu,3)
        eval_weights=self.w[:,::-1]
        for cap in (.025,.05,.25,.5):
            threshold,mass=M.weighted_cutoff(groups,self.w,cap)
            count=M.weighted_above(groups,eval_weights,threshold)
            for j,w in enumerate(self.w):
                values=np.repeat(self.negative,w[self.nu])
                self.assertEqual(mass[j],len(values))
                if not len(values):
                    self.assertTrue(np.isnan(threshold[j]));self.assertTrue(np.isnan(count[j]));continue
                expected=I.P.threshold_at_cap(values,cap)
                self.assertEqual(threshold[j],expected)
                evaluation=np.repeat(self.negative,eval_weights[j,self.nu])
                self.assertEqual(count[j],int((evaluation>=expected).sum()))

    def test_two_group_max_missing_group_is_not_zero_risk(self):
        first=M.score_groups(self.negative,self.nu,3)
        second=M.score_groups(np.array([.7,1.2]),np.array([1,2]),3)
        a,am=M.weighted_cutoff(first,self.w,.25);b,bm=M.weighted_cutoff(second,self.w,.25)
        combined=np.maximum(a,b)
        self.assertTrue(np.isnan(combined[(am==0)|(bm==0)]).all())
        for j,w in enumerate(self.w):
            x=np.repeat(self.negative,w[self.nu]);y=np.repeat([.7,1.2],w[[1,2]])
            if len(x) and len(y):self.assertEqual(combined[j],max(I.P.threshold_at_cap(x,.25),I.P.threshold_at_cap(y,.25)))

    def test_stratified_complete_units_repeated_rows_and_independent_seeds(self):
        unit=np.repeat([10,11,12,13],4);mode=np.repeat([0,0,1,1],4);mirror=np.zeros(len(unit),bool)
        ids,inv,a=M.stratified_weights(unit,mode,mirror,40,101)
        _,_,b=M.stratified_weights(unit,mode,mirror,40,102)
        np.testing.assert_array_equal(a[:,:2].sum(1),np.full(40,2));np.testing.assert_array_equal(a[:,2:].sum(1),np.full(40,2))
        self.assertFalse(np.array_equal(a,b))
        for j,u in enumerate(ids):self.assertTrue((inv[unit==u]==j).all())


if __name__=='__main__':unittest.main()
