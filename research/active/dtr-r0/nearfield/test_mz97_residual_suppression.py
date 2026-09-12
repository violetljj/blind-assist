import unittest
import numpy as np
import mz97_residual_suppression as r


class ResidualTest(unittest.TestCase):
    def test_disabled_and_subset(self):
        base=np.array([True,False,True]);score=np.array([.9,.0,.1])
        np.testing.assert_array_equal(r.apply(base,score,None),base)
        np.testing.assert_array_equal(r.apply(base,score,.5),[False,False,True])

    def test_validation_noop_when_only_rejection_loses_event(self):
        ids=np.array(['e']*4);base=np.ones(4,bool)
        labels=dict(truth=np.array([True,False,False,False]),future_only_truth=np.array([True,False,False,False]))
        outcome=r.select(ids,labels,base,np.array([.99,.8,.8,.8]))
        self.assertTrue(outcome['disabled'])

    def test_safe_rejection_selected(self):
        ids=np.array(['e']*4);base=np.ones(4,bool)
        labels=dict(truth=np.array([True,True,False,False]),future_only_truth=np.array([True,False,False,False]))
        outcome=r.select(ids,labels,base,np.array([.1,.1,.99,.99]))
        self.assertFalse(outcome['disabled']);self.assertEqual(outcome['selected']['removed_FP'],2)
        self.assertEqual(outcome['selected']['lost_TP'],0)

    def test_aggregate_floor_cannot_hide_lost_short_event(self):
        ids=np.array(['e']*103);truth=np.ones(103,bool);truth[100:102]=False
        candidate=truth.copy();candidate[102]=False
        labels=dict(truth=truth,future_only_truth=truth)
        row=r.compare(ids,labels,truth,candidate)
        self.assertTrue(row['gates']['tp_retention_ge_98'])
        self.assertFalse(row['gates']['no_lost_reference_event'])

if __name__=='__main__':unittest.main()
