import unittest
from collections import Counter
import numpy as np
from boundary_supervision_data import schedule,training_labels,decision
from contact_boundary_data import queries,contact_labels


class DenseScheduleTests(unittest.TestCase):
    def test_fixed_budget_and_balanced_exposure(self):
        q,m=schedule();self.assertEqual(q.shape,(100,72,3))
        self.assertEqual(len({tuple(x) for x in q.reshape(-1,3)}),258)
        for layer in (0,1):
            for start in (0,12,24):
                counts=Counter(tuple(row) for epoch in q for row in epoch[layer*36+start:layer*36+start+12])
                self.assertLessEqual(max(counts.values())-min(counts.values()),1)
        np.testing.assert_array_equal(q,schedule()[0])

    def test_mapping_matches_geometry_with_censored_negatives(self):
        source=queries();q,m=schedule()
        scenes=[[(np.array([.22,.5,1.43]),np.array([.5,.8,1.6]))],
                [(np.array([.8,-.1,1.4]),np.array([1.,.2,1.6]))],[]]
        target={k:np.stack([contact_labels(s,v)[0] for s in scenes]) for k,v in source.items()}
        y=training_labels(target,m)
        for e in (0,16,45,99):
            for i,s in enumerate(scenes):np.testing.assert_array_equal(y[e,i],contact_labels(s,q[e])[0])
        self.assertFalse(y[:,2].any())

    def test_untrained_combined_interpolation_queries(self):
        q,_=schedule();trained={tuple(np.round(x,6)) for x in q.reshape(-1,3)}
        self.assertFalse(trained & {tuple(np.round(x,6)) for x in queries()['both']})

    def test_gain_does_not_erase_false_costs(self):
        import copy
        one=dict(boundaries={k:dict(joint_within_5cm=.2,wrong_crossings_on_right_censored=10) for k in ('width','horizon')},
                 both=dict(recall=.8,FPR=.05),component_gate=False)
        old={s:copy.deepcopy(one) for s in ('train','evaluation')};new=copy.deepcopy(old)
        for s in new:
            for k in new[s]['boundaries']:new[s]['boundaries'][k]['joint_within_5cm']=.31
        self.assertTrue(decision(old,new)['mechanism_supported'])
        new['evaluation']['both']['FPR']=.061
        self.assertEqual(decision(old,new)['decision'],'ACCURACY_COST_TRADEOFF')


if __name__=='__main__':unittest.main()
