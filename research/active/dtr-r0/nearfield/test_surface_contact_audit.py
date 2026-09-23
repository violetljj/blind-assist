"""Focused count/UNKNOWN and episode-state fixtures, not cohort scoring."""
import unittest
import surface_contact_audit_20260923 as a


class AuditAccountingTests(unittest.TestCase):
    def test_missing_keeps_fixed_positive_denominator(self):
        rows=[dict(pred=1.02,truth=1.),dict(pred=None,truth=2.),dict(pred=3.,truth=2.),
              dict(pred=1.,truth=None),dict(pred=None,truth=None)]
        r=a.statistics(rows)
        self.assertEqual((r['queries'],r['truth_contact'],r['missing_contact'],r['unknown']), (5,3,1,2))
        self.assertEqual(r['correct_5cm'],1);self.assertEqual(r['hit_5cm'],1/3)
        self.assertAlmostEqual(r['conditional_mae_m'],.51)
        self.assertEqual((r['false_contact'],r['negative_queries'],r['false_contact_rate']),(1,2,.5))

    def test_empty_and_zero_width_not_missing(self):
        self.assertIsNone(a.statistics([])['hit_5cm'])
        r=a.statistics([dict(pred=0.,truth=0.)])
        self.assertEqual((r['unknown'],r['correct_5cm'],r['hit_5cm']),(0,1,1.))
        self.assertEqual(a.union(None,0.),0.)

    def test_two_on_two_off_and_episode_reset(self):
        values=[1.,1.,None,1.,None,None,1.,1.]
        rows=[dict(pred=v,episode='a' if i<7 else 'b') for i,v in enumerate(values)]
        self.assertEqual(a.debounce(rows),[False,True,True,True,True,False,False,False])

    def test_horizon_inclusive(self):
        rows=[dict(pred=3.,episode='a'),dict(pred=3.,episode='a')]
        self.assertEqual(a.debounce(rows),[False,True])
        self.assertEqual(a.alert_counts([False,True],[True,False]),dict(TP=0,FP=1,FN=1,TN=0))


if __name__=='__main__':unittest.main()
