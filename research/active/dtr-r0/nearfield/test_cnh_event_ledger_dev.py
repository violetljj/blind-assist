"""Focused exact-tie, event-window, pairing and diagnostic contracts."""
import unittest
import numpy as np
import cnh_event_ledger_dev as E


class EventLedgerTest(unittest.TestCase):
    def test_integer_budget_ties_and_complete_groups(self):
        values=np.array([.1,.2,.2,.2,.8,.8,.9])
        for budget in range(len(values)+1):
            r=E.integer_budget_threshold(values,budget)
            self.assertLessEqual(r['actual_fa_count'],budget)
            self.assertEqual(r['residual_fa_count'],budget-r['actual_fa_count'])
            if budget<len(values):self.assertGreater((values>=np.nextafter(r['threshold'],-np.inf)).sum(),budget)
        result=E.integer_budget_threshold(values,2)
        self.assertEqual(result['actual_fa_count'],1)
        self.assertEqual(result['residual_fa_count'],1)
        groups=E.control_ties(values)
        self.assertEqual(sum(g['count'] for g in groups),7)
        self.assertEqual([g['count'] for g in groups],[1,3,2,1])
        with self.assertRaises(ValueError):E.integer_budget_threshold(values,2.1)

    def test_bruteforce_exact_threshold_family(self):
        rng=np.random.default_rng(43)
        for n in range(1,17):
            values=rng.integers(-2,4,n).astype(float)
            candidates=np.r_[-np.inf,np.unique(values),np.nextafter(np.unique(values),np.inf)]
            for budget in range(n+1):
                feasible=sorted(t for t in candidates if (values>=t).sum()<=budget)
                self.assertEqual(E.integer_budget_threshold(values,budget)['threshold'],feasible[0])

    def test_warmup_deadline_and_late_alarm(self):
        scores=np.zeros(13);scores[[0,4,8]]=1
        warm=E.event_trace(scores,.5,3)
        self.assertTrue(warm['any_before_timely']);self.assertFalse(warm['timely_excluding_warmup'])
        self.assertEqual(warm['first_any_before_output'],0)
        exact=E.event_trace(scores,.5,4)
        self.assertTrue(exact['timely_excluding_warmup']);self.assertEqual(exact['first_primary_output'],4)
        self.assertEqual(exact['first_primary_native_frame'],7)
        self.assertFalse(E.event_trace(np.zeros(13),.5,12)['any_before_timely'])

    def test_pair_membership_query_change_allowed_invalid_retained(self):
        a=dict(anchor_id='a',unit=1,config=0,valid=True,evaluable=True,contact=True,deadline_index=5,contact_query=[True,False])
        b=dict(a,contact_query=[True,True])
        good=E.physical_membership(a,b)
        self.assertTrue(good['eligible']);self.assertTrue(good['contact_query_changed'])
        bad=E.physical_membership(a,dict(b,valid=False,evaluable=False,contact=None))
        self.assertFalse(bad['eligible']);self.assertIn('HB/invalid',bad['reasons'])
        mismatch=E.physical_membership(a,dict(b,deadline_index=4))
        self.assertIn('contact_deadline_mismatch',mismatch['reasons'])

    def test_nondecrease_entire_window_and_changed_threshold(self):
        a=np.zeros((13,2));b=np.ones((13,2));b[4,0]=-1
        r=E.nondecrease_diagnostic(a,b,2,5,.3,.4)
        self.assertTrue(r['at_original_first_alarm']['both_nondecreasing'])
        self.assertTrue(r['any_window_both_nondecreasing']);self.assertFalse(r['all_window_both_nondecreasing'])
        self.assertEqual(len(r['timely_window']),4);self.assertTrue(r['threshold_changed'])
        self.assertIn('cannot be assigned',r['interpretation'])


if __name__=='__main__':unittest.main()
