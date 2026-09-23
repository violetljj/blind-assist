import unittest
import numpy as np
from local_rescue_diagnostic import causal_controls, range_evidence, GATES
from inherit_spatial_model import QUERIES
from local_rescue_gate import select_cutoff,apply_gate


class RescueTest(unittest.TestCase):
    def fixture(self,n=4):
        prob=np.full((n,6),.9)
        base=[dict(alert=False,score=.2) for _ in range(n)]
        tof=np.zeros((n,64,6));tof[:,:,2:]=[.4,.4,.6,.6]
        tof[:,:,0]=2/8;tof[:,:,1]=1
        feat=np.zeros((n,6,961))
        return prob,base,tof,feat

    def test_history_resets_and_A_preserved(self):
        p,b,t,f=self.fixture();b[2]['alert']=True
        flags,_=causal_controls(p,b,t,f,[('a',0),('a',1),('b',0),('b',1)],.8)
        self.assertFalse(flags[0]['two_frames']);self.assertTrue(flags[1]['two_frames'])
        for arm in GATES:self.assertTrue(flags[2][arm])
        b[2]['alert']=False
        flags,_=causal_controls(p,b,t,f,[('a',0),('a',1),('b',0),('b',1)],.8)
        self.assertFalse(flags[2]['two_frames'])

    def test_prefix_causality(self):
        p,b,t,f=self.fixture();seq=[('a',i) for i in range(4)]
        a,x=causal_controls(p,b,t,f,seq,.8)
        p[2:]=0;t[2:,:,0]=7/8
        c,y=causal_controls(p,b,t,f,seq,.8)
        self.assertEqual(a[:2],c[:2]);self.assertEqual(x[:2],y[:2])

    def test_range_not_interval_ownership(self):
        _,_,t,_=self.fixture();t[0,:,0]=3/8
        r=range_evidence(t[0],QUERIES[4])
        self.assertGreater(r['nominal_count'],0);self.assertEqual(r['upper_count'],0)
        t[0,:,0]=3.15/8
        self.assertEqual(range_evidence(t[0],QUERIES[4])['nominal_count'],0)
        t[0,:,1]=0
        self.assertEqual(range_evidence(t[0],QUERIES[4])['minimum_range'],8.)

    def test_same_query_vs_any_query(self):
        p,b,t,f=self.fixture(2);p[0,1]=.95;p[0,4]=.1;p[1,4]=.99
        flags,_=causal_controls(p,b,t,f,[('a',0),('a',1)],.8)
        self.assertTrue(flags[1]['two_frames']);self.assertFalse(flags[1]['same_query'])

    def test_missing_history_rejected(self):
        p,b,t,f=self.fixture(1)
        with self.assertRaises(ValueError):causal_controls(p,b,t,f,[('a',2)],.8)

    def test_one_cutoff_selection_and_no_overlap_rescue(self):
        self.assertEqual(select_cutoff([.01,.02],[.04,.05]),.03)
        with self.assertRaises(ValueError):select_cutoff([.04],[.03])

    def test_gate_equality_A_and_unknown(self):
        rows=[dict(predictions=dict(A_current=dict(alert=a,unknown=True),local=dict(alert=l,unknown=True)))
              for a,l in [(False,True),(False,True),(True,True),(False,False)]]
        out=apply_gate(rows,[dict(possible_fraction=x) for x in [.03,.031,.8,0]],.03)
        self.assertEqual([r['predictions']['ambiguity_gate']['alert'] for r in out],[True,False,True,False])
        self.assertTrue(all(r['predictions']['ambiguity_gate']['unknown'] for r in out))


if __name__=='__main__':unittest.main()
