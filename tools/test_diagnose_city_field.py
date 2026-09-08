import unittest
import numpy as np
from diagnose_city_field import confusion,target_decision,truly_clear


class DiagnosticTests(unittest.TestCase):
    def test_unknown_not_negative_and_inclusive_cutoff(self):
        self.assertEqual(confusion([.5,.8,.1,.6],[1,-1,1,0],.5),dict(TP=1,FN=1,FP=1,TN=0,UNKNOWN=1))

    def test_native_obstruction_and_unknown_not_clear_controls(self):
        self.assertTrue(truly_clear('clear',[0,0]))
        self.assertFalse(truly_clear('clear',[1,0]))
        self.assertFalse(truly_clear('clear',[-1,0]))
        self.assertFalse(truly_clear('route_baseline',[0,0]))

    def test_absent_target_not_miss_and_overlap_not_alert(self):
        mask=np.zeros((18,32),int);support=np.ones((18,32))
        self.assertIsNone(target_decision(0,support,mask,.5))
        mask[2,3]=1;result=target_decision(.2,support,mask,.5)
        self.assertEqual(result['outcome'],'ALERT_MISS');self.assertTrue(result['overlap']);self.assertFalse(result['joint'])
        self.assertFalse(result['peak_hit'])


if __name__=='__main__':unittest.main()
