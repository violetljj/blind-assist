"""Decision-relevant notice contracts, no GPU or model fitting."""
import unittest
import numpy as np
import cnh_sector_notice_dev as S


class NoticeContract(unittest.TestCase):
    def test_whole_ties_and_multilabel_cost(self):
        score=np.array([[[3.,3.,0.],[2.,1.,0.]]])
        top=S.select_threshold(score,1,'top1')
        all_=S.select_threshold(score,1,'all_sectors')
        self.assertEqual(S.emitted(score,top['threshold'],'top1').sum(),1)
        self.assertEqual(S.emitted(score,all_['threshold'],'all_sectors').sum(),0)
        self.assertEqual(all_['unused_notices'],1)
        np.testing.assert_array_equal(S.emitted(score,top['threshold'],'top1')[0,0], [True,False,False])

    def test_first_wrong_is_not_repaired_by_later_correct(self):
        n=np.zeros((1,13,3),bool); t=np.zeros_like(n)
        n[0,0,1]=True; n[0,2,0]=True; n[0,3,1]=True
        t[:,:,1]=True
        e=S.events(n,t,np.array([True]),np.array([4]))
        self.assertEqual(e['first'][0],2)
        self.assertTrue(e['timely'][0]); self.assertFalse(e['first_clean'][0])
        self.assertTrue(e['ever_supported'][0])
        n[0,3,0]=True
        e=S.events(n,t,np.array([True]),np.array([2]))
        self.assertFalse(e['ever_supported'][0])

    def test_correct_plus_wrong_is_not_clean(self):
        n=np.zeros((1,13,3),bool); t=np.zeros_like(n)
        n[0,2,:2]=True; t[0,2,1]=True
        e=S.events(n,t,np.array([True]),np.array([4]))
        self.assertTrue(e['first_supported'][0]); self.assertFalse(e['first_clean'][0])

    def test_query_axes_and_no_future_inputs(self):
        noisy=np.repeat(np.eye(4)[None,None],16,axis=1)
        disp=np.zeros((1,16,3))
        q=S.sector_queries(noisy,disp)
        self.assertAlmostEqual(S.RC.A_yaw(q[0][0,0]),20.)
        self.assertAlmostEqual(S.RC.A_yaw(q[2][0,0]),-20.)
        changed=noisy.copy(); changed[0,8:,:3,3]=123.
        changed_q=S.sector_queries(changed,disp)
        for a,b in zip(q,changed_q): np.testing.assert_array_equal(a[:,:8],b[:,:8])

    def test_per_configuration_decision(self):
        self.assertEqual(S.decision([1,0,0,0]),'SUPPORT_DEDUP_CHECK')
        self.assertEqual(S.decision([-1,0,0,0]),'LOWER_CURRENT_SECTOR_READOUT_PRIORITY')
        self.assertEqual(S.decision([0]*4),'RETAIN_MIXED_OR_ZERO_NO_AUTOMATIC_FOLLOWUP')
        self.assertEqual(S.decision([1,-1,0,0]),'RETAIN_MIXED_OR_ZERO_NO_AUTOMATIC_FOLLOWUP')

    def test_analysis_keeps_other_scenes_separate(self):
        import cnh_sector_notice_analyze_dev as A
        n=np.zeros((3,13,3),bool); t=np.zeros_like(n)
        n[:,2,1]=True; t[0,2,1]=True
        rec,_=A.summarize(n,t,np.array([True,False,False]),np.array([False,True,False]),
            np.array([4,-1,-1]),dict())
        self.assertEqual(rec['total_notices'],3)
        self.assertEqual(rec['strict_clear_notices'],1)
        self.assertEqual(rec['other_window_notices'],1)
        self.assertEqual(rec['first_clean'],1)
        self.assertEqual(rec['label_removed_timely'],rec['timely_main'])


if __name__ == '__main__': unittest.main()
