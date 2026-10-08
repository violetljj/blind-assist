import unittest
import numpy as np
import cnh_ema_bearing_desmooth_dev as D


class DesmoothContract(unittest.TestCase):
    def test_smoothing_before_height_fusion(self):
        raw=np.zeros((3,3,1,13,2));raw[0,0,0,0,0]=9;raw[0,0,0,1,1]=9
        before=D.fuse(D.B.R.smooth(raw))
        self.assertEqual(before[0,0,1,0],6.)
        self.assertEqual(D.fuse(raw)[0,0,1,0],9.)
        self.assertNotEqual(before[0,0,1,0],(9.+2*9.)/3)

    def test_query_and_branch_axes(self):
        raw=np.zeros((3,3,1,13,2));raw[2,0]=5.;raw[1,2]=9.
        score=D.fuse(raw)
        self.assertEqual(score[0,0,2].argmax(),2)
        self.assertEqual(score[1,0,2].argmax(),1)

    def test_yaw_history_wrap_and_return_including_startup(self):
        yaw=np.zeros(16);yaw[3:6]=[179.,-170.,179.]
        poses=np.broadcast_to(np.eye(4),(1,16,4,4)).copy()
        poses[0,:,0,2]=np.sin(np.radians(yaw));poses[0,:,2,2]=np.cos(np.radians(yaw))
        self.assertAlmostEqual(D.yaw_change(poses)[0,2],11.)

    def test_unique_guard_and_zero_rule(self):
        self.assertEqual(D.decision([1,0,0,0],[0,0,0,0]),'PRIORITIZE_DESMOOTH_FOLLOWUP')
        self.assertEqual(D.decision([1,0,0,0],[-1,0,0,0]),'RETAIN_ORIGINAL_L2')
        self.assertEqual(D.decision([0,0,0,0],[1,0,0,0]),'RETAIN_ORIGINAL_L2')

    def test_exact_tie_order_and_stability(self):
        score=np.zeros((1,13,3));score[0,1,2]=2
        labels,margin,stable=D.descriptors(score)
        self.assertEqual((labels[0,0],margin[0,0]),(0,0.))
        self.assertFalse(stable[0,2]);self.assertTrue(stable[0,6])


if __name__=='__main__':unittest.main()
