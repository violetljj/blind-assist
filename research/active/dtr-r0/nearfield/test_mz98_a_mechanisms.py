import unittest
import numpy as np
import audit_mz98_a_mechanisms as audit


class AuditTests(unittest.TestCase):
    def test_current_extended_does_not_require_prediction(self):
        admitted,branch,mean=audit.classify([[0,3.4,0.,.4]])
        self.assertTrue(admitted);self.assertEqual(branch,'R_extended');self.assertIsNone(mean)

    def test_actual_later_entry(self):
        admitted,branch,_=audit.classify([[0,3.,22.,-.5],[1,2.95,20.,-.5],[2,2.9,18.,-.5]])
        self.assertTrue(admitted);self.assertEqual(branch,'P')

    def test_observer_parity_and_restoration(self):
        n=5;obs=dict(episode_id=np.array(['e']*n),delta_yaw=np.zeros(n),imu_valid=np.ones(n,bool),
            radar_valid=np.zeros((n,4),bool),radar_range_m=np.full((n,4),np.nan),radar_angle=np.full((n,4),np.nan),radar_velocity=np.full((n,4),np.nan))
        obs['radar_valid'][:,0]=True;obs['radar_range_m'][:,0]=3.;obs['radar_angle'][:,0]=[24,22,20,18,16];obs['radar_velocity'][:,0]=-.3
        original=audit.a.trajectory_admission;expected,_=audit.a.horizon_admission(obs)
        tags,actual,rows=audit.instrument(obs)
        np.testing.assert_array_equal(actual,expected);self.assertIs(audit.a.trajectory_admission,original);self.assertEqual(len(rows),n)

if __name__=='__main__':unittest.main()
