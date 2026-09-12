import unittest

import numpy as np

import mz89_joint_geometry as m


def observations(n=5):
    return {'episode_id':np.full(n,'e0'),'tof_known':np.ones(n,bool),
            'tof_range_m':np.tile([2.8,np.nan],(n,1)),
            'tof_theta_deg':np.tile([4.,np.nan],(n,1)),
            'tof_lateral':np.tile([True,False],(n,1)),
            'tof_height':np.full((n,2),'BODY'),
            'radar_range_m':np.zeros((n,4)),'radar_velocity':np.zeros((n,4)),
            'radar_angle':np.zeros((n,4)),'radar_valid':np.zeros((n,4),bool),
            'delta_yaw':np.zeros(n),'delta_pitch':np.zeros(n),'imu_valid':np.ones(n,bool)}


class JointGeometryTest(unittest.TestCase):
    def test_shared_calibration_does_not_become_independent_velocity_noise(self):
        previous = np.asarray([2.,0.])
        current = previous.copy()
        self.assertAlmostEqual(m.forecast_sigma(previous,current,0.1),2.)
        self.assertGreater(m.forecast_sigma(previous,current,0.1,False),29.)
        independent = np.asarray([0.,2.])
        self.assertAlmostEqual(m.forecast_sigma(previous,independent,0.1),
                               m.forecast_sigma(previous,independent,0.1,False))

    def test_covariance_psd_and_dropout_survives_recovery(self):
        load = m.pose_loadings(np.asarray([2.,0.,2.,2.]),np.asarray([True,False,True,True]))
        self.assertAlmostEqual(load[0,1],0.2)
        self.assertAlmostEqual(load[1,1],0.4)
        self.assertGreaterEqual(np.linalg.eigvalsh(load@load.T).min(),-1e-9)
        self.assertEqual(load[1,8],load[3,8])
        self.assertGreater(load[3,8],0)

    def test_elapsed_gap_uses_three_steps_not_one(self):
        obs = observations(4)
        obs['tof_known'][1:3] = False
        obs['tof_theta_deg'][:,0] = [25,24,23,22]
        load = m.pose_loadings(obs['delta_yaw'],obs['imu_valid'])
        result = m.joint_state(obs,np.zeros(4),np.zeros(4))
        self.assertAlmostEqual(result['sigma_future'][3],m.forecast_sigma(load[0],load[3],0.3))

    def test_return_permutation_and_ambiguous_pairing(self):
        prior = [(2.5,1,'BODY',True),(2.8,10,'HEAD',True)]
        current = [(2.55,2,'BODY',True),(2.75,9,'HEAD',True)]
        self.assertEqual(m.matches(prior,current),{0:0,1:1})
        self.assertEqual(m.matches(prior,list(reversed(current))),{0:1,1:0})
        self.assertEqual(m.matches([prior[0],prior[0]],[current[0]]),{})
        obs = observations()
        obs['tof_range_m'][:,1]=3.
        obs['tof_theta_deg'][:,1]=24.
        shuffled = {k:v.copy() for k,v in obs.items()}
        for k in ('tof_range_m','tof_theta_deg','tof_height','tof_lateral'):
            shuffled[k]=shuffled[k][:,::-1]
        a,_,_=m.predict(obs); b,_,_=m.predict(shuffled)
        np.testing.assert_array_equal(a['joint_spatial'],b['joint_spatial'])

    def test_radar_needs_same_range_angle_and_corridor(self):
        def match(rr,ra,valid=True):
            return m.compatible(2.5,7,3,np.array([rr]),np.array([-0.6]),
                                np.array([ra]),np.array([valid]))
        self.assertTrue(match(2.55,0))
        self.assertFalse(match(1.5,0))
        self.assertFalse(match(2.55,20))
        self.assertFalse(match(2.55,0,False))

    def test_missing_empty_height_and_no_radar_veto_of_certain_tof(self):
        obs=observations(4)
        obs['tof_known'][1]=False
        obs['tof_range_m'][2]=np.nan
        obs['tof_theta_deg'][3,0]=40
        obs['tof_lateral'][:]=False
        values,unknown,debug=m.predict(obs)
        np.testing.assert_array_equal(values['joint_spatial'],[True,False,False,False])
        np.testing.assert_array_equal(unknown['joint_spatial'],[False,True,False,False])
        self.assertEqual(debug['joint_spatial_height'][1],'UNKNOWN')
        self.assertTrue(values['hard_hold'][1])

    def test_prefix_causality_reset_and_schema_rejects_truth(self):
        obs=observations(6)
        obs['tof_theta_deg'][:,0]=[22,20,18,16,14,12]
        full,_,_=m.predict(obs)
        prefix,_,_=m.predict({k:v[:4] for k,v in obs.items()})
        for name in full:
            np.testing.assert_array_equal(prefix[name],full[name][:4])
        reset={k:v.copy() for k,v in obs.items()}
        reset['episode_id'][3:]='e1'
        a,_,_=m.predict(reset)
        b,_,_=m.predict({k:v[3:] for k,v in reset.items()})
        for name in a:
            np.testing.assert_array_equal(a[name][3:],b[name])
        with self.assertRaises(ValueError):
            m.predict(dict(obs,truth=np.ones(6,bool)))

    def test_disjoint_truth_events_are_not_compressed(self):
        ids=np.full(7,'e0'); truth=np.array([0,1,1,0,1,1,0],bool)
        rows=m.event_rows(ids,truth,{'a':np.array([1,1,0,1,0,1,1],bool)})
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['arms']['a']['first'],1)
        self.assertEqual(rows[1]['arms']['a']['first'],5)


if __name__=='__main__':
    unittest.main()
