import unittest

import numpy as np

import mz90_observable_contract as m


def raw_fixture(n=5):
    raw={'episode_id':np.full(n,'e0'),'time_s':np.arange(n)*0.1,
         'tof_packet_received':np.ones(n,bool),'tof_range_m':np.full((n,8),np.nan),
         'tof_theta_deg':np.full((n,8),np.nan),'tof_range_sigma_m':np.full((n,8),0.05),
         'tof_status':np.full((n,8),255,np.uint8),'radar_packet_received':np.ones(n,bool),
         'radar_range_m':np.full((n,4),np.nan),'radar_velocity':np.full((n,4),np.nan),
         'radar_angle':np.full((n,4),np.nan),'radar_valid':np.zeros((n,4),bool),
         'delta_yaw':np.zeros(n),'delta_pitch':np.zeros(n),'imu_valid':np.ones(n,bool)}
    raw['tof_range_m'][:,3]=2.5
    raw['tof_theta_deg'][:,3]=0
    raw['tof_status'][:,3]=5
    return raw


class ObservableContractTest(unittest.TestCase):
    def test_invalid_hidden_returns_do_not_change_predictions(self):
        raw=raw_fixture()
        raw['tof_status'][1:4]=255
        raw['tof_packet_received'][2]=False
        clean={k:v.copy() for k,v in raw.items()}
        clean['tof_range_m'][1:4]=np.nan
        clean['tof_theta_deg'][1:4]=np.nan
        a,va,_,_=m.predict(raw);b,vb,_,_=m.predict(clean)
        self.assertFalse(a['tof_known'][1:4].any())
        for name in va: np.testing.assert_array_equal(va[name],vb[name])
        for key in ('tof_range_m','tof_theta_deg'):np.testing.assert_array_equal(a[key],b[key])

    def test_no_return_and_packet_gap_both_unknown_not_clear(self):
        raw=raw_fixture(3)
        raw['tof_status'][:]=255
        raw['tof_packet_received'][1]=False
        _,values,unknown,_=m.predict(raw)
        for name in values:
            self.assertFalse(values[name].any())
            self.assertTrue(unknown[name].all())

    def test_shared_imu_transform_and_no_height_or_motion_truth(self):
        raw=raw_fixture(3)
        raw['delta_yaw'][1:]=5
        raw['tof_theta_deg'][:,3]=[0,-5,-10]
        raw['radar_angle'][:,0]=[0,-5,-10]
        raw['radar_range_m'][:,0]=2.5
        raw['radar_velocity'][:,0]=-.7
        raw['radar_valid'][:,0]=True
        obs=m.adapt(raw)
        np.testing.assert_allclose(obs['radar_angle'][:,0],0,atol=1e-9)
        self.assertTrue((obs['tof_height']=='UNKNOWN').all())
        self.assertTrue(obs['tof_lateral'][:,0].all())
        with self.assertRaises(ValueError):m.adapt(dict(raw,truth=np.ones(3,bool)))

    def test_budget_selection_is_order_independent(self):
        raw=raw_fixture(3)
        raw['tof_range_m'][:,1]=1.5;raw['tof_theta_deg'][:,1]=18;raw['tof_status'][:,1]=5
        raw['tof_range_m'][:,7]=3.5;raw['tof_theta_deg'][:,7]=-18;raw['tof_status'][:,7]=5
        a=m.adapt(raw)
        raw={k:v[:,::-1] if k.startswith('tof_') and v.ndim==2 else v for k,v in raw.items()}
        b=m.adapt(raw)
        for key in ('tof_range_m','tof_theta_deg'):np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a['tof_range_m'][0],[1.5,2.5])

    def test_hold_cannot_reseed_and_prefix_is_causal(self):
        raw=raw_fixture(5);raw['tof_status'][1:]=255
        _,values,_,_=m.predict(raw)
        np.testing.assert_array_equal(values['matched_hold'],[True,True,False,False,False])
        _,prefix,_,_=m.predict({k:v[:3] for k,v in raw.items()})
        for name in values:np.testing.assert_array_equal(prefix[name],values[name][:3])

    def test_evaluator_counts_each_truth_event_and_miss(self):
        ids=np.full(5,'e0');truth=np.array([0,1,0,1,0],bool)
        values={n:truth.copy() for n in m.ARM_NAMES}
        values['joint_spatial'][3]=False
        unknown={n:np.zeros(5,bool) for n in values}
        result=m.evaluate(ids,{'truth':truth,'future_only_truth':truth},values,unknown)
        self.assertEqual(result['truth_events'],2)
        self.assertEqual(result['timing']['missed_baseline_events'],1)
        self.assertFalse(result['full_gates']['no_lost_baseline_event_or_delay_over_0p2'])


if __name__=='__main__': unittest.main()
