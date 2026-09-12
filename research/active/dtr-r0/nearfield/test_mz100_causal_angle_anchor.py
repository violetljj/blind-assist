import unittest
import numpy as np
import mz100_causal_angle_anchor as a


def packets(n=18):
    p=dict(episode_id=np.array(['a']*n),time_s=np.arange(n)*.1,
        tof_packet_received=np.ones(n,bool),tof_status=np.full((n,8),255),
        tof_range_m=np.full((n,8),np.nan),tof_theta_deg=np.tile(np.arange(8)*5.625,(n,1)),
        tof_range_sigma_m=np.full((n,8),.04),radar_packet_received=np.ones(n,bool),
        radar_valid=np.zeros((n,4),bool),radar_range_m=np.full((n,4),np.nan),
        radar_angle=np.full((n,4),np.nan),radar_velocity=np.full((n,4),np.nan),
        delta_yaw=np.zeros(n),delta_pitch=np.zeros(n),imu_valid=np.ones(n,bool))
    p['radar_valid'][:,0]=True;p['radar_range_m'][:,0]=2;p['radar_velocity'][:,0]=-.5;p['radar_angle'][:,0]=0
    p['tof_status'][:3,0]=5;p['tof_range_m'][:3,0]=2;p['radar_angle'][:3,0]=[-7,7,0]
    return p


class AnchorTests(unittest.TestCase):
    def test_unique_and_ambiguous(self):
        p=packets();self.assertEqual(a.frame_anchor(p,0)[1]['status'],'anchor')
        p['radar_valid'][0,1]=True;p['radar_range_m'][0,1]=2;p['radar_angle'][0,1]=0;p['radar_velocity'][0,1]=-.5
        self.assertEqual(a.frame_anchor(p,0)[1]['status'],'ambiguous')

    def test_prior_only_expiry_and_reset(self):
        p=packets();t,r=a.estimate(p)
        self.assertFalse(t['available'][:3].any());self.assertTrue(t['applied'][3:13].all());self.assertFalse(t['applied'][13:].any())
        self.assertEqual(r[2]['status'],'validated_anchor');self.assertAlmostEqual(t['age_s'][12],1.)
        p['episode_id'][8:]='b';self.assertFalse(a.estimate(p)[0]['available'][8:].any())

    def test_prefix_causality_and_only_angle(self):
        p=packets();full,_=a.estimate(p)
        for n in range(1,len(p['episode_id'])):
            t,_=a.estimate({k:v[:n] for k,v in p.items()})
            for k in t:np.testing.assert_array_equal(t[k],full[k][:n])
        full['bias_deg'][full['applied']]=4.;out=a.correct(p,full)
        for k in p:
            if k!='radar_angle':np.testing.assert_array_equal(out[k],p[k])
        self.assertEqual(out['radar_angle'][3,0],-4.)
        np.testing.assert_array_equal(out['radar_angle'][:3],p['radar_angle'][:3])

    def test_conflict_and_invalid_packets(self):
        p=packets();p['tof_theta_deg'][0,0]=-10;p['tof_range_m'][0,0]=1
        p['tof_status'][0,7]=5;p['tof_range_m'][0,7]=3;p['tof_theta_deg'][0,7]=10
        p['radar_valid'][0,:2]=True;p['radar_range_m'][0,:2]=[1,3];p['radar_angle'][0,:2]=[-30,30];p['radar_velocity'][0,:2]=-.5
        self.assertEqual(a.frame_anchor(p,0)[1]['status'],'conflict')
        p=packets();p['tof_packet_received'][:]=False;self.assertFalse(a.estimate(p)[0]['available'].any())
        p=packets();p['radar_packet_received'][:]=False;self.assertFalse(a.estimate(p)[0]['applied'].any())

    def test_adjacent_cluster_boundary(self):
        p=packets();p['tof_status'][0,1:3]=5;p['tof_range_m'][0,1:3]=[2.1,3.]
        self.assertEqual([g['zones'] for g in a.clusters(p,0)],[[0,1],[2]])

    def test_temporal_conflict_clears_calibration(self):
        p=packets();p['tof_status'][3,0]=5;p['tof_range_m'][3,0]=2;p['radar_angle'][3,0]=20
        t,r=a.estimate(p)
        self.assertEqual(r[3]['status'],'temporal_conflict')
        self.assertFalse(t['applied'][3]);self.assertFalse(t['available'][4:].any())


if __name__=='__main__':unittest.main()
