import unittest
import numpy as np
import mz95_coverage_authority as m
import mz90_observation_source as source

class CoverageChecks(unittest.TestCase):
    def test_causality_permutation_and_noop(self):
        raw=source.materialize(source.build_source()[:3],'sensor_proxy')[0]
        full=m.predict(raw);prefix=m.predict({k:v[:65] for k,v in raw.items()})
        perm=m.predict({k:v[:,::-1] if v.ndim==2 else v for k,v in raw.items()})
        for name in full[1]:
            np.testing.assert_array_equal(full[1][name][:65],prefix[1][name])
            np.testing.assert_array_equal(full[1][name],perm[1][name])
        obs=full[0]
        np.testing.assert_array_equal(m.combine(obs,full[3]['admitted'],np.ones(len(raw['episode_id']),bool))[0],full[1]['unchanged_control'])

    def test_coverage_needs_valid_colocated_support(self):
        raw=source.materialize(source.build_source()[:1],'ideal')[0]
        raw['delta_yaw'][:]=0;raw['tof_status'][:]=255;raw['radar_valid'][:]=False
        raw['radar_valid'][:,0]=True;raw['radar_range_m'][:,0]=2;raw['radar_angle'][:,0]=18;raw['radar_velocity'][:,0]=-.7
        raw['tof_status'][:,0]=5;raw['tof_range_m'][:,0]=2;raw['tof_theta_deg'][:,0]=18
        self.assertTrue(m.predict(raw)[3]['coverage'].all())
        raw['tof_theta_deg'][:,0]=-18
        self.assertFalse(m.predict(raw)[3]['coverage'].any())
        raw['tof_theta_deg'][:,0]=18;raw['tof_packet_received'][:]=False
        self.assertFalse(m.predict(raw)[3]['coverage'].any())

if __name__=='__main__':unittest.main()
