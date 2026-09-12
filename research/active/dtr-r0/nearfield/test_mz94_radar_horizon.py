import unittest
import numpy as np
import mz94_radar_horizon as m
import mz90_observation_source as source


class HorizonChecks(unittest.TestCase):
    def test_current_far_receding_and_unknown_transverse(self):
        self.assertTrue(m.trajectory_admission([[0,3.4,0,1]]))
        self.assertFalse(m.trajectory_admission([[0,3.4,25,-1]]))

    def test_continuous_crossing_and_outside(self):
        self.assertTrue(m.corridor_hit(1,2,-2,0))
        self.assertFalse(m.corridor_hit(1,2,2,0))
        self.assertFalse(m.corridor_hit(0,.2,0,0))

    def test_prefix_permutation_and_baseline(self):
        raw=source.materialize(source.build_source()[:3],'sensor_proxy')[0]
        full=m.predict(raw);prefix=m.predict({k:v[:65] for k,v in raw.items()})
        perm=m.predict({k:v[:,::-1] if v.ndim==2 else v for k,v in raw.items()})
        for name in full[1]:
            np.testing.assert_array_equal(full[1][name][:65],prefix[1][name])
            np.testing.assert_array_equal(full[1][name],perm[1][name])
        obs=full[0];q=obs['radar_valid']&(obs['radar_range_m']<3.18)&(obs['radar_velocity']<=-.35)&(np.abs(obs['radar_angle'])<=20)
        np.testing.assert_array_equal(m.assemble_admission(obs,q)[0],m.contract.simple_controls(obs)['matched_hold'])

    def test_episode_reset_no_future_motion(self):
        raw=source.materialize(source.build_source()[:1],'sensor_proxy')[0]
        raw['episode_id'][2:]='new'
        obs=m.contract.adapt(raw);_,length=m.horizon_admission(obs)
        self.assertTrue((length[2]<=1).all())


if __name__=='__main__':unittest.main()
