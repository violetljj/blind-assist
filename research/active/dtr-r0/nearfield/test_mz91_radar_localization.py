import unittest
import numpy as np
import mz91_radar_localization as m
import mz90_observation_source as source


class LocalizationChecks(unittest.TestCase):
    def setUp(self):
        self.raw,_=source.materialize(source.build_source()[:3],'sensor_proxy')

    def test_prefix_and_slot_permutation(self):
        full=m.predict(self.raw)
        prefix=m.predict({k:v[:17] for k,v in self.raw.items()})
        shuffled={k:(v[:,::-1] if k.startswith('radar_') and v.ndim==2 else v.copy()) for k,v in self.raw.items()}
        perm=m.predict(shuffled)
        for arm in full[1]:
            np.testing.assert_array_equal(full[1][arm][:17],prefix[1][arm])
            np.testing.assert_array_equal(full[1][arm],perm[1][arm])
            np.testing.assert_allclose(full[3][arm+'_probability'][:,::-1],perm[3][arm+'_probability'],equal_nan=True)

    def test_unrestricted_acceptance_baseline_parity(self):
        obs=m.contract.adapt(self.raw)
        hold,unknown,_=m.assemble(obs,np.ones_like(obs['radar_range_m']))
        np.testing.assert_array_equal(hold,m.contract.simple_controls(obs)['matched_hold'])
        np.testing.assert_array_equal(unknown,~obs['tof_known']&~hold)

    def test_shared_bias_floor_and_shift(self):
        h=[[i,2-.07*i,8.,-.7] for i in range(5)]
        mean,cov=m.localize(h,3)
        shifted=[[i,r,a+10,v] for i,r,a,v in h]
        mean2,cov2=m.localize(shifted,3)
        self.assertGreaterEqual(cov[2,2],109)
        np.testing.assert_allclose(mean2-mean,[0,0,10,0],atol=1e-10)
        np.testing.assert_allclose(cov,cov2)

    def test_deterministic_collision_and_missing(self):
        self.assertEqual(m.risk(np.array([2.,-.7,0.,0.]),np.zeros((4,4))),1)
        self.assertEqual(m.risk(np.array([2.,-.7,40.,0.]),np.zeros((4,4))),0)
        raw={k:v.copy() for k,v in self.raw.items()}
        raw['tof_status'][:]=255; raw['radar_valid'][:]=False
        _,values,unknown,_=m.predict(raw)
        for arm in values:
            self.assertFalse(values[arm].any()); self.assertTrue(unknown[arm].all())

    def test_continuous_geometry_matches_contract(self):
        rng=np.random.default_rng(19)
        for x,z,vx,vz in rng.uniform([-3,.1,-2,-2],[3,4,2,1],size=(100,4)):
            r=np.hypot(x,z)
            mean=np.array([r,(x*vx+z*vz)/r,np.degrees(np.arctan2(x,z)),
                           np.degrees((z*vx-x*vz)/r**2)])
            self.assertEqual(m.risk(mean,np.zeros((4,4))),float(source.geometric_hazard(x,z,vx,vz)))


if __name__=='__main__':
    unittest.main()
