import unittest
import numpy as np
import mz93_supported_prediction as m
import mz91_radar_localization as old
import mz90_observation_source as source


class SupportedPredictionChecks(unittest.TestCase):
    def test_direction_requires_history_and_rate_evidence(self):
        mean=np.array([2.,-.7,20.,-5.]);cov=np.diag([.01,.01,9.,1.])
        self.assertTrue(m.motion_supported(mean,cov,3))
        self.assertFalse(m.motion_supported(mean,cov,2))
        cov[3,3]=16;self.assertFalse(m.motion_supported(mean,cov,3))
        mean[3]=20;self.assertFalse(m.motion_supported(mean,cov,3))

    def test_risk_decomposition_matches_frozen(self):
        rng=np.random.default_rng(93)
        for _ in range(30):
            mean=np.array([rng.uniform(1,4),-.7,rng.uniform(-30,30),rng.uniform(-20,20)])
            cov=np.diag([.01,.01,25,36])
            total,current=m.risk_components(mean,cov)
            self.assertEqual(total,old.risk(mean,cov));self.assertLessEqual(current,total)

    def test_prefix_permutation_and_subset(self):
        raw=source.materialize(source.build_source()[:3],'sensor_proxy')[0]
        full=m.predict(raw)
        prefix=m.predict({k:v[:65] for k,v in raw.items()})
        perm=m.predict({k:v[:,::-1] if v.ndim==2 else v for k,v in raw.items()})
        for name in full[1]:
            np.testing.assert_array_equal(full[1][name][:65],prefix[1][name])
            np.testing.assert_array_equal(full[1][name],perm[1][name])
        self.assertFalse((full[1]['mz93_supported']&~full[1]['mz92_control']).any())

    def test_future_crossing_and_current_evidence(self):
        mean=np.array([2.,-.7,20.,-20.])
        total,current=m.risk_components(mean,np.zeros((4,4)))
        self.assertEqual((total,current),(1,0))
        self.assertEqual(m.risk_components(np.array([2.,-.7,0.,0.]),np.zeros((4,4))),(1,1))

    def test_missing_is_unknown(self):
        raw=source.materialize(source.build_source()[:1],'sensor_proxy')[0]
        raw['tof_packet_received'][:]=False;raw['radar_valid'][:]=False
        _,values,unknown,_=m.predict(raw)
        for name in values:
            self.assertFalse(values[name].any());self.assertTrue(unknown[name].all())


if __name__=='__main__':unittest.main()
