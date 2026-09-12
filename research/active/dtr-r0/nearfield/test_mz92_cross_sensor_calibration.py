import unittest
import numpy as np
import mz92_cross_sensor_calibration as m
import mz90_observation_source as source


class CalibrationChecks(unittest.TestCase):
    def raw(self):
        return source.materialize(source.build_source()[:3],'sensor_proxy')[0]

    def test_prefix_and_permutation(self):
        raw=self.raw(); full=m.predict(raw)
        prefix=m.predict({k:v[:65] for k,v in raw.items()})
        perm=m.predict({k:v[:,::-1] if v.ndim==2 else v for k,v in raw.items()})
        for name in full[1]:
            np.testing.assert_array_equal(full[1][name][:65],prefix[1][name])
            np.testing.assert_array_equal(full[1][name],perm[1][name])
        np.testing.assert_allclose(full[3]['bias_deg'],perm[3]['bias_deg'])
        np.testing.assert_allclose(full[3]['bias_deg'][:65],prefix[3]['bias_deg'])

    def test_known_offset_missing_and_expiry(self):
        raw=self.raw()
        raw['tof_status'][:]=255;raw['radar_valid'][:]=False
        raw['tof_packet_received'][:]=True;raw['radar_packet_received'][:]=True
        for i in range(3):
            raw['tof_status'][i,0]=5;raw['tof_range_m'][i,0]=2;raw['tof_theta_deg'][i,0]=0
            raw['radar_valid'][i,0]=True;raw['radar_range_m'][i,0]=2;raw['radar_angle'][i,0]=10
        mean,var,count,_=m.estimate_bias(raw)
        self.assertGreater(mean[2],9);self.assertLess(mean[2],10)
        self.assertGreater(var[2],2.5**2);self.assertLess(var[2],100)
        self.assertEqual(mean[24],0);self.assertEqual(var[24],100)
        self.assertEqual(count[40],0)
        raw['tof_packet_received'][:]=False
        self.assertFalse(m.estimate_bias(raw)[2].any())

    def test_ambiguous_range_is_not_pair(self):
        raw=self.raw();raw['tof_status'][:]=255;raw['radar_valid'][:]=False
        raw['tof_packet_received'][:]=True;raw['radar_packet_received'][:]=True
        raw['tof_status'][:,0]=5;raw['tof_range_m'][:,0]=2;raw['tof_theta_deg'][:,0]=0
        raw['radar_valid'][:,:2]=True;raw['radar_range_m'][:,:2]=2;raw['radar_angle'][:,:2]=10
        self.assertFalse(m.estimate_bias(raw)[2].any())

    def test_common_sensor_rotation_cancels(self):
        raw=self.raw(); shifted={k:v.copy() for k,v in raw.items()}
        shifted['radar_angle']+=37;shifted['tof_theta_deg']+=37
        a=m.estimate_bias(raw);b=m.estimate_bias(shifted)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,equal_nan=True,atol=1e-12)

    def test_no_calibration_equals_prior_and_no_false_angular_rate(self):
        raw=self.raw();raw['tof_packet_received'][:]=False
        _,values,_,_=m.predict(raw)
        np.testing.assert_array_equal(values['mz91_control'],values['mz92_calibrated'])
        history=[[i,2-.07*i,15,-.7] for i in range(5)]
        mean,cov=m.calibrated_localize(history,0,10,9)
        self.assertAlmostEqual(mean[2],5);self.assertAlmostEqual(mean[3],0)
        self.assertGreaterEqual(cov[2,2],9)


if __name__=='__main__':unittest.main()
