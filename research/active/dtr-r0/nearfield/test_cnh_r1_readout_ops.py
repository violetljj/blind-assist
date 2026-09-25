import unittest
import numpy as np
from cnh_r1_readout_ops import (train_median_bias, known_xtalk, raw_query_weights,
    matched_correlation, query_scores, zone_cosines, shift_toward_near, temporal_mean)
from cnh_route_sensor import RAW_BIN_M, SensorParameters, _pulse_matrix


class ReadoutOpsTest(unittest.TestCase):
    def test_bias_train_only(self):
        raw=np.zeros((4,64,128));raw[0]=2;raw[1]=4;raw[2:]=999
        np.testing.assert_array_equal(train_median_bias(raw,[0,1]),3)
        raw[2:]=-1e9
        np.testing.assert_array_equal(train_median_bias(raw,np.array([1,1,0,0],bool)),3)
        with self.assertRaises(ValueError):train_median_bias(raw,[])

    def test_shift_sign_amount_and_zero_boundary(self):
        past=np.zeros((64,128));past[:,10]=1
        shifted=shift_toward_near(past,2*RAW_BIN_M)
        np.testing.assert_allclose(shifted[:,8],1)
        np.testing.assert_allclose(shifted[:,10],0)
        shifted=shift_toward_near(past,.5*RAW_BIN_M)
        np.testing.assert_allclose(shifted[:,9:11],.5)
        np.testing.assert_array_equal(shift_toward_near(past,200*RAW_BIN_M),0)

    def test_temporal_causal_clips_k1(self):
        raw=np.arange(5*64*128,dtype=np.float32).reshape(5,64,128)
        clips=np.array(['a','a','b','a','a']);steps=np.array([0,1,0,2,3])
        one=temporal_mean(raw,clips,steps,1,compensate=True,nominal_step_m=.1)
        self.assertEqual(one.dtype,raw.dtype)
        self.assertEqual(one.tobytes(),raw.tobytes())
        baseline=temporal_mean(raw,clips,steps,3)
        np.testing.assert_allclose(baseline[3],raw[[0,1,3]].astype(float).mean(0))
        np.testing.assert_array_equal(baseline[2],raw[2])
        changed=raw.copy();changed[4]=1e20
        np.testing.assert_array_equal(temporal_mean(changed,clips,steps,3)[:4],baseline[:4])

    def test_matched_pulse_signed_and_known_xtalk(self):
        matrix=_pulse_matrix(SensorParameters())
        raw=-7*matrix[30]
        self.assertAlmostEqual(float(matched_correlation(raw,matrix)[30]),-7,places=12)
        np.testing.assert_allclose(known_xtalk(4000,matrix,2),80*matrix[2])

    def test_temporal_compensation_uses_step_gap_and_zone_cosine(self):
        raw=np.zeros((2,64,128));raw[0,:,20]=1;raw[1,:,18]=1
        step=RAW_BIN_M
        result=temporal_mean(raw,['one','one'],np.array([0,2]),2,
                             nominal_step_m=step,compensate=True)
        expected=(raw[1]+shift_toward_near(raw[0],2*step*zone_cosines()))/2
        np.testing.assert_allclose(result[1],expected,atol=0,rtol=0)
        np.testing.assert_array_equal(result[0],raw[0])

    def test_raw_geometry_scores_cosines(self):
        weights=raw_query_weights()
        self.assertEqual(weights.shape,(6,64,128))
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_allclose(query_scores(np.ones((2,64,128)),weights),
                                   np.broadcast_to(weights.sum((1,2)),(2,6)))
        cosine=zone_cosines()
        self.assertTrue(np.all((cosine>0)&(cosine<=1)))
        self.assertGreater(cosine.reshape(8,8)[3,3],cosine.reshape(8,8)[0,0])

    def test_runner_k1_is_exactly_r2(self):
        from cnh_r1_readout_run import arms, h3
        rng=np.random.default_rng(7)
        raw=rng.normal(size=(3,64,128))
        data=dict(histogram=h3(raw).astype(np.float32),
                  clip_ids=np.array(['layout-a/centre']*3),steps=np.arange(3))
        scores,full,matched=arms(raw,data,np.zeros((64,128)))
        np.testing.assert_array_equal(scores['R3_K1_AVG'],scores['R2_MEDIAN'])
        np.testing.assert_array_equal(scores['R3_K1_MC'],scores['R2_MEDIAN'])
        np.testing.assert_array_equal(full['R3_K2_MC'],[False,True,True])


if __name__=='__main__':unittest.main()
