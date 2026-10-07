"""Frozen torso-input perturbations: clip state, complete queries, fixed sensor."""
import unittest
from unittest.mock import patch

import numpy as np

import cnh_torso_input_sensitivity_dev as S


def rotation(degrees):
    v = np.radians(np.asarray(degrees))
    r = np.broadcast_to(np.eye(4), v.shape+(4,4)).copy()
    r[...,0,0] = np.cos(v)
    r[...,0,2] = np.sin(v)
    r[...,2,0] = -np.sin(v)
    r[...,2,2] = np.cos(v)
    return r


def clip_bank(n=600):
    t = np.arange(n)/60.
    return dict(pelvis=np.c_[t,np.zeros(n)][None], torso=np.full((1,n),20.))


class TorsoInputSensitivityChecks(unittest.TestCase):
    def setUp(self):
        self.config = dict(tau_seconds=2.,straight_rate_deg_s=6.,torso_rate_deg_s=12.,
                           min_speed=.3,max_bias_rate_deg_s=3.,boxcar_frames=24,
                           midpoint_frames=42,earliest_update_frame=90)

    def test_fixed_constant_and_pulse_values_symmetry_and_clip_age(self):
        n = 600
        np.testing.assert_array_equal(S.perturbation('zero',n),np.zeros(n))
        positive = S.perturbation('const_pos',n)
        negative = S.perturbation('const_neg',n)
        np.testing.assert_array_equal(positive,np.full(n,5.))
        np.testing.assert_array_equal(negative,-positive)
        pulse = S.perturbation('pulse_pos',n)
        np.testing.assert_array_equal(S.perturbation('pulse_neg',n),-pulse)
        np.testing.assert_array_equal(pulse[:120],0.)
        np.testing.assert_array_equal(pulse[241:],0.)
        self.assertAlmostEqual(pulse[120],0.)
        self.assertAlmostEqual(pulse[240],0.)
        self.assertAlmostEqual(pulse[150],5.)
        self.assertAlmostEqual(pulse[180],10.)
        self.assertAlmostEqual(pulse[210],5.)
        self.assertGreaterEqual(pulse.min(),0.)
        self.assertLessEqual(pulse.max(),10.)
        np.testing.assert_allclose(pulse[120:181],pulse[180:241][::-1],atol=1e-12)
        for end in (30,120,150,180,240,301):
            np.testing.assert_array_equal(S.perturbation('pulse_pos',end),pulse[:end])

    def test_rotate_query_full_se3_broadcast_and_preserve_source_inputs(self):
        q = rotation(np.linspace(-30.,40.,16))
        q[:,:3,3] = np.c_[np.linspace(.1,.3,16),np.full(16,.05),np.full(16,.2)]
        prior = q.copy()
        delta = np.linspace(-5.,5.,16)
        observed = S.rotate_query(q,delta)
        np.testing.assert_allclose(observed,rotation(-delta)@q,atol=1e-12)
        np.testing.assert_array_equal(q,prior)
        self.assertGreater(np.abs(observed[:,:3,3]-q[:,:3,3]).max(),.001)
        batch = np.stack([q,q])
        deltas = np.stack([delta,-delta])
        batched = S.rotate_query(batch,deltas)
        np.testing.assert_allclose(batched,rotation(-deltas)@batch,atol=1e-12)
        np.testing.assert_allclose(S.rotate_query(q,np.zeros(16)),q,atol=1e-12)
        # The same world sensor must be recovered after changing only the
        # estimated travel frame's yaw; its current pelvis origin stays fixed.
        old_estimated = rotation(np.full(16,35.))
        old_estimated[:,:3,3] = np.c_[np.linspace(-1.,0.,16),np.zeros(16),np.full(16,.1)]
        old_sensor = old_estimated@q
        sensor_copy = old_sensor.copy()
        new_estimated = old_estimated@rotation(delta)
        np.testing.assert_allclose(new_estimated@observed,old_sensor,atol=1e-12)
        np.testing.assert_array_equal(old_sensor,sensor_copy)
        np.testing.assert_allclose(S.rotate_query(observed,-delta),q,atol=1e-12)

    def test_full_clip_state_uses_only_pelvis_torso_and_frozen_config(self):
        bank = clip_bank()
        original = {key:value.copy() for key,value in bank.items()}
        config = self.config.copy()
        import cnh_torso_gait_bias_dev as G
        import cnh_torso_bias_dev as T
        prepared = {'clip': np.array([0])}
        for name in S.NAMES:
            input_yaw = T.B.wrap(bank['torso'][0]+S.perturbation(name,600))
            gait,bias,updated,_ = G.causal_correct(bank['pelvis'][0],input_yaw,config)
            with patch.object(T,'labels',side_effect=AssertionError('future labels accessed')):
                observed = S.estimate_clip(bank,0,name,config)
            for key,expected in dict(torso=input_yaw,gait=gait,bias=bias,updated=updated,
                                     seen=np.cumsum(updated)>0).items():
                np.testing.assert_array_equal(observed[key],expected)
                self.assertEqual(len(observed[key]),600)
            # A sampled late window inherits native state, not a new zero bias.
            self.assertGreater(observed['bias'][300],0.)
            f = 132+12*np.arange(16)
            np.testing.assert_array_equal(observed['gait'][f],gait[f])
            for key,value in observed.items():
                prepared[name+'/'+key] = value[None]
        native_bank = dict(bank,corrected_gait=prepared['zero/gait'],exact=np.zeros((1,600)))
        frames = np.stack([120+12*np.arange(16),180+12*np.arange(16)])
        signs = np.array([1.,-1.])
        original_query = np.stack([rotation(np.full(16,15.)),rotation(np.full(16,-15.))])
        original_query[:,:,:3,3] = [.1,.03,.2]
        old_gait_delta = signs[:,None]*T.B.wrap(native_bank['corrected_gait'][0,frames]-bank['torso'][0,frames])
        saved = dict(clip_index=np.array([0,0]),native_frame_index=frames,sign=signs,
                     torso_query=original_query.copy(),
                     corrected_gait_query=S.rotate_query(original_query,old_gait_delta),
                     sensor=np.tile(np.eye(4),(2,16,1,1)))
        sensor_copy = saved['sensor'].copy()
        for name in S.NAMES:
            queries,states = S.unit_queries(saved,prepared,native_bank,name)
            for arm,original_arm in (('torso','torso'),('gait','corrected_gait')):
                sampled = prepared[name+'/'+arm][0,frames]
                delta = signs[:,None]*T.B.wrap(sampled-native_bank[original_arm][0,frames])
                np.testing.assert_allclose(queries[arm],rotation(-delta)@saved[original_arm+'_query'],atol=1e-12)
                np.testing.assert_array_equal(states[arm+'_heading_error'],signs[:,None]*T.B.wrap(sampled))
            for key in ('updated','seen','bias'):
                np.testing.assert_array_equal(states[key],prepared[name+'/'+key][0,frames])
            if name == 'zero':
                np.testing.assert_allclose(queries['gait'],saved['corrected_gait_query'],atol=1e-12)
                np.testing.assert_allclose(queries['torso'],saved['torso_query'],atol=1e-12)
        np.testing.assert_array_equal(saved['sensor'],sensor_copy)
        for key in bank:
            np.testing.assert_array_equal(bank[key],original[key])
        self.assertEqual(config,self.config)

    def test_future_perturbation_and_clip_truncation_leave_deadline_prefix_unchanged(self):
        bank = clip_bank()
        cutoff = 301
        changed = {key:value.copy() for key,value in bank.items()}
        changed['pelvis'][0,cutoff:] += np.c_[np.arange(600-cutoff),-np.arange(600-cutoff)]
        changed['torso'][0,cutoff:] = -130.
        prefix = {key:value[:,:cutoff].copy() for key,value in bank.items()}
        import cnh_torso_bias_dev as T
        q = rotation(np.linspace(-20.,30.,16))
        q[:,:3,3] = [.1,0.,.2]
        f = 120+12*np.arange(16)  # through nativeframe300 inclusive
        for name in ('zero','const_pos','const_neg','pulse_pos','pulse_neg'):
            reference = S.estimate_clip(bank,0,name,self.config)
            future = S.estimate_clip(changed,0,name,self.config)
            short = S.estimate_clip(prefix,0,name,self.config)
            for key in ('torso','gait','bias','updated','seen'):
                np.testing.assert_array_equal(reference[key][:cutoff],future[key][:cutoff])
                np.testing.assert_array_equal(reference[key][:cutoff],short[key])
            baseline = S.estimate_clip(bank,0,'zero',self.config)['gait']
            # This is the new-minus-old estimated heading supplied to query
            # rebuilding, without evaluator truth or sensor yaw perturbation.
            for sign in (1.,-1.):
                expected_delta = sign*T.B.wrap(reference['gait'][f]-baseline[f])
                future_delta = sign*T.B.wrap(future['gait'][f]-baseline[f])
                np.testing.assert_array_equal(S.rotate_query(q,expected_delta),S.rotate_query(q,future_delta))


if __name__ == '__main__':
    unittest.main()
