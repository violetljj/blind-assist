import unittest
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from cnh_track_a_readout import cell_points, _transform, _indices, query_weights, transport
from cnh_scan_development import scan
from cnh_track_a_v12_readout import (transport_matrix, accumulate, scan_transported,
                                    s1, memory_scan, sequence_readouts)


class TrackAV12Test(unittest.TestCase):
    def test_identity_transport_and_current_gain(self):
        a,w,d=transport_matrix(np.eye(4))
        np.testing.assert_allclose(a.toarray(),np.eye(1024),atol=1e-14)
        np.testing.assert_array_equal(w,1)
        r=np.full((1,8,8,16),3.);v=np.full_like(r,2.)
        acc=accumulate(r,v,np.eye(4)[None])
        np.testing.assert_array_equal(acc['mean'],r)
        np.testing.assert_array_equal(acc['mean_variance'],v)

    def test_gain_per_point_before_summing_and_coverage_unchanged(self):
        t=np.eye(4);t[2,3]=-.08
        a,w,_=transport_matrix(t)
        points=cell_points();now=_transform(points,t);ids,valid=_indices(now)
        gain=(np.linalg.norm(points,axis=-1)/np.linalg.norm(now,axis=-1))**2
        source=100
        expected=np.bincount(ids[source,valid[source]],weights=gain[source,valid[source]]/16,minlength=1024)
        np.testing.assert_allclose(a[:,source].toarray().ravel(),expected,atol=1e-14)
        _,old_w=transport(np.ones((8,8,16)),t)
        np.testing.assert_array_equal(w,old_w)

    def test_rigid_quarter_turn_and_static_variance(self):
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
        a,w,_=transport_matrix(t)
        r=np.arange(1024,dtype=float).reshape(8,8,16)
        old,_=transport(r,t)
        np.testing.assert_allclose((a@r.ravel()).reshape(8,8,16),old,atol=1e-10)
        np.testing.assert_array_equal(w,1)
        values=np.ones((4,8,8,16));poses=np.repeat(np.eye(4)[None],4,axis=0)
        out=accumulate(values,values,poses)
        np.testing.assert_allclose(out['mean'][-1],1,atol=1e-14)
        np.testing.assert_allclose(out['mean_variance'][-1],.25,atol=1e-14)

    def test_window_covariance_combines_same_source(self):
        # One noisy original cell splat equally into two adjacent bins.
        a=sparse.csr_matrix(([.5,.5],([0,1],[0,0])),shape=(1024,1024))
        r=np.zeros((1,8,8,16));v=np.zeros_like(r)
        r.flat[0]=4;v.flat[0]=9
        weights=np.zeros((6,64,16));weights[:,0,:2]=1
        scores=scan_transported(r,v,[(0,a)],weights,.5)
        np.testing.assert_allclose(scores,4/3,atol=1e-14)

    def test_s1_exact_existing_scan_and_k1_s2(self):
        rng=np.random.default_rng(3)
        r=rng.normal(size=(1,8,8,16));v=np.ones_like(r)
        weights=query_weights(np.eye(4))
        expected=scan(r.reshape(1,64,16),v.reshape(1,64,16),weights,.5)
        np.testing.assert_array_equal(s1(r,v,weights,.5),expected)
        output=sequence_readouts(r,v,np.eye(4)[None],np.eye(4),tau_s1=.5,tau_s2=.5)
        np.testing.assert_array_equal(output['S1'],output['S2'])
        np.testing.assert_array_equal(output['S3'],output['S2'])

    def test_memory_empty_identity_and_causal(self):
        r=np.zeros((3,8,8,16));v=np.ones_like(r)
        poses=np.repeat(np.eye(4)[None],3,axis=0)
        # Turn current ToF away from Q; retain historical forward observations.
        poses[1,:3,:3]=Rotation.from_euler('y',40,degrees=True).as_matrix()
        tq=np.eye(4)@poses[1] # Q stays world-aligned at same origin.
        r[0]=1
        before=memory_scan(r,v,poses,1,tq)
        self.assertTrue(np.any(before>-50))
        r[2]=1e9
        np.testing.assert_array_equal(memory_scan(r,v,poses,1,tq),before)
        np.testing.assert_array_equal(memory_scan(r,v,poses,0,np.eye(4)), -50)

    def test_all_readouts_causal_and_h3_only(self):
        rng=np.random.default_rng(7)
        r=rng.normal(size=(3,8,8,16));v=np.ones_like(r)
        poses=np.repeat(np.eye(4)[None],3,axis=0);poses[:,2,3]=[0,.1,.2]
        before=sequence_readouts(r,v,poses,np.eye(4),tau_s1=.25,tau_s2=.5)
        r[2]=1e8;v[2]=1e6;poses[2,2,3]=100
        after=sequence_readouts(r,v,poses,np.eye(4),tau_s1=.25,tau_s2=.5)
        for arm in ('B1_R','S1','S2','S3'):
            np.testing.assert_array_equal(before[arm][:2],after[arm][:2])
        with self.assertRaises(ValueError):
            accumulate(np.zeros((3,8,8,128)),np.ones((3,8,8,128)),poses)


if __name__=='__main__':unittest.main()
