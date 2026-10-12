"""Focused causal geometry and gating behavior checks."""
import unittest
import numpy as np
import cnh_dual_gated_geometry as G

class GeometryChecks(unittest.TestCase):
    def test_exact_identity_volume(self):
        z=np.linspace(.9,2.5,200001)
        for q,(yl,yh) in enumerate(G.HEIGHTS):
            width=np.minimum(.29,z*G.EDGE)*2
            height=np.maximum(0,np.minimum(yh,z*G.EDGE)-np.maximum(yl,-z*G.EDGE))
            expected=np.trapezoid(width*height,z)/(.58*(yh-yl)*1.6)
            self.assertAlmostEqual(G.query_coverage(np.eye(4),q),expected,places=8)
    def test_mirror_symmetry(self):
        np.testing.assert_allclose(G.query_coverage(G.extrinsic(-15)),G.query_coverage(G.extrinsic(15)),atol=1e-10)
    def test_direction_causal(self):
        pose=np.repeat(np.eye(4)[None],16,axis=0);pose[:,2,3]=np.arange(16)*.16
        before=G.estimated_query_poses(pose,frames=[3]);pose[4:,:3,3]=1000
        np.testing.assert_array_equal(before,G.estimated_query_poses(pose,frames=[3]))
        pose[:]=np.eye(4)
        self.assertTrue((G.query_coverage(G.estimated_query_poses(pose))==0).all())
    def test_fusion(self):
        logits=np.array([[2.,3.,4.],[6.,1.,8.]])
        coverage=np.array([[.8,.1,.5],[.2,.7,.5]])
        np.testing.assert_array_equal(G.fuse(logits,coverage,'G1_0.5'),[2,1,8])
        np.testing.assert_array_equal(G.fuse(logits,coverage,'G2'),[2,1,8])
        np.testing.assert_allclose(G.fuse(logits,coverage,'G3'),[2.8,1.25,6])
        self.assertTrue(np.isneginf(G.fuse(logits,np.zeros_like(coverage),'G2')).all())
    def test_finite_cutoff_and_tied_clear_cost(self):
        from cnh_dual_gated_evaluate import cutoff
        values=np.array([-np.inf,1.,2.,2.,3.])
        threshold=cutoff(values,2,values)
        self.assertTrue(np.isfinite(threshold))
        self.assertEqual((values>=threshold).sum(),1)
        self.assertEqual(threshold,np.nextafter(2.,np.inf))
        absent=np.full(5,-np.inf)
        threshold=cutoff(absent,3,absent)
        self.assertTrue(np.isfinite(threshold))
        self.assertFalse((absent>=threshold).any())

if __name__=='__main__':unittest.main()
