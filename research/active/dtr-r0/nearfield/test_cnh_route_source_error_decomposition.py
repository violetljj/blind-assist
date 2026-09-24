"""Ensure diagnostic splitting preserves failures and all-ray denominators."""
import unittest
import numpy as np
from cnh_route_source_error_decomposition import edge_samples,group_metrics


class ErrorDecompositionTests(unittest.TestCase):
    def test_discontinuity_and_interior_are_independent_of_residual(self):
        depth=np.ones((9,9));depth[:,5:]=2
        yy=np.array([4,4,4]);xx=np.array([2,4,6])
        np.testing.assert_array_equal(edge_samples(depth,yy,xx),[False,True,False])
        np.testing.assert_array_equal(edge_samples(depth,yy,xx,3),[True,True,True])

    def test_validity_boundary_and_flat_plane(self):
        depth=np.ones((5,5));depth[2,3]=np.inf
        self.assertTrue(edge_samples(depth,np.array([2]),np.array([2]))[0])
        self.assertFalse(edge_samples(np.ones((5,5)),np.array([0]),np.array([0]))[0])

    def test_large_error_retained_and_denominator_includes_empty(self):
        metrics=group_metrics(np.ones(4,dtype=bool),np.array([.881,0,0,0]),
            np.array([True,True,False,False]),np.array([True,False,True,False]),np.array([1.,2.,3.,4.]))
        self.assertEqual(metrics['radial_absolute']['maximum_m'],.881)
        self.assertEqual(metrics['over_75mm'],1)
        self.assertEqual(metrics['missing_fraction'],.2)
        self.assertEqual(metrics['extra_fraction'],.3)


if __name__=='__main__':unittest.main()
