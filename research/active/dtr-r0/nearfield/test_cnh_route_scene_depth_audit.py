"""Analytic CUDA checks for nearest-hit depth units and missing geometry."""
import unittest
import numpy as np
import torch
from cnh_route_scene_depth_audit import trace_cuda


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required by the actual diagnostic')
class NativeDepthAuditTests(unittest.TestCase):
    def test_nearest_surface_axial_units_and_large_world_origin(self):
        origin = np.array([10000., -20000., 30000.])
        triangles = np.array([[[-4,-4,4], [4,-4,4], [0,4,4]],
                              [[-4,-4,2], [4,-4,2], [0,4,2]]]) + origin
        depth, owner = trace_cuda(origin, np.array([[0,0,1], [.1,.1,1], [0,0,-1]]), triangles)
        np.testing.assert_allclose(depth[:2], [2,2])
        np.testing.assert_array_equal(owner, [1,1,-1])
        self.assertTrue(np.isinf(depth[2]))

    def test_empty_geometry_is_missing_not_far(self):
        depth, owner = trace_cuda(np.zeros(3), np.array([[0,0,1]]), np.empty((0,3,3)))
        self.assertTrue(np.isinf(depth[0]))
        self.assertEqual(owner[0], -1)


if __name__ == '__main__':
    unittest.main()
