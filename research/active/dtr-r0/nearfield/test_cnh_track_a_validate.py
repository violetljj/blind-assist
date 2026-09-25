import unittest
import numpy as np
from cnh_track_a_validate import exact_voxels


class ExactVoxelTest(unittest.TestCase):
    def test_sliver_and_closed_face(self):
        tiny=np.array([[[.011,.012,.013],[.011001,.012,.013],[.011,.012001,.013]]])
        self.assertEqual(exact_voxels(tiny),{(1,1,1)})
        on_face=tiny.copy();on_face[:,:,0]=.01
        self.assertEqual(exact_voxels(on_face),{(0,1,1),(1,1,1)})

    def test_not_aabb_occupancy(self):
        diagonal=np.array([[[.001,.001,.001],[.029,.001,.001],[.001,.029,.001]]])
        occupied=exact_voxels(diagonal)
        self.assertIn((0,0,0),occupied)
        self.assertNotIn((2,2,0),occupied)


if __name__=='__main__':unittest.main()
