import unittest
import numpy as np
from motion_return_diagnostic import auc, aggregate
from motion_return_public import sample_grid, zones_for_points
from tof_fov45_core import boxes45
from query_occupancy_data import observation_tokens

class MotionDiagnosticTest(unittest.TestCase):
    def test_auc_direction_ties_and_missing_class(self):
        self.assertEqual(auc([True,False],[0,1]),1)
        self.assertEqual(auc([True,False],[1,0]),0)
        self.assertEqual(auc([True,False],[1,1]),.5)
        self.assertIsNone(auc([True,True],[0,1]))
        with self.assertRaises(AssertionError):auc([True,False],[0,np.nan])

    def test_public_grid_matches_sensor_box_membership(self):
        flat,xy=sample_grid(); boxes=boxes45(); tof=observation_tokens(np.full(64,2.),boxes)
        predicted=zones_for_points(xy,tof[:,2:]);expected=np.full(len(flat),-1)
        y,x=flat//256,flat%256
        for j,(y0,x0,y1,x1) in enumerate(boxes):
            expected[(y>=y0)&(y<y1)&(x>=x0)&(x<x1)]=j
        np.testing.assert_array_equal(predicted,expected)

    def test_empty_population_not_perfect(self):
        result=aggregate([])['contributor']
        self.assertEqual(result['evaluable_zones'],0)
        self.assertIsNone(result['auc']['residual'])

if __name__=='__main__':unittest.main()
