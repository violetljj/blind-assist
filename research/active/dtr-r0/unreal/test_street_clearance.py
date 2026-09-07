"""Counterexamples for visible-surface clearance; no simulator truth inputs."""
import unittest
import numpy as np

from street_live_policy import MotionPolicy, depth_corridors


class ClearanceTest(unittest.TestCase):
    def test_off_axis_visible_obstacle_is_retained_by_buffer(self):
        depth = np.full((120, 160), 20.0)
        # At 90 degree FoV and 2 m depth these samples are ~.46-.61 m right.
        depth[42:82, 98:105] = 2.0
        old = depth_corridors(depth, 90, 1.6)
        new = depth_corridors(depth, 90, 1.6, buffered=True)
        self.assertIsNone(old['front_obstacle_m'])
        self.assertAlmostEqual(new['front_obstacle_m'], 2.0)
        self.assertNotIn('side_targets_m', old)

    def test_expanded_target_is_actually_reachable_by_nominal_selection(self):
        corridors = {'clearance_m': {str(k): 12.0 for k in (0.0,-1.2,1.2,-.72,.72,-.52,.52)},
                     'side_targets_m': [-1.2,1.2,-.72,.72,-.52,.52],
                     'front_obstacle_m': 2.0, 'valid_fraction': 1.0}
        policy = MotionPolicy('CANDIDATE_CLEARANCE')
        command = policy.command(t=0,x=0,y=0,goal_x=8,dtr_risk=True,corridors=corridors)
        self.assertEqual(command['target_y_m'], -1.2)
        self.assertLess(command['vy_mps'], 0)
        self.assertFalse(command['candidate_intervention'])
        self.assertEqual(command['candidate_evaluation']['admitted_tracks'], 0)

    def test_unknown_depth_still_stops(self):
        corridors = depth_corridors(np.zeros((120,160)),90,1.6,buffered=True)
        command = MotionPolicy('CANDIDATE_CLEARANCE').command(
            t=0,x=0,y=0,goal_x=8,dtr_risk=False,corridors=corridors)
        self.assertEqual(command['action'],'WAIT_UNKNOWN_DEPTH')
        self.assertEqual((command['vx_mps'],command['vy_mps']),(0,0))


if __name__ == '__main__':
    unittest.main()
