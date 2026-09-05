import unittest
import dtr_final_s10_temporal_design as design


class S10TemporalDesignTests(unittest.TestCase):
    def test_joint_negative_occlusion_and_reonset(self):
        receipt = design.analytic_receipt()
        self.assertEqual(len(receipt['future_contact_runs']), 2)
        self.assertGreaterEqual(max(r['sample_cell_duration_s'] for r in receipt['known_negative_in_full_window_runs']), .8)
        self.assertGreaterEqual(receipt['planned_full_disappearance_frames'], 6)
        self.assertGreaterEqual(receipt['pre_window_frustum_samples'], 8)
        self.assertGreaterEqual(receipt['post_exit_frustum_samples'], 8)
        self.assertEqual(receipt['analytic_frustum_intersection_samples'], 91)

    def test_shell_midpoint_during_window(self):
        value = design.design()
        for i in range(18,39):
            t=i/10
            target = design.c2.trajectory_position(value['target'],t)
            shell = design.c2.trajectory_position(value['shell'],t)
            wearer = design.c2.trajectory_position(value['wearer'],t)
            self.assertAlmostEqual(shell[0], (wearer[0]+.08+target[0])/2)
            self.assertAlmostEqual(shell[1], 0.)
        self.assertFalse(value['shell_collision_relevant'])
        self.assertFalse(value['shell_collisions_enabled'])


if __name__ == '__main__':
    unittest.main()
