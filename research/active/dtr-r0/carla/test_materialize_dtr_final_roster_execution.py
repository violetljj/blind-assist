import unittest
import materialize_dtr_final_roster_execution as execution


class ExecutionCandidateTest(unittest.TestCase):
    def test_groups_keep_frozen_seeds_and_exclude_references(self):
        for group, seed in [('FIT_ONLY',517031),('FINAL_A',517131),('FINAL_B',517231)]:
            protocol, annex = execution.materialize(group)
            self.assertEqual(seed, protocol['capture']['seed'])
            self.assertEqual(10, len(annex['main_episode_ids']))
            self.assertFalse(set(annex['main_episode_ids']) & set(annex['auxiliary_reference_ids']))
            self.assertFalse(annex['capture_authorized'])
            self.assertFalse(protocol['source_disjoint_contract']['capture_authorized'])

    def test_reference_changes_only_shell_and_temporal_gap_is_supported(self):
        protocol, annex = execution.materialize('FIT_ONLY')
        a, b = protocol['scenarios'][9], protocol['scenarios'][11]
        self.assertEqual(a['issued_plan'], b['issued_plan'])
        self.assertEqual(a['wearer_trajectory'], b['wearer_trajectory'])
        delta = [k for k in a['asset_trajectories'] if a['asset_trajectories'][k] != b['asset_trajectories'][k]]
        self.assertEqual(['c8_l01_c16_shell_02'], delta)
        negative = annex['s10_analytic']['known_negative_in_full_window_runs']
        self.assertGreaterEqual(negative[0]['end_inclusive_s']-negative[0]['start_s'], .8)


if __name__ == '__main__':
    unittest.main()
