from __future__ import annotations

import unittest

import materialize_dtr_final_visual_shell_probe as probe


class VisualShellMaterializerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = probe.materialize()

    def test_only_visual_shells_change_paired_sources(self):
        scenarios = {s["episode_id"]: s for s in self.protocol["scenarios"]}
        self.assertEqual(12, len(scenarios))
        for pair in self.protocol["final_visual_shell_probe"]["pairs"]:
            primary = scenarios[pair["episode_id"]]
            reference = scenarios[pair["reference_episode_id"]]
            for key in ("issued_plan", "wearer_trajectory", "expected_responsible_assets", "layout_id"):
                self.assertEqual(primary[key], reference[key])
            changes = [key for key in primary["asset_trajectories"]
                       if primary["asset_trajectories"][key] != reference["asset_trajectories"][key]]
            self.assertEqual([pair["shell_asset"]], changes)
            assets = probe.base.c2.materialize_layout_assets(self.protocol, primary["layout_id"])
            shell = next(a for a in assets if a["asset_key"] == pair["shell_asset"])
            self.assertFalse(shell["collision_relevant"])
            self.assertFalse(shell["collisions_enabled"])

    def test_full_panel_tracks_midpoint_during_frozen_window(self):
        library = self.protocol["trajectory_library"]
        position = probe.base.c2.trajectory_position
        for t in (5.0, 5.2, 5.4, 5.6):
            wearer = position(library["fr_wearer_two_contact"], t)
            target = position(library["fr_target_two_crossings"], t)
            shell = position(library["fr_visual_full"], t)
            for axis in (0, 1):
                self.assertAlmostEqual((wearer[axis] + target[axis]) / 2, shell[axis])

    def test_probe_cannot_supply_formal_pixels(self):
        self.assertTrue(self.protocol["source_disjoint_contract"]["probe_seed_disjoint_from_fit_and_final"])
        self.assertFalse(self.protocol["final_visual_shell_probe"]["probe_pixels_reusable_as_fit_or_final"])
        self.assertFalse(self.protocol["final_visual_shell_probe"]["method_predictions_or_scores_allowed"])


if __name__ == "__main__":
    unittest.main()
