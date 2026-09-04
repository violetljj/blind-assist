from __future__ import annotations

import json
import unittest

import materialize_dtr_final_reckoning_source_probe as probe


class FinalReckoningSourceProbeTest(unittest.TestCase):
    def test_materialized_probe_has_ten_distinct_nonreusable_cells(self) -> None:
        protocol = probe.materialize(
            probe.read_json(probe.BASE_PROTOCOL),
            probe.read_json(probe.ROSTER_PROTOCOL),
        )
        cells = protocol["final_reckoning_source_probe"]["cells"]
        self.assertEqual(10, len(cells))
        self.assertEqual(10, len({row["stratum_id"] for row in cells}))
        self.assertFalse(protocol["final_reckoning_source_probe"]["probe_pixels_reusable_as_fit_or_final"])
        self.assertFalse(protocol["source_disjoint_contract"]["probe_pixels_reusable"])
        self.assertTrue(all(row["twin_role"] == "probe_only_unpaired" for row in protocol["scenarios"]))
        self.assertEqual(10, probe.analytic_receipt(protocol)["episode_count"])

    def test_probe_keeps_final_seeds_untouched_and_schedules_real_ego_yaw(self) -> None:
        protocol = probe.materialize(
            probe.read_json(probe.BASE_PROTOCOL),
            probe.read_json(probe.ROSTER_PROTOCOL),
        )
        roster = probe.read_json(probe.ROSTER_PROTOCOL)
        final_seeds = {
            row["capture_seed"] for row in roster["source_design"]["seed_groups"]
        }
        self.assertNotIn(probe.PROBE_SEED, final_seeds)
        self.assertEqual(
            60.0,
            sum(
                row["yaw_rate_degrees_per_second"]
                * (next_row["start_s"] - row["start_s"])
                for row, next_row in zip(
                    protocol["trajectory_library"]["fr_wearer_turn_safe"][
                        "yaw_segments"
                    ],
                    protocol["trajectory_library"]["fr_wearer_turn_safe"][
                        "yaw_segments"
                    ][1:],
                )
            ),
        )

    def test_roster_protocol_hash_is_bound(self) -> None:
        protocol = probe.materialize(
            probe.read_json(probe.BASE_PROTOCOL),
            probe.read_json(probe.ROSTER_PROTOCOL),
        )
        self.assertEqual(
            probe.c2.sha256_file(probe.ROSTER_PROTOCOL),
            protocol["final_reckoning_source_probe"]["roster_protocol_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
