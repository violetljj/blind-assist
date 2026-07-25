from __future__ import annotations

import json
import unittest
from pathlib import Path

import audit_jrdb_person_3d_trajectory_far_range_denominator_adequacy_metadata_blind_replication_r0 as audit
import run_jrdb_person_3d_trajectory_far_range_denominator_adequacy_metadata_blind_replication_r0 as runner


class FarRangeDenominatorReplicationR0Test(unittest.TestCase):
    def test_range_band_uses_frozen_boundaries(self) -> None:
        self.assertEqual(audit.range_band(9.999), "0-10")
        self.assertEqual(audit.range_band(10.0), "10-20")
        self.assertEqual(audit.range_band(20.0), "20-40")
        self.assertEqual(audit.range_band(40.0), "40-plus")

    def test_valid_box_requires_positive_finite_dimensions(self) -> None:
        valid = {"box": {"cx": 3, "cy": 4, "cz": 12, "w": 1, "l": 1, "h": 2, "rot_z": 0}}
        invalid = {"box": {**valid["box"], "h": 0}}
        self.assertIsNotNone(audit.valid_box(valid)[0])
        self.assertIsNone(audit.valid_box(invalid)[0])

    def test_support_fraction_preserves_four_class_denominator(self) -> None:
        rows = [
            {"classification": "sensor-supported"},
            {"classification": "annotation-only"},
            {"classification": "abstained"},
            {"classification": "invalid"},
        ]
        self.assertEqual(runner.support_fraction(rows), 0.25)

    def test_grouped_keeps_empty_result_out_of_scope(self) -> None:
        rows = [
            {
                "classification": "annotation-only",
                "point_support_band": "zero",
                "centroid_residual_3d_m": None,
            }
        ]
        grouped = runner.grouped(rows, "point_support_band")
        self.assertEqual(grouped["zero"]["denominator"], 1)
        self.assertEqual(grouped["zero"]["support_fraction"], 0.0)

    def test_preregistered_gate_is_cluster_level_and_strict(self) -> None:
        config = json.loads(
            Path(
                "configs/ustrf_jrdb_person_3d_trajectory_far_range_denominator_adequacy_metadata_blind_replication_r0.json"
            ).read_text(encoding="utf-8")
        )
        gate = config["denominator_adequacy"]
        self.assertEqual(gate["minimum_40_plus_object_frames_per_sequence"], 100)
        self.assertEqual(gate["minimum_adequate_sequences"], 3)
        self.assertEqual(gate["failure_terminal"], "DENOMINATOR_INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
