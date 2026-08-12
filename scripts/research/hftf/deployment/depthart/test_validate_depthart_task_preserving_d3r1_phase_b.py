from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart import materialize_depthart_task_preserving_d3r1_phase_b as producer
from scripts.research.hftf.deployment.depthart import validate_depthart_task_preserving_d3r1_phase_b as subject


class D3R1PhaseBValidatorUnitTest(unittest.TestCase):
    def test_independent_gate_and_selection_cover_pass_and_fail(self) -> None:
        thresholds = {
            "minimum_truth_known_cells": 18,
            "minimum_truth_clear_cells": 9,
            "minimum_truth_occupied_cells": 9,
            "minimum_truth_clear_cells_per_band_horizon": 1,
            "minimum_truth_occupied_cells_per_band_horizon": 1,
            "minimum_valid_band_clearances": 3,
        }
        counts = producer.empty_counts()
        counts["valid_band_clearances"] = 3
        for key in counts["known_by_grid"]:
            counts["known_by_grid"][key] = 2
            counts["clear_by_grid"][key] = 1
            counts["occupied_by_grid"][key] = 1
        counts["known_cells"], counts["clear_cells"], counts["occupied_cells"] = 18, 9, 9
        self.assertTrue(subject.independently_qualifies(counts, thresholds)[0])
        rows = [
            {
                "selection_order": index,
                "pool_order": index,
                "visit_id": str(index),
                "video_id": str(100 + index),
                "selected_frame_plan_sha256": f"{index:064X}",
                "source_truth_support_qualified": index <= 15,
            }
            for index in range(1, 33)
        ]
        self.assertEqual(subject.independently_finalize(rows), (False, []))
        rows[15]["source_truth_support_qualified"] = True
        passed, selected = subject.independently_finalize(rows)
        self.assertTrue(passed)
        self.assertEqual(len(selected), 16)

    def test_comparable_checkpoint_excludes_transport_but_keeps_scientific_payload(self) -> None:
        payload = {
            "frame_count": 300,
            "selected_frame_plan_sha256": "A" * 64,
            "archive_validation": {},
            "trajectory_row_count": 2,
            "maximum_pose_bracketing_gap_seconds": 0.1,
            "depth_sizes_wh": [[256, 192]],
            "confidence_values": [0, 1, 2],
            "orientation_counts": {"0": 0, "1": 300, "2": 0, "3": 0},
            "truth_support": {"known_cells": 1},
            "source_truth_support_qualified": False,
            "qualification_failures": ["fixture"],
            "phase_a_sources": [],
            "rgb_read": False,
            "model_output_read": False,
            "per_frame_truth_retained": False,
            "source_assets": [{"transport": "not replay output"}],
        }
        comparable = subject.comparable_checkpoint(payload)
        self.assertNotIn("source_assets", comparable)
        self.assertEqual(comparable["truth_support"], {"known_cells": 1})

    def test_inventory_rejects_extra_source_root_file_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source" / "Training"
            receipts = root / "receipts"
            source_root.mkdir(parents=True)
            receipts.mkdir()
            (source_root / "unexpected.txt").write_text("x", encoding="utf-8")
            selected = [
                {"selection_order": index, "pool_order": index, "visit_id": str(index), "video_id": str(100 + index), "fold": "Training"}
                for index in range(1, 33)
            ]
            with self.assertRaises(ValueError):
                subject.validate_inventory(root, selected, {}, {})


if __name__ == "__main__":
    unittest.main()
