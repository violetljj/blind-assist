import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_stage_c_d6_veto_review_candidates import (
    build_windows,
    conservative_veto_summary,
    consensus_cell_rows,
)


class VetoReviewCandidateExportTest(unittest.TestCase):
    def test_build_windows_requires_consecutive_local_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in (0, 1, 2, 3, 4, 6):
                path = root / f"{index:06d}.png"
                path.write_bytes(b"x")
                rows.append(
                    {
                        "source_session_id": "session",
                        "camera": "chest",
                        "view": "left",
                        "frame_index": index,
                        "rgb_local_path": str(path),
                    }
                )

            windows = build_windows(rows)

            self.assertEqual(len(windows), 1)
            self.assertEqual(
                windows[0]["history_frame_indices"],
                [0, 1, 2, 3, 4],
            )

    def test_consensus_rows_rank_votes_before_score(self):
        windows = [
            {
                "window_id": "window",
                "source_session_id": "session",
                "anchor_frame_index": 4,
                "anchor_rgb_path": "frame.png",
            }
        ]
        shape = (3, 1, 3, 3, 6, 6)
        scores = np.zeros(shape, dtype=np.float32)
        active = np.zeros(shape, dtype=bool)
        risk = np.full(shape, 0.8, dtype=np.float32)
        known = np.full(shape, 0.9, dtype=np.float32)
        active[:1, 0, 1, 1, 0, 0] = True
        scores[:1, 0, 1, 1, 0, 0] = 0.6
        active[:, 0, 1, 1, 0, 1] = True
        scores[:, 0, 1, 1, 0, 1] = 0.5
        veto = np.zeros_like(active)
        veto[:2, 0, 1, 1, 0, 1] = True

        rows = consensus_cell_rows(
            windows,
            scores,
            active,
            risk,
            known,
            veto,
        )

        self.assertEqual(rows[0]["grid_column"], 1)
        self.assertEqual(rows[0]["active_vote_count"], 3)
        self.assertEqual(rows[0]["conservative_veto_vote_count"], 2)
        self.assertTrue(rows[0]["conservative_veto_consensus"])
        self.assertTrue(rows[0]["consensus_review_eligible"])
        self.assertFalse(rows[1]["consensus_review_eligible"])

    def test_conservative_summary_counts_full_window_clearance(self):
        active = np.zeros((2, 2, 3, 3, 6, 2), dtype=bool)
        active[:, :, 1, 1, 2, 0] = True
        veto = np.zeros_like(active)
        veto[0, 0, 1, 1, 2, 0] = True
        veto[1, 1, 1, 1, 2, 0] = True

        summary = conservative_veto_summary(
            active,
            veto,
            [(17, 0), (29, 1)],
        )

        self.assertEqual(
            summary["total_baseline_active_model_windows"], 4
        )
        self.assertEqual(
            summary["total_fully_cleared_model_windows"], 2
        )
        self.assertEqual(
            summary["window_count_fully_cleared_by_any_model"], 2
        )
        self.assertEqual(
            summary["window_count_fully_cleared_by_majority_models"],
            0,
        )
        self.assertEqual(
            summary[
                "central_total_fully_cleared_model_windows"
            ],
            2,
        )
        self.assertEqual(
            summary[
                "central_window_count_fully_cleared_by_any_model"
            ],
            2,
        )


if __name__ == "__main__":
    unittest.main()
