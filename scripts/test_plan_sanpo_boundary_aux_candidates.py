from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_sanpo_boundary_aux_candidates as planner


def candidate(session_id: str, pixels: int, *, split: str = "train", profile: str = "step_curb") -> dict:
    return {
        "session_id": session_id,
        "official_split": split,
        "camera": "camera_chest",
        "lens": "left",
        "selection_profile": profile,
        "recommended_start_frame": 42,
        "license": "Creative Commons Attribution 4.0 International",
        "sparse_frame_evidence": [
            {"profiles": {"step_curb": True, "best_boundary_target": {"pixel_count": pixels}}},
            {"profiles": {"step_curb": True, "best_boundary_target": {"pixel_count": pixels // 2}}},
        ],
    }


class PlanSanpoBoundaryAuxCandidatesTests(unittest.TestCase):
    def test_ranks_by_boundary_coverage_and_excludes_canonical_sessions(self) -> None:
        planned = planner.plan_candidates(
            [candidate("keep-low", 10), candidate("excluded", 1000), candidate("keep-high", 100)],
            excluded_session_ids={"excluded"},
            limit=2,
        )
        self.assertEqual(["keep-high", "keep-low"], [item["session_id"] for item in planned])
        self.assertEqual(2, planned[0]["sparse_boundary_frame_count"])
        self.assertEqual(150, planned[0]["sparse_boundary_pixel_sum"])

    def test_rejects_non_train_or_non_step_candidates(self) -> None:
        planned = planner.plan_candidates(
            [candidate("dev", 100, split="dev"), candidate("other", 100, profile="center_obstacle")],
            excluded_session_ids=set(),
            limit=8,
        )
        self.assertEqual([], planned)

    def test_rejects_invalid_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit"):
            planner.plan_candidates([], excluded_session_ids=set(), limit=0)


if __name__ == "__main__":
    unittest.main()
