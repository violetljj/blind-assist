from __future__ import annotations

import copy
import unittest

from scripts.research.taro_o1r_r11_abstention_runtime import abstention_candidate as candidate


def feature(*, base: bool = False, stronger_cell: tuple[int, int, int] | None = None) -> dict:
    hits = [[[False for _ in range(3)] for _ in range(3)] for _ in range(4)]
    if stronger_cell is not None:
        base = True
        pixel, height, forward = stronger_cell
        hits[pixel][height][forward] = True
    if base:
        hits[0][0][2] = True
    return {
        "query_id": "q",
        "query_receipt": {},
        "grid_index": 0,
        "r6_state": "UNKNOWN",
        "occupied_hits": hits,
        "positive_obstacle_veto": bool(base),
        "far_valid_anchor_count": 9,
        "far_fractions": [0.0, 0.0, 0.0],
        "observed_support_points": 128,
        "reason_codes": [],
    }


class AbstentionCandidateTests(unittest.TestCase):
    def test_weak_two_pixel_positive_abstains(self) -> None:
        self.assertEqual(candidate.state_from_feature(feature(base=True))[0], "UNKNOWN")

    def test_any_adjacent_margin_positive_is_occupied(self) -> None:
        for cell in candidate.STRONGER_CELLS_ANY:
            with self.subTest(cell=cell):
                self.assertEqual(candidate.state_from_feature(feature(stronger_cell=cell))[0], "OCCUPIED_OBSERVED")

    def test_prior_occupied_is_preserved(self) -> None:
        row = feature()
        row["r6_state"] = "OCCUPIED_OBSERVED"
        self.assertEqual(candidate.state_from_feature(row)[0], "OCCUPIED_OBSERVED")

    def test_result_side_field_and_non_subset_grid_fail_closed(self) -> None:
        leaked = feature()
        leaked["truth_state"] = "CLEAR_OBSERVED"
        with self.assertRaisesRegex(candidate.AbstentionCandidateError, "feature surface"):
            candidate.state_from_feature(leaked)
        invalid = feature()
        invalid["occupied_hits"][3][0][2] = True
        with self.assertRaisesRegex(candidate.AbstentionCandidateError, "not a subset"):
            candidate.state_from_feature(invalid)

    def test_algorithm_is_no_clear_no_training(self) -> None:
        frozen = copy.deepcopy(candidate.FROZEN_ALGORITHM)
        self.assertFalse(frozen["clear_output_allowed"])
        self.assertFalse(frozen["unknown_is_negative"])
        self.assertEqual(frozen["training_steps"], 0)
        self.assertEqual(len(frozen["stronger_cells_any"]), 3)
        self.assertEqual(frozen["stronger_cells_any"][0]["minimum_connected_confidence2_pixels"], 16)


if __name__ == "__main__":
    unittest.main()
