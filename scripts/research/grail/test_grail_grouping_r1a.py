from __future__ import annotations

import unittest

import numpy as np

from grail_grouping_r1a import (
    aligned_context_score,
    pair_affinity,
    predict_groups,
    predicted_ordinals,
    select_by_predicted_ordinal,
    shifted_context_score,
)


def candidate(name: str, bbox: list[int], value: float = 1.0) -> dict:
    embedding = np.zeros(4, dtype=np.float32)
    embedding[0] = value
    embedding /= np.linalg.norm(embedding)
    return {"object_type": name, "bbox": bbox, "embedding": embedding}


class GroupingR1ATest(unittest.TestCase):
    def test_adjacent_same_type_parts_group_but_distant_parts_do_not(self) -> None:
        rows = [candidate("Drawer", [0, 0, 10, 10]), candidate("Drawer", [10, 0, 20, 10]),
                candidate("Drawer", [80, 0, 90, 10])]
        groups = predict_groups(rows)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        self.assertTrue(pair_affinity(rows[0], rows[1])["linked"])

    def test_ordinal_is_deterministic_within_predicted_group(self) -> None:
        rows = [candidate("Drawer", [0, 0, 10, 10]), candidate("Drawer", [10, 0, 20, 10])]
        self.assertEqual(predicted_ordinals(rows, [0, 0]), [("LEFT", "SINGLE"), ("RIGHT", "SINGLE")])

    def test_spatial_context_scores_reward_alignment(self) -> None:
        reference = np.eye(4, dtype=np.float32)
        aligned = reference.copy()
        reversed_tokens = reference[::-1].copy()
        self.assertGreater(aligned_context_score(aligned, reference), aligned_context_score(reversed_tokens, reference))
        grid = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (4, 1))
        self.assertAlmostEqual(shifted_context_score(grid, grid, radius=1), 1.0)

    def test_ordinal_selector_uses_appearance_only_for_collision(self) -> None:
        rows = [candidate("Drawer", [0, 0, 10, 10]), candidate("Drawer", [10, 0, 20, 10])]
        selected = select_by_predicted_ordinal(
            "Drawer", ("LEFT", "SINGLE"), rows,
            [("LEFT", "SINGLE"), ("RIGHT", "SINGLE")], [0.1, 0.9], ["a", "b"],
        )
        self.assertEqual(selected, (0, 1.0, "UNIQUE_PREDICTED_ORDINAL_MATCH"))


if __name__ == "__main__":
    unittest.main()
