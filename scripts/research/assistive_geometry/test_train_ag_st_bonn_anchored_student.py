from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_ag_st_factor_labels import (  # noqa: E402
    PROVENANCE_SOURCE_NATIVE,
    PROVENANCE_UNKNOWN,
)
from evaluate_ag_st_student_bonn_depth import (  # noqa: E402
    BONN_HEIGHT,
    BONN_WIDTH,
    FIXED_FRAME_INDICES_BY_SEQUENCE,
    load_cohort_indices,
)
from train_ag_st_bonn_anchored_student import (  # noqa: E402
    DEFAULT_COHORT_MANIFEST,
    _unknown_factor_targets,
)
from train_ag_st_masked_student import TIER_A_SOURCE, TIER_UNKNOWN  # noqa: E402


class BonnAnchoredStudentTest(unittest.TestCase):
    def test_cohort_roles_are_eight_by_eight_and_disjoint(self) -> None:
        fit = load_cohort_indices(DEFAULT_COHORT_MANIFEST, "fit")
        evaluation = load_cohort_indices(DEFAULT_COHORT_MANIFEST, "evaluation")
        self.assertEqual((8, 8), (len(fit), len(evaluation)))
        self.assertFalse(set(fit) & set(evaluation))
        self.assertFalse(set(fit) & set(FIXED_FRAME_INDICES_BY_SEQUENCE))
        self.assertFalse(set(evaluation) & set(FIXED_FRAME_INDICES_BY_SEQUENCE))

    def test_bonn_targets_expose_only_source_depth(self) -> None:
        depth = np.ones((BONN_HEIGHT, BONN_WIDTH), dtype=np.float32)
        valid = np.ones((BONN_HEIGHT, BONN_WIDTH), dtype=np.bool_)
        valid[0, 0] = False
        targets = _unknown_factor_targets(depth, valid)
        self.assertEqual(TIER_A_SOURCE, int(targets["metric_tier"][0, 0, 1, 1]))
        self.assertEqual(TIER_UNKNOWN, int(targets["metric_tier"][0, 0, 0, 0]))
        self.assertEqual(
            PROVENANCE_SOURCE_NATIVE,
            int(targets["metric_provenance"][0, 0, 1, 1]),
        )
        self.assertEqual(
            PROVENANCE_UNKNOWN,
            int(targets["metric_provenance"][0, 0, 0, 0]),
        )
        self.assertEqual(0, int(targets["support_valid"].sum()))
        self.assertEqual(0, int(targets["evidence_valid"].sum()))


if __name__ == "__main__":
    unittest.main()
