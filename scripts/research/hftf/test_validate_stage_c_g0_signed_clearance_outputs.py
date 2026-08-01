from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_stage_c_g0_signed_clearance_outputs import (  # noqa: E402
    _metric_passes,
    _validate_role_ids,
)


def _records(prefix: str, count: int) -> list[dict[str, str]]:
    return [{"session_id": f"{prefix}-{index}"} for index in range(count)]


def _metric() -> dict[str, int]:
    return {
        "positive_known": 5,
        "negative_known": 20,
        "binary_equivalence_violations": 0,
        "known_nonfinite_clipped_target": 0,
        "unknown_nonnull_target_violations": 0,
        "unknown_to_safe_violations": 0,
        "distinct_clipped_target_millimeter_bins": 20,
        "known_near_boundary": 5,
        "risk_not_clip_min": 1,
        "safe_not_clip_max": 1,
    }


class StageCG0SignedClearanceOutputValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gates = {
            "each_source_height_distinct_clipped_"
            "millimeter_bins_minimum": 20,
            "each_source_height_known_near_boundary_count_minimum": 5,
        }

    def test_exact_disjoint_roles_pass(self) -> None:
        _validate_role_ids(
            _records("development", 9),
            _records("fresh", 3),
            _records("heldout", 3),
        )

    def test_role_overlap_fails(self) -> None:
        development = _records("development", 9)
        fresh = _records("fresh", 3)
        heldout = _records("heldout", 3)
        heldout[0] = dict(fresh[0])
        with self.assertRaisesRegex(ValueError, "multiple roles"):
            _validate_role_ids(development, fresh, heldout)

    def test_exact_metric_gate_passes(self) -> None:
        self.assertTrue(_metric_passes(_metric(), self.gates))

    def test_unknown_nonnull_violation_fails(self) -> None:
        metric = _metric()
        metric["unknown_nonnull_target_violations"] = 1
        self.assertFalse(_metric_passes(metric, self.gates))

    def test_clip_degenerate_class_fails(self) -> None:
        metric = _metric()
        metric["safe_not_clip_max"] = 0
        self.assertFalse(_metric_passes(metric, self.gates))


if __name__ == "__main__":
    unittest.main()
