from __future__ import annotations

import unittest

from evaluate_dav2_model_variant_gate_r1 import (
    truth_change_summary,
    truth_geometry_summary,
)


def valid_field(occupied: bool) -> dict:
    return {
        "status": "VALID",
        "bands": {
            band: {
                "occupied_by_horizon": {
                    "1.0": occupied,
                    "1.5": occupied,
                    "2.0": occupied,
                }
            }
            for band in ("left", "center", "right")
        },
    }


class TruthReferencedGeometryTest(unittest.TestCase):
    def test_false_block_is_distinct_from_false_clear(self) -> None:
        rows = [
            {
                "sequence_id": "s",
                "sensor": valid_field(False),
                "candidate": valid_field(True),
            }
        ]
        result = truth_geometry_summary(rows, "candidate")
        self.assertEqual(result["false_clear_count"], 0)
        self.assertEqual(result["false_block_count"], 9)
        self.assertEqual(result["truth_geometry_state_exact_agreement"], 0.0)

    def test_truth_improvement_is_beneficial_not_harmful(self) -> None:
        rows = [
            {
                "sequence_id": "s",
                "sensor": valid_field(True),
                "baseline": valid_field(False),
                "candidate": valid_field(True),
            }
        ]
        result = truth_change_summary(rows)
        self.assertEqual(result["beneficial_changes"], 9)
        self.assertEqual(result["harmful_changes"], 0)
        self.assertEqual(result["net_beneficial_changes"], 9)


if __name__ == "__main__":
    unittest.main()
