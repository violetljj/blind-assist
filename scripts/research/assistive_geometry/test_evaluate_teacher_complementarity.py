from __future__ import annotations

import copy
import unittest

from scripts.research.assistive_geometry.evaluate_teacher_complementarity import (
    BANDS,
    HORIZONS,
    TEACHERS,
    compute_metrics,
)


GATES = {
    "oracle_clearance_relative_gain_min": 0.05,
    "oracle_false_clear_absolute_gain_min": 0.01,
    "exclusive_correct_parent_count_min": 2,
    "disagreement_error_rate_excess_min": 0.10,
    "temporal_delta_mae_advantage_min_m": 0.01,
}


def fixture() -> list[dict]:
    rows = []
    for parent_index, parent in enumerate(("p0", "p1")):
        for sequence in range(4):
            bands = []
            for band_index, band in enumerate(BANDS):
                truth_clearance = 0.8 + 0.05 * sequence + 0.1 * band_index
                truth_states = []
                teacher_states = {teacher: [] for teacher in TEACHERS}
                for horizon_index, _ in enumerate(HORIZONS):
                    truth = "OCCUPIED_OBSERVED" if (parent_index + sequence + band_index + horizon_index) % 2 == 0 else "CLEAR_OBSERVED"
                    truth_states.append(truth)
                    pattern = (sequence + band_index + horizon_index) % 4
                    opposite = "CLEAR_OBSERVED" if truth == "OCCUPIED_OBSERVED" else "OCCUPIED_OBSERVED"
                    teacher_states["metric_teacher"].append(opposite if pattern == 0 else truth)
                    teacher_states["temporal_geometry_teacher"].append(opposite if pattern == 1 else truth)
                bands.append(
                    {
                        "band": band,
                        "truth_clearance_valid": True,
                        "truth_clearance_m": truth_clearance,
                        "truth_states": truth_states,
                        "teachers": {
                            "metric_teacher": {
                                "clearance_valid": True,
                                "clearance_m": truth_clearance + (0.12 if sequence % 2 == 0 else -0.12),
                                "states": teacher_states["metric_teacher"],
                            },
                            "temporal_geometry_teacher": {
                                "clearance_valid": True,
                                "clearance_m": truth_clearance + 0.02,
                                "states": teacher_states["temporal_geometry_teacher"],
                            },
                        },
                    }
                )
            rows.append(
                {
                    "schema": "blindassist_assistive_geometry_teacher_complementarity_frame_v1",
                    "data_role": "TEACHER_EVALUATION",
                    "parent_id": parent,
                    "session_id": f"{parent}-session",
                    "sequence_index": sequence,
                    "bands": bands,
                }
            )
    return rows


class TeacherComplementarityTests(unittest.TestCase):
    def test_complementary_fixture_passes_kill_gate(self) -> None:
        result = compute_metrics(fixture(), GATES)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["c1_training_authorized"])
        self.assertTrue(all(result["gates"].values()))
        self.assertGreaterEqual(result["complementarity"]["oracle_false_clear_absolute_gain"], 0.01)
        self.assertGreater(result["complementarity"]["temporal_teacher_delta_mae_advantage_m"], 0.01)

    def test_stricter_parent_support_closes_c1(self) -> None:
        gates = {**GATES, "exclusive_correct_parent_count_min": 3}
        result = compute_metrics(fixture(), gates)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["gates"]["bidirectional_parent_support"])
        self.assertFalse(result["c1_training_authorized"])

    def test_unknown_truth_is_not_counted_as_clear(self) -> None:
        rows = copy.deepcopy(fixture())
        rows[0]["bands"][0]["truth_states"][0] = "UNKNOWN"
        for teacher in TEACHERS:
            rows[0]["bands"][0]["teachers"][teacher]["states"][0] = "CLEAR_OBSERVED"
        result = compute_metrics(rows, GATES)
        for teacher in TEACHERS:
            self.assertEqual(result["teachers"][teacher]["occupancy"]["truth_known_support"], 71)


if __name__ == "__main__":
    unittest.main()
