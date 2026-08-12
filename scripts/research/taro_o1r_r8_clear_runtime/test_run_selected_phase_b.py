from __future__ import annotations

import unittest

from scripts.research.taro_o1r_r8_clear_runtime import run_selected_phase_b as runner


class SelectedPhaseBTests(unittest.TestCase):
    def test_unchanged_gates_pass_on_well_formed_dual_class_fixture(self) -> None:
        identities = [(str(index), str(index)) for index in range(8)]
        sources = []
        labels = []
        for parent, video in identities:
            features = []
            query_labels = []
            for index in range(9):
                features.append({"r6_state": "OCCUPIED_OBSERVED", "query_receipt": {}, "occupied_hits": [[[True] * 3] * 3] * 3})
                query_labels.append({"state": "OCCUPIED_OBSERVED" if index < 2 else "CLEAR_OBSERVED" if index == 2 else "UNKNOWN"})
            sources.append({"parent_id": parent, "video_id": video, "query_features": features})
            labels.append({"query_labels": query_labels})
        result = runner.summarize(identities, sources * 7, labels * 7)
        self.assertTrue(result["scientifically_evaluable"])
        self.assertFalse(result["passed"])
        self.assertGreater(result["occupied_false_positive_against_definite_clear"], 0)

    def test_unknown_is_not_negative(self) -> None:
        self.assertEqual(runner.NOT_EVALUABLE_TERMINAL, "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_NOT_EVALUABLE_DUAL_CLASS_COVERAGE")


if __name__ == "__main__":
    unittest.main()
