from __future__ import annotations

import inspect
import unittest

from scripts.research.taro_o1r_r7_canary_runtime.test_r7_canary import _feature, _unavailable_source
from scripts.research.taro_o1r_r8_clear_runtime import clear_enrichment as runtime


class R8ClearEnrichmentTests(unittest.TestCase):
    def test_public_api_has_no_result_side(self) -> None:
        runtime.assert_public_api_truth_blind()
        for function in (runtime.score_parent, runtime.select_final_parents):
            self.assertFalse(any(token in name.lower() for name in inspect.signature(function).parameters for token in ("faro", "truth", "label", "outcome")))

    def test_candidate_requires_all_frozen_source_conditions(self) -> None:
        feature = _feature()
        self.assertTrue(runtime.query_is_clear_negative_control_candidate(feature))
        feature["far_valid_anchor_count"] = 8
        self.assertFalse(runtime.query_is_clear_negative_control_candidate(feature))

    def test_positive_evidence_vetoes_enrichment(self) -> None:
        feature = _feature()
        feature["occupied_hits"][0][0][2] = True
        self.assertFalse(runtime.query_is_clear_negative_control_candidate(feature))

    def test_unavailable_parent_scores_zero_without_clear_output(self) -> None:
        score = runtime.score_parent([_unavailable_source()])
        self.assertEqual(score["eligible_query_count"], 0)
        self.assertEqual(score["faro_reads"], 0)
        self.assertFalse(score["clear_output_emitted"])

    def test_rank_uses_source_score_then_frozen_tie(self) -> None:
        scores = []
        for index in range(24):
            scores.append({"selector_id": runtime.SELECTOR_ID, "parent_id": f"p{index}", "video_id": f"v{index}", "frame_count": 1, "query_count": 9, "available_query_count": 9, "eligible_query_count": index, "eligible_fraction_of_available": index / 24, "tie_break_sha256": f"{index:064X}", "faro_reads": 0, "truth_reads": 0, "clear_output_emitted": False})
        selected = runtime.select_final_parents(scores)
        self.assertEqual([row["eligible_query_count"] for row in selected], list(range(23, 15, -1)))


if __name__ == "__main__":
    unittest.main()
