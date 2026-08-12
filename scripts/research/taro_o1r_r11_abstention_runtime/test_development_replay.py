from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.research.taro_o1r_r11_abstention_runtime import abstention_candidate
from scripts.research.taro_o1r_r11_abstention_runtime import development_replay


ROOT = Path(__file__).resolve().parents[3]


class DevelopmentReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = development_replay.build_actual_replay(ROOT)

    def test_actual_r10_replay_is_exact_and_development_only(self) -> None:
        result = development_replay.validate_development_result(self.result)
        summary = result["summary"]
        self.assertEqual(summary["suppressed_base_positive_by_truth"], {"CLEAR_OBSERVED": 1, "OCCUPIED_OBSERVED": 1, "UNKNOWN": 0})
        self.assertEqual(summary["candidate_metrics"]["occupied_false_positive_against_definite_clear"], 0)
        self.assertFalse(result["fresh_confirmation_authority"])

    def test_candidate_replay_keeps_r10_terminal_immutable(self) -> None:
        self.assertEqual(self.result["r10_lineage"]["terminal"], development_replay.EXPECTED_R10_TERMINAL)
        self.assertFalse(self.result["route_promotion"])

    def test_resealed_metric_or_authority_mutation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.result)
        mutated["summary"]["candidate_metrics"]["occupied_true_positive"] += 1
        mutated.pop("content_sha256")
        mutated["content_sha256"] = development_replay.adapter.canonical_sha256(mutated)
        with self.assertRaises(development_replay.DevelopmentReplayError):
            development_replay.validate_development_result(mutated)
        mutated = copy.deepcopy(self.result)
        mutated["fresh_confirmation_authority"] = True
        mutated.pop("content_sha256")
        mutated["content_sha256"] = development_replay.adapter.canonical_sha256(mutated)
        with self.assertRaises(development_replay.DevelopmentReplayError):
            development_replay.validate_development_result(mutated)

    def test_algorithm_identity_is_bound(self) -> None:
        self.assertEqual(self.result["candidate"], abstention_candidate.FROZEN_ALGORITHM)


if __name__ == "__main__":
    unittest.main()
