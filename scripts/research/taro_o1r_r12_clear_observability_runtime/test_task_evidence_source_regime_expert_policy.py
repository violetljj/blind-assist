from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_multi_candidate_reliability_consistency as r31
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_source_regime_expert_policy as subject


def candidate(index: int, translation: float) -> scorer.CandidateRecord:
    reference = bonn.Frame("parent", 0.0, Path("reference"), Path("reference"), np.eye(4))
    neighbor = bonn.Frame(
        "parent", float(index + 1), Path(f"rgb-{index}"), Path(f"depth-{index}"), np.eye(4)
    )
    pair = bonn.Pair(reference, neighbor, float(index + 1), translation, 0.0)
    return scorer.CandidateRecord(
        "parent",
        "OUTCOME_BLIND_TEST",
        reference.frame_id,
        pair,
        np.zeros(len(r31.GLOBAL_FEATURE_NAMES), dtype=np.float64),
        {},
    )


class SourceRegimeExpertPolicyTest(unittest.TestCase):
    def test_r31_gating_does_not_require_fresh_targets(self) -> None:
        records = [candidate(0, 0.1), candidate(1, 0.3), candidate(2, 0.2)]
        utility = np.asarray([[0.0, 1.0, 0.2], [0.0, 0.9, 0.1], [0.0, 1.1, 0.3]])
        opportunity = np.asarray([[0.0, 0.8, 0.1], [0.0, 0.9, 0.2], [0.0, 0.7, 0.1]])
        scores, receipt = r31.gated_scores(records, utility, opportunity)
        self.assertEqual(1, int(np.sum(scores)))
        self.assertFalse(receipt["outcome_diagnostics_available"])
        self.assertIsNone(receipt["harmful_override_count"])

    def test_source_regime_opportunity_policy_is_outcome_blind(self) -> None:
        records = [candidate(0, 0.1), candidate(1, 0.3), candidate(2, 0.2)]
        utility = np.zeros((3, 3), dtype=np.float64)
        opportunity = np.asarray([[0.0, 0.2, 1.0], [0.0, 0.1, 0.9], [0.0, 0.3, 1.1]])
        geometry = np.full(3, np.nan)
        scores, receipt = subject.select(
            records, utility, opportunity, geometry, "OPPORTUNITY_MARGIN_TOP"
        )
        self.assertEqual(1, int(np.sum(scores)))
        self.assertEqual(1, receipt["reference_count"])


if __name__ == "__main__":
    unittest.main()
