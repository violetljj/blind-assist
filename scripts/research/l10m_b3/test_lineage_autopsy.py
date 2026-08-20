from __future__ import annotations

import unittest

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, render_raw
from scripts.research.l10m_b3.lineage_autopsy import (
    INITIAL_SCORE,
    TARGET_CANONICAL_SHA256,
    reconstruct_trajectory,
)


class LineageAutopsyTest(unittest.TestCase):
    def test_reconstructs_strict_incumbent_and_repeated_target(self) -> None:
        initial = render_raw(INITIAL_SPEC)
        target = initial.replace("TURN_THRESHOLD = 0.20", "TURN_THRESHOLD = 0.10")
        events = []
        for generation in range(1, 9):
            is_target = generation in (2, 4, 8)
            events.append(
                {
                    "arm": "raw",
                    "seed": 89,
                    "generation": generation,
                    "request_id": f"request-{generation}",
                    "candidate_output": target if is_target else initial,
                    "semantic_valid": True,
                    "unsafe_candidate": False,
                    "behavioral_score": 0.993103448275862 if is_target else INITIAL_SCORE,
                }
            )

        result = reconstruct_trajectory(events)

        self.assertEqual(result["target_proposed_generations"], [2, 4, 8])
        self.assertEqual(result["target_accepted_generations"], [2])
        self.assertEqual(result["final_incumbent_canonical_sha256"], TARGET_CANONICAL_SHA256)
        self.assertEqual(result["lineage"][1]["disposition"], "accepted_strict_improvement")
        self.assertEqual(result["lineage"][3]["disposition"], "not_retained_not_strictly_better")
        self.assertFalse(result["dedup_mechanism_present"])

    def test_rejects_incomplete_generation_lineage(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "generations 1..8"):
            reconstruct_trajectory(
                [
                    {
                        "arm": "raw",
                        "seed": 89,
                        "generation": 1,
                        "request_id": "only-one",
                        "candidate_output": render_raw(INITIAL_SPEC),
                        "semantic_valid": True,
                        "unsafe_candidate": False,
                        "behavioral_score": INITIAL_SCORE,
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
