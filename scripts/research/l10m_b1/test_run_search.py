from __future__ import annotations

import unittest

from .policy_space import INITIAL_SPEC
from .run_search import _feedback, _parse_output, _render_candidate


class L10MB1RunSearchTest(unittest.TestCase):
    def test_runner_round_trips_both_candidate_formats(self) -> None:
        for arm in ("raw", "structured"):
            self.assertEqual(_parse_output(arm, _render_candidate(arm, INITIAL_SPEC)), INITIAL_SPEC)

    def test_feedback_does_not_include_hidden_episode_ledger(self) -> None:
        result = {"semantic_valid": True, "unsafe_candidate": False, "behavioral_score": 0.5, "behavioral_vector": {"x": 1}, "episode_ledger": [{"hidden": True}]}
        self.assertNotIn("hidden", _feedback(result, generation=1, best_score=0.5))


if __name__ == "__main__":
    unittest.main()
