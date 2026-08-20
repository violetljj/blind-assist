from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC

from .hard_benchmark import evaluate_instance, load_benchmark
from .protocol_b4a import PAIRED_IDENTITIES, build_protocol_manifest
from .run_b4a import _balanced_prompt


class B4AProtocolTest(unittest.TestCase):
    def test_fresh_pair_grid_and_budget_are_frozen(self) -> None:
        manifest = build_protocol_manifest(Path("."))
        identities = [int(row["paired_identity"]) for row in PAIRED_IDENTITIES]

        self.assertEqual(len(identities), 9)
        self.assertEqual(len(set(identities)), 9)
        self.assertEqual(manifest["execution"]["planned_model_calls"], 144)
        self.assertFalse(manifest["anti_post_hoc"]["generation_budget_reduced_to_two"])
        self.assertEqual(
            manifest["harder_cohort"]["model_calls_used_to_construct_or_qualify"], 0
        )

    def test_search_prompt_does_not_expose_hidden_instance_outcomes(self) -> None:
        instance = load_benchmark()["instances"][0]
        result = evaluate_instance(INITIAL_SPEC, instance)
        prompt = _balanced_prompt(519302, 1, INITIAL_SPEC, result, [])

        self.assertNotIn("amber", prompt)
        self.assertNotIn("accepted_actions", prompt)
        self.assertNotIn("episode_ledger", prompt)


if __name__ == "__main__":
    unittest.main()
