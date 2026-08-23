import unittest

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.freeze_groundbench_referent_union_confirmation import (
    ordered_eligible,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_referent_union_confirmation import (
    DOMAIN_LEXICON,
    brain_command,
    paired_verdict,
)


class GroundBenchReferentUnionConfirmationTest(unittest.TestCase):
    def test_domain_lexicon_is_fixed_and_unique(self) -> None:
        self.assertEqual(len(DOMAIN_LEXICON), len(set(DOMAIN_LEXICON)))
        self.assertIn("car", DOMAIN_LEXICON)
        self.assertIn("umbrella", DOMAIN_LEXICON)

    def test_brain_prompt_is_not_passed_in_argv(self) -> None:
        command = brain_command(
            executable=__import__("pathlib").Path("codex.exe"),
            schema_path=__import__("pathlib").Path("schema.json"),
            raw_path=__import__("pathlib").Path("raw.json"),
            rendered=[__import__("pathlib").Path("image.jpg")],
            model="model", reasoning_effort="medium",
        )
        self.assertNotIn("a very long prompt", command)
        self.assertNotIn("--", command)

    def test_ordered_eligible_is_deterministic(self) -> None:
        def row(index: int) -> dict:
            return {
                "image": f"train2014/{index}.jpg",
                "annotations": {
                    "dataset": "refcoco", "image_id": index, "ann_id": 100 + index,
                    "category_group": "vehicle", "same_class_distractors": 1,
                },
            }
        rows = [row(index) for index in range(10)]
        self.assertEqual(ordered_eligible(rows), ordered_eligible(list(reversed(rows))))

    def test_paired_verdict_requires_mechanism_benefit_and_no_safety_regression(self) -> None:
        def evaluation(proposal: int, correct: int, wrong: int) -> dict:
            return {
                "proposal_availability": {"numerator": proposal},
                "outcome_counts": {"CORRECT_GROUNDING": correct},
                "wrong_confident_guidance_all_observations": {"numerator": wrong},
            }
        supported = paired_verdict(evaluation(50, 40, 8), evaluation(55, 44, 8))
        unsafe = paired_verdict(evaluation(50, 40, 8), evaluation(55, 44, 9))
        self.assertEqual(supported["verdict"], "DOMAIN_LEXICON_PROPOSAL_UNION_SUPPORTED")
        self.assertEqual(unsafe["verdict"], "DOMAIN_LEXICON_PROPOSAL_UNION_NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
