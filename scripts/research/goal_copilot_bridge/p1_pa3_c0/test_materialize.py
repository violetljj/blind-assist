from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_pa3_c0.materialize import MaterializationError, content_sha256, materialize


PROMPT_MAP = {
    "schema_version": "blindassist_p1_pa3_c0_prompt_map_v1",
    "mapping_id": "test-map-v1",
    "mapping_rule": "EXACT_GLOBAL_GOAL_TYPE_LOOKUP_NO_EPISODE_OVERRIDE",
    "entries": [{"goal_type": "NAMED_BUILDING_ENTRANCE", "canonical_prompt": "building entrance"}],
}


def intake() -> dict:
    return {
        "schema_version": "blindassist_p1_pa3_c0_goal_intake_v1",
        "intake_id": "prospective-test-v1",
        "provenance_contract": {
            "allowed_source_authorities": ["USER_TASK_INPUT", "PRODUCT_TASK_INPUT"],
            "capture_state_at_goal_recording": "NOT_STARTED",
            "truth_state_at_goal_recording": "NOT_CREATED",
        },
        "episodes": [
            {
                "episode_id": "prospective-001",
                "goal_text_original": "帮我找入口",
                "goal_recorded_at_utc": "2026-08-22T00:00:00Z",
                "goal_source": {"authority": "USER_TASK_INPUT", "source_record_sha256": "a" * 64},
                "goal_contract": {
                    "goal_type": "NAMED_BUILDING_ENTRANCE",
                    "reference_mode": "AMBIGUOUS",
                    "task_semantics": "find a building entrance satisfying the user task",
                },
            }
        ],
    }


class Pa3C0MaterializationTest(unittest.TestCase):
    def test_materializes_global_prompt_without_truth(self) -> None:
        result = materialize(intake(), PROMPT_MAP)
        self.assertEqual("building entrance", result["episodes"][0]["canonical_prompt"])
        self.assertFalse(result["private_truth_access"])
        self.assertFalse(result["pa3_inference_authorized"])
        self.assertEqual("PENDING_FUTURE_TRUTH_BINDING_TO_THIS_RECEIPT", result["created_before_truth"])
        self.assertTrue(result["future_truth_must_bind_receipt_body_sha256"])
        self.assertEqual(64, len(result["receipt_body_sha256"]))
        body_hash = result.pop("receipt_body_sha256")
        self.assertEqual(body_hash, content_sha256(result))

    def test_rejects_episode_prompt_override(self) -> None:
        value = intake()
        value["episodes"][0]["canonical_prompt"] = "closet door"
        with self.assertRaisesRegex(MaterializationError, "provider-forbidden"):
            materialize(value, PROMPT_MAP)

    def test_rejects_target_truth_key(self) -> None:
        value = intake()
        value["episodes"][0]["target_bbox_xyxy"] = [0, 0, 1, 1]
        with self.assertRaisesRegex(MaterializationError, "provider-forbidden"):
            materialize(value, PROMPT_MAP)

    def test_rejects_retrofitted_truth_provenance(self) -> None:
        value = intake()
        value["provenance_contract"]["truth_state_at_goal_recording"] = "ALREADY_CREATED"
        with self.assertRaisesRegex(MaterializationError, "before truth"):
            materialize(value, PROMPT_MAP)

    def test_rejects_empty_non_cohort_template(self) -> None:
        value = intake()
        value["episodes"] = []
        with self.assertRaisesRegex(MaterializationError, "no Goal Contract episodes"):
            materialize(value, PROMPT_MAP)


if __name__ == "__main__":
    unittest.main()
