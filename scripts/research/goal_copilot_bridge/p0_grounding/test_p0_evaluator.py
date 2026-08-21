from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p0_evaluator as evaluator


TARGET_NAME = "XX Hospital"
TARGET_REGION = {
    "frame_id": "frame-0",
    "coordinate_space": "NORMALIZED_XYXY",
    "x_min": 0.10,
    "y_min": 0.10,
    "x_max": 0.40,
    "y_max": 0.80,
}
DISTRACTOR_REGION = {
    "frame_id": "frame-0",
    "coordinate_space": "NORMALIZED_XYXY",
    "x_min": 0.60,
    "y_min": 0.10,
    "x_max": 0.90,
    "y_max": 0.80,
}
SECOND_TARGET_REGION = {
    "frame_id": "frame-0",
    "coordinate_space": "NORMALIZED_XYXY",
    "x_min": 0.65,
    "y_min": 0.15,
    "x_max": 0.78,
    "y_max": 0.70,
}
MISLOCALIZED_REGION = {
    "frame_id": "frame-0",
    "coordinate_space": "NORMALIZED_XYXY",
    "x_min": 0.42,
    "y_min": 0.15,
    "x_max": 0.54,
    "y_max": 0.70,
}


def region(value: dict) -> dict:
    return copy.deepcopy(value)


def episode(
    *,
    expectation: str = "MUST_GROUND",
    target_visible: bool = True,
    observation_valid: bool = True,
    with_distractor: bool = True,
    reference_resolution: str = "UNIQUE",
) -> dict:
    if expectation == "MUST_BE_AMBIGUOUS":
        reference_resolution = "AMBIGUOUS"
        target_visible = False
    visible = target_visible
    valid_targets = []
    if reference_resolution == "UNIQUE":
        valid_targets = [{
            "target_instance_id": "target-entrance",
            "target_name": TARGET_NAME,
            "relation": "entrance_of",
            "regions": [region(TARGET_REGION)] if visible else [],
        }]
    elif reference_resolution == "SET_VALUED":
        valid_targets = [
            {"target_instance_id": "target-east", "target_name": TARGET_NAME, "relation": "entrance_of", "regions": [region(TARGET_REGION)] if visible else []},
            {"target_instance_id": "target-west", "target_name": TARGET_NAME, "relation": "entrance_of", "regions": [region(SECOND_TARGET_REGION)] if visible else []},
        ]
    return {
        "schema_version": 1,
        "episode_id": "mock-episode",
        "goal_spec": {
            "goal_type": "NAMED_BUILDING_ENTRANCE",
            "target_name": TARGET_NAME,
            "requested_relation": "entrance_of",
        },
        "observation_window": {
            "frame_ids": ["frame-0"],
            "start_timestamp_ms": 0,
            "end_timestamp_ms": 1000,
        },
        "observation_valid": observation_valid,
        "goal_reference_resolution": reference_resolution,
        "target_visible": visible,
        "valid_target_instances": valid_targets,
        "acceptable_spatial_regions": [item for target in valid_targets for item in target["regions"]],
        "distractor_instances": (
            [
                {
                    "distractor_instance_id": "other-entrance",
                    "semantic_role": "OTHER_BUILDING_ENTRANCE",
                    "region": region(DISTRACTOR_REGION),
                }
            ]
            if with_distractor else []
        ),
        "target_min_side_px": 120.0 if visible else None,
        "visibility_fraction": 0.95 if visible else None,
        "text_support": "READABLE" if visible else "NOT_APPLICABLE",
        "scene_condition": {
            "target_size": "MEDIUM" if visible else "ABSENT",
            "visibility": "FULL" if visible else "ABSENT",
            "entrance_count": "MULTIPLE" if with_distractor else "SINGLE",
            "same_class_distractor": with_distractor,
            "illumination": "DAYLIGHT",
            "view_angle": "FRONTAL",
        },
        "grounding_expectation": expectation,
    }


def evidence(
    evidence_id: str,
    item_region: dict,
    *,
    target_name: str | None,
    evidence_type: str,
    validity: str = "VALID",
    expiry_timestamp_ms: int | None = 5000,
) -> dict:
    return {
        "provider_id": "mock-provider",
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "source_frame_id": "frame-0",
        "source_timestamp_ms": 0,
        "region_in_source_frame": region(item_region),
        "confidence": 0.9,
        "validity": validity,
        "expiry_timestamp_ms": expiry_timestamp_ms,
        "identity_claim": {
            "target_name": target_name,
            "relation": "entrance_of" if target_name is not None else "none",
        },
        "provenance": {
            "implementation_id": "mock-v1",
            "config_id": "mock-config-v1",
            "source_kind": "MOCK",
        },
    }


def candidate(
    candidate_id: str,
    item_region: dict,
    evidence_ids: list[str],
    provider_rank: int,
    *,
    identity_hypothesis: str | None = TARGET_NAME,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "region": region(item_region),
        "category_label": "building entrance",
        "identity_hypothesis": identity_hypothesis,
        "confidence": 0.8,
        "provider_rank": provider_rank,
        "provider_ids": ["mock-provider"],
        "evidence_ids": list(evidence_ids),
    }


def system_output(
    *,
    candidates: list[dict],
    evidence_items: list[dict],
    status: str,
    selected_candidate_id: str | None = None,
    ranked_candidate_ids: list[str] | None = None,
    abstention_reason: str | None = None,
) -> dict:
    candidate_ids = [item["candidate_id"] for item in candidates]
    evidence_ids = [item["evidence_id"] for item in evidence_items]
    ranked = list(ranked_candidate_ids if ranked_candidate_ids is not None else candidate_ids)
    if status == "GROUNDED":
        selected = next(item for item in candidates if item["candidate_id"] == selected_candidate_id)
        supporting = list(selected["evidence_ids"])
        selected_region = region(selected["region"])
        handoff = {
            "handoff_id": "handoff-1",
            "candidate_id": selected_candidate_id,
            "source_frame_id": selected_region["frame_id"],
            "spatial_region": region(selected_region),
            "evidence_ids": list(supporting),
        }
        decision = {
            "status": status,
            "selected_candidate_id": selected_candidate_id,
            "ranked_candidate_ids": ranked,
            "source_frame_id": selected_region["frame_id"],
            "decision_timestamp_ms": 2000,
            "spatial_region": selected_region,
            "goal_identity_support": "SUPPORTED",
            "spatial_support": "SUPPORTED",
            "confidence": 0.8,
            "supporting_evidence_ids": supporting,
            "competing_candidate_ids": [item for item in ranked if item != selected_candidate_id],
            "abstention_reason": None,
            "persistence_handoff_token": handoff,
        }
    else:
        decision = {
            "status": status,
            "selected_candidate_id": None,
            "ranked_candidate_ids": ranked,
            "source_frame_id": None,
            "decision_timestamp_ms": 2000,
            "spatial_region": None,
            "goal_identity_support": "NOT_EVALUABLE",
            "spatial_support": "NOT_EVALUABLE",
            "confidence": None,
            "supporting_evidence_ids": [],
            "competing_candidate_ids": ranked,
            "abstention_reason": abstention_reason or "NO_CANDIDATE",
            "persistence_handoff_token": None,
        }
    return {
        "schema_version": 1,
        "episode_id": "mock-episode",
        "provider_runs": [
            {
                "provider_id": "mock-provider",
                "status": "RUN_SUCCESS",
                "source_frame_ids": ["frame-0"],
                "evidence_ids": evidence_ids,
                "candidate_ids": candidate_ids,
                "failure_reason": None,
            }
        ],
        "evidence": evidence_items,
        "candidates": candidates,
        "decision": decision,
    }


def correct_components(*, identity_name: str | None = TARGET_NAME, expiry: int | None = 5000) -> tuple[list[dict], list[dict]]:
    items = [
        evidence("ev-target-structure", TARGET_REGION, target_name=None, evidence_type="ENTRANCE_STRUCTURE", expiry_timestamp_ms=expiry),
        evidence("ev-target-identity", TARGET_REGION, target_name=identity_name, evidence_type="OCR_TEXT", expiry_timestamp_ms=expiry),
    ]
    return [candidate("candidate-target", TARGET_REGION, [item["evidence_id"] for item in items], 1)], items


class P0GroundingEvaluatorTest(unittest.TestCase):
    def test_contract_files_are_machine_readable(self) -> None:
        episode_schema = json.loads((HERE / "p0_episode_schema.json").read_text(encoding="utf-8"))
        output_schema = json.loads((HERE / "p0_output_schema.json").read_text(encoding="utf-8"))
        protocol_path = HERE.parents[3] / "docs" / "research" / "goal-copilot" / "p0_grounding_protocol_v1.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual("https://json-schema.org/draft/2020-12/schema", episode_schema["$schema"])
        self.assertFalse(episode_schema["additionalProperties"])
        self.assertFalse(output_schema["additionalProperties"])
        self.assertEqual(evaluator.PROTOCOL_ID, protocol["protocol_id"])
        self.assertFalse(protocol["execution_authority"]["baseline_execution"])

    def test_correct_candidate_exists_and_is_selected(self) -> None:
        candidates, items = correct_components()
        result = evaluator.evaluate_episode(
            episode(),
            system_output(candidates=candidates, evidence_items=items, status="GROUNDED", selected_candidate_id="candidate-target"),
        )
        self.assertTrue(result["provider_availability"]["correct_candidate_available"])
        self.assertTrue(result["brain_selection"]["top1_correct_given_available"])
        self.assertEqual("CORRECT_GROUNDING", result["end_to_end"]["outcome"])
        self.assertTrue(result["p1_handoff"]["valid"])

    def test_correct_candidate_exists_but_brain_selects_distractor(self) -> None:
        target_candidates, target_items = correct_components()
        wrong_items = [
            evidence("ev-wrong-structure", DISTRACTOR_REGION, target_name=None, evidence_type="ENTRANCE_STRUCTURE"),
            evidence("ev-wrong-identity", DISTRACTOR_REGION, target_name=TARGET_NAME, evidence_type="OCR_TEXT"),
        ]
        candidates = target_candidates + [
            candidate("candidate-distractor", DISTRACTOR_REGION, [item["evidence_id"] for item in wrong_items], 2)
        ]
        result = evaluator.evaluate_episode(
            episode(),
            system_output(
                candidates=candidates,
                evidence_items=target_items + wrong_items,
                status="GROUNDED",
                selected_candidate_id="candidate-distractor",
                ranked_candidate_ids=["candidate-distractor", "candidate-target"],
            ),
        )
        self.assertTrue(result["provider_availability"]["correct_candidate_available"])
        self.assertTrue(result["brain_selection"]["wrong_instance_given_available"])
        self.assertEqual("WRONG_INSTANCE", result["end_to_end"]["outcome"])

    def test_correct_candidate_unavailable_is_not_brain_failure(self) -> None:
        result = evaluator.evaluate_episode(
            episode(),
            system_output(
                candidates=[],
                evidence_items=[],
                status="ABSTAIN_NO_RELIABLE_EVIDENCE",
                abstention_reason="NO_CANDIDATE",
            ),
        )
        self.assertFalse(result["provider_availability"]["correct_candidate_available"])
        self.assertIn("RUN_SUCCESS_NO_CANDIDATE", result["provider_availability"]["provider_failure_classes"])
        self.assertEqual("NOT_IDENTIFIABLE", result["brain_selection"]["identifiability"])
        self.assertIsNone(result["brain_selection"]["top1_correct_given_available"])
        self.assertEqual("PROVIDER_CORRECT_CANDIDATE_UNAVAILABLE", result["end_to_end"]["outcome"])

    def test_false_grounding_when_target_is_absent(self) -> None:
        items = [
            evidence("ev-false-structure", DISTRACTOR_REGION, target_name=None, evidence_type="ENTRANCE_STRUCTURE"),
            evidence("ev-false-identity", DISTRACTOR_REGION, target_name=TARGET_NAME, evidence_type="OCR_TEXT"),
        ]
        candidates = [candidate("candidate-false", DISTRACTOR_REGION, [item["evidence_id"] for item in items], 1)]
        result = evaluator.evaluate_episode(
            episode(expectation="MUST_ABSTAIN", target_visible=False),
            system_output(candidates=candidates, evidence_items=items, status="GROUNDED", selected_candidate_id="candidate-false"),
        )
        self.assertEqual("FALSE_GROUNDING_TARGET_ABSENT", result["end_to_end"]["outcome"])

    def test_ambiguity_is_preserved(self) -> None:
        target_candidates, target_items = correct_components()
        wrong_items = [evidence("ev-other", DISTRACTOR_REGION, target_name=None, evidence_type="ENTRANCE_STRUCTURE")]
        candidates = target_candidates + [candidate("candidate-other", DISTRACTOR_REGION, ["ev-other"], 2, identity_hypothesis=None)]
        result = evaluator.evaluate_episode(
            episode(expectation="MUST_BE_AMBIGUOUS"),
            system_output(
                candidates=candidates,
                evidence_items=target_items + wrong_items,
                status="AMBIGUOUS",
                abstention_reason="AMBIGUOUS_CANDIDATES",
            ),
        )
        self.assertTrue(result["brain_selection"]["correct_abstention_under_ambiguity"])
        self.assertEqual("CORRECT_AMBIGUITY", result["end_to_end"]["outcome"])

    def test_set_valued_goal_accepts_any_valid_physical_target(self) -> None:
        items = [
            evidence("ev-west-structure", SECOND_TARGET_REGION, target_name=None, evidence_type="ENTRANCE_STRUCTURE"),
            evidence("ev-west-identity", SECOND_TARGET_REGION, target_name=TARGET_NAME, evidence_type="OCR_TEXT"),
        ]
        candidates = [candidate("candidate-west", SECOND_TARGET_REGION, [item["evidence_id"] for item in items], 1)]
        result = evaluator.evaluate_episode(
            episode(reference_resolution="SET_VALUED"),
            system_output(candidates=candidates, evidence_items=items, status="GROUNDED", selected_candidate_id="candidate-west"),
        )
        self.assertTrue(result["provider_availability"]["correct_candidate_available"])
        self.assertTrue(result["brain_selection"]["top1_correct_given_available"])
        self.assertEqual("CORRECT_GROUNDING", result["end_to_end"]["outcome"])

    def test_ambiguous_goal_allows_fail_closed_abstention(self) -> None:
        result = evaluator.evaluate_episode(
            episode(expectation="MUST_BE_AMBIGUOUS", with_distractor=False),
            system_output(candidates=[], evidence_items=[], status="ABSTAIN_NO_RELIABLE_EVIDENCE", abstention_reason="INSUFFICIENT_IDENTITY_EVIDENCE"),
        )
        self.assertTrue(result["brain_selection"]["correct_abstention_under_ambiguity"])
        self.assertEqual("CORRECT_ABSTENTION", result["end_to_end"]["outcome"])

    def test_set_valued_goal_requires_multiple_physical_instances(self) -> None:
        invalid = episode(reference_resolution="SET_VALUED")
        invalid["valid_target_instances"] = invalid["valid_target_instances"][:1]
        invalid["acceptable_spatial_regions"] = invalid["acceptable_spatial_regions"][:1]
        with self.assertRaises(evaluator.EpisodeContractError):
            evaluator.validate_episode(invalid)

    def test_acceptable_regions_cannot_hide_single_target_truth(self) -> None:
        invalid = episode(reference_resolution="SET_VALUED")
        invalid["acceptable_spatial_regions"] = invalid["acceptable_spatial_regions"][:1]
        with self.assertRaises(evaluator.EpisodeContractError):
            evaluator.validate_episode(invalid)

    def test_expired_slow_evidence_is_not_accepted(self) -> None:
        candidates, items = correct_components(expiry=1000)
        result = evaluator.evaluate_episode(
            episode(),
            system_output(candidates=candidates, evidence_items=items, status="GROUNDED", selected_candidate_id="candidate-target"),
        )
        self.assertTrue(result["valid_system_output"])
        self.assertTrue(result["brain_selection"]["stale_evidence_used"])
        self.assertEqual("STALE_EVIDENCE_USED", result["end_to_end"]["outcome"])

    def test_region_correct_but_goal_identity_wrong(self) -> None:
        candidates, items = correct_components(identity_name="Other Hospital")
        result = evaluator.evaluate_episode(
            episode(),
            system_output(candidates=candidates, evidence_items=items, status="GROUNDED", selected_candidate_id="candidate-target"),
        )
        self.assertTrue(result["brain_selection"]["spatial_match"])
        self.assertFalse(result["brain_selection"]["identity_match"])
        self.assertEqual("GOAL_IDENTITY_ERROR", result["end_to_end"]["outcome"])

    def test_goal_identity_correct_but_region_wrong(self) -> None:
        items = [
            evidence("ev-mislocalized-structure", MISLOCALIZED_REGION, target_name=None, evidence_type="ENTRANCE_STRUCTURE"),
            evidence("ev-correct-identity", MISLOCALIZED_REGION, target_name=TARGET_NAME, evidence_type="OCR_TEXT"),
        ]
        candidates = [candidate("candidate-mislocalized", MISLOCALIZED_REGION, [item["evidence_id"] for item in items], 1)]
        result = evaluator.evaluate_episode(
            episode(with_distractor=False),
            system_output(candidates=candidates, evidence_items=items, status="GROUNDED", selected_candidate_id="candidate-mislocalized"),
        )
        self.assertEqual("NOT_IDENTIFIABLE", result["brain_selection"]["identifiability"])
        self.assertEqual("SPATIAL_LOCALIZATION_ERROR", result["end_to_end"]["outcome"])

    def test_invalid_observation_is_handled(self) -> None:
        result = evaluator.evaluate_episode(
            episode(expectation="INVALID_OBSERVATION", target_visible=False, observation_valid=False, with_distractor=False),
            system_output(
                candidates=[],
                evidence_items=[],
                status="INVALID_OBSERVATION",
                abstention_reason="INVALID_INPUT",
            ),
        )
        self.assertEqual("INVALID_OBSERVATION_HANDLED", result["end_to_end"]["outcome"])

    def test_handoff_binding_drift_fails_closed(self) -> None:
        candidates, items = correct_components()
        output = system_output(candidates=candidates, evidence_items=items, status="GROUNDED", selected_candidate_id="candidate-target")
        output["decision"]["persistence_handoff_token"]["candidate_id"] = "different-candidate"
        result = evaluator.evaluate_episode(episode(), output)
        self.assertFalse(result["valid_system_output"])
        self.assertEqual("INVALID_SYSTEM_OUTPUT", result["end_to_end"]["outcome"])
        self.assertIn("handoff candidate binding drift", result["contract_error"])

    def test_zero_denominator_metrics_are_null(self) -> None:
        invalid_episode = episode(expectation="INVALID_OBSERVATION", target_visible=False, observation_valid=False, with_distractor=False)
        invalid_output = system_output(
            candidates=[], evidence_items=[], status="INVALID_OBSERVATION", abstention_reason="INVALID_INPUT"
        )
        summary = evaluator.evaluate_batch([(invalid_episode, invalid_output)])
        self.assertEqual(0, summary["aggregate"]["brain_top1_correct_given_available"]["denominator"])
        self.assertIsNone(summary["aggregate"]["brain_top1_correct_given_available"]["value"])


if __name__ == "__main__":
    unittest.main()
