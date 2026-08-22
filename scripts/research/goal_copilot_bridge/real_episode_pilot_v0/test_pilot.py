from __future__ import annotations

import json
from pathlib import Path
import unittest

from .annotation import make_annotation
from .audit_abotn_poibench_truth_source import classify_source, inspect_task, summarize_tasks
from .audit_abotn_render_runtime import classify_runtime
from .baseline import run_baseline
from .evaluate import evaluate
from .public_real_mining import mine_prospective
from .truth_contract import validate_observation_truth


def public_manifest():
    return {
        "schema_version": "blindassist_real_episode_public_manifest_v0",
        "episodes": [{
            "episode_id": "ep-001",
            "goal_contract": {"goal_contract": {"reference_mode": "UNIQUE"}},
            "observations": [
                {"observation_id": "o1", "timestamp_ms": 0},
                {"observation_id": "o2", "timestamp_ms": 250},
            ],
        }],
    }


def config():
    return json.loads((Path(__file__).parent / "baseline_config.json").read_text(encoding="utf-8"))


def freeze(annotation):
    annotation["truth_frozen"] = True
    return annotation


def authorize_native(row):
    row.update({
        "truth_authority_tier": "NATIVE_GT",
        "functional_authority": "ESTABLISHED",
        "functional_authority_sources": ["NATIVE_GT"],
    })
    return row


class RealEpisodePilotTest(unittest.TestCase):
    def test_abotn_source_is_arrival_only_and_official_envelope_leaks_private_geometry(self):
        task = {
            "trajectory": [
                {"x": 1.02, "y": 2.01, "z": 0.65, "yaw": 0.0},
                {"x": 3.01, "y": 4.02, "z": 0.65, "yaw": 0.5},
            ],
            "instruction": "前往测试商店",
            "label": {"extend": {
                "goal_label": "测试商店",
                "start_point": [1.0, 2.0],
                "end_point": [3.0, 4.0],
            }},
        }
        inspected = inspect_task(task)
        self.assertTrue(inspected["endpoint_consistent"])
        self.assertEqual([], inspected["explicit_region_keys"])
        summary = summarize_tasks([("annotations/scene-a/traj_0.json", task)])
        result = classify_source(
            summary,
            dataset_license=None,
            dataset_root_files=[".gitattributes"],
            repository_readme="License: Apache-2.0",
            repository_license_present=False,
            evaluator_source=(
                "target_position=base_obs.target_position\n"
                "distance_to_goal=base_obs.distance_to_goal\n"
            ),
        )
        self.assertEqual("ARRIVAL_TRUTH_ONLY_INTERNAL_RESEARCH_CANDIDATE", result["overall"])
        self.assertEqual(
            "NOT_EVALUABLE_NO_EXPLICIT_ENTRANCE_FRAME_OR_PIXEL_REGION",
            result["functional_frame_region_truth"],
        )
        self.assertEqual(
            ["target_position", "distance_to_goal"],
            result["private_fields_exposed_to_official_agent"],
        )

    def test_abotn_render_runtime_fails_before_payload_download_when_vram_is_below_official_minimum(self):
        result = classify_runtime(
            host_os="Windows",
            gpu_count=1,
            maximum_gpu_vram_mib=8151,
            torch_cuda_available=True,
            cuda_compiler_available=False,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(
            "NOT_EVALUABLE_LOCAL_RENDER_RUNTIME_VRAM_BELOW_OFFICIAL_MINIMUM",
            result["terminal"],
        )
        self.assertIn("VRAM_BELOW_OFFICIAL_24GB_MINIMUM", result["failures"])

    def test_annotation_starts_truth_unknown(self):
        annotation = make_annotation(public_manifest())
        self.assertTrue(annotation["private_evaluator_only"])
        self.assertFalse(annotation["truth_frozen"])
        self.assertEqual("UNKNOWN", annotation["episodes"][0]["observations"][0]["truth_authority_tier"])
        self.assertEqual(
            {"teacher_A", "teacher_B", "teacher_C"},
            set(annotation["episodes"][0]["observations"][0]["teacher_outputs"]),
        )
        self.assertEqual("UNKNOWN", annotation["episodes"][0]["observations"][0]["target_visibility"])

    def test_baseline_contests_without_selection_authority_and_handoffs_near(self):
        provider = {"observations": [
            {"observation_id": "o1", "candidate_cardinality": "SET_VALUED", "selection_authorized": False, "candidates": [
                {"candidate_id": "a", "rank": 1, "x_center_fraction": 0.2, "range_m": 3.0}
            ]},
            {"observation_id": "o2", "candidate_cardinality": "UNIQUE", "selection_authorized": True, "candidates": [
                {"candidate_id": "b", "rank": 1, "x_center_fraction": 0.5, "range_m": 1.2}
            ]},
        ]}
        result = run_baseline(public_manifest(), provider, config())
        self.assertEqual("CONTESTED", result["predictions"][0]["decision_state"])
        self.assertEqual("HANDOFF_READY", result["predictions"][1]["decision_state"])
        self.assertFalse(result["identity_persistence"])

    def test_evaluator_keeps_conditioned_denominators_and_attribution(self):
        annotation = freeze(make_annotation(public_manifest()))
        episode = annotation["episodes"][0]
        authorize_native(episode["observations"][0]).update({"target_visibility": "VISIBLE", "legal_candidate_ids": ["a"], "allowed_decision_states": ["FOUND"], "range_truth": "RANGE_FAR"})
        authorize_native(episode["observations"][1]).update({"target_visibility": "NOT_VISIBLE", "allowed_decision_states": ["NOT_VISIBLE"]})
        prediction = {
            "schema_version": "blindassist_real_episode_baseline_prediction_v0",
            "predictions": [
                {"observation_id": "o1", "candidate_ids": ["a"], "selected_referent": "a", "decision_state": "FOUND", "command": "GUIDE_LEFT", "range_bucket": "RANGE_FAR", "confident_spoken_guidance": True},
                {"observation_id": "o2", "candidate_ids": [], "selected_referent": None, "decision_state": "NOT_VISIBLE", "command": None, "range_bucket": "RANGE_UNKNOWN", "confident_spoken_guidance": False},
            ],
        }
        result = evaluate(annotation, prediction)
        recall = result["observation_metrics"]["proposal_recall_at_k_given_visible"]["1"]
        self.assertEqual(1, recall["eligible"])
        self.assertEqual(1.0, recall["rate"])
        self.assertEqual(2, result["observation_metrics"]["unconditional_target_visibility"]["eligible"])

    def test_unknown_truth_never_becomes_wrong_referent(self):
        annotation = freeze(make_annotation(public_manifest()))
        prediction = {
            "schema_version": "blindassist_real_episode_baseline_prediction_v0",
            "predictions": [
                {"observation_id": "o1", "candidate_ids": ["a"], "selected_referent": "a", "decision_state": "FOUND", "command": "GUIDE_LEFT", "range_bucket": "RANGE_UNKNOWN", "confident_spoken_guidance": True},
                {"observation_id": "o2", "candidate_ids": [], "selected_referent": None, "decision_state": "ABSTAIN", "command": None, "range_bucket": "RANGE_UNKNOWN", "confident_spoken_guidance": False},
            ],
        }
        result = evaluate(annotation, prediction)
        metric = result["observation_metrics"]["confident_wrong_referent_guidance"]
        self.assertEqual(0, metric["eligible"])
        self.assertEqual(1, metric["unknown"])
        self.assertIsNone(metric["rate"])

    def test_frozen_legal_region_matches_later_provider_candidate_without_id_dependency(self):
        annotation = freeze(make_annotation(public_manifest()))
        first, second = annotation["episodes"][0]["observations"]
        authorize_native(first).update({
            "target_visibility": "VISIBLE",
            "legal_regions_normalized_xyxy": [[0.1, 0.2, 0.4, 0.8]],
        })
        prediction = {
            "schema_version": "blindassist_real_episode_baseline_prediction_v0",
            "predictions": [
                {
                    "observation_id": "o1",
                    "candidate_ids": ["provider-generated-after-freeze"],
                    "candidate_regions_normalized_xyxy": [[0.1, 0.2, 0.4, 0.8]],
                    "selected_referent": "provider-generated-after-freeze",
                    "decision_state": "FOUND",
                    "command": "GUIDE_LEFT",
                    "range_bucket": "RANGE_UNKNOWN",
                    "confident_spoken_guidance": True,
                },
                {
                    "observation_id": "o2",
                    "candidate_ids": [],
                    "candidate_regions_normalized_xyxy": [],
                    "selected_referent": None,
                    "decision_state": "ABSTAIN",
                    "command": None,
                    "range_bucket": "RANGE_UNKNOWN",
                    "confident_spoken_guidance": False,
                },
            ],
        }
        result = evaluate(annotation, prediction)
        self.assertEqual(
            {"correct": 1, "eligible": 1, "rate": 1.0},
            result["observation_metrics"]["selection_accuracy_given_legal_candidate_present"],
        )

    def test_evaluator_rejects_unfrozen_truth(self):
        annotation = make_annotation(public_manifest())
        prediction = {
            "schema_version": "blindassist_real_episode_baseline_prediction_v0",
            "predictions": [],
        }
        with self.assertRaisesRegex(ValueError, "frozen truth"):
            evaluate(annotation, prediction)

    def test_teacher_consensus_cannot_establish_functional_truth(self):
        row = make_annotation(public_manifest())["episodes"][0]["observations"][0]
        row.update({
            "truth_authority_tier": "TEACHER_ONLY_WEAK",
            "teacher_agreement": "AGREE",
            "functional_authority": "ESTABLISHED",
            "functional_authority_sources": [],
        })
        for index, key in enumerate(("teacher_A", "teacher_B", "teacher_C"), start=1):
            row["teacher_outputs"][key].update({
                "teacher_id": f"independent-{index}",
                "implementation_id": f"teacher-impl-{index}",
                "status": "RUN_SUCCESS",
                "raw_output": {"region": [index, index, index + 1, index + 1]},
                "independent_of_evaluated_provider": True,
                "provider_family_overlap": False,
            })
        with self.assertRaisesRegex(ValueError, "native or map/trajectory"):
            validate_observation_truth(row, finalized=True)

    def test_tier_stratification_preserves_unknown_outside_accuracy_denominator(self):
        annotation = freeze(make_annotation(public_manifest()))
        first, second = annotation["episodes"][0]["observations"]
        authorize_native(first).update({
            "target_visibility": "VISIBLE",
            "legal_candidate_ids": ["a"],
            "allowed_decision_states": ["FOUND"],
        })
        prediction = {
            "schema_version": "blindassist_real_episode_baseline_prediction_v0",
            "predictions": [
                {"observation_id": "o1", "candidate_ids": ["a"], "selected_referent": "a", "decision_state": "FOUND", "command": "GUIDE_LEFT", "range_bucket": "RANGE_UNKNOWN", "confident_spoken_guidance": True},
                {"observation_id": "o2", "candidate_ids": ["x"], "selected_referent": "x", "decision_state": "FOUND", "command": "GUIDE_RIGHT", "range_bucket": "RANGE_UNKNOWN", "confident_spoken_guidance": True},
            ],
        }
        result = evaluate(annotation, prediction)
        metrics = result["observation_metrics"]
        self.assertEqual({"NATIVE_GT": 1, "MAP_TRAJECTORY_DERIVED": 0, "TEACHER_SUPPORTED": 0, "TEACHER_ONLY_WEAK": 0, "UNKNOWN": 1}, metrics["truth_authority_distribution"])
        self.assertEqual(1, metrics["confident_spoken_guidance_referent_correctness"]["eligible"])
        self.assertEqual(1, metrics["confident_spoken_guidance_referent_correctness"]["unknown"])
        self.assertEqual(1, metrics["by_truth_authority_tier"]["UNKNOWN"]["observations"])

    def test_prospective_miner_enforces_precedence_and_keeps_set_valued_goal(self):
        roster = {
            "schema_version": "blindassist_public_goal_roster_v0",
            "precedence": {
                "mapillary_metadata_accessed": False,
                "mapillary_pixels_accessed": False,
                "model_outputs_created": False,
                "evaluator_truth_created": False,
            },
            "goals": [{
                "goal_id": "g1", "goal_type": "NAMED_BUILDING_ENTRANCE", "target_name": "venue",
                "public_entrance_candidates": [
                    {"candidate_id": "a", "coordinates": [0.0, 0.0]},
                    {"candidate_id": "b", "coordinates": [0.0, 0.0]},
                ],
            }],
        }
        metadata = {
            "schema_version": "blindassist_mapillary_sequence_metadata_v0",
            "images": [
                {"image_id": "i1", "sequence_id": "s", "captured_at_ms": 0, "coordinates": [-0.00027, 0.0], "computed_compass_angle": 90.0},
                {"image_id": "i2", "sequence_id": "s", "captured_at_ms": 1000, "coordinates": [-0.00018, 0.0], "computed_compass_angle": 90.0},
                {"image_id": "i3", "sequence_id": "s", "captured_at_ms": 2000, "coordinates": [-0.00009, 0.0], "computed_compass_angle": 90.0},
            ],
        }
        result = mine_prospective(roster, metadata)
        self.assertEqual(1, result["episode_count"])
        self.assertEqual("SET_VALUED", result["episodes"][0]["goal_contract"]["goal_contract"]["reference_mode"])
        self.assertTrue(result["goal_before_mapillary_metadata_and_pixels"])

    def test_prospective_miner_rejects_post_pixel_goal_roster(self):
        roster = {
            "schema_version": "blindassist_public_goal_roster_v0",
            "precedence": {
                "mapillary_metadata_accessed": True,
                "mapillary_pixels_accessed": False,
                "model_outputs_created": False,
                "evaluator_truth_created": False,
            },
            "goals": [],
        }
        with self.assertRaises(ValueError):
            mine_prospective(roster, {"schema_version": "blindassist_mapillary_sequence_metadata_v0", "images": []})

if __name__ == "__main__":
    unittest.main()
