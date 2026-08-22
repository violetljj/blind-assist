from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_PROVIDER_CONFIGURATION,
    EXPECTED_ULTRALYTICS_VERSION,
    PRIVATE_SCHEMA,
    PROTOCOL_ID,
    PUBLIC_SCHEMA,
    PREDICTION_SCHEMA,
    Pa3ContractError,
    content_sha256,
    evaluate,
    private_truth_body,
    sha256,
    validate_public,
)
from scripts.research.goal_copilot_bridge.p1_proposal_availability.materialize_pa3_inputs import (
    CAPTURE_SCHEMA,
    TRUTH_SCHEMA,
    materialize_inputs,
)


PROMPT_MAP = {
    "schema_version": "blindassist_p1_pa3_c0_prompt_map_v1",
    "mapping_id": "test-map",
    "mapping_rule": "EXACT_GLOBAL_GOAL_TYPE_LOOKUP_NO_EPISODE_OVERRIDE",
    "entries": [{"goal_type": "NAMED_BUILDING_ENTRANCE", "canonical_prompt": "building entrance"}],
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Pa3PublicContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "frame.jpg"
        self.image.write_bytes(b"synthetic-frame-identity")
        self.body_sha = "a" * 64
        self.receipt = self.root / "precedence.json"
        write_json(self.receipt, {
            "schema_version": "blindassist_p1_pa3_goal_truth_precedence_receipt_v1",
            "goal_receipt_body_sha256": self.body_sha,
            "goal_recorded_at_utc": "2026-08-22T00:00:00Z",
            "capture_created_at_utc": "2026-08-22T00:01:00Z",
            "truth_created_at_utc": "2026-08-22T00:02:00Z",
            "created_before_capture": True,
            "created_before_truth": True,
            "private_truth_body_sha256": "b" * 64,
        })

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def public(self) -> dict:
        return {
            "schema_version": PUBLIC_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "private_truth_access": False,
            "prompt_map_sha256": content_sha256(PROMPT_MAP),
            "provider_contract": {
                "input": "CURRENT_FRAME_PLUS_PRETRUTH_GOAL_CONTRACT",
                "maximum_candidates": 10,
                "identity_selection": "FORBIDDEN",
            },
            "cases": [{
                "case_id": "case-001",
                "query": {"image_path": str(self.image), "image_sha256": sha256(self.image)},
                "goal_contract": {
                    "goal_text_original": "帮我找入口",
                    "goal_type": "NAMED_BUILDING_ENTRANCE",
                    "reference_mode": "SET_VALUED",
                    "canonical_prompt": "building entrance",
                    "c0_goal_receipt_body_sha256": self.body_sha,
                    "precedence_receipt_path": str(self.receipt),
                    "precedence_receipt_sha256": sha256(self.receipt),
                },
            }],
        }

    def test_accepts_bound_pretruth_goal_and_current_frame(self) -> None:
        cases = validate_public(self.public(), PROMPT_MAP, self.root)
        self.assertEqual("building entrance", cases[0]["canonical_prompt"])
        self.assertEqual("SET_VALUED", cases[0]["reference_mode"])

    def test_rejects_episode_prompt_override(self) -> None:
        value = self.public()
        value["cases"][0]["goal_contract"]["canonical_prompt"] = "closet door"
        with self.assertRaisesRegex(Pa3ContractError, "canonical prompt override"):
            validate_public(value, PROMPT_MAP, self.root)

    def test_rejects_truth_in_public_input(self) -> None:
        value = self.public()
        value["cases"][0]["target_bbox_xyxy"] = [0, 0, 1, 1]
        with self.assertRaisesRegex(Pa3ContractError, "evaluator/identity truth"):
            validate_public(value, PROMPT_MAP, self.root)

    def test_rejects_unconfirmed_goal_truth_precedence(self) -> None:
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        receipt["created_before_truth"] = False
        write_json(self.receipt, receipt)
        value = self.public()
        with self.assertRaisesRegex(Pa3ContractError, "goal-before-truth"):
            validate_public(value, PROMPT_MAP, self.root)

    def test_rejects_timestamp_that_does_not_precede_capture(self) -> None:
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        receipt["capture_created_at_utc"] = "2026-08-21T23:59:00Z"
        write_json(self.receipt, receipt)
        value = self.public()
        with self.assertRaisesRegex(Pa3ContractError, "does not precede capture"):
            validate_public(value, PROMPT_MAP, self.root)


class Pa3EvaluationTest(unittest.TestCase):
    def test_unique_and_set_valued_enter_primary_ambiguous_stays_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public_path = root / "public.json"
            private_path = root / "private.json"
            prediction_path = root / "prediction.json"
            public = {
                "schema_version": PUBLIC_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "cases": [
                    {"case_id": "unique", "goal_contract": {"reference_mode": "UNIQUE"}},
                    {"case_id": "set", "goal_contract": {"reference_mode": "SET_VALUED"}},
                    {"case_id": "ambiguous", "goal_contract": {"reference_mode": "AMBIGUOUS"}},
                    {"case_id": "not-visible", "goal_contract": {"reference_mode": "SET_VALUED"}},
                ],
            }
            write_json(public_path, public)
            private = {
                "schema_version": PRIVATE_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "public_input_sha256": sha256(public_path),
                "primary_iou_threshold": 0.30,
                "diagnostic_iou_thresholds": [0.10, 0.50],
                "recall_at_k": [1, 3, 5, 10],
                "cases": [
                    {"case_id": "unique", "reference_mode": "UNIQUE", "legal_target_bboxes_xyxy": [[0, 0, 10, 10]]},
                    {"case_id": "set", "reference_mode": "SET_VALUED", "legal_target_bboxes_xyxy": [[20, 20, 30, 30], [40, 40, 50, 50]]},
                    {"case_id": "ambiguous", "reference_mode": "AMBIGUOUS", "legal_target_bboxes_xyxy": [[60, 60, 70, 70]]},
                    {"case_id": "not-visible", "reference_mode": "SET_VALUED", "target_visibility": "NOT_VISIBLE", "legal_target_bboxes_xyxy": []},
                ],
            }
            write_json(private_path, private)
            body_sha = content_sha256(private_truth_body(private))
            receipts = {}
            for case_id in ("unique", "set", "ambiguous", "not-visible"):
                receipt_path = root / f"{case_id}-precedence.json"
                write_json(receipt_path, {
                    "schema_version": "blindassist_p1_pa3_goal_truth_precedence_receipt_v1",
                    "goal_receipt_body_sha256": "a" * 64,
                    "goal_recorded_at_utc": "2026-08-22T00:00:00Z",
                    "capture_created_at_utc": "2026-08-22T00:01:00Z",
                    "truth_created_at_utc": "2026-08-22T00:02:00Z",
                    "created_before_capture": True,
                    "created_before_truth": True,
                    "private_truth_body_sha256": body_sha,
                })
                receipts[case_id] = receipt_path
            for case in public["cases"]:
                receipt_path = receipts[case["case_id"]]
                case["goal_contract"].update({
                    "c0_goal_receipt_body_sha256": "a" * 64,
                    "precedence_receipt_path": str(receipt_path),
                    "precedence_receipt_sha256": sha256(receipt_path),
                })
            write_json(public_path, public)
            private["public_input_sha256"] = sha256(public_path)
            write_json(private_path, private)
            prediction = {
                "schema_version": PREDICTION_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "public_input_sha256": sha256(public_path),
                "private_truth_access": False,
                "provider": {
                    "model_sha256": EXPECTED_MODEL_SHA256,
                    "ultralytics_version": EXPECTED_ULTRALYTICS_VERSION,
                    **EXPECTED_PROVIDER_CONFIGURATION,
                },
                "cases": [
                    {"case_id": "unique", "latency_ms": 3.0, "candidates": [
                        {"rank": 1, "bbox_xyxy": [80, 80, 90, 90]},
                        {"rank": 2, "bbox_xyxy": [0, 0, 10, 10]},
                    ]},
                    {"case_id": "set", "latency_ms": 4.0, "candidates": [
                        {"rank": 1, "bbox_xyxy": [40, 40, 50, 50]},
                    ]},
                    {"case_id": "ambiguous", "latency_ms": 5.0, "candidates": []},
                    {"case_id": "not-visible", "latency_ms": 2.0, "candidates": []},
                ],
            }
            write_json(prediction_path, prediction)

            result = evaluate(public_path, private_path, prediction_path)

            self.assertEqual(2, result["primary_evaluable_case_count"])
            self.assertEqual(1, result["ambiguous_diagnostic_case_count"])
            self.assertEqual(1, result["target_not_visible_case_count"])
            self.assertEqual(0.5, result["candidate_availability"]["recall_at_1"])
            self.assertEqual(1.0, result["candidate_availability"]["recall_at_3"])
            self.assertEqual("P1_PA3_FULL_BOUNDED_SEMANTIC_TARGET_AVAILABILITY_ON_COHORT", result["terminal"])
            set_row = next(row for row in result["rows"] if row["case_id"] == "set")
            self.assertEqual(0.5, set_row["legal_target_recall_at_10"]["0.3"])
            self.assertEqual("NOT_EVALUABLE_BY_CONTRACT", result["ambiguous_specific_referent_accuracy"])

            prediction["provider"]["model_sha256"] = "0" * 64
            write_json(prediction_path, prediction)
            with self.assertRaisesRegex(Pa3ContractError, "model identity drift"):
                evaluate(public_path, private_path, prediction_path)
            prediction["provider"]["model_sha256"] = EXPECTED_MODEL_SHA256
            prediction["provider"]["text_encoder_sha256"] = "0" * 64
            write_json(prediction_path, prediction)
            with self.assertRaisesRegex(Pa3ContractError, "text_encoder_sha256"):
                evaluate(public_path, private_path, prediction_path)


class Pa3InputMaterializationTest(unittest.TestCase):
    def inputs(self, root: Path) -> tuple[dict, dict, dict]:
        c0 = {
            "schema_version": "blindassist_p1_pa3_c0_public_goal_cohort_v1",
            "prompt_map_sha256": content_sha256(PROMPT_MAP),
            "private_truth_access": False,
            "pa3_inference_authorized": False,
            "episodes": [{
                "episode_id": "episode-001",
                "goal_text_original": "帮我找入口",
                "goal_contract": {
                    "goal_type": "NAMED_BUILDING_ENTRANCE",
                    "reference_mode": "SET_VALUED",
                    "task_semantics": "find a building entrance satisfying the user task",
                },
                "canonical_prompt": "building entrance",
                "goal_provenance": {"goal_recorded_at_utc": "2026-08-22T00:00:00Z"},
            }],
        }
        c0["receipt_body_sha256"] = content_sha256(c0)
        image = root / "frame.jpg"
        image.write_bytes(b"prospective-frame")
        capture = {
            "schema_version": CAPTURE_SCHEMA,
            "cases": [{
                "case_id": "case-001",
                "episode_id": "episode-001",
                "capture_created_at_utc": "2026-08-22T00:01:00Z",
                "image_path": str(image),
                "image_sha256": sha256(image),
            }],
        }
        truth = {
            "schema_version": TRUTH_SCHEMA,
            "truth_created_at_utc": "2026-08-22T00:02:00Z",
            "primary_iou_threshold": 0.30,
            "diagnostic_iou_thresholds": [0.10, 0.50],
            "recall_at_k": [1, 3, 5, 10],
            "cases": [{
                "case_id": "case-001",
                "reference_mode": "SET_VALUED",
                "legal_target_bboxes_xyxy": [[1, 2, 11, 12]],
            }],
        }
        return c0, capture, truth

    def test_materializer_closes_goal_capture_truth_hash_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c0, capture, truth = self.inputs(root)
            output = root / "pa3"
            public_path, private_path = materialize_inputs(
                c0=c0,
                prompt_map=PROMPT_MAP,
                capture=capture,
                truth=truth,
                output_dir=output,
                source_base_dir=root,
            )
            public = json.loads(public_path.read_text(encoding="utf-8"))
            private = json.loads(private_path.read_text(encoding="utf-8"))
            self.assertEqual(sha256(public_path), private["public_input_sha256"])
            cases = validate_public(public, PROMPT_MAP, output)
            self.assertEqual("case-001", cases[0]["case_id"])
            receipt_path = Path(public["cases"][0]["goal_contract"]["precedence_receipt_path"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(content_sha256(private_truth_body(private)), receipt["private_truth_body_sha256"])

    def test_materializer_exposes_only_hash_bound_pretruth_public_spatial_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c0, capture, truth = self.inputs(root)
            spatial = {
                "schema_version": "blindassist_p1_pa3_public_spatial_goal_contract_v2",
                "protocol_id": "P1-PA3-S0-PUBLIC-SPATIAL-GOAL-CONTRACT-V1",
                "goal_receipt_body_sha256": c0["receipt_body_sha256"],
                "provider_public": True,
                "private_truth_access": False,
                "source_authority": "OPENSTREETMAP_PRETRUTH",
                "spatial_goal_role": "PRODUCT_NAVIGATION_ROUTE_ENDPOINT_CANDIDATE",
                "created_before_mapillary_metadata_pixels_and_truth": True,
                "episodes": [{
                    "episode_id": "episode-001",
                    "osm_type": "way",
                    "osm_id": 10,
                    "lat": 52.0,
                    "lon": 4.0,
                    "selected_parent": {"osm_type": "way", "osm_id": 10},
                    "selected_entrance": {
                        "osm_node_id": 11,
                        "lat": 52.0001,
                        "lon": 4.0001,
                        "entrance_tag": "main",
                        "access_tag": "yes",
                    },
                }],
            }
            spatial["roster_body_sha256"] = content_sha256(spatial)
            capture["cases"][0].update({
                "osm_entrance_node_id": 11,
                "public_spatial_contract_body_sha256": spatial["roster_body_sha256"],
            })
            output = root / "pa3"
            public_path, _ = materialize_inputs(
                c0=c0,
                prompt_map=PROMPT_MAP,
                capture=capture,
                truth=truth,
                output_dir=output,
                source_base_dir=root,
                public_spatial_contract=spatial,
            )
            public = json.loads(public_path.read_text(encoding="utf-8"))
            self.assertTrue(public["provider_contract"]["public_spatial_context"])
            context = public["cases"][0]["goal_contract"]["public_spatial_context"]
            self.assertEqual(11, context["route_endpoint_candidate"]["osm_node_id"])
            self.assertNotIn("legal_target_bboxes_xyxy", json.dumps(context))
            validate_public(public, PROMPT_MAP, output)

            capture["cases"][0]["osm_entrance_node_id"] = 12
            with self.assertRaisesRegex(Pa3ContractError, "route endpoint candidate drift"):
                materialize_inputs(
                    c0=c0,
                    prompt_map=PROMPT_MAP,
                    capture=capture,
                    truth=truth,
                    output_dir=root / "invalid",
                    source_base_dir=root,
                    public_spatial_contract=spatial,
                )

    def test_materializer_rejects_truth_created_before_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c0, capture, truth = self.inputs(root)
            truth["truth_created_at_utc"] = "2026-08-22T00:00:30Z"
            output = root / "pa3"
            with self.assertRaisesRegex(Pa3ContractError, "truth must be created after capture"):
                materialize_inputs(
                    c0=c0,
                    prompt_map=PROMPT_MAP,
                    capture=capture,
                    truth=truth,
                    output_dir=output,
                    source_base_dir=root,
                )
            self.assertFalse(output.exists())

    def test_materializer_resolves_relative_output_receipt_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c0, capture, truth = self.inputs(root)
            previous = Path.cwd()
            try:
                os.chdir(root)
                public_path, _ = materialize_inputs(
                    c0=c0,
                    prompt_map=PROMPT_MAP,
                    capture=capture,
                    truth=truth,
                    output_dir=Path("relative-pa3"),
                    source_base_dir=root,
                )
            finally:
                os.chdir(previous)
            public = json.loads(public_path.read_text(encoding="utf-8"))
            receipt_path = Path(public["cases"][0]["goal_contract"]["precedence_receipt_path"])
            self.assertTrue(receipt_path.is_absolute())
            self.assertTrue(receipt_path.is_file())


if __name__ == "__main__":
    unittest.main()
