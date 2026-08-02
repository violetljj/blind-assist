from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .common import (
    ACTION_REVIEW_SCHEMA,
    ARMS,
    COHORT_SCHEMA,
    EXCLUSION_SCHEMA,
    FULL_EVENT_FACTS_SCHEMA,
    PROTOCOL_ID,
    REPRESENTATION_ARMS,
    SCENE_FRAME_SCHEMA,
    TRACE_MANIFEST_SCHEMA,
    TRACE_SCHEMA,
    sha256_file,
    sha256_json,
)
from .evaluate import _validate_contract, run_audit, run_frozen_audit
from .discover_normal_candidates import normal_walkable_candidate
from .freeze_screening_cohort import ScreeningCohortError, freeze_screening_cohort
from .materialize_screening_inputs import MaterializationError, build_plan
from .reconcile_screening_windows import reconcile_screening_windows
from .prepare_p0_review_packets import prepare_packets
from .audit_data_admission import _dct_matrix, _hash_features, _p_hash_candidates, audit_admission
from .prepare_phash_manual_review import prepare_packets as prepare_phash_packets
from .finalize_phash_manual_review import HOLD_STATUS as PHASH_HOLD_STATUS, PASSED_STATUS as PHASH_PASSED_STATUS, finalize_review as finalize_phash_review
from .reconcile_phash_data_admission import PASSED_STATUS_AFTER_REVIEW, reconcile as reconcile_phash_admission
from .finalize_p0_anchor_agreement import P0AgreementError, P0_PASSED_STATUS, P0_STOP_STATUS, finalize_p0
from .prepare_p1_review_packets import prepare_packets as prepare_p1_packets
from .finalize_p1_action_facts import P1_PASSED_STATUS, P1_STOP_STATUS, finalize_p1
from .finalize_scene_facts import FINAL_STATUS, finalize_scene_facts
from .select_unseen_discovery import select_unseen_candidates
from .render_report import render


BUCKETS = (
    "blocking_obstacle_positive",
    "boundary_level_change_positive",
    "parallel_curb_negative",
    "normal_walkable_negative",
)


def contract() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PRE_OUTPUT_LOCKED",
        "cohort_requirements": {
            "minimum_parent_events": 4,
            "minimum_source_sessions": 4,
            "one_event_per_session": True,
            "minimum_bucket_parent_events": {bucket: 1 for bucket in BUCKETS},
        },
        "gates": {
            "minimum_actionability_exact_agreement": 1.0,
            "minimum_clearance_exact_agreement": 1.0,
            "minimum_knownness_exact_agreement": 1.0,
            "minimum_sequence_exact_agreement": 1.0,
            "maximum_unknown_burden": 0.2,
            "maximum_response_delay_frames": 0,
        },
        "shared_execution": {
            "decision_kernel_contract_id": "eval-validity-test-kernel",
            "risk_config_id": "test-fixed",
            "feedback_profile": "TEST",
            "clock": "frozen_anchor_frame",
        },
    }


def registry() -> dict:
    return {
        "schema_version": EXCLUSION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "excluded_source_sessions": ["consumed-session"],
    }


def cohort() -> dict:
    items = []
    for index, bucket in enumerate(BUCKETS, start=1):
        items.append({
            "parent_event_id": f"event-{index}",
            "source_session_id": f"fresh-session-{index}",
            "bucket": bucket,
            "frame_indices": list(range(20)),
            "anchor_frame_indices": [4, 8, 12, 16],
            "scene_fact_manifest_sha256": "a" * 64,
        })
    return {
        "schema_version": COHORT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "SCENE_FACTS_FROZEN_ACTION_REVIEWS_PENDING",
        "candidate_outputs_opened": False,
        "items": items,
    }


def review(cohort_value: dict, role: str, disagreement: bool = False) -> tuple[dict, dict[str, dict[str, int | str]]]:
    review_map: dict[str, dict[str, int | str]] = {}
    items = []
    for index, item in enumerate(cohort_value["items"], start=1):
        positive = item["bucket"].endswith("_positive")
        for frame in item["anchor_frame_indices"]:
            opaque = f"opaque-{index}-{frame}"
            review_map[opaque] = {"parent_event_id": item["parent_event_id"], "anchor_frame_index": frame}
            action = "YES" if positive and frame in (4, 8) else "NO"
            cleared = "YES" if positive and frame in (12, 16) else "NO"
            if disagreement and frame in (4, 8):
                action = "NO" if action == "YES" else "YES"
            items.append({"review_item_id": opaque, "anchor": {"frame_index": frame, "reminder_now": action, "cleared": cleared, "knownness": "KNOWN"}})
    return {
        "schema_version": ACTION_REVIEW_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "reviewer_role": role,
        "cohort_sha256": sha256_json(cohort_value),
        "isolated_context": True,
        "other_review_visible_before_submission": False,
        "model_or_oracle_output_visible": False,
        "items": items,
    }, review_map


def scene_rows(cohort_value: dict) -> list[dict]:
    output: list[dict] = []
    for item in cohort_value["items"]:
        for arm in REPRESENTATION_ARMS:
            for frame in item["frame_indices"]:
                intersection = {"current_yolo": 5, "truth_box": 7, "truth_mask": 8}[arm]
                predicted = {"current_yolo": 8, "truth_box": 8, "truth_mask": 8}[arm]
                output.append({
                    "schema_version": SCENE_FRAME_SCHEMA,
                    "protocol_id": PROTOCOL_ID,
                    "parent_event_id": item["parent_event_id"],
                    "arm": arm,
                    "frame_index": frame,
                    "scene_fact_manifest_sha256": item["scene_fact_manifest_sha256"],
                    "frame_area_px": 100,
                    "truth_area_px": 10,
                    "intersection_area_px": intersection,
                    "predicted_area_px": predicted,
                    "truth_component_count": 1,
                    "matched_truth_component_count": 1,
                    "predicted_component_count": 1,
                    "unmatched_predicted_component_count": 0,
                    "previous_prediction_iou": None if frame == 0 else 0.9,
                })
    return output


def trace_manifest(cohort_value: dict, full_facts: dict) -> dict:
    return {
        "schema_version": TRACE_MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "cohort_sha256": sha256_json(cohort_value),
        "full_event_facts_sha256": sha256_json(full_facts),
        "action_reviews_passed_before_trace_access": True,
        "full_event_facts_frozen_before_trace_access": True,
        "shared_execution": contract()["shared_execution"],
        "arms": {
            "current_yolo": {"input_kind": "CURRENT_YOLO_OUTPUT"},
            "truth_box": {"input_kind": "SCENE_FACT_TRUTH_BOX"},
            "truth_mask": {"input_kind": "SCENE_FACT_TRUTH_MASK"},
            "synthetic_oracle": {"input_kind": "EVENT_FACT_SYNTHETIC_DIRECTIVE"},
        },
    }


def traces(cohort_value: dict) -> list[dict]:
    output = []
    for item in cohort_value["items"]:
        positive = item["bucket"].endswith("_positive")
        for arm in ARMS:
            for frame in item["frame_indices"]:
                output.append({
                    "schema_version": TRACE_SCHEMA,
                    "protocol_id": PROTOCOL_ID,
                    "parent_event_id": item["parent_event_id"],
                    "arm": arm,
                    "frame_index": frame,
                    "feedback_alert": positive and frame == 4,
                })
    return output


def full_event_facts(cohort_value: dict) -> dict:
    agreement = {
        "metrics": {
            "reminder_now_exact_agreement": 1.0,
            "cleared_exact_agreement": 1.0,
            "knownness_exact_agreement": 1.0,
            "parent_event_sequence_exact_agreement": 1.0,
            "unknown_anchor_burden": 0.0,
            "anchor_count": 16,
        },
        "passed": True,
    }
    items = []
    for item in cohort_value["items"]:
        positive = item["bucket"].endswith("_positive")
        items.append({
            "parent_event_id": item["parent_event_id"],
            "bucket": item["bucket"],
            "alertable_interval_frames": [4, 8] if positive else None,
            "passed_interval_frames": [12, 16] if positive else None,
        })
    return {
        "schema_version": FULL_EVENT_FACTS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "cohort_sha256": sha256_json(cohort_value),
        "status": "FULL_EVENT_FACTS_FROZEN_AFTER_ANCHOR_CONSISTENCY",
        "anchor_consistency_sha256": sha256_json(agreement),
        "independent_full_review_evidence": {
            "review_a_sha256": "b" * 64,
            "review_b_sha256": "c" * 64,
            "reviewers_isolated": True,
            "model_or_oracle_output_visible": False,
            "agreement_passed": True,
            "unknown_anchor_or_frame_count": 0,
        },
        "items": items,
    }


class EvalValidityR0Test(unittest.TestCase):
    def valid_inputs(self) -> tuple[dict, dict, dict, dict, dict, dict, dict, list[dict], dict, list[dict]]:
        cohort_value = cohort()
        review_a, review_map = review(cohort_value, "ACTION_REVIEW_A")
        review_b, _ = review(cohort_value, "ACTION_REVIEW_B")
        facts = full_event_facts(cohort_value)
        return (
            contract(), registry(), cohort_value, review_map, review_a, review_b, facts,
            scene_rows(cohort_value), trace_manifest(cohort_value, facts), traces(cohort_value),
        )

    def test_valid_ladder_reports_two_layers(self) -> None:
        result = run_audit(*self.valid_inputs())
        self.assertEqual("VALID_EVALUATION_CONSTRUCT_AND_ORACLE_LADDER", result["status"])
        self.assertTrue(result["actionability_consistency"]["passed"])
        self.assertEqual(0.5, result["scene_representation"]["current_yolo"]["coverage"])
        self.assertTrue(result["oracle_monotonicity"]["passed"])
        report = render(result)
        self.assertIn("## 表征层", report)
        self.assertIn("## 事件层", report)

    def test_frozen_p0_p1_route_keeps_the_same_ladder_gate(self) -> None:
        cohort_value = cohort()
        cohort_value["status"] = "SCENE_AND_EVENT_FACTS_FROZEN_AFTER_P0_P1"
        screening_sha = "d" * 64
        for item in cohort_value["items"]:
            item["screening_cohort_sha256"] = screening_sha
        agreement = {
            "metrics": {
                "reminder_now_exact_agreement": 1.0, "cleared_exact_agreement": 1.0,
                "knownness_exact_agreement": 1.0, "parent_event_sequence_exact_agreement": 1.0,
                "unknown_anchor_burden": 0.0, "anchor_count": 16,
            }, "passed": True,
        }
        p0 = {
            "schema_version": "blindassist.eval_validity_r0.p0_anchor_agreement.v1", "protocol_id": PROTOCOL_ID,
            "status": "P0_ANCHOR_CONSISTENCY_PASSED", "screening_cohort_sha256": screening_sha,
            "candidate_outputs_opened": False, "anchor_agreement": agreement,
        }
        p1 = {
            "schema_version": "blindassist.eval_validity_r0.p1_action_facts.v1", "protocol_id": PROTOCOL_ID,
            "status": "P1_ACTION_FACTS_FROZEN_AFTER_P0_CONSISTENCY", "screening_cohort_sha256": screening_sha,
            "p0_anchor_agreement_sha256": sha256_json(p0), "candidate_outputs_opened": False,
            "independent_full_review_evidence": {"agreement_passed": True, "unknown_or_disagreement_event_count": 0},
            "items": [{"screening_event_id": item["parent_event_id"], "resolved": True, "p0_anchor_compatible": True} for item in cohort_value["items"]],
        }
        facts = full_event_facts(cohort_value)
        facts["anchor_consistency_sha256"] = sha256_json(agreement)
        facts["p0_anchor_agreement_receipt_sha256"] = sha256_json(p0)
        facts["p1_action_facts_sha256"] = sha256_json(p1)
        result = run_frozen_audit(
            contract(), registry(), cohort_value, p0, p1, facts, scene_rows(cohort_value),
            trace_manifest(cohort_value, facts), traces(cohort_value),
        )
        self.assertEqual("VALID_EVALUATION_CONSTRUCT_AND_ORACLE_LADDER", result["status"])

    def test_inconsistent_actionability_stops_before_traces(self) -> None:
        inputs = list(self.valid_inputs())
        cohort_value = inputs[2]
        review_b, _ = review(cohort_value, "ACTION_REVIEW_B", disagreement=True)
        # Disagree on every action field so the consistency gate has no ambiguity.
        for item in review_b["items"]:
            anchor = item["anchor"]
            anchor["reminder_now"] = "NO" if anchor["reminder_now"] == "YES" else "YES"
        inputs[5] = review_b
        result = run_audit(*inputs)
        self.assertEqual("STOP_EVENT_FACT_CONSISTENCY_NOT_ESTABLISHED", result["status"])
        self.assertIsNone(result["event_quality"])

    def test_mask_false_alert_fails_monotonicity(self) -> None:
        inputs = list(self.valid_inputs())
        trace_rows = copy.deepcopy(inputs[9])
        for row in trace_rows:
            if row["arm"] == "truth_mask" and row["parent_event_id"] == "event-3" and row["frame_index"] == 4:
                row["feedback_alert"] = True
        inputs[9] = trace_rows
        result = run_audit(*inputs)
        self.assertEqual("STOP_ORACLE_MONOTONICITY_NOT_ESTABLISHED", result["status"])
        self.assertIn("more_false_alert_events", result["oracle_monotonicity"]["checks"][1]["problems"])

    def test_mask_premature_positive_alert_fails_monotonicity(self) -> None:
        inputs = list(self.valid_inputs())
        trace_rows = copy.deepcopy(inputs[9])
        for row in trace_rows:
            if row["arm"] == "truth_mask" and row["parent_event_id"] == "event-1" and row["frame_index"] == 0:
                row["feedback_alert"] = True
        inputs[9] = trace_rows
        result = run_audit(*inputs)
        self.assertEqual("STOP_ORACLE_MONOTONICITY_NOT_ESTABLISHED", result["status"])
        self.assertIn("more_premature_alert_events", result["oracle_monotonicity"]["checks"][1]["problems"])

    def test_synthetic_oracle_failure_is_evaluator_integrity_failure(self) -> None:
        inputs = list(self.valid_inputs())
        trace_rows = copy.deepcopy(inputs[9])
        for row in trace_rows:
            if row["arm"] == "synthetic_oracle" and row["parent_event_id"] == "event-3" and row["frame_index"] == 4:
                row["feedback_alert"] = True
        inputs[9] = trace_rows
        result = run_audit(*inputs)
        self.assertEqual("STOP_EVALUATOR_INTEGRITY_NOT_ESTABLISHED", result["status"])
        self.assertFalse(result["oracle_monotonicity"]["synthetic_integrity_passed"])

    def test_trace_cannot_use_anchor_screen_as_full_event_fact_substitute(self) -> None:
        inputs = list(self.valid_inputs())
        inputs[6]["status"] = "P0_ONLY_NO_FULL_EVENT_FACTS"
        with self.assertRaisesRegex(ValueError, "full event facts: wrong status"):
            run_audit(*inputs)

    def test_trace_manifest_must_bind_the_frozen_full_event_facts(self) -> None:
        inputs = list(self.valid_inputs())
        inputs[8]["full_event_facts_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "full event facts binding mismatch"):
            run_audit(*inputs)

    def test_review_packet_with_source_session_identity_is_rejected(self) -> None:
        inputs = list(self.valid_inputs())
        inputs[4]["items"][0]["source_session_id"] = "should-not-be-visible"
        with self.assertRaisesRegex(ValueError, "leaks forbidden context"):
            run_audit(*inputs)

    def test_p0_uses_one_opaque_item_per_causal_anchor(self) -> None:
        cohort_value = cohort()
        review_a, review_map = review(cohort_value, "ACTION_REVIEW_A")
        self.assertEqual(16, len(review_a["items"]))
        self.assertTrue(all("anchor" in item and "anchors" not in item for item in review_a["items"]))
        self.assertEqual(16, len(review_map))

    def test_consumed_session_is_rejected(self) -> None:
        inputs = list(self.valid_inputs())
        inputs[2]["items"][0]["source_session_id"] = "consumed-session"
        with self.assertRaisesRegex(ValueError, "excluded source session"):
            run_audit(*inputs)

    def test_checked_in_contract_matches_validator_schema(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        path = repository / "docs" / "research" / "dual-loop" / "EVAL_VALIDITY_R0_CONTRACT_2026-08-02.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        _validate_contract(value)
        self.assertEqual(0.0, value["gates"]["maximum_unknown_burden"])

    def test_sparse_discovery_rejects_consumed_session_without_creating_truth(self) -> None:
        discovery = {"candidates": [
            {"session_id": "fresh-session", "selection_profile": "center_obstacle"},
            {"session_id": "consumed-session", "selection_profile": "step_curb"},
        ]}
        result = select_unseen_candidates(discovery, registry())
        self.assertEqual("SOURCE_MASK_DISCOVERY_ONLY_NOT_EVENT_TRUTH", result["status"])
        self.assertEqual(["fresh-session"], result["eligible_source_sessions"])
        self.assertEqual({"center_obstacle": 1}, result["eligible_profile_session_counts"])
        self.assertEqual("excluded_source_session", result["rejected_candidates"][0]["reason"])

    def test_strict_normal_shortlist_rejects_any_center_or_boundary_signal(self) -> None:
        base = {
            "path_geometry_usable": True,
            "has_center_hazard": False,
            "step_curb": False,
            "center_obstacle": False,
            "center_lateral_target": False,
        }
        self.assertTrue(normal_walkable_candidate(base))
        for field in ("has_center_hazard", "step_curb", "center_obstacle", "center_lateral_target"):
            candidate = dict(base)
            candidate[field] = True
            self.assertFalse(normal_walkable_candidate(candidate))

    def test_screening_cohort_is_48_session_disjoint_and_not_event_truth(self) -> None:
        def candidate(session: str, profile: str, start: int) -> dict:
            return {
                "session_id": session, "official_split": "train", "camera": "camera_chest", "lens": "left",
                "selection_profile": profile, "recommended_start_frame": start, "geometry_matching_source_frames": [start + 15],
            }
        hazard = {
            "protocol_id": PROTOCOL_ID, "status": "SOURCE_MASK_DISCOVERY_ONLY_NOT_EVENT_TRUTH",
            "eligible_candidates": [
                *[candidate(f"center-{index:02d}", "center_obstacle", index) for index in range(12)],
                *[candidate(f"curb-{index:02d}", "step_curb", index) for index in range(24)],
            ],
        }
        normal = {
            "protocol_id": PROTOCOL_ID, "status": "SOURCE_MASK_DISCOVERY_ONLY_NOT_EVENT_TRUTH",
            "candidates": [candidate(f"normal-{index:02d}", "strict_normal_walkable_source_mask_only", index) for index in range(12)],
        }
        result = freeze_screening_cohort(hazard, normal, registry())
        self.assertEqual("OUTPUT_BLIND_SCREENING_COHORT_FROZEN", result["status"])
        self.assertEqual(48, result["source_session_count"])
        self.assertTrue(all(item["event_bucket"] is None for item in result["items"]))
        self.assertTrue(all(item["candidate_outputs_opened"] is False for item in result["items"]))

    def test_screening_cohort_rejects_source_session_collisions_that_prevent_a_stratum(self) -> None:
        hazard = {
            "protocol_id": PROTOCOL_ID, "status": "SOURCE_MASK_DISCOVERY_ONLY_NOT_EVENT_TRUTH",
            "eligible_candidates": [{
                "session_id": "same", "official_split": "train", "camera": "camera_chest", "lens": "left",
                "selection_profile": "center_obstacle", "recommended_start_frame": 0, "geometry_matching_source_frames": [15],
            }] * 12,
        }
        normal = {
            "protocol_id": PROTOCOL_ID, "status": "SOURCE_MASK_DISCOVERY_ONLY_NOT_EVENT_TRUTH",
            "candidates": [{
                "session_id": f"normal-{index:02d}", "official_split": "train", "camera": "camera_chest", "lens": "left",
                "selection_profile": "strict_normal_walkable_source_mask_only", "recommended_start_frame": index, "geometry_matching_source_frames": [index + 15],
            } for index in range(12)],
        }
        with self.assertRaisesRegex(ScreeningCohortError, "center_obstacle_candidate"):
            freeze_screening_cohort(hazard, normal, registry())

    def test_native_materialization_plan_requires_exact_contiguous_rgb_and_mask_frames(self) -> None:
        cohort_value = {
            "schema_version": "blindassist.eval_validity_r0.screening_cohort.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "OUTPUT_BLIND_SCREENING_COHORT_FROZEN",
            "candidate_outputs_opened": False,
            "final_event_facts_frozen": False,
            "items": [{
                "screening_event_id": f"event-{index:02d}", "source_session_id": f"session-{index:02d}",
                "camera": "camera_chest", "lens": "left", "screening_stratum": "test",
                "source_selection_profile": "test", "source_window": {"start_frame": 10, "frame_count": 20, "p0_anchor_offsets": [1, 5, 10, 19]},
            } for index in range(48)],
        }
        def objects(prefix: str) -> list[dict]:
            return [{"name": f"{prefix}{frame:06d}.png", "generation": "1", "size": "7", "md5Hash": "x"} for frame in range(10, 30)]
        plan = build_plan(cohort_value, list_objects=objects, get_source_fps=lambda session, camera: 15.0)
        self.assertEqual(48, plan["screening_event_count"])
        self.assertEqual(960, plan["frame_count"])
        self.assertEqual(48 * 20 * 7, plan["total_rgb_bytes"])
        with self.assertRaisesRegex(MaterializationError, "incomplete continuous window"):
            build_plan(cohort_value, list_objects=lambda prefix: objects(prefix)[:-1], get_source_fps=lambda session, camera: 15.0)

    def test_window_reconciliation_preserves_reference_without_opening_pixels(self) -> None:
        cohort_value = {
            "schema_version": "blindassist.eval_validity_r0.screening_cohort.v1", "protocol_id": PROTOCOL_ID,
            "status": "OUTPUT_BLIND_SCREENING_COHORT_FROZEN", "candidate_outputs_opened": False, "final_event_facts_frozen": False,
            "items": [{
                "screening_event_id": f"event-{index:02d}", "source_session_id": f"session-{index:02d}",
                "camera": "camera_chest", "lens": "left", "source_window": {
                    "start_frame": 19, "frame_count": 60, "p0_anchor_offsets": [8, 20, 36, 52], "source_screening_reference_frame": 34,
                },
            } for index in range(48)],
        }
        def objects(prefix: str) -> list[dict]:
            return [{"name": f"{prefix}{frame:06d}.png"} for frame in range(64)]
        result = reconcile_screening_windows(cohort_value, list_objects=objects)
        self.assertEqual("OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN", result["status"])
        self.assertTrue(all(item["source_window"]["start_frame"] == 4 for item in result["items"]))
        self.assertTrue(all(item["source_window"]["metadata_admission"]["pixel_payload_read"] is False for item in result["items"]))
        for item in cohort_value["items"]:
            item["source_window"]["start_frame"] = 4
        retained = reconcile_screening_windows(cohort_value, list_objects=objects)
        self.assertTrue(all(item["source_window"]["start_frame"] == 4 for item in retained["items"]))

    def test_p0_packets_split_every_anchor_into_distinct_opaque_causal_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_value = {
                "schema_version": "blindassist.eval_validity_r0.screening_cohort.v1", "protocol_id": PROTOCOL_ID,
                "status": "OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN",
                "candidate_outputs_opened": False, "final_event_facts_frozen": False,
                "items": [{
                    "screening_event_id": f"event-{index:02d}", "source_session_id": f"session-{index:02d}",
                    "source_window": {"frame_count": 20, "p0_anchor_offsets": [1, 5, 10, 19]},
                } for index in range(48)],
            }
            materialized = {"schema_version": "blindassist.eval_validity_r0.continuous_native_inputs.v1", "protocol_id": PROTOCOL_ID,
                            "status": "CONTINUOUS_NATIVE_RGB_AND_MASKS_MATERIALIZED_OUTPUT_BLIND",
                            "screening_cohort_sha256": sha256_json(cohort_value), "candidate_outputs_opened": False, "items": []}
            for item in cohort_value["items"]:
                frames = []
                for ordinal in range(20):
                    path = root / "raw" / item["screening_event_id"] / f"{ordinal:03d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{item['screening_event_id']}:{ordinal}".encode())
                    frames.append({"ordinal": ordinal, "rgb_path": path.relative_to(root).as_posix(), "rgb_sha256": sha256_file(path)})
                materialized["items"].append({"screening_event_id": item["screening_event_id"], "source_session_id": item["source_session_id"], "frames": frames})
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(materialized), encoding="utf-8")
            admission = {
                "schema_version": "blindassist.eval_validity_r0.data_admission_receipt.v1", "protocol_id": PROTOCOL_ID, "status": "EVAL_VALIDITY_DATA_ADMISSION_PASSED",
                "screening_cohort_sha256": sha256_json(cohort_value), "materialized_manifest_sha256": sha256_file(manifest_path),
                "candidate_outputs_opened": False,
            }
            result = prepare_packets(
                cohort=cohort_value, admission=admission, materialized_root=root,
                reviewer_a_root=root / "reviewer-a", reviewer_b_root=root / "reviewer-b", private_root=root / "private",
            )
            self.assertEqual(192, result["packet_a_item_count"])
            packet = json.loads((root / "reviewer-a" / "packet.json").read_text(encoding="utf-8"))
            self.assertTrue(all(len(item["causal_rgb_frames"]) == item["current_frame_ordinal"] + 1 for item in packet["items"]))
            self.assertFalse(any("session-" in json.dumps(item) for item in packet["items"]))
            private = json.loads((root / "private" / "private-review-map.json").read_text(encoding="utf-8"))
            self.assertEqual(192, len(private["reviewer_a_map"]))
            self.assertEqual(sha256_json(cohort_value), packet["submission_shape"]["screening_cohort_sha256"])

            def submission(role: str, review_map: dict[str, dict[str, int | str]]) -> dict:
                return {
                    "schema_version": ACTION_REVIEW_SCHEMA, "protocol_id": PROTOCOL_ID, "reviewer_role": role,
                    "screening_cohort_sha256": sha256_json(cohort_value), "isolated_context": True,
                    "other_review_visible_before_submission": False, "model_or_oracle_output_visible": False,
                    "items": [{
                        "review_item_id": opaque_id,
                        "anchor": {
                            "frame_index": item["anchor_frame_index"],
                            "reminder_now": "YES" if item["anchor_frame_index"] in (5, 10) else "NO",
                            "cleared": "YES" if item["anchor_frame_index"] == 19 else "NO",
                            "knownness": "KNOWN",
                        },
                    } for opaque_id, item in review_map.items()],
                }

            packet_b = json.loads((root / "reviewer-b" / "packet.json").read_text(encoding="utf-8"))
            review_a = submission("ACTION_REVIEW_A", private["reviewer_a_map"])
            review_b = submission("ACTION_REVIEW_B", private["reviewer_b_map"])
            review_a_path, review_b_path = root / "review-a.json", root / "review-b.json"
            review_a_path.write_text(json.dumps(review_a), encoding="utf-8")
            review_b_path.write_text(json.dumps(review_b), encoding="utf-8")
            agreement = finalize_p0(
                screening_cohort=cohort_value, admission_receipt=admission, private_map=private, packet_a=packet, packet_b=packet_b,
                review_a=review_a, review_b=review_b,
                packet_a_sha256=sha256_file(root / "reviewer-a" / "packet.json"), packet_b_sha256=sha256_file(root / "reviewer-b" / "packet.json"),
                review_a_sha256=sha256_file(review_a_path), review_b_sha256=sha256_file(review_b_path),
            )
            self.assertEqual(P0_PASSED_STATUS, agreement["status"])
            self.assertTrue(agreement["anchor_agreement"]["passed"])
            p1_packets = prepare_p1_packets(
                cohort=cohort_value, admission=admission, p0_agreement=agreement, materialized_root=root,
                reviewer_a_root=root / "p1-reviewer-a", reviewer_b_root=root / "p1-reviewer-b", private_root=root / "p1-private",
            )
            self.assertEqual(48, p1_packets["packet_a_item_count"])
            p1_packet = json.loads((root / "p1-reviewer-a" / "packet.json").read_text(encoding="utf-8"))
            self.assertTrue(all(len(item["causal_rgb_frames"]) == 20 for item in p1_packet["items"]))
            self.assertFalse(any("session-" in json.dumps(item) for item in p1_packet["items"]))
            self.assertEqual(sha256_json(agreement), p1_packet["submission_shape"]["p0_anchor_agreement_sha256"])

            p1_private = json.loads((root / "p1-private" / "private-review-map.json").read_text(encoding="utf-8"))

            def p1_submission(role: str, review_map: dict[str, dict[str, int | str]]) -> dict:
                return {
                    "schema_version": "blindassist.eval_validity_r0.p1_action_review.v1", "protocol_id": PROTOCOL_ID, "reviewer_role": role,
                    "screening_cohort_sha256": sha256_json(cohort_value), "p0_anchor_agreement_sha256": sha256_json(agreement),
                    "isolated_context": True, "reviewer_is_not_a_p0_reviewer": True,
                    "other_review_visible_before_submission": False, "model_or_oracle_output_visible": False,
                    "items": [{
                        "review_item_id": opaque_id,
                        "event_fact": {"knownness": "KNOWN", "reminder_now_interval": [5, 10], "cleared_interval": [11, 19]},
                    } for opaque_id in review_map],
                }

            p1_review_a = p1_submission("P1_FULL_EVENT_REVIEW_A", p1_private["reviewer_a_map"])
            p1_review_b = p1_submission("P1_FULL_EVENT_REVIEW_B", p1_private["reviewer_b_map"])
            p1_a_path, p1_b_path = root / "p1-review-a.json", root / "p1-review-b.json"
            p1_a_path.write_text(json.dumps(p1_review_a), encoding="utf-8")
            p1_b_path.write_text(json.dumps(p1_review_b), encoding="utf-8")
            p1_facts = finalize_p1(
                screening_cohort=cohort_value, admission_receipt=admission, p0_agreement=agreement, private_map=p1_private,
                packet_a=p1_packet, packet_b=json.loads((root / "p1-reviewer-b" / "packet.json").read_text(encoding="utf-8")),
                review_a=p1_review_a, review_b=p1_review_b,
                packet_a_sha256=sha256_file(root / "p1-reviewer-a" / "packet.json"), packet_b_sha256=sha256_file(root / "p1-reviewer-b" / "packet.json"),
                review_a_sha256=sha256_file(p1_a_path), review_b_sha256=sha256_file(p1_b_path),
            )
            self.assertEqual(P1_PASSED_STATUS, p1_facts["status"])
            self.assertTrue(all(item["p0_anchor_compatible"] for item in p1_facts["items"]))
            p1_review_b["items"][0]["event_fact"]["reminder_now_interval"] = [6, 10]
            p1_disagreement = finalize_p1(
                screening_cohort=cohort_value, admission_receipt=admission, p0_agreement=agreement, private_map=p1_private,
                packet_a=p1_packet, packet_b=json.loads((root / "p1-reviewer-b" / "packet.json").read_text(encoding="utf-8")),
                review_a=p1_review_a, review_b=p1_review_b,
                packet_a_sha256=sha256_file(root / "p1-reviewer-a" / "packet.json"), packet_b_sha256=sha256_file(root / "p1-reviewer-b" / "packet.json"),
                review_a_sha256=sha256_file(p1_a_path), review_b_sha256=sha256_file(p1_b_path),
            )
            self.assertEqual(P1_STOP_STATUS, p1_disagreement["status"])
            review_b["items"][0]["anchor"]["reminder_now"] = "YES"
            disagreement = finalize_p0(
                screening_cohort=cohort_value, admission_receipt=admission, private_map=private, packet_a=packet, packet_b=packet_b,
                review_a=review_a, review_b=review_b,
                packet_a_sha256=sha256_file(root / "reviewer-a" / "packet.json"), packet_b_sha256=sha256_file(root / "reviewer-b" / "packet.json"),
                review_a_sha256=sha256_file(review_a_path), review_b_sha256=sha256_file(review_b_path),
            )
            self.assertEqual(P0_STOP_STATUS, disagreement["status"])
            self.assertFalse(disagreement["anchor_agreement"]["passed"])
            bad_admission = dict(admission)
            bad_admission["status"] = "HOLD_EVAL_VALIDITY_DATA"
            with self.assertRaisesRegex(P0AgreementError, "data admission did not pass"):
                finalize_p0(
                    screening_cohort=cohort_value, admission_receipt=bad_admission, private_map=private, packet_a=packet, packet_b=packet_b,
                    review_a=review_a, review_b=review_b,
                    packet_a_sha256=sha256_file(root / "reviewer-a" / "packet.json"), packet_b_sha256=sha256_file(root / "reviewer-b" / "packet.json"),
                    review_a_sha256=sha256_file(review_a_path), review_b_sha256=sha256_file(review_b_path),
                )

    def test_admission_holds_on_any_new_to_excluded_phash_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_value = {
                "schema_version": "blindassist.eval_validity_r0.screening_cohort.v1", "protocol_id": PROTOCOL_ID,
                "status": "OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN", "candidate_outputs_opened": False,
                "items": [{"screening_event_id": f"event-{index:02d}", "source_session_id": f"new-{index:02d}", "source_window": {"frame_count": 1}} for index in range(48)],
            }
            manifest = {"schema_version": "blindassist.eval_validity_r0.continuous_native_inputs.v1", "protocol_id": PROTOCOL_ID,
                        "screening_cohort_sha256": sha256_json(cohort_value), "candidate_outputs_opened": False, "items": []}
            for item in cohort_value["items"]:
                rgb = root / "new" / item["screening_event_id"] / "rgb.png"
                mask = root / "new" / item["screening_event_id"] / "mask.png"
                rgb.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), (255, 0, 0)).save(rgb)
                Image.new("RGB", (8, 8), (0, 0, 0)).save(mask)
                manifest["items"].append({"screening_event_id": item["screening_event_id"], "source_session_id": item["source_session_id"], "frames": [{
                    "ordinal": 0, "rgb_path": rgb.relative_to(root).as_posix(), "rgb_sha256": sha256_file(rgb),
                    "source_mask_path": mask.relative_to(root).as_posix(), "source_mask_sha256": sha256_file(mask),
                }]})
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            old = root / "old.png"
            Image.new("RGB", (8, 8), (255, 0, 0)).save(old)
            old_features = _hash_features(old, _dct_matrix())
            cache = root / "cache.jsonl"
            cache.write_text(json.dumps({
                "session_ids": ["old-session"], "pixel_domain": "rgb", "decode_error": "", "rel_path": "old.png",
                "file_sha256": sha256_file(old), "rgb_pixel_sha256": old_features["rgb_pixel_sha256"],
                "phash": old_features["phash_variants"]["original"], "phash_variants": old_features["phash_variants"],
            }) + "\n", encoding="utf-8")
            ledger = root / "truth.jsonl"
            ledger.write_text("{}\n", encoding="utf-8")
            registry_value = {"schema_version": "blindassist.eval_validity_r0.exclusion_registry.v1", "protocol_id": PROTOCOL_ID, "excluded_source_sessions": ["old-session"]}
            result = audit_admission(
                cohort=cohort_value, manifest=manifest, materialized_root=root, registry=registry_value,
                truth_ledger=ledger, cache_path=cache,
            )
            self.assertEqual("HOLD_EVAL_VALIDITY_DATA", result["status"])
            self.assertFalse(result["checks"]["p_hash_no_unresolved_new_to_excluded_candidate"])
            self.assertGreater(result["evidence"]["p_hash_candidate_count_lower_bound"], 0)
            self.assertTrue(result["evidence"]["p_hash_candidate_enumeration_complete"])

    def test_scene_finalization_freezes_only_the_predeclared_balanced_four_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items, manifest_items = [], []
            for index in range(48):
                if index < 12:
                    profile, source_id, alertable, passed = "center_obstacle", 20, [5, 10], [11, 19]
                elif index < 24:
                    profile, source_id, alertable, passed = "step_curb", 2, [5, 10], [11, 19]
                elif index < 36:
                    profile, source_id, alertable, passed = "step_curb", 2, None, None
                else:
                    profile, source_id, alertable, passed = "strict_normal_walkable_source_mask_only", 1, None, None
                event_id, session_id = f"event-{index:02d}", f"scene-session-{index:02d}"
                items.append({
                    "screening_event_id": event_id, "source_session_id": session_id,
                    "source_selection_profile": profile, "screening_stratum": "test",
                    "source_window": {"start_frame": 100, "frame_count": 20, "p0_anchor_offsets": [1, 5, 10, 19]},
                    "candidate_outputs_opened": False, "final_event_facts_frozen": False,
                })
                frames = []
                for ordinal in range(20):
                    mask = root / "native" / event_id / f"{ordinal:03d}.png"
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    array = np.zeros((4, 4, 3), dtype=np.uint8)
                    array[:, :, 0] = source_id
                    array[:, :, 1] = 1
                    Image.fromarray(array, mode="RGB").save(mask)
                    frames.append({"ordinal": ordinal, "source_mask_path": mask.relative_to(root).as_posix(), "source_mask_sha256": sha256_file(mask)})
                manifest_items.append({"screening_event_id": event_id, "source_session_id": session_id, "frames": frames})
            screening = {
                "schema_version": "blindassist.eval_validity_r0.screening_cohort.v1", "protocol_id": PROTOCOL_ID,
                "status": "OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN", "candidate_outputs_opened": False,
                "final_event_facts_frozen": False, "items": items,
            }
            screening_sha = sha256_json(screening)
            admission = {"schema_version": "blindassist.eval_validity_r0.data_admission_receipt.v1", "protocol_id": PROTOCOL_ID, "status": "EVAL_VALIDITY_DATA_ADMISSION_PASSED", "screening_cohort_sha256": screening_sha, "candidate_outputs_opened": False}
            p0 = {
                "schema_version": "blindassist.eval_validity_r0.p0_anchor_agreement.v1", "protocol_id": PROTOCOL_ID,
                "status": "P0_ANCHOR_CONSISTENCY_PASSED", "screening_cohort_sha256": screening_sha,
                "admission_receipt_sha256": sha256_json(admission), "candidate_outputs_opened": False,
                "anchor_agreement": {"metrics": {}, "passed": True},
            }
            p0_sha = sha256_json(p0)
            p1_items = []
            for index, item in enumerate(items):
                alertable = [5, 10] if index < 24 else None
                passed = [11, 19] if index < 24 else None
                p1_items.append({"screening_event_id": item["screening_event_id"], "resolved": True, "p0_anchor_compatible": True, "alertable_interval_frames": alertable, "passed_interval_frames": passed})
            p1 = {
                "schema_version": "blindassist.eval_validity_r0.p1_action_facts.v1", "protocol_id": PROTOCOL_ID,
                "status": "P1_ACTION_FACTS_FROZEN_AFTER_P0_CONSISTENCY", "screening_cohort_sha256": screening_sha,
                "admission_receipt_sha256": sha256_json(admission), "p0_anchor_agreement_sha256": p0_sha,
                "candidate_outputs_opened": False,
                "independent_full_review_evidence": {"agreement_passed": True, "unknown_or_disagreement_event_count": 0, "review_a_sha256": "a" * 64, "review_b_sha256": "b" * 64},
                "items": p1_items,
            }
            manifest = {
                "schema_version": "blindassist.eval_validity_r0.continuous_native_inputs.v1", "protocol_id": PROTOCOL_ID,
                "status": "CONTINUOUS_NATIVE_RGB_AND_MASKS_MATERIALIZED_OUTPUT_BLIND", "screening_cohort_sha256": screening_sha,
                "candidate_outputs_opened": False, "items": manifest_items,
            }
            receipt = finalize_scene_facts(
                screening_cohort=screening, admission_receipt=admission, p0_agreement=p0, p1_action_facts=p1,
                manifest=manifest, materialized_root=root, output_root=root / "frozen",
            )
            self.assertEqual(FINAL_STATUS, receipt["status"])
            final_cohort = json.loads((root / "frozen" / "cohort-v1.json").read_text(encoding="utf-8"))
            self.assertEqual({bucket: 12 for bucket in BUCKETS}, dict(sorted(Counter(item["bucket"] for item in final_cohort["items"]).items())))
            self.assertEqual(960, len((root / "frozen" / "scene-facts.jsonl").read_text(encoding="utf-8").splitlines()))

    def test_p_hash_evidence_cap_is_a_fail_closed_lower_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            Image.new("RGB", (8, 8), (255, 0, 0)).save(image)
            features = _hash_features(image, _dct_matrix())
            new_rows = [{"event_id": "new-event", "session_id": "new-session", "ordinal": 0, "path": image}]
            prior_rows = [{
                "rel_path": f"old-{index:03d}.png", "session_ids": [f"old-{index:03d}"],
                "file_sha256": f"{index:064x}", "rgb_pixel_sha256": features["rgb_pixel_sha256"],
                "phash": features["phash_variants"]["original"], "phash_variants": features["phash_variants"],
            } for index in range(201)]
            _, edges, errors, truncated = _p_hash_candidates(new_rows, prior_rows)
            self.assertFalse(errors)
            self.assertEqual(200, len(edges))
            self.assertTrue(truncated)

    def test_phash_manual_review_requires_exact_two_reviewer_distinct_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialized = root / "materialized"
            new_image = materialized / "events" / "screen-001" / "rgb" / "000.png"
            prior_image = root / "artifacts.local" / "prior.png"
            new_image.parent.mkdir(parents=True, exist_ok=True)
            prior_image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (10, 20, 30)).save(new_image)
            Image.new("RGB", (8, 8), (30, 20, 10)).save(prior_image)
            prior_features = _hash_features(prior_image, _dct_matrix())
            candidate = {
                "new_event_id": "screen-001", "new_session_id": "new-session", "new_ordinal": 0,
                "prior_path": "artifacts.local/prior.png", "prior_sessions": ["old-session"],
                "comparison": "new:original:old:original", "hamming": 6,
            }
            checks = {
                "session_disjoint": True, "old_truth_session_disjoint": True, "parent_identity_disjoint": True,
                "exact_rgb_disjoint": True, "decoded_rgb_disjoint": True, "exact_source_mask_disjoint": True,
                "p_hash_prior_session_coverage_complete": True, "p_hash_prior_decode_complete": True,
                "p_hash_new_decode_complete": True, "p_hash_no_unresolved_new_to_excluded_candidate": False,
            }
            admission = {
                "schema_version": "blindassist.eval_validity_r0.data_admission_receipt.v1", "protocol_id": PROTOCOL_ID,
                "status": "HOLD_EVAL_VALIDITY_DATA", "candidate_outputs_opened": False, "checks": checks,
                "screening_cohort_sha256": "c" * 64, "materialized_manifest_sha256": "m" * 64,
                "source_session_count": 1, "frame_counts": {"rgb": 1, "source_mask": 1},
                "evidence": {"p_hash_candidate_enumeration_complete": True, "p_hash_candidate_count_lower_bound": 1, "p_hash_candidates": [candidate]},
            }
            cache = root / "cache.jsonl"
            cache.write_text(json.dumps({"rel_path": "artifacts.local/prior.png", "rgb_pixel_sha256": prior_features["rgb_pixel_sha256"], "decode_error": ""}) + "\n", encoding="utf-8")
            prepared = prepare_phash_packets(
                admission=admission, materialized_root=materialized, workspace_root=root, prior_cache=cache,
                reviewer_a_root=root / "review-a", reviewer_b_root=root / "review-b", private_root=root / "private",
            )
            packet_a, packet_b, private = prepared["packet_a"], prepared["packet_b"], prepared["private"]
            self.assertEqual(1, prepared["candidate_case_count"])
            self.assertNotIn("screen-001", json.dumps(packet_a))
            self.assertNotIn("old-session", json.dumps(packet_b))

            def review(packet: dict, role: str, decision: str) -> dict:
                return {
                    "schema_version": "blindassist.eval_validity_r0.phash_manual_review.v1", "protocol_id": PROTOCOL_ID,
                    "reviewer_role": role, "admission_receipt_sha256": sha256_json(admission), "isolated_context": True,
                    "other_review_visible_before_submission": False, "model_or_oracle_output_visible": False,
                    "items": [{"review_item_id": item["review_item_id"], "same_natural_capture": decision} for item in packet["items"]],
                }

            review_a = review(packet_a, "PHASH_RGB_REVIEW_A", "DISTINCT_CAPTURE")
            review_b = review(packet_b, "PHASH_RGB_REVIEW_B", "DISTINCT_CAPTURE")
            result = finalize_phash_review(
                admission=admission, private_map=private, packet_a=packet_a, packet_b=packet_b,
                review_a=review_a, review_b=review_b,
                packet_a_sha256=sha256_file(root / "review-a" / "packet.json"), packet_b_sha256=sha256_file(root / "review-b" / "packet.json"),
                review_a_sha256=sha256_json(review_a), review_b_sha256=sha256_json(review_b),
            )
            self.assertEqual(PHASH_PASSED_STATUS, result["status"])
            reconciled = reconcile_phash_admission(held_admission=admission, phash_resolution=result)
            self.assertEqual(PASSED_STATUS_AFTER_REVIEW, reconciled["status"])
            review_b["items"][0]["same_natural_capture"] = "UNKNOWN"
            held = finalize_phash_review(
                admission=admission, private_map=private, packet_a=packet_a, packet_b=packet_b,
                review_a=review_a, review_b=review_b,
                packet_a_sha256=sha256_file(root / "review-a" / "packet.json"), packet_b_sha256=sha256_file(root / "review-b" / "packet.json"),
                review_a_sha256=sha256_json(review_a), review_b_sha256=sha256_json(review_b),
            )
            self.assertEqual(PHASH_HOLD_STATUS, held["status"])


if __name__ == "__main__":
    unittest.main()
