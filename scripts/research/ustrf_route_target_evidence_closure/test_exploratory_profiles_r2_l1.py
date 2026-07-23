from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from exploratory_profiles_r2_l1 import (
    ExecutionAborted,
    InputBlocked,
    assert_candidate_input_uncontaminated,
    base_terminal_receipt,
    compute_discontinuities,
    identity,
    load_and_verify_config,
    replay_candidate_ledger,
    validate_exhausted_resource_guard_receipt,
    validate_compact_ledger,
    validate_profile_contract,
)
from validate_exploratory_profiles_r2_l1 import forbidden_key_fragments


class ExploratoryProfilesR2L1MutationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = {
            "source_id": "source",
            "sequence_id": "sequence",
            "frame_count": 3,
            "frame_mask_sha256": "mask",
        }
        self.rows = [
            {
                "source_id": "source",
                "sequence_id": "sequence",
                "frame_id": frame,
                "source_capture_timestamp_ns": timestamp,
                "unit_id": f"unit-{frame}",
            }
            for frame, timestamp in [(1, 100), (3, 200), (4, 1_000_000_201)]
        ]

    def test_frame_gap_and_large_time_gap_both_reset(self) -> None:
        resets = compute_discontinuities([(self.descriptor, self.rows)])
        self.assertEqual(2, len(resets))
        self.assertEqual(["frame_id_not_consecutive"], resets[0]["reasons"])
        self.assertEqual(["timestamp_gap_exceeds_one_second"], resets[1]["reasons"])

    def test_nonpositive_timestamp_resets(self) -> None:
        rows = copy.deepcopy(self.rows[:2])
        rows[1]["frame_id"] = 2
        rows[1]["source_capture_timestamp_ns"] = 100
        resets = compute_discontinuities([(self.descriptor, rows)])
        self.assertEqual(["timestamp_nonpositive"], resets[0]["reasons"])

    def test_compact_ledger_file_presence_is_not_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "ledger.json"
            successor = root / "successor.json"
            ledger.write_text("{}", encoding="utf-8")
            successor.write_text("{}", encoding="utf-8")
            self.assertFalse(
                validate_compact_ledger(
                    ledger, successor, self.descriptor, self.rows
                )
            )

    def test_terminal_receipt_contains_no_selection_fields(self) -> None:
        config = {
            "candidate_roster": ["C1", "C2", "C3"],
            "resource_guards": {},
        }
        gaps = [
            {
                "expected_frame_count": 3,
                "missing_fields": ["android_canvas_canonical_detector_raw_successor"],
            }
        ]
        receipt = base_terminal_receipt(
            "FAIL_CLOSED_INPUT_BLOCKED",
            {},
            [(self.descriptor, self.rows)],
            [],
            gaps,
            config,
            "raw_missing",
        )
        self.assertEqual([], forbidden_key_fragments(receipt))
        self.assertFalse(receipt["candidate_execution"]["started"])
        self.assertEqual([], receipt["profiles"])
        mutated = copy.deepcopy(receipt)
        mutated["profiles"] = [{"nested": {"winner": "C1"}}]
        self.assertTrue(forbidden_key_fragments(mutated))

    def test_illegal_terminal_state_rejected(self) -> None:
        with self.assertRaises(ValueError):
            base_terminal_receipt(
                "PARTIAL_RESULTS",
                {},
                [],
                [],
                [],
                {"candidate_roster": [], "resource_guards": {}},
                None,
            )

    def test_authority_mutation_fails_before_external_inputs(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        config_path = repo / "configs/ustrf_route_target_l1_exploratory_profile_r1.json"
        config = __import__("json").loads(config_path.read_text(encoding="utf-8"))
        config["authority"]["selection"] = True
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "config.json"
            mutated.write_text(__import__("json").dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(InputBlocked, "authority_must_remain_closed"):
                load_and_verify_config(repo, mutated)

    def test_candidate_sha_drift_fails_closed(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        config_path = repo / "configs/ustrf_route_target_l1_exploratory_profile_r1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["parent_bindings"]["candidate_implementation"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "config.json"
            mutated.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(InputBlocked, "candidate_implementation_sha256_mismatch"):
                load_and_verify_config(repo, mutated)

    def test_truth_clear_eligibility_and_scoring_fields_cannot_enter_state(self) -> None:
        for key in (
            "truth",
            "truth_clear",
            "clear",
            "critical",
            "eligibility",
            "metric_eligibility",
            "scoring_label",
            "score",
            "event_id",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    ExecutionAborted, "candidate_input_forbidden_field"
                ):
                    assert_candidate_input_uncontaminated({"safe": {"nested": {key: 1}}})

    def test_discontinuity_and_sequence_start_reset_candidate_state(self) -> None:
        frames = []
        for frame_id, timestamp in ((1, 100), (2, 200), (10, 2_000_000_000)):
            frames.append(
                {
                    "source_id": "source",
                    "sequence_id": "sequence",
                    "frame_id": frame_id,
                    "source_capture_timestamp_ns": timestamp,
                    "source_size": [640, 480],
                    "android_raw_output_sha256": "a" * 64,
                    "detector_processing_latency_ns": 10,
                    "host_decode_latency_ns": 20,
                    "person_detections": [
                        {
                            "prediction_index": 0,
                            "class_id": 0,
                            "label": "person",
                            "confidence": 0.9,
                            "box": [100.0, 100.0, 300.0, 400.0],
                        }
                    ],
                }
            )
        ledger = {"frames": frames}
        routes = {
            identity(frame): {"status": "known", "uv": [200.0, 350.0]}
            for frame in frames
        }
        tracker_config = {
            "fixed_kernel": {
                "min_alert_frames": 2,
                "min_clear_frames": 3,
                "route_point_margin_fraction": 0.08,
            }
        }
        config = {"resource_guards": {"host_maximum_rss_bytes": 8 * 1024**3}}
        trace, _ = replay_candidate_ledger(
            "C2_ROUTE_OCCUPANCY_EPISODE_FSM",
            ledger,
            routes,
            {identity(frames[2])},
            tracker_config,
            config,
        )
        self.assertEqual([1], trace[1]["deliveries"])
        self.assertEqual([], trace[2]["deliveries"])
        self.assertTrue(trace[0]["state_reset_before_frame"])
        self.assertTrue(trace[2]["state_reset_before_frame"])
        self.assertGreater(
            trace[0]["candidate_consume_timestamp_ns"],
            trace[0]["source_capture_timestamp_ns"],
        )
        second_sequence = copy.deepcopy(ledger)
        for frame in second_sequence["frames"]:
            frame["sequence_id"] = "other"
        second_routes = {
            identity(frame): {"status": "known", "uv": [200.0, 350.0]}
            for frame in second_sequence["frames"]
        }
        other_trace, _ = replay_candidate_ledger(
            "C2_ROUTE_OCCUPANCY_EPISODE_FSM",
            second_sequence,
            second_routes,
            {identity(second_sequence["frames"][2])},
            tracker_config,
            config,
        )
        self.assertEqual([], other_trace[0]["deliveries"])

    def _valid_profile(self) -> dict:
        return {
            "metrics": {
                "critical_miss": {"denominator": 8},
                "clearance": {
                    "denominator": 12,
                    "pre_clear_units_excluded": 6357,
                },
                "unknown_or_stale_alert": {"denominator": 62229},
                "repeat": {
                    "denominator": 0,
                    "denominator_source": "first_delivery_then_complete_observation",
                    "eligibility_status": "not_evaluable",
                },
                "evidence_age": {
                    "timestamp_frame_count": 0,
                    "eligibility_status": "not_evaluable",
                },
                "event_recall": {
                    "level": "L0",
                    "eligibility_status": "diagnostic_only",
                    "denominator": 0,
                },
                "regeneration": {
                    "level": "L0",
                    "eligibility_status": "diagnostic_only",
                    "denominator": 0,
                },
                "false_alerts_per_minute": {
                    "level": "L0",
                    "eligibility_status": "diagnostic_only",
                },
            }
        }

    def _profile_config(self) -> dict:
        return {"metric_permissions": {"repeat": {"minimum_actual_denominator": 5}}}

    def test_zero_over_zero_cannot_be_evaluable(self) -> None:
        profile = self._valid_profile()
        profile["metrics"]["event_recall"]["eligibility_status"] = "evaluable"
        with self.assertRaisesRegex(ExecutionAborted, "L0_metric_authority_opened"):
            validate_profile_contract(profile, self._profile_config())

    def test_preclear_units_cannot_enter_clearance(self) -> None:
        profile = self._valid_profile()
        profile["metrics"]["clearance"]["pre_clear_units_excluded"] = 6356
        with self.assertRaisesRegex(ExecutionAborted, "pre_clear_entered_clearance"):
            validate_profile_contract(profile, self._profile_config())

    def test_repeat_truth_pool_cannot_replace_actual_denominator(self) -> None:
        profile = self._valid_profile()
        profile["metrics"]["repeat"]["denominator_source"] = "truth_pool"
        with self.assertRaisesRegex(ExecutionAborted, "repeat_truth_pool"):
            validate_profile_contract(profile, self._profile_config())

    def test_evidence_age_one_missing_frame_invalidates_metric(self) -> None:
        profile = self._valid_profile()
        profile["metrics"]["evidence_age"] = {
            "timestamp_frame_count": 62228,
            "eligibility_status": "evaluable",
        }
        with self.assertRaisesRegex(ExecutionAborted, "evidence_age_missing_frame"):
            validate_profile_contract(profile, self._profile_config())

    def test_l0_metric_cannot_gain_gate_field(self) -> None:
        profile = self._valid_profile()
        profile["metrics"]["false_alerts_per_minute"]["gate"] = True
        with self.assertRaisesRegex(ExecutionAborted, "L0_metric_gate_field"):
            validate_profile_contract(profile, self._profile_config())

    def test_exhausted_resource_guard_is_machine_locked(self) -> None:
        implementations = {"core_implementation_sha256": "a" * 64}
        receipt = {
            "schema": "blindassist_ustrf_route_target_l1e_resource_guard_attempts_r1",
            "stage": "R2-L1E",
            "config_sha256": "b" * 64,
            "implementation_bindings": implementations,
            "guard": "system_available_physical_memory_bytes",
            "required_minimum_bytes": 100,
            "maximum_attempts": 3,
            "attempts": [],
            "device_attempt_created": False,
            "canonical_raw_shard_created": False,
            "candidate_execution_started": False,
            "candidate_trace_created": False,
            "profile_authority": False,
            "automatic_retry_allowed_after_receipt": False,
            "retry_limit_exhausted": True,
        }
        for attempt in range(1, 4):
            receipt["attempts"].append(
                {
                    "attempt_number": attempt,
                    "attempt_id": f"attempt-{attempt}",
                    "observation_time_utc": None,
                    "observation_time_status": "not_recorded_pre_receipt_contract",
                    "observed_available_bytes": 90 + attempt,
                    "required_available_bytes": 100,
                    "wall_time_seconds": 0.0,
                    "process_exit_code": None,
                    "system_event": "host_pre_device_memory_guard",
                    "last_safe_checkpoint": {
                        "verified_sequence_ledgers": 2,
                        "verified_frames": 4594,
                        "candidate_execution_started": False,
                    },
                    "outcome": "STOPPED_BEFORE_DEVICE_ATTEMPT",
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guard.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            loaded = validate_exhausted_resource_guard_receipt(
                path, "b" * 64, implementations, 100, 3
            )
            self.assertFalse(loaded["automatic_retry_allowed_after_receipt"])

    def test_identity_includes_sequence_and_timestamp(self) -> None:
        first = self.rows[0]
        second = copy.deepcopy(first)
        second["sequence_id"] = "other"
        self.assertNotEqual(identity(first), identity(second))


if __name__ == "__main__":
    unittest.main()
