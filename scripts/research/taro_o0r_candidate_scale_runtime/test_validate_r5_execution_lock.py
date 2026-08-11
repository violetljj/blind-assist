#!/usr/bin/env python3
"""Mutation tests for the future TARO R5 one-shot execution lock validator."""

from __future__ import annotations

import copy
import hashlib
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import validate_r5_execution_lock as validator


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _payload() -> dict[str, object]:
    implementation = validator.REPO_ROOT / validator.IMPLEMENTATION_LOCK_RELATIVE
    relative_lock = validator.DEFAULT_LOCK_PATH.relative_to(validator.REPO_ROOT).as_posix()
    return {
        "schema": validator.SCHEMA,
        "lock_id": validator.LOCK_ID,
        "date": "2026-08-11",
        "research_mode": "WILD_LAB",
        "status": "ONE_SHOT_EXECUTION_AUTHORIZED_NOT_YET_CONSUMED",
        "implementation_lock_binding": {"path": validator.IMPLEMENTATION_LOCK_RELATIVE, "bytes": implementation.stat().st_size, "sha256": _sha(implementation)},
        "roots": {"source_root": validator.SOURCE_ROOT_RELATIVE, "r3_evidence_root": validator.R3_ROOT_RELATIVE, "r5_evidence_root": validator.R5_ROOT_RELATIVE},
        "candidate_identity": {
            "model_id": "depthart-s-metric-indoor-448-official-fp32",
            "source_root": validator.DEPTHART_SOURCE,
            "source_commit": "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c",
            "checkpoint_path": validator.CHECKPOINT,
            "checkpoint_bytes": 32871942,
            "checkpoint_sha256": "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65",
            "preprocess_id": "DEPTHART_OFFICIAL_LOWER_BOUND_448_RGB_CUBIC_IMAGENET_V1",
            "postprocess_id": "TARO_TORCH_CPU_BILINEAR_ALIGN_CORNERS_TRUE_FLOAT32_448X608_TO_1440X1920_V1",
        },
        "exact_cohort": {"parent_count": 8, "physical_frame_count": r5.EXPECTED_FRAME_COUNT, "query_slot_count": r5.EXPECTED_QUERY_COUNT, "canonical_frame_identity_sequence_sha256": r5.EXPECTED_IDENTITY_SEQUENCE_SHA256},
        "phase_firewall": {"all_candidates_before_decisions": True, "all_decisions_before_faro": True, "phase_a_completion_reload_before_faro": True, "prior_eval_truth_or_outcome_roots_allowed": False, "branch_reselection_after_truth_allowed": False},
        "unique_argv": ["E:/codex-tools/tools/venvs/blindassist-venv-export312/Scripts/python.exe", "-m", validator.MODULE, "--execution-lock", relative_lock],
        "argv_alternatives": [],
        "resource_budget": {"maximum_wall_seconds": 28800, "maximum_peak_rss_bytes": 17179869184, "maximum_cuda_allocated_bytes": 8500000000, "maximum_evidence_bytes": 2147483648},
        "side_effects": {"network_requests": 0, "training_steps": 0, "device_actions": 0},
        "activation": {"root_must_be_absent": True, "one_shot_consumed_on_root_creation": True, "overwrite": False, "rerun": False},
        "user_authority": {"explicit_model_execution_authority": True, "scope": "EXACT_R5_211_DEPTHART_INFERENCES_PHASE_A_SOURCE_DECISIONS_AND_SAME_FRAME_PHASE_B_FARO_SCORING", "authorization_sha256": "A" * 64, "recorded_at_utc": "2026-08-11T00:00:00Z"},
        "execution_authority": {"one_shot_execution_lock": True, "user_model_execution_authority": True, "depthart_inference": True, "phase_a_source_decisions": True, "phase_b_truth_scoring": True, "training": False, "network": False, "device": False, "product": False, "safety": False},
        "claim_ceiling": "synthetic validator fixture",
    }


class R5ExecutionLockValidatorTests(unittest.TestCase):
    def assert_rejected(self, mutation, code: str) -> None:
        value = _payload()
        mutation(value)
        with self.assertRaises(validator.R5ExecutionLockError) as caught:
            validator.validate_payload(value, lock_path=validator.DEFAULT_LOCK_PATH, verify_files=False, enforce_argv=False)
        self.assertEqual(code, caught.exception.code)

    def test_exact_future_lock_shape_passes_without_activation(self) -> None:
        result = validator.validate_payload(_payload(), lock_path=validator.DEFAULT_LOCK_PATH, verify_files=False, enforce_argv=False)
        self.assertEqual(str((validator.REPO_ROOT / validator.R5_ROOT_RELATIVE).resolve()), result["roots"]["r5_evidence_root"])
        self.assertFalse((validator.REPO_ROOT / validator.R5_ROOT_RELATIVE).exists())

    def test_user_authority_and_execution_scope_cannot_be_fabricated_or_expanded(self) -> None:
        self.assert_rejected(lambda value: value["user_authority"].__setitem__("explicit_model_execution_authority", False), "R5_EXEC_USER_AUTHORITY_MISSING")
        self.assert_rejected(lambda value: value["execution_authority"].__setitem__("training", True), "R5_EXEC_AUTHORITY_DRIFT")

    def test_phase_transform_and_argv_drift_are_rejected(self) -> None:
        self.assert_rejected(lambda value: value["phase_firewall"].__setitem__("all_decisions_before_faro", False), "R5_EXEC_PHASE_FIREWALL_DRIFT")
        self.assert_rejected(lambda value: value["candidate_identity"].__setitem__("postprocess_id", "ALIGN_CORNERS_FALSE"), "R5_EXEC_TRANSFORM_DRIFT")
        self.assert_rejected(lambda value: value["unique_argv"].append("--rerun"), "R5_EXEC_ARGV_DRIFT")


if __name__ == "__main__":
    unittest.main()
