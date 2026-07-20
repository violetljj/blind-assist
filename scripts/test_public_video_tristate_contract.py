#!/usr/bin/env python3
"""Pure tests for the frozen prospective tri-state lifecycle contract."""

from __future__ import annotations

import copy
import unittest

import public_video_tristate_contract as subject


def contract_fixture() -> dict[str, object]:
    return {
        "schema": subject.SCHEMA,
        "contract_id": "test-v1",
        "model": {
            "weights_sha256": "a" * 64,
            "text_prompt_used": False,
        },
        "scan": {
            "sample_interval_ms": 1000,
            "image_size": 640,
            "confidence": 0.05,
            "require_nearfield_corridor": False,
            "baseline_groups": list(subject.SELECTED_GROUPS),
            "workzone_marker_additions": sorted(subject.WORKZONE_MARKER_ADDITIONS),
        },
        "lifecycle": {
            "selected_groups": list(subject.SELECTED_GROUPS),
            "entry_window_samples": 3,
            "entry_min_active_samples": 2,
            "clear_absent_samples": 3,
        },
        "source_eligibility": {
            "item_level_reusable_license_required": True,
            "original_temporal_order_required": True,
            "continuous_capture_required": True,
            "hard_cut_or_montage_allowed": False,
            "prospective_source_must_not_have_influenced_r78_parameters": True,
        },
        "acceptance": {
            "exactly_one_exit_interval_required": True,
            "gpt_visual_reference_must_be_contained": True,
            "post_clear_single_frame_reopen_allowed": False,
            "minimum_risk_present_active_fraction": 0.5,
            "maximum_stable_clear_active_fraction": 0.1,
        },
        "authorization": {
            key: False for key in subject.REQUIRED_FALSE_AUTHORIZATIONS
        },
    }


class ProspectiveTristateContractTest(unittest.TestCase):
    def test_frozen_contract_shape_passes(self) -> None:
        contract = contract_fixture()
        self.assertIs(contract, subject.validate_contract(contract))

    def test_marker_drift_fails_closed(self) -> None:
        contract = copy.deepcopy(contract_fixture())
        contract["scan"]["workzone_marker_additions"].append("street sign")
        with self.assertRaisesRegex(ValueError, "marker additions"):
            subject.validate_contract(contract)

    def test_promotion_flag_fails_closed(self) -> None:
        contract = copy.deepcopy(contract_fixture())
        contract["authorization"]["training_execution_authorized"] = True
        with self.assertRaisesRegex(ValueError, "unauthorized promotion"):
            subject.validate_contract(contract)

    def test_scan_binding_rejects_parameter_override(self) -> None:
        contract = subject.validate_contract(contract_fixture())
        with self.assertRaisesRegex(ValueError, "sample interval"):
            subject.validate_scan_binding(
                contract,
                weights_sha256="a" * 64,
                sample_interval_ms=2000,
                image_size=640,
                confidence=0.05,
                require_nearfield_corridor=False,
                include_workzone_markers=True,
            )

    def test_scan_binding_accepts_exact_contract(self) -> None:
        contract = subject.validate_contract(contract_fixture())
        subject.validate_scan_binding(
            contract,
            weights_sha256="a" * 64,
            sample_interval_ms=1000,
            image_size=640,
            confidence=0.05,
            require_nearfield_corridor=False,
            include_workzone_markers=True,
        )


if __name__ == "__main__":
    unittest.main()
