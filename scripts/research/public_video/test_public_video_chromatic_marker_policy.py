#!/usr/bin/env python3
"""Pure tests for the frozen chromatic marker policy."""

from __future__ import annotations

import copy
import unittest

import public_video_chromatic_marker_policy as subject


def contract_fixture() -> dict[str, object]:
    return {
        "risk_evidence_policy": {
            "policy_id": subject.POLICY_ID,
            "target_classes": list(subject.TARGET_CLASSES),
            "detection_acceptance": "high_saturation_fraction > dark_fraction",
            "minimum_accepted_detections_per_active_frame": 1,
            "absolute_color_threshold_used": False,
            "geometry_gate_used": False,
        },
        "model": {"weights_sha256": "a" * 64},
        "scan": {"sample_interval_ms": 1000, "image_size": 640, "confidence": 0.05},
    }


class ChromaticMarkerPolicyTest(unittest.TestCase):
    def test_exact_policy_passes(self) -> None:
        contract = contract_fixture()
        self.assertIs(contract["risk_evidence_policy"], subject.validate_policy(contract))

    def test_color_rule_drift_fails_closed(self) -> None:
        contract = copy.deepcopy(contract_fixture())
        contract["risk_evidence_policy"]["detection_acceptance"] = "warm > dark"
        with self.assertRaisesRegex(ValueError, "color rule"):
            subject.validate_policy(contract)

    def test_apply_policy_accepts_only_chromatic_target(self) -> None:
        policy = subject.validate_policy(contract_fixture())
        rows = [{
            "timestamp_ms": 0,
            "detections": [
                {"class_name": "barricade", "features": {"high_saturation_fraction": 0.2, "dark_fraction": 0.1}},
                {"class_name": "traffic cone", "features": {"high_saturation_fraction": 0.1, "dark_fraction": 0.2}},
                {"class_name": "construction site", "features": {"high_saturation_fraction": 0.9, "dark_fraction": 0.0}},
            ],
        }]
        filtered = subject.apply_policy(rows, policy)[0]
        self.assertEqual({"barrier_structure": 1}, filtered["semantic_group_counts"])

    def test_extractor_binding_rejects_target_drift(self) -> None:
        contract = contract_fixture()
        with self.assertRaisesRegex(ValueError, "target classes"):
            subject.validate_extractor_binding(
                contract,
                weights_sha256="a" * 64,
                sample_interval_ms=1000,
                image_size=640,
                confidence=0.05,
                target_classes={"traffic cone"},
            )


if __name__ == "__main__":
    unittest.main()
