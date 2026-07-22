#!/usr/bin/env python3
"""Focused tests for the fail-closed risk/lifecycle prototype contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sanpo_risk_lifecycle_prototype as subject


HAZARDS = ("parallel_boundary", "step_curb", "center_obstacle", "lateral_pedestrian_or_ebike")


def positive() -> dict:
    return {
        "format": subject.TARGET_FORMAT,
        "episode_id": "positive",
        "duration_ms": 10000,
        "expected_should_alert": True,
        "pixel_supervision_role": "auxiliary_only",
        "risk_profile": {
            "primary_hazard_type": "step_curb",
            "corridor_relation": "enters_or_blocks",
            "lifecycle": "approach_alertable_clear",
        },
        "lifecycle_intervals_ms": {
            "approach": [0, 2000], "alertable": [2000, 8000], "post_event": [8000, 10000],
        },
    }


def negative() -> dict:
    return {
        "format": subject.TARGET_FORMAT,
        "episode_id": "negative",
        "duration_ms": 10000,
        "expected_should_alert": False,
        "pixel_supervision_role": "auxiliary_only",
        "risk_profile": {
            "primary_hazard_type": "parallel_boundary",
            "corridor_relation": "outside_or_nonblocking",
            "lifecycle": "no_alert",
        },
        "lifecycle_intervals_ms": {"non_alert": [0, 10000]},
    }


class RiskLifecyclePrototypeTest(unittest.TestCase):
    def test_reviewed_intervals_become_deterministic_half_open_labels(self) -> None:
        subject.validate_target(positive(), allowed_hazard_types=HAZARDS)
        labels = subject.lifecycle_labels_for_timestamps(positive(), [0, 1999, 2000, 7999, 8000, 9999])
        self.assertEqual([1, 1, 2, 2, 3, 3], labels)
        self.assertEqual(
            {"hazard": 1, "corridor_relation": 1, "episode_should_alert": 1},
            subject.risk_profile_labels(positive(), hazard_types=HAZARDS),
        )

    def test_matched_negative_stays_non_alert(self) -> None:
        subject.validate_target(negative(), allowed_hazard_types=HAZARDS)
        self.assertEqual([0, 0, 0], subject.lifecycle_labels_for_timestamps(negative(), [0, 5000, 9999]))
        self.assertEqual(0, subject.risk_profile_labels(negative(), hazard_types=HAZARDS)["episode_should_alert"])

    def test_rejects_pixel_or_interval_shortcuts(self) -> None:
        target = positive()
        target["pixel_supervision_role"] = "primary"
        with self.assertRaisesRegex(subject.TargetContractError, "auxiliary_only"):
            subject.validate_target(target, allowed_hazard_types=HAZARDS)
        target = positive()
        target["lifecycle_intervals_ms"]["alertable"] = [2100, 8000]
        with self.assertRaisesRegex(subject.TargetContractError, "contiguous"):
            subject.validate_target(target, allowed_hazard_types=HAZARDS)
        with self.assertRaisesRegex(subject.TargetContractError, "half-open"):
            subject.lifecycle_labels_for_timestamps(positive(), [10000])

    def test_loader_requires_complete_model_consensus_targets_and_matching_hash(self) -> None:
        targets = [positive(), negative()]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target_path = root / "risk_lifecycle_targets.jsonl"
            report_path = root / "risk_lifecycle_target_report.json"
            target_path.write_text("".join(json.dumps(row) + "\n" for row in targets), encoding="utf-8")
            report = {
                "format": subject.REPORT_FORMAT,
                "target_sha256": subject.canonical_sha256(targets),
                "supervision_tier": "hash_bound_model_consensus",
                "review_authority": "gpt_codex_isolated_consensus_v1",
                "validated_collection": {"training_eligible": True},
                "training_execution_authorized": True,
                "production_model_replacement_authorized": False,
                "pixel_supervision_role": "auxiliary_only",
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            loaded = subject.load_attested_targets(targets_path=target_path, report_path=report_path, allowed_hazard_types=HAZARDS)
            self.assertEqual(["positive", "negative"], [row["episode_id"] for row in loaded])
            report["validated_collection"] = {"training_eligible": False}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(subject.TargetContractError, "complete GPT/Codex-reviewed"):
                subject.load_attested_targets(targets_path=target_path, report_path=report_path, allowed_hazard_types=HAZARDS)

    def test_loader_accepts_explicit_hash_bound_model_silver_provisional_targets(self) -> None:
        targets = [positive(), negative()]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target_path = root / "risk_lifecycle_targets.jsonl"
            report_path = root / "risk_lifecycle_target_report.json"
            target_path.write_text("".join(json.dumps(row) + "\n" for row in targets), encoding="utf-8")
            report_path.write_text(json.dumps({
                "format": subject.REPORT_FORMAT,
                "target_sha256": subject.canonical_sha256(targets),
                "supervision_tier": "hash_bound_model_silver_provisional",
                "provisional_training_only": True,
                "training_execution_authorized": True,
                "production_model_replacement_authorized": False,
                "pixel_supervision_role": "auxiliary_only",
                "source_manifest_sha256": "a" * 64,
                "labeler": {"provider": "openai", "model": "gpt-5", "prompt_sha256": "b" * 64},
            }), encoding="utf-8")
            loaded = subject.load_attested_targets(targets_path=target_path, report_path=report_path, allowed_hazard_types=HAZARDS)
        self.assertEqual(2, len(loaded))


if __name__ == "__main__":
    unittest.main()
