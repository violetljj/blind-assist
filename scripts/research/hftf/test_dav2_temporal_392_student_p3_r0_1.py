from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dav2_temporal_392_student_p3_r0_1 import (
    DecoupledTemporalStateHead,
    STATE_TO_INDEX,
    TRANSITIONS,
    activation_transition_weights,
    build_temporal_evidence,
    effective_number_transition_weights,
    temporal_distillation_loss,
    validate_clip_manifest,
    validate_sealed_coverage_receipt,
)


SHA = "A" * 64


def _evidence(disagreement: float = 0.0, teacher_valid: bool = True, tof_valid: bool = True):
    timestamps = torch.tensor([[1_000_000_000 + index * 100_000_000 for index in range(4)]])
    return build_temporal_evidence(
        timestamps,
        timestamps,
        torch.full((1, 4), teacher_valid),
        torch.full((1, 4), tof_valid),
        torch.full((1, 4), disagreement),
    )


def _frame(role: str, parent: str, index: int) -> dict:
    common = {
        "frame_id": f"{role}-{index}",
        "video_id": f"video-{parent}",
        "parent_id": parent,
        "timestamp_ns": 1_000_000_000 + index * 100_000_000,
        "rgb_identity": f"rgb-{role}-{index}",
        "rgb_sha256": SHA,
    }
    if role == "sealed_holdout":
        return {**common, "sealed_target_id": f"sealed-{index}"}
    return {
        **common,
        "teacher_depth_ref": f"teacher-{role}-{index}",
        "teacher_depth_sha256": SHA,
        "teacher_timestamp_ns": common["timestamp_ns"],
        "teacher_valid": True,
        "tof_valid": True,
        "frozen_a2_mean_abs_log_depth_disagreement": 0.1,
        "clearance_m": [1.0, 1.5, 2.0],
        "geometry_state": ["CLEAR", "OCCUPIED", "UNKNOWN_GROUND"],
        "geometry_target_valid": [True, True, True],
    }


def _manifest() -> dict:
    clips = []
    for role, parent in (
        ("train", "train-parent"),
        ("validation", "validation-parent"),
        ("sealed_holdout", "holdout-parent"),
    ):
        clips.append(
            {
                "clip_id": f"clip-{role}",
                "role": role,
                "video_id": f"video-{parent}",
                "parent_id": parent,
                "frames": [_frame(role, parent, index) for index in range(4)],
            }
        )
    return {
        "schema": "blindassist_dav2_temporal_392_student_p3_r0_1_clip_manifest",
        "protocol_sha256": SHA,
        "clip_length": 4,
        "holdout": {
            "status": "SEALED_NOT_OPENED",
            "outcomes_opened": False,
        },
        "clips": clips,
    }


class EvidenceAndLossTest(unittest.TestCase):
    def test_transition_weights_require_all_positive_train_classes(self) -> None:
        weights = effective_number_transition_weights(
            {name: index + 1 for index, name in enumerate(TRANSITIONS)}
        )
        self.assertEqual(weights.shape, (9,))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        with self.assertRaisesRegex(ValueError, "positive train support"):
            effective_number_transition_weights({name: 0 for name in TRANSITIONS})

    def test_evidence_is_single_detached_source(self) -> None:
        disagreement = torch.full((1, 4), 0.3, requires_grad=True)
        timestamps = torch.tensor([[1_000_000_000 + index for index in range(4)]])
        evidence = build_temporal_evidence(
            timestamps,
            timestamps,
            torch.ones((1, 4), dtype=torch.bool),
            torch.ones((1, 4), dtype=torch.bool),
            disagreement,
        )
        self.assertFalse(evidence.head_features.requires_grad)
        self.assertTrue(torch.all(evidence.external_abstain_target))
        self.assertTrue(torch.all(evidence.clearance_frame_usable))

    def test_external_evidence_cannot_change_geometry_logits(self) -> None:
        torch.manual_seed(5)
        head = DecoupledTemporalStateHead(hidden_width=8)
        depth = torch.full((1, 4, 8, 8), 2.0)
        delta = torch.full((1, 3), 0.1)
        low = head(depth, _evidence(0.0), delta)
        high = head(depth, _evidence(0.4), delta)
        self.assertTrue(
            torch.equal(
                low["geometry_transition_logits"],
                high["geometry_transition_logits"],
            )
        )
        self.assertFalse(
            torch.equal(
                low["external_abstention_logits"],
                high["external_abstention_logits"],
            )
        )

    def test_invalid_teacher_is_excluded_from_all_depth_components(self) -> None:
        evidence = _evidence(teacher_valid=False)
        student = torch.full((1, 4, 8, 8), 9.0, requires_grad=True)
        teacher = torch.ones((1, 4, 8, 8))
        head = DecoupledTemporalStateHead(hidden_width=8)
        output = head(student, evidence, torch.full((1, 3), 0.1))
        total, parts = temporal_distillation_loss(
            student,
            teacher,
            torch.ones_like(teacher, dtype=torch.bool),
            evidence,
            output,
            torch.ones((1, 4, 3)),
            torch.full((1, 4, 3), STATE_TO_INDEX["CLEAR"]),
            torch.ones((1, 4, 3), dtype=torch.bool),
            torch.ones(9),
        )
        self.assertEqual(int(parts["valid_depth_pixels"]), 0)
        self.assertEqual(float(parts["log_depth"].detach()), 0.0)
        self.assertEqual(float(parts["gradient"].detach()), 0.0)
        self.assertEqual(float(parts["scale"].detach()), 0.0)
        total.backward()

    def test_frozen_disagreement_does_not_mask_clearance_delta(self) -> None:
        evidence = _evidence(disagreement=0.4)
        student = torch.ones((1, 4, 8, 8), requires_grad=True)
        teacher = torch.ones((1, 4, 8, 8))
        head = DecoupledTemporalStateHead(hidden_width=8)
        output = head(student, evidence, torch.full((1, 3), 0.1))
        _total, parts = temporal_distillation_loss(
            student,
            teacher,
            torch.ones_like(teacher, dtype=torch.bool),
            evidence,
            output,
            torch.tensor([[[1.0] * 3, [0.8] * 3, [0.6] * 3, [0.4] * 3]]),
            torch.full((1, 4, 3), STATE_TO_INDEX["OCCUPIED"]),
            torch.ones((1, 4, 3), dtype=torch.bool),
            torch.ones(9),
        )
        self.assertEqual(int(parts["clearance_delta_pairs"]), 9)
        self.assertGreater(float(parts["clearance_delta"].detach()), 0.0)


class ManifestAndCoverageTest(unittest.TestCase):
    def test_strict_manifest_reports_training_transition_distribution(self) -> None:
        summary = validate_clip_manifest(_manifest(), consumed_parent_ids={"old-p1"})
        self.assertEqual(summary.clips_by_role["sealed_holdout"], 1)
        self.assertEqual(summary.transitions_by_role["train"]["CLEAR_TO_CLEAR"], 3)
        self.assertEqual(summary.transitions_by_role["train"]["OCCUPIED_TO_OCCUPIED"], 3)
        with self.assertRaisesRegex(ValueError, "lacks transition support"):
            activation_transition_weights(summary)

    def test_holdout_renamed_label_field_is_rejected(self) -> None:
        manifest = _manifest()
        manifest["clips"][2]["frames"][0]["target_state"] = "CLEAR"
        with self.assertRaisesRegex(ValueError, "fields must be exact"):
            validate_clip_manifest(manifest, consumed_parent_ids=set())

    def test_parent_leakage_and_consumed_parent_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "consumed parent reused"):
            validate_clip_manifest(_manifest(), consumed_parent_ids={"holdout-parent"})
        manifest = _manifest()
        manifest["clips"][1]["parent_id"] = "train-parent"
        for frame in manifest["clips"][1]["frames"]:
            frame["parent_id"] = "train-parent"
        with self.assertRaisesRegex(ValueError, "parent leakage"):
            validate_clip_manifest(manifest, consumed_parent_ids=set())

    def test_sealed_coverage_receipt_requires_all_counts(self) -> None:
        receipt = {
            "schema": "blindassist_dav2_temporal_392_student_p3_r0_1_sealed_coverage_receipt",
            "status": "SEALED_COVERAGE_VERIFIED",
            "protocol_sha256": SHA,
            "identity_manifest_sha256": SHA,
            "sealed_bundle_sha256": SHA,
            "coverage_producer_sha256": SHA,
            "created_before_training_activation": True,
            "label_rows_disclosed": False,
            "evaluable_clip_count": 32,
            "video_parent_count": 8,
            "key_transition_counts": {
                "CLEAR_TO_OCCUPIED": 8,
                "OCCUPIED_TO_CLEAR": 8,
                "KNOWN_TO_UNKNOWN_GROUND": 8,
                "UNKNOWN_GROUND_TO_KNOWN": 8,
            },
            "geometry_transition_counts": {name: 1 for name in TRANSITIONS},
        }
        validate_sealed_coverage_receipt(
            receipt,
            expected_identity_manifest_sha256=SHA,
            expected_protocol_sha256=SHA,
        )
        receipt["key_transition_counts"]["CLEAR_TO_OCCUPIED"] = 7
        with self.assertRaisesRegex(ValueError, "insufficient key transitions"):
            validate_sealed_coverage_receipt(
                receipt,
                expected_identity_manifest_sha256=SHA,
                expected_protocol_sha256=SHA,
            )


if __name__ == "__main__":
    unittest.main()
