from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dav2_temporal_392_student_p3_r0 import (
    LightweightTemporalStateHead,
    STATE_TO_INDEX,
    temporal_distillation_loss,
    validate_clip_manifest,
)


def frame(frame_id: str, timestamp_ns: int, role: str, parent: str) -> dict:
    base = {
        "frame_id": frame_id,
        "video_id": f"video-{parent}",
        "parent_id": parent,
        "timestamp_ns": timestamp_ns,
    }
    if role == "sealed_holdout":
        return {**base, "sealed_target_id": f"target-{frame_id}"}
    return {
        **base,
        "teacher_depth_ref": f"teacher-{frame_id}",
        "clearance_m": [1.0, 1.5, 2.0],
        "state": ["OCCUPIED", "CLEAR", "CLEAR"],
        "teacher_timestamp_ns": timestamp_ns,
        "teacher_valid": True,
        "tof_valid": True,
    }


def manifest() -> dict:
    clips = []
    for role, parent in zip(
        ("train", "validation", "sealed_holdout"), ("p-train", "p-val", "p-hold")
    ):
        frames = [frame(f"{role}-{index}", 1_000_000_000 + index * 100_000_000, role, parent) for index in range(4)]
        clips.append(
            {
                "clip_id": f"clip-{role}",
                "role": role,
                "video_id": f"video-{parent}",
                "parent_id": parent,
                "frames": frames,
            }
        )
    return {
        "schema": "blindassist_dav2_temporal_392_student_p3_r0_clip_manifest",
        "clip_length": 4,
        "holdout": {"status": "SEALED_NOT_OPENED", "outcomes_opened": False},
        "clips": clips,
    }


class ClipManifestTest(unittest.TestCase):
    def test_valid_manifest_is_video_parent_disjoint(self) -> None:
        summary = validate_clip_manifest(manifest(), consumed_parent_ids={"old-p1"})
        self.assertEqual(summary.clips_by_role["train"], 1)
        self.assertEqual(summary.transitions_by_role["sealed_holdout"], 3)

    def test_parent_leakage_fails_closed(self) -> None:
        value = manifest()
        value["clips"][1]["parent_id"] = "p-train"
        for item in value["clips"][1]["frames"]:
            item["parent_id"] = "p-train"
        with self.assertRaisesRegex(ValueError, "video-parent leakage"):
            validate_clip_manifest(value)

    def test_holdout_label_leak_fails_closed(self) -> None:
        value = manifest()
        value["clips"][2]["frames"][0]["state"] = ["CLEAR"] * 3
        with self.assertRaisesRegex(ValueError, "holdout label leaked"):
            validate_clip_manifest(value)

    def test_stale_teacher_requires_unknown(self) -> None:
        value = manifest()
        item = value["clips"][0]["frames"][1]
        item["teacher_timestamp_ns"] = item["timestamp_ns"] - 600_000_000
        with self.assertRaisesRegex(ValueError, "must target UNKNOWN_GROUND"):
            validate_clip_manifest(value)

    def test_consumed_parent_cannot_be_reused(self) -> None:
        with self.assertRaisesRegex(ValueError, "consumed parent reused"):
            validate_clip_manifest(manifest(), consumed_parent_ids={"p-hold"})


class TemporalLossTest(unittest.TestCase):
    def test_head_shapes_and_four_supervision_families(self) -> None:
        torch.manual_seed(7)
        student = torch.full((2, 4, 8, 8), 2.0, requires_grad=True)
        teacher = torch.full((2, 4, 8, 8), 2.0)
        evidence = torch.ones((2, 4, 4))
        evidence[..., 0] = 0.0
        delta_s = torch.full((2, 3), 0.1)
        head = LightweightTemporalStateHead(hidden_width=16)
        output = head(student, evidence, delta_s)
        self.assertEqual(output["clearance_delta_m"].shape, (2, 3, 3))
        self.assertEqual(output["transition_logits"].shape, (2, 3, 3, 9))
        self.assertEqual(output["unknown_logits"].shape, (2, 4, 3))

        clearance = torch.tensor([1.0, 0.9, 0.8, 0.7]).view(1, 4, 1).expand(2, 4, 3)
        states = torch.full((2, 4, 3), STATE_TO_INDEX["OCCUPIED"])
        total, parts = temporal_distillation_loss(
            student,
            teacher,
            output,
            clearance,
            states,
            teacher_age_s=torch.zeros((2, 4)),
            tof_valid=torch.ones((2, 4), dtype=torch.bool),
            teacher_valid=torch.ones((2, 4), dtype=torch.bool),
        )
        self.assertEqual(
            set(parts),
            {"log_depth", "gradient", "scale", "clearance_delta", "state_transition", "unknown", "total"},
        )
        total.backward()
        self.assertIsNotNone(student.grad)

    def test_disagreement_forces_unknown_target(self) -> None:
        student = torch.full((1, 4, 8, 8), 4.0)
        teacher = torch.ones((1, 4, 8, 8))
        evidence = torch.tensor([[[0.0, 1.0, 1.0, 0.0]] * 4])
        output = LightweightTemporalStateHead(hidden_width=8)(
            student, evidence, torch.full((1, 3), 0.1)
        )
        clearance = torch.ones((1, 4, 3))
        state = torch.zeros((1, 4, 3), dtype=torch.long)
        _total, parts = temporal_distillation_loss(
            student,
            teacher,
            output,
            clearance,
            state,
            teacher_age_s=torch.zeros((1, 4)),
            tof_valid=torch.ones((1, 4), dtype=torch.bool),
            teacher_valid=torch.ones((1, 4), dtype=torch.bool),
        )
        self.assertGreater(float(parts["unknown"].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
