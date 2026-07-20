#!/usr/bin/env python3
"""Focused deterministic checks for the train-only distance ablation contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import run_sanpo_balanced_distance_ablation as subject
import train_export_sanpo_segmentation as shared


def record(session: str) -> shared.Record:
    return shared.Record(
        sample_id=session,
        split="train",
        session_id=session,
        image_path=Path("unused.png"),
        masks={},
        semantic_mask_path=Path("unused-mask.png"),
        scene_bucket="test",
        label_authority="human_reviewed",
    )


class BalancedDistanceAblationTest(unittest.TestCase):
    def test_train_only_loader_never_resolves_dev_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(root / "train.png")
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(root / "train-mask.png")
            rows = [
                {
                    "id": "train", "split": "train", "session_id": "train-session",
                    "image_path": "train.png", "semantic_mask_path": "train-mask.png",
                    "label_authority": "source_ground_truth",
                },
                {
                    "id": "dev", "split": "dev", "session_id": "dev-session",
                    "image_path": "missing-dev.png", "semantic_mask_path": "missing-dev-mask.png",
                    "label_authority": "source_ground_truth",
                },
            ]
            (root / "training_manifest.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
            )
            records = subject.load_canonical_train_records_only(root)
        self.assertEqual([record.sample_id for record in records], ["train"])

    def test_partition_requires_matched_nonzero_boundary_coverage(self) -> None:
        masks = [
            np.array([[1, 0], [0, 0]], dtype=np.uint8),
            np.array([[1, 0], [0, 0]], dtype=np.uint8),
            np.array([[1, 0], [0, 0]], dtype=np.uint8),
            np.array([[1, 0], [0, 0]], dtype=np.uint8),
        ]
        partition = subject.partition_train_sessions(
            [record("a"), record("b"), record("c"), record("d")], masks, ["a", "b"],
            minimum_ratio=0.95, maximum_ratio=1.05,
        )
        self.assertEqual(partition.evaluation_indices, (0, 1))
        self.assertEqual(partition.train_indices, (2, 3))
        self.assertEqual(partition.evaluation_boundary_fraction, 0.25)

    def test_partition_rejects_unknown_or_mismatched_sessions(self) -> None:
        masks = [
            np.array([[1, 0]], dtype=np.uint8),
            np.array([[0, 0]], dtype=np.uint8),
            np.array([[1, 0]], dtype=np.uint8),
            np.array([[1, 0]], dtype=np.uint8),
        ]
        rows = [record("a"), record("b"), record("c"), record("d")]
        with self.assertRaisesRegex(ValueError, "not all canonical train"):
            subject.partition_train_sessions(rows, masks, ["a", "missing"], minimum_ratio=0.8, maximum_ratio=1.25)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            subject.partition_train_sessions(rows, masks, ["a", "c"], minimum_ratio=0.95, maximum_ratio=1.05)

    def test_distance_targets_are_signed_and_keep_deterministic_weight(self) -> None:
        masks = np.array([[[0, 1, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
        values = subject.distance_targets(masks, truncate=4.0, signed=True)
        self.assertEqual(values.shape, (1, 3, 3, 2))
        self.assertLess(values[0, 0, 1, 0], 0.0)
        self.assertGreater(values[0, 2, 2, 0], 0.0)
        self.assertGreater(values[0, 0, 1, 1], values[0, 2, 2, 1])

    def test_boundary_probability_targets_are_binary_and_full_resolution(self) -> None:
        masks = np.array([[[0, 1, 2], [1, 0, 0]]], dtype=np.uint8)
        values = subject.boundary_probability_targets(masks)
        self.assertEqual(values.shape, (1, 2, 3, 1))
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_array_equal(values[..., 0], [[ [0.0, 1.0, 0.0], [1.0, 0.0, 0.0] ]])

    def test_parser_refuses_single_evaluation_session(self) -> None:
        with self.assertRaises(SystemExit):
            subject.parse_args(["--evaluation-session", "a", "--output", "report.json"])


if __name__ == "__main__":
    unittest.main()
