#!/usr/bin/env python3
"""Tests for controlled train-only static counterfactual construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import build_public_silver_synthetic_static_counterfactuals as subject


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class SyntheticStaticCounterfactualBuilderTest(unittest.TestCase):
    def test_compose_obstacle_returns_exact_nonempty_mask_and_bbox(self) -> None:
        background = np.full((80, 120, 3), 90, dtype=np.uint8)
        obstacle = np.zeros((30, 50, 4), dtype=np.uint8)
        obstacle[5:28, 8:44, :3] = (20, 30, 220)
        obstacle[5:28, 8:44, 3] = 255
        composed, mask, bbox = subject.compose_obstacle(
            background,
            obstacle,
            width_fraction=0.4,
            center_x=0.5,
            bottom_y=0.9,
        )
        self.assertEqual(background.shape, composed.shape)
        self.assertEqual(background.shape[:2], mask.shape)
        self.assertGreater(int(np.count_nonzero(mask)), 0)
        x1, y1, x2, y2 = bbox
        self.assertTrue(np.all(mask[y1:y2, x1:x2] >= 0))
        self.assertEqual(0, int(np.count_nonzero(mask[:y1])))
        self.assertFalse(np.array_equal(background, composed))

    def test_rejects_independent_direction_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "independent model direction"):
            subject.reject_independent_direction(Path("secondary-corridor-causal/example"))

    def test_build_creates_valid_train_only_pair_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            package = parent / "base"
            images = root / "parent_images"
            package.mkdir(parents=True)
            images.mkdir()
            hashes: list[str] = []
            frames: list[dict[str, object]] = []
            for index, value in enumerate((50, 80, 110)):
                path = images / f"frame_{index}.png"
                cv2.imwrite(str(path), np.full((72, 128, 3), value, dtype=np.uint8))
                digest = subject.sha256_file(path)
                hashes.append(digest)
                frames.append({
                    "frame_index": index,
                    "file_name": path.name,
                    "sha256": digest,
                })
            source = {
                "format": "blindassist_public_rgb_timeline_source_manifest_v2",
                "source_id": "test_parent_source",
                "source": {"dataset": "test", "license": "CC0-1.0"},
                "frame_count": 3,
                "frames": frames,
                "privacy_audit_required": True,
                "human_event_truth_present": False,
                "provisional_training_authorized": True,
                "training_execution_authorized": True,
                "production_model_replacement_authorized": False,
                "promotion": {
                    "image_root": str(images.resolve()),
                    "mode": "provisional_model_supervision",
                },
            }
            source_path = package / "source_manifest_v2.json"
            write_json(source_path, source)
            silver = {
                "schema": "blindassist_public_video_silver_labels_v2",
                "source": {
                    "source_id": "test_parent_source",
                    "source_manifest_path": "source_manifest_v2.json",
                    "source_manifest_sha256": subject.sha256_file(source_path),
                    "human_event_truth_present": False,
                    "privacy_audit_required": True,
                },
                "labeler": {
                    "provider": "test",
                    "model": "test",
                    "prompt_id": "test",
                    "prompt_sha256": "test",
                    "review_mode": "multiframe_temporal",
                },
                "episodes": [{
                    "episode_id": "base-clear",
                    "evidence_frame_sha256": hashes,
                    "silver_should_alert": "candidate_no_alert",
                    "confidence": 0.75,
                    "risk_profile": {"lifecycle": "no_alert"},
                    "negative_decision_quality": {
                        "corridor_heading_stability": 0.8,
                        "near_field_visibility": 0.8,
                        "corridor_clearance": 0.8,
                        "near_field_lateral_intrusion_absent": 0.8,
                    },
                    "uncertainty_reasons": ["test only"],
                }],
                "training_execution_authorized": True,
                "calibration_authorized": False,
                "blind_evaluation_authorized": False,
                "production_model_replacement_authorized": False,
                "training_mode": "provisional_model_supervision",
            }
            write_json(package / "silver_labels_v2.json", silver)
            asset = np.zeros((40, 60, 4), dtype=np.uint8)
            asset[5:38, 5:55, :3] = (0, 0, 255)
            asset[5:38, 5:55, 3] = 255
            barricade = root / "barricade.png"
            sand = root / "sand.png"
            cv2.imwrite(str(barricade), asset)
            cv2.imwrite(str(sand), asset)
            output = root / "output"
            receipt = subject.build(
                parent_root=parent,
                barricade_asset=barricade,
                sand_pile_asset=sand,
                output_root=output,
                pair_specs=[{
                    "slug": "test_pair",
                    "base_episode_id": "base-clear",
                    "asset_name": "barricade",
                    "center_x": 0.5,
                    "bottom_y": 0.95,
                    "width_fractions": (0.15, 0.25, 0.4),
                }],
            )
            self.assertEqual(1, receipt["pair_count"])
            synthetic_package = output / "packages" / "synthetic-static-test_pair-20260717"
            synthetic_source = json.loads(
                (synthetic_package / "source_manifest_v2.json").read_text(encoding="utf-8")
            )
            self.assertTrue(synthetic_source["synthetic_counterfactual"]["train_only"])
            self.assertEqual("test_parent_source", synthetic_source["synthetic_counterfactual"]["parent_source_id"])
            alert_frames = [
                row for row in synthetic_source["frames"]
                if row.get("synthetic_variant") == "static_obstacle_composite"
            ]
            self.assertEqual(3, len(alert_frames))
            self.assertTrue(all((output / row["mask_path"]).is_file() for row in alert_frames))
            self.assertTrue((output / "generation_records.jsonl").is_file())
            self.assertNotIn(
                "secondary-corridor-causal",
                (output / "build_receipt.json").read_text(encoding="utf-8").lower(),
            )


if __name__ == "__main__":
    unittest.main()
