from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import build_sanpo_sequence_evalset as sanpo
import create_sanpo_v2_review_decisions as v2_review
import finalize_sanpo_sequence_evalset as finalize
import clone_sanpo_event_phase_evalset as event_clone


class SanpoSequenceEvalsetTest(unittest.TestCase):
    def test_resample_15_fps_to_10_fps_without_duplicates(self) -> None:
        actual = sanpo.resample_indices(range(15), source_fps=15.0, target_fps=10.0, start_frame=0, max_frames=10)
        self.assertEqual([0, 2, 3, 5, 6, 8, 9, 11, 12, 14], actual)

    def test_resample_rejects_upsampling(self) -> None:
        with self.assertRaises(ValueError):
            sanpo.resample_indices(range(10), source_fps=10.0, target_fps=15.0, start_frame=0, max_frames=3)

    def test_mask_regions_keep_source_classes_and_only_exact_coco_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mask.png"
            mask = np.zeros((6, 8, 3), dtype=np.uint8)
            mask[1:5, 1:3] = [12, 0, 1]  # pedestrian instance 1
            mask[2:6, 4:8] = [20, 0, 1]  # generic obstacle instance 1
            Image.fromarray(mask, mode="RGB").save(path)
            regions, objects = sanpo.parse_mask_regions(path, {12: "pedestrian", 20: "obstacle"})
        self.assertEqual({"pedestrian", "obstacle"}, {item["class"] for item in regions})
        self.assertEqual(["person"], [item["class"] for item in objects])
        self.assertEqual([1, 1, 3, 5], objects[0]["bbox_xyxy"])

    def test_validation_rejects_unreviewed_non_null_risk_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "images/test/sample.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (4, 4), "black").save(image)
            row = {
                "id": "sample",
                "image_path": "images/test/sample.png",
                "status": "pending_review",
                "sequence_id": "seq",
                "frame_index": 0,
                "expected_risk_direction": "CENTER",
                "expected_distance_band": None,
                "expected_should_alert": None,
                "expected_risk_level": None,
                "expected_approach_state": None,
                "expected_approach_alert": None,
                "expected_time_to_alert_frames": None,
                "source": {
                    "dataset": "SANPO-Real v0",
                    "license": sanpo.LICENSE_NAME,
                    "license_url": sanpo.LICENSE_URL,
                    "original_url_or_id": "https://example.test/frame.png",
                    "sha256": "placeholder",
                    "redistribution_policy": "local_only",
                },
            }
            result = sanpo.validate_rows([row], root)
        self.assertFalse(result["ok"])
        self.assertIn("unreviewed risk field is not null", json.dumps(result))

    def test_finalize_requires_explicit_manual_acceptance(self) -> None:
        row = {"id": "sample"}
        review = {
            "review_status": "pending_manual_risk_review",
            "expected_risk_direction": "CENTER",
            "expected_distance_band": "NEAR",
            "expected_should_alert": "true",
            "expected_risk_level": "HIGH",
            "expected_approach_state": "APPROACHING",
            "expected_approach_alert": "true",
            "expected_time_to_alert_frames": "3",
        }
        with self.assertRaisesRegex(ValueError, "accepted_manual_review"):
            finalize.finalize_row(row, review)

    def test_gcs_md5_verification_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.bin"
            path.write_bytes(b"sanpo")
            expected = sanpo.base64_encode(hashlib.md5(b"sanpo", usedforsecurity=False).digest())
            sanpo.verify_gcs_md5(path, {"md5Hash": expected})
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                sanpo.verify_gcs_md5(path, {"md5Hash": sanpo.base64_encode(b"wrong")})

    def test_finalize_accepts_complete_review_and_known_primary_region(self) -> None:
        row = {
            "id": "sample",
            "objects": [],
            "source_regions": [{"id": "sanpo_20_1", "class": "obstacle"}],
        }
        review = {
            "review_status": "accepted_manual_review",
            "primary_object_id": "",
            "source_primary_region_id": "sanpo_20_1",
            "expected_risk_direction": "CENTER",
            "expected_distance_band": "NEAR",
            "expected_should_alert": "true",
            "expected_risk_level": "HIGH",
            "expected_approach_state": "APPROACHING",
            "expected_approach_alert": "true",
            "expected_time_to_alert_frames": "3",
            "review_notes": "controlled review",
        }
        actual = finalize.finalize_row(row, review)
        self.assertEqual("accepted", actual["status"])
        self.assertEqual("sanpo_20_1", actual["source_primary_region_id"])
        self.assertEqual(3, actual["expected_time_to_alert_frames"])

    def test_finalize_preserves_scene_and_event_labels_for_benchmark_loader(self) -> None:
        row = {
            "id": "sample",
            "objects": [],
            "source_regions": [{"id": "sanpo_20_1", "class": "obstacle"}],
            "attributes": {"existing_annotation": "preserved"},
        }
        review = {
            "review_status": "accepted_manual_review",
            "source_primary_region_id": "sanpo_20_1",
            "expected_risk_direction": "CENTER",
            "expected_distance_band": "NEAR",
            "expected_should_alert": "true",
            "expected_risk_level": "HIGH",
            "expected_approach_state": "APPROACHING",
            "expected_approach_alert": "true",
            "expected_time_to_alert_frames": "0",
            "scene_bucket": "center_obstacle",
            "risk_event_id": "obstacle_event_0",
        }
        actual = finalize.finalize_row(row, review)
        self.assertEqual("center_obstacle", actual["scene_bucket"])
        self.assertEqual("obstacle_event_0", actual["risk_event_id"])
        self.assertEqual("center_obstacle", actual["attributes"]["scene_bucket"])
        self.assertEqual("obstacle_event_0", actual["attributes"]["risk_event_id"])
        self.assertEqual("preserved", actual["attributes"]["existing_annotation"])

    def test_v2_profiles_assign_parallel_curb_and_stable_event_id(self) -> None:
        rows = v2_review.build("parallel_curb_negative", 3)
        self.assertEqual({"parallel_curb"}, {row["scene_bucket"] for row in rows})
        self.assertEqual({"parallel_curb_event_0"}, {row["risk_event_id"] for row in rows})
        self.assertFalse(any(row["expected_should_alert"] for row in rows))

    def test_event_clone_recovers_all_three_reviewed_scene_buckets(self) -> None:
        self.assertEqual("parallel_curb", event_clone.scene_bucket_for_sequence([
            {"expected_should_alert": False, "source_primary_region_id": None},
        ]))
        self.assertEqual("front_stairs", event_clone.scene_bucket_for_sequence([
            {"expected_should_alert": True, "source_primary_region_id": "sanpo_15_1"},
        ]))
        self.assertEqual("center_obstacle", event_clone.scene_bucket_for_sequence([
            {"expected_should_alert": True, "source_primary_region_id": "sanpo_20_7"},
        ]))

    def test_finalize_ai_review_requires_explicit_opt_in(self) -> None:
        row = {"id": "sample", "objects": [], "source_regions": [{"id": "sanpo_20_1"}]}
        review = {
            "review_status": "accepted_ai_review",
            "reviewer_type": "ai_assistant",
            "reviewer_id": "test_consensus",
            "review_confidence": "0.80",
            "independent_review_count": "3",
            "source_primary_region_id": "sanpo_20_1",
            "expected_risk_direction": "CENTER",
            "expected_distance_band": "MID",
            "expected_should_alert": "false",
            "expected_risk_level": "LOW",
            "expected_approach_state": "APPROACHING",
            "expected_approach_alert": "false",
            "expected_time_to_alert_frames": "",
        }
        with self.assertRaisesRegex(ValueError, "explicit --allow-ai-review"):
            finalize.finalize_row(row, review)

        actual = finalize.finalize_row(row, review, allow_ai_review=True)
        self.assertEqual("ai_assistant", actual["review_provenance"]["reviewer_type"])
        self.assertEqual(3, actual["review_provenance"]["independent_review_count"])
        self.assertEqual("multi_agent_consensus_v1", actual["review_provenance"]["policy"])

    def test_finalize_ai_review_rejects_weak_provenance(self) -> None:
        row = {"id": "sample", "objects": [], "source_regions": []}
        review = {
            "review_status": "accepted_ai_review",
            "reviewer_type": "ai_assistant",
            "reviewer_id": "single_pass",
            "review_confidence": "0.60",
            "independent_review_count": "1",
        }
        with self.assertRaisesRegex(ValueError, "confidence"):
            finalize.finalize_row(row, review, allow_ai_review=True)

    def test_finalize_rejects_source_region_as_detection_primary(self) -> None:
        row = {"id": "sample", "objects": [], "source_regions": [{"id": "sanpo_20_1"}]}
        review = {
            "review_status": "accepted_manual_review",
            "primary_object_id": "sanpo_20_1",
            "expected_risk_direction": "CENTER",
            "expected_distance_band": "NEAR",
            "expected_should_alert": "true",
            "expected_risk_level": "HIGH",
            "expected_approach_state": "APPROACHING",
            "expected_approach_alert": "true",
            "expected_time_to_alert_frames": "3",
        }
        with self.assertRaisesRegex(ValueError, "detection GT objects"):
            finalize.finalize_row(row, review)

    def test_finalize_rejects_blocking_issue_tags(self) -> None:
        row = {"id": "sample", "objects": [], "source_regions": []}
        review = {"review_status": "accepted_manual_review", "issue_tags": "unsafe_or_sensitive"}
        with self.assertRaisesRegex(ValueError, "blocking issue_tags"):
            finalize.finalize_row(row, review)

    def test_full_final_validation_rechecks_image_and_mask_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "images/test/sample.png"
            mask = root / "source_masks/test/sample.png"
            image.parent.mkdir(parents=True)
            mask.parent.mkdir(parents=True)
            Image.new("RGB", (4, 4), "black").save(image)
            Image.new("RGB", (4, 4), "black").save(mask)
            row = {
                "id": "sample",
                "image_path": "images/test/sample.png",
                "width": 4,
                "height": 4,
                "objects": [],
                "source_regions": [],
                "sequence_id": "seq",
                "frame_index": 0,
                "status": "accepted",
                "review_status": "accepted_manual_review",
                "primary_object_id": None,
                "source_primary_region_id": None,
                "source": {
                    "sha256": finalize.sha256_file(image),
                    "mask_sha256": finalize.sha256_file(mask),
                    "official_split": "test",
                },
            }
            self.assertEqual([], finalize.validate_final([row], root, {"person"}))
            Image.new("RGB", (4, 4), "white").save(image)
            errors = finalize.validate_final([row], root, {"person"})
        self.assertTrue(any("SHA256 differs" in error for error in errors))

    def test_full_final_validation_rejects_machine_annotated_detection_gt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "images/test/sample.png"
            mask = root / "source_masks/test/sample.png"
            image.parent.mkdir(parents=True)
            mask.parent.mkdir(parents=True)
            Image.new("RGB", (4, 4), "black").save(image)
            Image.new("RGB", (4, 4), "black").save(mask)
            row = {
                "id": "sample",
                "image_path": "images/test/sample.png",
                "width": 4,
                "height": 4,
                "objects": [{"id": "person_1", "class": "person", "bbox_xyxy": [0, 0, 2, 2]}],
                "source_regions": [],
                "source_annotation_quality": "MACHINE_ANNOTATED",
                "sequence_id": "seq",
                "frame_index": 0,
                "status": "accepted",
                "review_status": "accepted_manual_review",
                "primary_object_id": "person_1",
                "source_primary_region_id": None,
                "source": {
                    "sha256": finalize.sha256_file(image),
                    "mask_sha256": finalize.sha256_file(mask),
                    "official_split": "test",
                },
            }
            errors = finalize.validate_final([row], root, {"person"})
        self.assertTrue(any("require HUMAN_ANNOTATED" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
