from __future__ import annotations

import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from PIL import Image

from scripts.research.goal_copilot_bridge.p1_proposal_availability.materialize_pa3_public_dataset import (
    CAPTURE_SCHEMA,
    FACADEELEMENTS_MD5,
    ROSTER_SCHEMA,
    _sealed_payload,
    download_doordetect_private_truth,
    extract_facadeelements_private_truth,
    extract_facadeelements_public,
    freeze_facadeelements_roster,
)
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


class DoorDetectTruthMaterializationTest(unittest.TestCase):
    def test_class_zero_boxes_are_private_and_other_classes_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source" / "public_images" / "sample.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (200, 100), "white").save(image)
            label = root / "source" / "private_labels" / "sample.txt"
            label.parent.mkdir(parents=True)
            label.write_text("0 0.5 0.5 0.4 0.6\n2 0.2 0.2 0.1 0.1\n", encoding="utf-8")
            roster = _sealed_payload({
                "schema_version": ROSTER_SCHEMA,
                "source_kind": "DOORDETECT_GITHUB_TREE",
                "cases": [{
                    "case_id": "case-1",
                    "source_stem": "sample",
                    "label_path": "labels/sample.txt",
                }],
            }, "roster_body_sha256")
            capture = _sealed_payload({
                "schema_version": CAPTURE_SCHEMA,
                "private_truth_access": False,
                "cases": [{"case_id": "case-1", "image_path": str(image)}],
            }, "capture_manifest_body_sha256")
            roster_path = root / "roster.json"
            capture_path = root / "capture.json"
            truth_path = root / "truth.json"
            roster_path.write_text(json.dumps(roster), encoding="utf-8")
            capture_path.write_text(json.dumps(capture), encoding="utf-8")

            truth = download_doordetect_private_truth(roster_path, capture_path, root / "source", truth_path)

            self.assertEqual(truth["cases"][0]["target_visibility"], "VISIBLE")
            self.assertEqual(truth["cases"][0]["legal_target_bboxes_xyxy"], [[60.0, 20.0, 140.0, 80.0]])

    def test_facadeelements_freezes_schema_before_pixels_then_materializes_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "facades.zip"
            image_buffer = io.BytesIO()
            Image.new("RGB", (100, 80), "white").save(image_buffer, format="JPEG")
            image_buffer.seek(0)
            image_bytes = image_buffer.read()
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("data.yaml", "nc: 3\nnames: ['window', 'metalDoor', 'woodDoor']\n")
                for stem in ("a", "b"):
                    archive.writestr(f"train/images/{stem}.jpg", image_bytes)
                archive.writestr("train/labels/a.txt", "1 0.5 0.5 0.4 0.5\n")
                archive.writestr("train/labels/b.txt", "0 0.5 0.5 0.4 0.5\n")
            c0 = {
                "receipt_body_sha256": "a" * 64,
                "episodes": [
                    {
                        "episode_id": f"test-leftmost-{index:03d}",
                        "goal_provenance": {"goal_recorded_at_utc": "2026-01-01T00:00:00Z"},
                    }
                    for index in (1, 2)
                ],
            }
            source_lock = {
                "created_before_archive_member_access": True,
                "created_before_project_pixel_access": True,
                "created_before_private_label_access": True,
                "goal_receipt_body_sha256": "a" * 64,
                "roster_rule": {
                    "take": 2,
                    "case_id_prefix": "test-leftmost-case",
                    "episode_id_prefix": "test-leftmost",
                },
                "public_class_schema_rule": {"legal_class_names_normalized": ["metaldoor", "wooddoor"]},
            }
            c0_path = root / "c0.json"
            lock_path = root / "lock.json"
            metadata_path = root / "metadata.json"
            roster_path = root / "roster.json"
            capture_path = root / "capture.json"
            truth_path = root / "truth.json"
            c0_path.write_text(json.dumps(c0), encoding="utf-8")
            lock_path.write_text(json.dumps(source_lock), encoding="utf-8")
            metadata_path.write_text(json.dumps({
                "created_before_archive_download": True,
                "source_lock_sha256": sha256(lock_path),
                "size_bytes": archive_path.stat().st_size,
            }), encoding="utf-8")
            with mock.patch(
                "scripts.research.goal_copilot_bridge.p1_proposal_availability.materialize_pa3_public_dataset._md5",
                return_value=FACADEELEMENTS_MD5,
            ):
                roster = freeze_facadeelements_roster(lock_path, metadata_path, c0_path, archive_path, roster_path)
            self.assertEqual(roster["legal_class_ids"], [1, 2])
            self.assertEqual(roster["cases"][0]["case_id"], "test-leftmost-case-001")
            self.assertEqual(roster["cases"][0]["episode_id"], "test-leftmost-001")
            capture = extract_facadeelements_public(roster_path, c0_path, archive_path, root / "source", capture_path)
            self.assertFalse(capture["private_truth_access"])
            truth = extract_facadeelements_private_truth(
                roster_path, capture_path, archive_path, root / "source", truth_path
            )
            self.assertEqual(
                sorted(case["target_visibility"] for case in truth["cases"]),
                ["NOT_VISIBLE", "VISIBLE"],
            )


if __name__ == "__main__":
    unittest.main()
