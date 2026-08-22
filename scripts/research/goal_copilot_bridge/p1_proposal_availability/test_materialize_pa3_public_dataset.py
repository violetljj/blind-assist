from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.research.goal_copilot_bridge.p1_proposal_availability.materialize_pa3_public_dataset import (
    CAPTURE_SCHEMA,
    ROSTER_SCHEMA,
    _sealed_payload,
    download_doordetect_private_truth,
)


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


if __name__ == "__main__":
    unittest.main()
