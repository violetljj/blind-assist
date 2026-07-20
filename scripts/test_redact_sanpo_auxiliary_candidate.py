from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redact_sanpo_auxiliary_candidate as redactor


class RedactSanpoAuxiliaryCandidateTests(unittest.TestCase):
    def _root_with_row(self, *, risk: object = None, split: str = "train") -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        image = root / "images" / "test" / "x.png"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (4, 3), color=(1, 2, 3)).save(image)
        row = {
            "id": "x", "image_path": "images/test/x.png", "status": "pending_review",
            "expected_should_alert": risk, "expected_risk_level": None, "expected_approach_state": None,
            "expected_risk_direction": None, "expected_distance_band": None, "expected_approach_alert": None,
            "source": {"official_split": split},
        }
        (root / "manifest.draft.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return root

    def test_accepts_pending_non_risk_official_train_row(self) -> None:
        rows = redactor.load_draft_rows(self._root_with_row())
        self.assertEqual(1, len(rows))
        self.assertTrue(rows[0]["image_path"].is_file())

    def test_rejects_risk_field_and_non_train_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-null risk"):
            redactor.load_draft_rows(self._root_with_row(risk=False))
        with self.assertRaisesRegex(ValueError, "official train"):
            redactor.load_draft_rows(self._root_with_row(split="test"))

    def test_rejects_blind_path_case_insensitively(self) -> None:
        root = self._root_with_row()
        row = json.loads((root / "manifest.draft.jsonl").read_text(encoding="utf-8"))
        row["image_path"] = "Blind_Holdout/x.png"
        (root / "manifest.draft.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe image path"):
            redactor.load_draft_rows(root)


if __name__ == "__main__":
    unittest.main()
