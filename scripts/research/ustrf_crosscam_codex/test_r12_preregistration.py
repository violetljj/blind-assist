from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validate_r12_preregistration import validate


class R12PreregistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[3]
        cls.prereg_path = cls.repo_root / "configs/ustrf_crosscam_geometry_multisource_r12_heldout_v1.json"

    def _mutated_prereg(self, transform) -> Path:
        prereg = json.loads(self.prereg_path.read_text(encoding="utf-8"))
        transform(prereg)
        root = Path(tempfile.mkdtemp())
        path = root / "prereg.json"
        path.write_text(json.dumps(prereg), encoding="utf-8")
        return path

    def test_live_preregistration_is_valid(self) -> None:
        result = validate(self.repo_root, self.prereg_path)
        self.assertEqual(6, result["source_count"])
        self.assertEqual({"inside": 3, "outside": 3}, result["relation_counts"])

    def test_rejects_inference_before_freeze(self) -> None:
        path = self._mutated_prereg(
            lambda prereg: prereg["chronology"].update(
                {"new_source_detector_inference_completed_before_freeze": True}
            )
        )
        with self.assertRaisesRegex(ValueError, "chronology"):
            validate(self.repo_root, path)

    def test_rejects_unfrozen_label(self) -> None:
        path = self._mutated_prereg(
            lambda prereg: prereg["held_out_events"][0].update(
                {"detector_label_allowlist": ["construction barrel"]}
            )
        )
        with self.assertRaisesRegex(ValueError, "unfrozen detector label"):
            validate(self.repo_root, path)

    def test_rejects_missing_source_even_when_other_fields_are_valid(self) -> None:
        path = self._mutated_prereg(
            lambda prereg: prereg["held_out_events"][0].update(
                {"local_video_path": "artifacts.local/does-not-exist.mp4"}
            )
        )
        with self.assertRaisesRegex(ValueError, "missing or hash mismatch"):
            validate(self.repo_root, path)

    def test_rejects_seen_r11_source_identity(self) -> None:
        path = self._mutated_prereg(
            lambda prereg: prereg["held_out_events"][0].update(
                {"source_id": "wikimedia_commons_japan_rural_riverside_walk_2025"}
            )
        )
        with self.assertRaisesRegex(ValueError, "R1.1 diagnostic source reused"):
            validate(self.repo_root, path)


if __name__ == "__main__":
    unittest.main()
