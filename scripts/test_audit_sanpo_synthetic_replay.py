from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import audit_sanpo_synthetic_replay as audit


class SanpoSyntheticReplayAuditTest(unittest.TestCase):
    def write_row(self, root: Path, index: int, native_id: int) -> dict:
        image = root / f"image-{index}.png"
        mask = root / f"mask-{index}.png"
        depth = root / f"depth-{index}.gz"
        Image.new("RGB", (2, 2), (1, 2, 3)).save(image)
        Image.new("RGB", (2, 2), (native_id, 0, 0)).save(mask)
        depth.write_bytes(f"depth-{index}".encode())
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "id": f"row-{index}", "image_path": image.name, "image_sha256": digest(image),
            "source_mask_path": mask.name, "source_mask_sha256": digest(mask),
            "source_depth_path": depth.name, "source_depth_sha256": digest(depth),
            "source_frame_index": index * 2,
            "source": {"source_id": "sanpo_synthetic_v0", "official_split": "train"},
            "authorization": {"real_finetune_or_eval": False, "human_event_truth": False, "calibration": False, "blind_evaluation": False, "android_runtime": False, "production_model_replacement": False},
        }

    def test_accepts_hash_bound_window_with_exact_four_class_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [self.write_row(root, index, native_id) for index, native_id in enumerate((1, 2, 4, 0))]
            (root / "manifest.replay.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            report = audit.audit(root)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual([0, 1, 2, 3], report["mapped_four_class_ids"])
            self.assertFalse(report["production_authorized"])

    def test_rejects_tampered_depth_and_unmapped_native_class(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = self.write_row(root, 0, 31)
            (root / "depth-0.gz").write_bytes(b"tampered")
            (root / "manifest.replay.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = audit.audit(root)
            self.assertFalse(report["ok"])
            self.assertTrue(any("source_depth_sha256 mismatch" in error for error in report["errors"]))
            self.assertTrue(any("unmapped SANPO class IDs" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
