from __future__ import annotations

import base64
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_stage_c_f0_1_sanpo_acquisition import (
    _md5_base64,
    _receipt_matches_file,
    _root_name,
    _safe_file,
    _validate_spec,
)


class StageCF01SanpoAcquisitionAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "role": "heldout",
            "official_split": "test",
            "session_id": "a" * 64,
            "source_fps": 20.0,
            "target_fps": 10.0,
            "selected_source_frames": list(range(0, 50, 2)),
            "description_object": {"name": "description"},
            "camera_poses_object": {"name": "poses"},
        }

    def test_root_name_binds_role_split_and_source(self) -> None:
        self.assertEqual(
            "hftf-f0-1-heldout-test-aaaaaaaa-25frames-20260801",
            _root_name(self.source),
        )

    def test_spec_validation_binds_identity_sampling_and_inventory(self) -> None:
        spec = {
            "source": {"official_split": "test", "session_id": "a" * 64},
            "sampling": {
                "source_fps": 20.0,
                "target_fps": 10.0,
                "selected_source_frames": list(range(0, 50, 2)),
            },
            "source_inventory": {
                "description": {"name": "description"},
                "camera_poses": {"name": "poses"},
                "official_split_receipt": {"generation": "1"},
                "rgb": [{}] * 25,
                "masks": [{}] * 25,
                "depth": [{}] * 25,
            },
        }
        self.assertEqual([], _validate_spec(spec, self.source))
        spec["source"]["official_split"] = "train"
        self.assertIn(
            "dataset_spec_source_identity_mismatch",
            _validate_spec(spec, self.source),
        )

    def test_receipt_matches_exact_size_and_md5(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload"
            path.write_bytes(b"source")
            receipt = {
                "size": 6,
                "md5_base64": base64.b64encode(
                    hashlib.md5(b"source", usedforsecurity=False).digest()
                ).decode("ascii"),
            }
            self.assertEqual(receipt["md5_base64"], _md5_base64(path))
            self.assertTrue(_receipt_matches_file(receipt, path))
            receipt["size"] = 7
            self.assertFalse(_receipt_matches_file(receipt, path))

    def test_safe_file_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            (root / "inside").write_text("ok", encoding="utf-8")
            self.assertEqual((root / "inside").resolve(), _safe_file(root, "inside"))
            with self.assertRaisesRegex(ValueError, "escapes"):
                _safe_file(root, "../outside")


if __name__ == "__main__":
    unittest.main()
