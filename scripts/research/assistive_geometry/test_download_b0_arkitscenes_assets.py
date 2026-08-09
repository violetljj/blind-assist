import tempfile
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (
    roster_rows,
    validate_video_receipt,
    write_json_exclusive,
)


class DownloadB0ArkitScenesAssetsTest(unittest.TestCase):
    def test_roster_rows_preserve_role_and_fold(self) -> None:
        roles = {}
        for role, fold, offset, count in (
            ("TRAIN", "Training", 0, 16),
            ("DEVELOPMENT", "Training", 20, 8),
            ("CONFIRMATION", "Validation", 40, 8),
        ):
            roles[role] = [
                {"visit_id": str(100000 + offset + index), "video_id": str(200000 + offset + index), "official_fold": fold}
                for index in range(count)
            ]
        rows = roster_rows({"roles": roles})
        self.assertEqual(32, len(rows))
        self.assertEqual(16, sum(row["role"] == "TRAIN" for row in rows))
        self.assertEqual(8, sum(row["role"] == "DEVELOPMENT" for row in rows))
        self.assertEqual(8, sum(row["official_fold"] == "Validation" for row in rows))

    def test_overlap_fails(self) -> None:
        roles = {
            "TRAIN": [{"visit_id": str(100000 + index), "video_id": str(200000 + index), "official_fold": "Training"} for index in range(16)],
            "DEVELOPMENT": [{"visit_id": str(300000 + index), "video_id": str(400000 + index), "official_fold": "Training"} for index in range(8)],
            "CONFIRMATION": [{"visit_id": str(500000 + index), "video_id": str(600000 + index), "official_fold": "Validation"} for index in range(8)],
        }
        roles["CONFIRMATION"][0]["visit_id"] = roles["TRAIN"][0]["visit_id"]
        with self.assertRaisesRegex(ValueError, "visit overlap"):
            roster_rows({"roles": roles})

    def test_receipt_is_exclusive_and_resume_validates_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "frame.bin"
            output.write_bytes(b"ok")
            parent = {"role": "TRAIN", "visit_id": "100000", "video_id": "200000", "official_fold": "Training"}
            receipt = parent | {"selected_frame_count": 1, "extracted": {"rgb": [{"path": str(output), "bytes": 2}]}}
            path = root / "receipt.json"
            write_json_exclusive(path, receipt)
            self.assertEqual(receipt, validate_video_receipt(path, parent, 1))
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, receipt)


if __name__ == "__main__":
    unittest.main()
