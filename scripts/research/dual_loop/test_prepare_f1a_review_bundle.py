import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_f1a_review_bundle import load_frame_ledger, render_contact_sheets, sample_ledger


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareF1aReviewBundleTest(unittest.TestCase):
    def test_ledger_identity_hash_sampling_and_contact_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "sequence"
            rgb = sequence / "rgb"
            rgb.mkdir(parents=True)
            rows = []
            for index, timestamp in enumerate((1_000_000_000, 1_500_000_000, 2_100_000_000)):
                image = rgb / f"{timestamp}.png"
                Image.new("RGB", (16, 12), (index * 50, 0, 0)).save(image)
                rows.append(
                    {
                        "frame_id": f"{index:06d}",
                        "source_id": "source",
                        "sequence_id": "session",
                        "source_capture_timestamp_ns": timestamp,
                        "rgb_path": f"rgb/{timestamp}.png",
                        "rgb_sha256": sha256_file(image),
                    }
                )
            ledger = sequence / "frames.jsonl"
            ledger.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            bundle = sequence / "bundle.json"
            bundle.write_text("{}\n", encoding="utf-8")
            item = {
                "input_id": "fixture",
                "source_id": "source",
                "session_id": "session",
                "frames_path": "sequence/frames.jsonl",
                "frames_sha256": sha256_file(ledger),
                "bundle_path": "sequence/bundle.json",
                "bundle_sha256": sha256_file(bundle),
                "expected_frame_count": 3,
                "review_sample_period_seconds": 1.0,
            }
            loaded = load_frame_ledger(root, item)
            samples = sample_ledger(root, item, loaded)
            self.assertEqual(["000000", "000002"], [sample["frame_id"] for sample in samples])
            output = root / "output"
            output.mkdir()
            sheets = render_contact_sheets(samples, output, "fixture")
            self.assertEqual(1, len(sheets))
            self.assertTrue(Path(sheets[0]["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
