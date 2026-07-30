import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.research.dual_loop_unseen_natural_event_r0.prepare_input as module


class PrepareInputTest(unittest.TestCase):
    def test_prepares_contiguous_hash_bound_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.webm"
            video.write_bytes(b"fixed-video")
            output = root / "input"

            def fake_ffmpeg(command, check):
                self.assertTrue(check)
                pattern = Path(command[-1])
                pattern.parent.mkdir(parents=True, exist_ok=True)
                (pattern.parent / "frame_000000.jpg").write_bytes(b"frame-0")
                (pattern.parent / "frame_000001.jpg").write_bytes(b"frame-1")

            expected = hashlib.sha256(video.read_bytes()).hexdigest()
            with patch.object(module.subprocess, "run", side_effect=fake_ffmpeg):
                receipt = module.prepare(
                    video,
                    output,
                    root / "ffmpeg",
                    expected_video_sha256=expected,
                    protocol_id="TEST_RANK_2",
                    source_id="test_source",
                )

            self.assertEqual(2, receipt["frame_count"])
            self.assertEqual(expected, receipt["video_sha256"])
            self.assertEqual("TEST_RANK_2", receipt["protocol_id"])
            self.assertEqual("test_source", receipt["source_id"])
            self.assertTrue((output / "manifest.jsonl").is_file())
            self.assertTrue((output / "input_receipt.json").is_file())

    def test_rejects_noncontiguous_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.webm"
            video.write_bytes(b"fixed-video")

            def fake_ffmpeg(command, check):
                self.assertTrue(check)
                pattern = Path(command[-1])
                pattern.parent.mkdir(parents=True, exist_ok=True)
                (pattern.parent / "frame_000000.jpg").write_bytes(b"frame-0")
                (pattern.parent / "frame_000002.jpg").write_bytes(b"frame-2")

            expected = hashlib.sha256(video.read_bytes()).hexdigest()
            with patch.object(module.subprocess, "run", side_effect=fake_ffmpeg):
                with self.assertRaisesRegex(ValueError, "non-contiguous"):
                    module.prepare(
                        video,
                        root / "input",
                        root / "ffmpeg",
                        expected_video_sha256=expected,
                    )


if __name__ == "__main__":
    unittest.main()
