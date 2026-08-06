import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_quality_gated_clearance_fusion_r0_raw_stream as subject


class RawStreamValidationTest(unittest.TestCase):
    def test_expected_state(self):
        self.assertEqual("UNKNOWN", subject.expected_state(None, False, 1.5))
        self.assertEqual("OCCUPIED", subject.expected_state(1.5, True, 1.5))
        self.assertEqual("CLEAR", subject.expected_state(1.6, True, 1.5))

    def test_verify_binding_rejects_sha_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "asset"
            path.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                subject.verify_binding(root, {"path": "asset", "sha256": "0" * 64}, "asset")

    def test_validate_rejects_duplicate_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            upstream = {}
            for name in ("development", "materialization", "producer", "receipt", "checkpoint"):
                path = root / name
                path.write_bytes(name.encode())
                upstream[name] = {"path": name, "sha256": subject.sha256_file(path)}
            catalog = {
                "schema": subject.CATALOG_SCHEMA,
                "labels_opened": False,
                "image_or_depth_bytes_decoded": False,
                "frames": [{"frame_id": "f", "parent_id": "p", "video_id": "v", "timestamp_ns": 1}],
            }
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            row = {
                "schema": subject.STREAM_SCHEMA,
                "frame_id": "f", "parent_id": "p", "video_id": "v", "timestamp_ns": 1,
                "raw_clearance_m": [1.0, 2.0, None],
                "raw_geometry_valid": [True, True, False],
                "raw_geometry_state": ["OCCUPIED", "CLEAR", "UNKNOWN"],
                "tof_valid": True, "teacher_age_s": 0.0, "frozen_a2_disagreement": 0.1,
                "rgb_sha256": "A" * 64, "metric_depth_sha256": "B" * 64,
                "geometry_status": "VALID",
            }
            stream_path = root / "stream.jsonl"
            stream_path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            protocol = {
                "schema": subject.PROTOCOL_SCHEMA,
                "parent_development_protocol": upstream["development"],
                "materialization_protocol": upstream["materialization"],
                "materializer_producer": upstream["producer"],
                "source_catalog": {"path": "catalog.json", "sha256": subject.sha256_file(catalog_path)},
                "a2_training_receipt": upstream["receipt"],
                "a2_checkpoint": upstream["checkpoint"],
                "raw_stream": {"path": "stream.jsonl", "sha256": subject.sha256_file(stream_path)},
                "expected_frame_count": 1,
                "expected_parent_frame_counts": {"p": 1},
                "required_frame_fields": list(row),
                "clearance_threshold_m": 1.5,
            }
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                subject.validate(root, protocol_path, catalog_path, stream_path)


if __name__ == "__main__":
    unittest.main()
