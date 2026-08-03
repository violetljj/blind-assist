import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from preflight_phone_session import qualify


class PhoneSessionPreflightTest(unittest.TestCase):
    def test_admits_complete_p0_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"protocol_id": "P", "model_id": "M", "common": {"camera_height_range_m": [0.8, 2.2], "maximum_camera_height_uncertainty_m": 0.02}, "phases": {"P0": {"minimum_admitted_frames_per_session": 2}}}
            (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            rows = []
            for index in range(2):
                image = root / f"{index}.jpg"
                image.write_bytes(f"rgb-{index}".encode())
                rows.append({"capture_timestamp_ns": index + 1, "rgb_file": image.name, "rgb_sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
            (root / "frames.json").write_text(json.dumps(rows), encoding="utf-8")
            reference = root / "reference.json"
            reference.write_text("{}", encoding="utf-8")
            receipt = {"phase": "P0", "protocol_id": "P", "model_id": "M", "session_id": "S1", "device_serial": "D", "camera_id": "0", "mount_profile_id": "fixed-1", "intrinsics_sha256": "A" * 64, "camera_height_m": 1.45, "camera_height_uncertainty_m": 0.01, "frame_manifest": "frames.json", "reference_manifest": "reference.json", "reference_manifest_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual("ADMITTED", qualify(receipt_path, root / "protocol.json")["status"])
            (root / "0.jpg").write_bytes(b"tampered")
            result = qualify(receipt_path, root / "protocol.json")
            self.assertEqual("HOLD", result["status"])
            self.assertIn("FRAME_0_RGB_SHA256_MISMATCH", result["failures"])

    def test_missing_physical_receipts_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"protocol_id": "P", "model_id": "M", "common": {"camera_height_range_m": [0.8, 2.2], "maximum_camera_height_uncertainty_m": 0.02}, "phases": {"P0": {"minimum_admitted_frames_per_session": 90}}}
            (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"phase": "P0", "protocol_id": "P", "model_id": "M"}), encoding="utf-8")
            result = qualify(receipt, root / "protocol.json")
            self.assertEqual("HOLD", result["status"])
            self.assertIn("INVALID_MEASURED_CAMERA_HEIGHT", result["failures"])
            self.assertIn("MISSING_INDEPENDENT_REFERENCE_MANIFEST", result["failures"])


if __name__ == "__main__":
    unittest.main()
