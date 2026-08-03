import json
import tempfile
import unittest
from pathlib import Path

from validate_protocol import validate


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = (
    ROOT
    / "docs"
    / "research"
    / "hftf"
    / "KNOWN_CAMERA_HEIGHT_GROUND_SCALE_R0_PROTOCOL_2026-08-04.json"
)


class ValidateKnownCameraHeightProtocolTest(unittest.TestCase):
    def test_committed_protocol_matches_implementation(self) -> None:
        self.assertEqual("VALID", validate(PROTOCOL)["status"])

    def test_rejects_drift(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        protocol["operator"]["ransac_iterations"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate(path)


if __name__ == "__main__":
    unittest.main()
