import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from scripts.research.hftf.deployment.depthart.reseal_depthart_task_preserving_d2r1_checkpoints import (
    canonical_json_bytes,
    verify_pair,
)


class D2R1CheckpointResealTest(unittest.TestCase):
    def test_pure_crlf_translation_is_resealed(self) -> None:
        value = {"schema": "fixture", "value": [1, 2, 3]}
        canonical = canonical_json_bytes(value)
        with tempfile.TemporaryDirectory() as root:
            receipt = Path(root) / "receipt.json"
            sidecar = Path(root) / "receipt.sha256.json"
            receipt.write_bytes(canonical.replace(b"\n", b"\r\n"))
            sidecar.write_text(
                json.dumps({"bytes": len(canonical), "sha256": sha256(canonical).hexdigest().upper()}),
                encoding="utf-8",
            )
            result = verify_pair(receipt, sidecar, value)
            self.assertTrue(result["only_lf_to_crlf_translation"])

    def test_semantic_change_fails(self) -> None:
        value = {"schema": "fixture", "value": 1}
        canonical = canonical_json_bytes(value)
        with tempfile.TemporaryDirectory() as root:
            receipt = Path(root) / "receipt.json"
            sidecar = Path(root) / "receipt.sha256.json"
            receipt.write_bytes(canonical)
            sidecar.write_text(json.dumps({"bytes": len(canonical), "sha256": sha256(canonical).hexdigest().upper()}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "semantic drift"):
                verify_pair(receipt, sidecar, {"schema": "fixture", "value": 2})


if __name__ == "__main__":
    unittest.main()
