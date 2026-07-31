from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from .build_readiness_lock import ReadinessLockError, verify_frozen_identities


class ReadinessLockTest(unittest.TestCase):
    def test_nested_frozen_identities_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "evidence" / "receipt.json"
            target.parent.mkdir()
            target.write_bytes(b"frozen\n")
            expected = hashlib.sha256(b"frozen\n").hexdigest()
            receipt = {
                "candidate": {
                    "identity": {
                        "relative_path": "evidence/receipt.json",
                        "sha256": expected,
                    }
                }
            }
            verified = verify_frozen_identities(root, receipt)
            self.assertEqual(verified[0]["sha256"], expected)

    def test_frozen_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "receipt.json"
            target.write_bytes(b"changed")
            receipt = {
                "relative_path": "receipt.json",
                "sha256": "0" * 64,
            }
            with self.assertRaises(ReadinessLockError):
                verify_frozen_identities(root, receipt)


if __name__ == "__main__":
    unittest.main()
