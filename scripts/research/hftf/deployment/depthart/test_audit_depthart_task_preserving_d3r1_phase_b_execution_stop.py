from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart import audit_depthart_task_preserving_d3r1_phase_b_execution_stop as subject


class D3R1PhaseBExecutionStopAuditTest(unittest.TestCase):
    def test_metadata_entry_does_not_hash_body(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            body = root / "body.zip"
            body.write_bytes(b"body")
            self.assertEqual(
                subject.metadata_entry(body, root),
                {"path": "body.zip", "bytes": 4},
            )

    def test_write_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-root" / "audit.json"
            subject.write_json_exclusive(output, {"schema": "fixture"})
            with self.assertRaises(FileExistsError):
                subject.write_json_exclusive(output, {"schema": "fixture"})


if __name__ == "__main__":
    unittest.main()
