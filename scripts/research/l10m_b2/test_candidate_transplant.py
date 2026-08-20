from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .candidate_transplant import (
    PROTOCOL_ID,
    build_protocol,
    execute,
)


class L10MB2CandidateTransplantTest(unittest.TestCase):
    def test_seed89_transplant_localizes_the_b1_difference(self) -> None:
        run_dir = Path("artifacts.local/evidence/l10m_b1/runs/b1-20260820T115002-98733875")
        if not run_dir.exists():
            self.skipTest("formal B1 source evidence is unavailable")
        protocol = build_protocol(run_dir)
        with tempfile.TemporaryDirectory() as temp_dir:
            protocol_path = Path(temp_dir) / "protocol.json"
            protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = execute(protocol_path, run_dir)
        self.assertEqual(result["protocol_id"], PROTOCOL_ID)
        self.assertEqual(result["scientific_verdict"], "B2_SEARCH_PATH_FAILURE_SIGNAL")
        self.assertTrue(result["semantic_equivalent"])
        self.assertTrue(result["behavior_equal"])
        self.assertEqual(result["behavior_sha256"]["raw"], result["behavior_sha256"]["structured"])
        self.assertEqual(result["raw_evaluation"]["behavioral_score"], 0.993103448275862)


if __name__ == "__main__":
    unittest.main()
