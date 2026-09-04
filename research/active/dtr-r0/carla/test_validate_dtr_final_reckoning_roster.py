from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import validate_dtr_final_reckoning_roster as roster


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROTOCOL = HERE / "dtr_final_reckoning_roster_protocol.json"


class FinalReckoningRosterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    def test_frozen_roster_and_byte_locks_validate(self) -> None:
        result = roster.validate(self.protocol, repo_root=REPO)
        self.assertEqual(30, result["episodes"])
        self.assertEqual(11, result["arms"])
        self.assertFalse(result["materialization_authorized"])

    def test_x97_or_missing_stratum_is_rejected(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["terminal_decision_contract"]["x97_forbidden"] = False
        with self.assertRaisesRegex(RuntimeError, "x97"):
            roster.validate(changed, repo_root=REPO)
        changed = copy.deepcopy(self.protocol)
        changed["source_design"]["strata"].pop()
        with self.assertRaisesRegex(RuntimeError, "strata"):
            roster.validate(changed, repo_root=REPO)


if __name__ == "__main__":
    unittest.main()
