from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import materialize_r12a_continuous_input as subject


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "configs/ustrf_crosscam_continuous_events_r12a_seen_v1.json"
PREREG = ROOT / "configs/ustrf_crosscam_continuous_events_r13_prereg_v1.json"


class R12aContinuousProtocolTest(unittest.TestCase):
    def test_frozen_protocol_and_r13_prereg_validate(self) -> None:
        subject.validate_protocol(json.loads(PROTOCOL.read_text(encoding="utf-8")))
        subject.validate_r13(json.loads(PREREG.read_text(encoding="utf-8")), PROTOCOL)

    def test_materialization_uses_only_seen_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subject.materialize(ROOT, PROTOCOL, PREREG, Path(directory))
            output = json.loads(Path(result["input"]).read_text(encoding="utf-8"))
        self.assertEqual(12, len(output["sources"]))
        self.assertTrue(all(row["dataset_role"] == "seen_diagnostic_not_held_out" for row in output["sources"]))
        self.assertFalse(output["authority"]["new_held_out_read"])
        vancouver = next(row for row in output["sources"] if row["event_id"].startswith("vancouver_"))
        self.assertFalse(vancouver["gate_eligible"])
        self.assertEqual("miss_lead_only", vancouver["diagnostic_role"])


if __name__ == "__main__":
    unittest.main()
