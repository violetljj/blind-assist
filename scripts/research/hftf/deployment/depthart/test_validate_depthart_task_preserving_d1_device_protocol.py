from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d1_device_protocol import (
    validate,
)


class ValidateDepthArtTaskPreservingD1DeviceProtocolTest(unittest.TestCase):
    def test_live_protocol_is_valid(self) -> None:
        repo = Path(__file__).resolve().parents[5]
        protocol = repo / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D1_SM8650_HTP_CONTEXT_AND_OUTCOME_ACTIVATION_PREFLIGHT_PROTOCOL_2026-08-10.json"
        self.assertEqual(validate(repo, protocol)["status"], "VALID_NO_DEVICE_OUTPUT_ACCESSED")

    def test_device_identity_mutation_fails_closed(self) -> None:
        repo = Path(__file__).resolve().parents[5]
        source = repo / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D1_SM8650_HTP_CONTEXT_AND_OUTCOME_ACTIVATION_PREFLIGHT_PROTOCOL_2026-08-10.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["device_lock"]["soc"] = "SM8550"
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "protocol.json"
            mutated.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "device lock mismatch"):
                validate(repo, mutated)

    def test_raw_depth_diagnostic_cannot_become_gate(self) -> None:
        repo = Path(__file__).resolve().parents[5]
        source = repo / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D1_SM8650_HTP_CONTEXT_AND_OUTCOME_ACTIVATION_PREFLIGHT_PROTOCOL_2026-08-10.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["reference_diagnostic"]["gate"] = True
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "protocol.json"
            mutated.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "illegally became"):
                validate(repo, mutated)


if __name__ == "__main__":
    unittest.main()
