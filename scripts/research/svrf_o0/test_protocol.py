from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.research.svrf_o0.validate_protocol import validate_protocol


PROTOCOL_PATH = Path("docs/research/svrf/SVRF_O0_SCALE_FREE_VISUAL_RISK_FIELD_PROTOCOL_2026-08-15.json")


class SvrfO0ProtocolTest(unittest.TestCase):
    def protocol(self) -> dict[str, object]:
        return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_frozen_protocol_is_executable_contract_valid(self) -> None:
        validate_protocol(self.protocol())

    def test_top_level_status_drift_is_rejected(self) -> None:
        protocol = deepcopy(self.protocol())
        protocol["status"] = "FROZEN_PREOUTCOME_BLOCKED_ON_FRESH_PARENT_CAPABILITY_AND_SOURCE_LOCK"
        with self.assertRaisesRegex(ValueError, "top-level status drift"):
            validate_protocol(protocol)

    def test_source_native_intrinsics_or_non_joint_coverage_is_rejected(self) -> None:
        protocol = deepcopy(self.protocol())
        protocol["candidate_input_firewall"]["candidate_intrinsics_policy"] = "SOURCE_NATIVE_INTRINSICS_ALLOWED"
        with self.assertRaisesRegex(ValueError, "candidate intrinsics policy drift"):
            validate_protocol(protocol)

        protocol = self.protocol()
        protocol["truth_unknown_contract"]["coverage_definitions"]["winner_rule_coverage"] = "candidate_valid_coverage"
        with self.assertRaisesRegex(ValueError, "winner coverage definition drift"):
            validate_protocol(protocol)

    def test_negative_control_transform_drift_is_rejected(self) -> None:
        protocol = self.protocol()
        protocol["negative_control_lock_contract"]["N2"] = "arbitrary outcome-selected warp"
        with self.assertRaisesRegex(ValueError, "N2 transform identity drift"):
            validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
