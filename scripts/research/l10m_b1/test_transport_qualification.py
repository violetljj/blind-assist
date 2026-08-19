from __future__ import annotations

import json
import unittest

from .transport_qualification import (
    ATTEMPT_COUNT,
    FAIL_TERMINAL,
    PASS_TERMINAL,
    PAYLOAD_ITEMS,
    PAYLOAD_VALUE,
    _diagnostic_flags,
    _validate_response,
    build_protocol_manifest,
)


class L10MB1TransportQualificationTest(unittest.TestCase):
    def test_protocol_is_infrastructure_only_and_requires_ten_of_ten(self) -> None:
        manifest = build_protocol_manifest("proxy")
        self.assertEqual(manifest["request_count"], ATTEMPT_COUNT)
        self.assertEqual(
            manifest["pass_gate"]["successful_terminal_responses"], ATTEMPT_COUNT
        )
        self.assertEqual(manifest["authority"]["scientific_instances_or_seeds_consumed"], 0)
        self.assertFalse(manifest["authority"]["candidate_generation"])
        self.assertEqual(manifest["terminals"]["pass"], PASS_TERMINAL)
        self.assertEqual(manifest["terminals"]["fail"], FAIL_TERMINAL)

    def test_strict_terminal_response_validation(self) -> None:
        output = json.dumps(
            {
                "canary_id": "B1-I0",
                "attempt": 3,
                "payload": [PAYLOAD_VALUE] * PAYLOAD_ITEMS,
            }
        )
        self.assertEqual(_validate_response(output, 3), (True, None))
        self.assertFalse(_validate_response("", 3)[0])
        self.assertFalse(_validate_response(output, 4)[0])

    def test_transport_failure_markers_are_separate(self) -> None:
        flags = _diagnostic_flags(
            "websocket reconnect attempts exhausted; UnicodeDecodeError"
        )
        self.assertTrue(flags["websocket_reconnect_exhaustion"])
        self.assertTrue(flags["decode_failure"])


if __name__ == "__main__":
    unittest.main()
