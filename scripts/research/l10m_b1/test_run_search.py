from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from .policy_space import INITIAL_SPEC
from .run_search import (
    _feedback,
    _parse_output,
    _render_candidate,
    _validate_transport_qualification,
)
from .transport_qualification import ATTEMPT_COUNT, PASS_TERMINAL, PROTOCOL_ID


class L10MB1RunSearchTest(unittest.TestCase):
    def test_runner_round_trips_both_candidate_formats(self) -> None:
        for arm in ("raw", "structured"):
            self.assertEqual(_parse_output(arm, _render_candidate(arm, INITIAL_SPEC)), INITIAL_SPEC)

    def test_feedback_does_not_include_hidden_episode_ledger(self) -> None:
        result = {"semantic_valid": True, "unsafe_candidate": False, "behavioral_score": 0.5, "behavioral_vector": {"x": 1}, "episode_ledger": [{"hidden": True}]}
        self.assertNotIn("hidden", _feedback(result, generation=1, best_score=0.5))

    def test_proxy_transport_qualification_is_required_exactly(self) -> None:
        result = {
            "protocol_id": PROTOCOL_ID,
            "route": "proxy",
            "terminal": PASS_TERMINAL,
            "b1_execution_authorized": True,
            "scientific_instances_or_seeds_consumed": 0,
            "scientific_verdict": "NO_SCIENTIFIC_VERDICT",
            "request_count": ATTEMPT_COUNT,
            "success_count": ATTEMPT_COUNT,
            "failure_count": 0,
            "websocket_reconnect_exhaustion_count": 0,
            "decode_failure_count": 0,
            "container_isolation_guard": "PASS",
            "run_id": "test-proxy-pass",
            "completed_at": "2026-08-20T00:00:00+00:00",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(_validate_transport_qualification(path)["route"], "proxy")
            result["route"] = "direct"
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not authorize"):
                _validate_transport_qualification(path)


if __name__ == "__main__":
    unittest.main()
