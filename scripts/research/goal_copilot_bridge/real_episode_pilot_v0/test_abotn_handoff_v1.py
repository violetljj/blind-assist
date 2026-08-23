from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.core import EpisodeState, State
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_abotn_v0_closed_loop import (
    HANDOFF_READY,
    _apply_termination_mode,
    _audit_call_mechanics,
)


class AbotnHandoffV1Test(unittest.TestCase):
    def test_arrival_cue_becomes_terminal_handoff_without_completion(self) -> None:
        state = EpisodeState.start(
            episode_id="abotn-test-traj-9", location_id="scene", goal_name="shop", started_at_ms=1
        )
        state.state = State.ARRIVAL_CONFIRM.value
        event = {
            "to_state": State.ARRIVAL_CONFIRM.value,
            "transitions": [State.CURRENT_CANDIDATE.value, State.ARRIVAL_CONFIRM.value],
            "completion": False,
        }
        state, transformed = _apply_termination_mode(state, event, termination_mode="HANDOFF_V1")
        self.assertEqual(HANDOFF_READY, state.state)
        self.assertEqual(HANDOFF_READY, transformed["to_state"])
        self.assertTrue(transformed["handoff_ready"])
        self.assertFalse(transformed["completion"])

    def test_v0_mode_remains_unchanged(self) -> None:
        state = EpisodeState.start(
            episode_id="abotn-test-traj-9", location_id="scene", goal_name="shop", started_at_ms=1
        )
        state.state = State.ARRIVAL_CONFIRM.value
        event = {"to_state": State.ARRIVAL_CONFIRM.value, "transitions": [], "completion": False}
        state, transformed = _apply_termination_mode(state, event, termination_mode="V0_COMPLETION")
        self.assertEqual(State.ARRIVAL_CONFIRM.value, state.state)
        self.assertEqual(event, transformed)

    def test_service_tier_diagnostic_is_not_an_external_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            call_dir = Path(directory)
            for name in (
                "brain-prompt.txt", "brain-output-schema.json", "brain-input.jpg", "observation.json"
            ):
                (call_dir / name).write_bytes(b"{}")
            (call_dir / "completion.json").write_text(
                json.dumps({"status": "RUN_SUCCESS"}), encoding="utf-8"
            )
            events = [
                {"type": "item.completed", "item": {"type": "error", "message": "Configured service tier `priority` is not advertised as supported for model `gpt-5.6-terra` and will be omitted from requests."}},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
            ]
            (call_dir / "attempt-1-stdout.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events), encoding="utf-8"
            )
            audit = _audit_call_mechanics(call_dir)
            self.assertTrue(audit["pass"])
            self.assertEqual(0, audit["external_action_event_count"])
            self.assertEqual(1, len(audit["benign_diagnostic_items"]))


if __name__ == "__main__":
    unittest.main()
