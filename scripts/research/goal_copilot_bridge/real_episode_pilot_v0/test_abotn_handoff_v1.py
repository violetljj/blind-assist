from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.core import EpisodeState, State
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_abotn_v0_closed_loop import (
    HANDOFF_READY,
    _apply_termination_mode,
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


if __name__ == "__main__":
    unittest.main()
