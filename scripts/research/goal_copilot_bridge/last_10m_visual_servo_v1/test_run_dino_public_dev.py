from __future__ import annotations

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.run_dino_public_dev import PROMPT


def test_public_prompt_is_goal_semantic() -> None:
    assert PROMPT == "door."
