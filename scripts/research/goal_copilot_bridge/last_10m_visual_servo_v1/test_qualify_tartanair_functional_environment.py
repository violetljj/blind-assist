from __future__ import annotations

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.qualify_tartanair_functional_environment import qualify


def test_qualifier_is_callable() -> None:
    assert callable(qualify)
