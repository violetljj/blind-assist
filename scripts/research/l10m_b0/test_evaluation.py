from __future__ import annotations

import unittest

from .evaluation import Action, Arm, Evidence, Hazard, Truth, run_episode, summarize


def episode(name: str, *, blocked: bool = False) -> tuple[list[Evidence], list[Truth]]:
    evidence = [
        Evidence(name, 0, 0.0, Hazard.HIGH if blocked else Hazard.LOW, 0.95),
        Evidence(name, 1, 0.0, Hazard.HIGH if blocked else Hazard.LOW, 0.95),
        Evidence(name, 2, 0.0, Hazard.LOW, 0.95),
    ]
    truth = [
        Truth(name, 0, 0.0, unsafe_forward=blocked),
        Truth(name, 1, 0.0, unsafe_forward=blocked),
        Truth(name, 2, 1.0, arrived=True),
    ]
    return evidence, truth


class L10MB0Test(unittest.TestCase):
    def test_safety_shield_blocks_unsafe_forward(self) -> None:
        rows = episode("blocked", blocked=True)
        reactive = run_episode(Arm.REACTIVE, *rows)
        safe = run_episode(Arm.STATEFUL_SAFETY, *rows)
        self.assertGreater(reactive.unsafe_actions, safe.unsafe_actions)
        self.assertEqual(safe.actions[0], Action.STOP)

    def test_unknown_evidence_is_not_promoted_to_progress(self) -> None:
        evidence = [Evidence("unknown", 0, 0.0, Hazard.LOW, 0.2, stale=True)]
        truth = [Truth("unknown", 0, 1.0, arrived=True)]
        result = run_episode(Arm.STATEFUL_SAFETY, evidence, truth)
        self.assertEqual(result.unknown_steps, 1)
        self.assertFalse(result.success)

    def test_summary_is_behavioral_vector_without_composite(self) -> None:
        rows = [episode("clear"), episode("blocked", blocked=True)]
        result = summarize({arm: [run_episode(arm, *row) for row in rows] for arm in Arm})
        self.assertEqual(set(result[Arm.REACTIVE.value]), {
            "task_success", "unsafe_action_rate", "false_arrival_rate",
            "stuck_detection_latency", "recovery_success", "UNKNOWN_rate",
            "oscillation_rate", "excess_action",
        })
        self.assertNotIn("score", result[Arm.REACTIVE.value])


if __name__ == "__main__":
    unittest.main()
