from __future__ import annotations

from collections import Counter
import unittest

import run_p1_a3_temporal_loss as a3


def candidate(frame_index: int):
    return {
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 33,
        "candidates": [{
            "candidate_id": f"c-{frame_index}",
            "identity_support": 0.9,
            "identity_contradiction": 0.1,
            "stability": 0.9,
            "oscillation": 0.0,
        }],
    }


def episode(length: int):
    frames = [candidate(index) for index in range(length)]
    return {
        "episode_id": "synthetic",
        "p1_input": {"frames": frames},
        "p1_output": {"protocol_id": "BA-P1-TARGET-PERSISTENCE-R0-V1", "referent_id": "target"},
    }


class FamilyTest(unittest.TestCase):
    def test_family_is_exactly_three_small_operator_classes(self):
        specs = a3.operator_family()
        self.assertEqual(len(specs), 40)
        self.assertEqual(Counter(spec["operator"] for spec in specs), {
            "CONSECUTIVE_HYSTERESIS": 16,
            "SLIDING_WINDOW_VOTE": 8,
            "LEAKY_EVIDENCE_ACCUMULATOR": 16,
        })

    def test_specs_do_not_contain_a2_frame_thresholds(self):
        forbidden = set(a3.RAW_FEATURES)
        for spec in a3.operator_family():
            self.assertTrue(forbidden.isdisjoint(spec))


class StateMachineTest(unittest.TestCase):
    def spec(self):
        return {
            "operator": "CONSECUTIVE_HYSTERESIS",
            "low_threshold": 0.3,
            "high_threshold": 0.7,
            "exit_run": 2,
            "loss_run": 5,
            "recovery_run": 2,
            "reacquisition_confirm_frames": 5,
        }

    def test_sustained_contradiction_declares_loss_and_recovery_is_strict(self):
        value = episode(13)
        scores = {f"c-{index}": score for index, score in enumerate(
            [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        )}
        output = a3.run_temporal_episode(value, scores, self.spec())
        self.assertEqual(output["frames"][2]["state"], "UNCERTAIN")
        self.assertEqual(output["frames"][5]["state"], "LOST")
        self.assertEqual(output["frames"][5]["event"], "LOSS_DETECTED")
        reacquired = [frame for frame in output["frames"] if frame["event"] == "REACQUIRED"]
        self.assertEqual(len(reacquired), 1)
        self.assertGreaterEqual(reacquired[0]["frame_index"], 10)

    def test_one_frame_positive_bounce_never_revives_lost_state(self):
        value = episode(10)
        scores = {f"c-{index}": score for index, score in enumerate(
            [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
        )}
        output = a3.run_temporal_episode(value, scores, self.spec())
        self.assertFalse(any(frame["event"] == "REACQUIRED" for frame in output["frames"]))
        self.assertEqual(output["frames"][-1]["state"], "LOST")


class TerminalTest(unittest.TestCase):
    def row(self, admitted=False, delayed=False, count=0, name="x"):
        return {
            "candidate_id": name,
            "admission_pass": admitted,
            "gate_pass_count": count,
            "gate_passes": {"correct": admitted or delayed, "wrong": admitted, "wrong_lock": admitted},
            "usability_pass": admitted or delayed,
            "loss_declaration_pass": admitted or delayed,
            "false_reacquisitions": 0,
            "false_loss_frames": 0,
            "wrong_assertions": 0 if admitted else 900,
            "max_wrong_lock_duration_ms": 0 if admitted else 7000,
            "correct_assertions": 80,
            "tracking_boundary_transitions": 0,
            "reacquisition_chatter_within_30_frames": 0,
        }

    def test_three_terminals_are_exhaustive(self):
        terminal, _ = a3.choose_terminal([self.row(admitted=True, name="success")])
        self.assertEqual(terminal, "TEMPORAL_LOSS_STATE_SIGNAL_ESTABLISHED")
        terminal, _ = a3.choose_terminal([self.row(delayed=True, name="delay")])
        self.assertEqual(terminal, "TEMPORAL_SMOOTHING_ONLY_DELAYS_FAILURE")
        terminal, _ = a3.choose_terminal([self.row(count=2, name="insufficient")])
        self.assertEqual(terminal, "TEMPORAL_POLICY_INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
