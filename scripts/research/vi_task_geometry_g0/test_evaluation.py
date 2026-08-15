from __future__ import annotations

import unittest

from scripts.research.vi_task_geometry_g0.evaluation import CandidateFrame, TruthFrame, evaluate_g0


class VitgG0EvaluationTest(unittest.TestCase):
    def fixture(self, *, false_wall: bool = False, unsafe_degeneracy: bool = False):
        episode_types = {"observable": "EXCITED_WALK_TURN", "control": "STATIC_CONTROL"}
        truth = []
        candidates = []
        for parent in range(8):
            for episode in episode_types:
                for frame in range(3):
                    parent_id = f"parent-{parent:02d}"
                    frame_id = f"frame-{frame:03d}"
                    truth.append(
                        TruthFrame(
                            parent_id, episode, frame_id, 1.5,
                            {"anchor": 2.0},
                            {"support": "NON_GROUND" if false_wall else "GROUND"},
                            clearance_m=0.8,
                        )
                    )
                    for arm in ("A0", "A1", "A2", "A3"):
                        valid = arm in {"A2", "A3"} and (
                            episode == "observable" or unsafe_degeneracy
                        )
                        candidates.append(
                            CandidateFrame(
                                arm, parent_id, episode, frame_id,
                                "VALID_METRIC_GEOMETRY" if valid else "UNKNOWN_OBSERVABILITY",
                                1.5 if valid else None,
                                {"anchor": 2.0} if valid else {},
                                ("support",) if valid else (),
                                clearance_m=0.8 if valid else None,
                            )
                        )
        return candidates, truth, episode_types

    def test_primary_arms_can_pass_without_promoting_comparators_or_clearance(self) -> None:
        result = evaluate_g0(*self.fixture())
        self.assertTrue(result["passed"])
        self.assertEqual(result["eligible_primary_arms_passing"], ["A2", "A3"])
        self.assertFalse(result["arms"]["A0"]["passed_absolute_gates"])
        self.assertFalse(result["clearance_metrics_affect_g0_pass"])

    def test_false_wall_support_fails_primary_arms(self) -> None:
        result = evaluate_g0(*self.fixture(false_wall=True))
        self.assertFalse(result["passed"])
        self.assertEqual(result["arms"]["A2"]["parent_macro"]["false_wall_ground_support_rate"], 1.0)

    def test_metric_valid_on_static_control_fails_closed(self) -> None:
        result = evaluate_g0(*self.fixture(unsafe_degeneracy=True))
        self.assertFalse(result["passed"])
        self.assertEqual(result["arms"]["A3"]["worst_parent"]["degeneracy_unsafe_valid_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
