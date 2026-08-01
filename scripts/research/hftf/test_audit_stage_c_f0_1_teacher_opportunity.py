from __future__ import annotations

import sys
import math
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_stage_c_f0_1_teacher_opportunity import (
    _causal_future_basis,
    _next_step_authorization,
    _pixel_lattices_disjoint,
    _structural_canaries,
    _target_counts,
    _timeline_contract,
    _union_support,
)
from audit_swept_envelope_label_mechanics import _swept_prism_counts


class StageCF01TeacherOpportunityTest(unittest.TestCase):
    def test_physical_time_contract_for_both_target_rates(self) -> None:
        five = _timeline_contract(5.0)
        self.assertEqual([-4, -3, -2, -1, 0], five["history_offsets"])
        self.assertEqual(-2, five["velocity_history_offset"])
        self.assertEqual(2, five["future_offset"])
        self.assertEqual(list(range(4, 23)), five["usable_anchor_indices"])
        ten = _timeline_contract(10.0)
        self.assertEqual([-8, -6, -4, -2, 0], ten["history_offsets"])
        self.assertEqual(-4, ten["velocity_history_offset"])
        self.assertEqual(4, ten["future_offset"])
        self.assertEqual(list(range(8, 21)), ten["usable_anchor_indices"])
        with self.assertRaisesRegex(ValueError, "5 or 10"):
            _timeline_contract(20.0)

    def test_future_basis_has_no_future_pose_input(self) -> None:
        plane = {
            "camera_ground_projection_m": [0.0, 0.0, 0.0],
            "normal_toward_camera": [0.0, 0.0, 1.0],
        }
        history = {
            "position_m": [-0.4, 0.0, 1.3],
            "quaternion_xyzw": [
                0.0,
                math.sqrt(0.5),
                0.0,
                math.sqrt(0.5),
            ],
        }
        anchor = {
            "position_m": [0.0, 0.0, 1.3],
            "quaternion_xyzw": [
                0.0,
                math.sqrt(0.5),
                0.0,
                math.sqrt(0.5),
            ],
        }
        current, future, velocity = _causal_future_basis(
            history, anchor, plane
        )
        self.assertTrue(np.allclose(velocity, [1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(future[0], [0.4, 0.0, 0.0]))
        for index in (1, 2, 3):
            self.assertTrue(np.allclose(current[index], future[index]))

    def test_union_uses_max_support_and_probe_or(self) -> None:
        anchor_counts = np.asarray([[[1], [3]]])
        future_counts = np.asarray([[[1], [0]]])
        anchor_passes = np.zeros((2, 9), dtype=bool)
        future_passes = np.zeros((2, 9), dtype=bool)
        anchor_passes[0, :3] = True
        future_passes[0, 3:5] = True
        counts, known = _union_support(
            anchor_counts,
            future_counts,
            anchor_passes,
            future_passes,
            (1, 2, 1),
        )
        self.assertEqual(1, counts[0, 0, 0])
        self.assertNotEqual(2, counts[0, 0, 0])
        self.assertTrue(known[0, 0, 0])
        self.assertFalse(known[0, 1, 0])

    def test_unknown_safe_violation_is_observable(self) -> None:
        known = np.asarray([False, True, True])
        support = np.asarray([0, 0, 2])
        clean = _target_counts(known, support)
        self.assertEqual(0, clean["unknown_to_safe_violations"])
        damaged = _target_counts(
            known,
            support,
            safe_assignment=np.asarray([True, True, False]),
        )
        self.assertEqual(1, damaged["unknown_to_safe_violations"])

    def test_candidate_and_reference_lattices_are_disjoint(self) -> None:
        self.assertTrue(_pixel_lattices_disjoint(2208, 1242))

    def test_next_step_never_authorizes_heldout_training_corpus(self) -> None:
        authorization = _next_step_authorization(True)
        self.assertTrue(
            authorization[
                "train_candidate_corpus_materialization_authorized"
            ]
        )
        self.assertTrue(
            authorization[
                "dev_reference_target_materialization_authorized"
            ]
        )
        self.assertFalse(
            authorization[
                "heldout_training_corpus_materialization_authorized"
            ]
        )
        self.assertFalse(
            authorization[
                "heldout_reference_target_materialization_authorized_before_frozen_checkpoint"
            ]
        )
        self.assertFalse(
            authorization[
                "student_training_authorized_before_corpus_validation"
            ]
        )

    def test_body_head_boundary_is_assigned_to_head_only(self) -> None:
        points = np.asarray([[1.0], [0.0], [1.35]])
        dynamic = np.asarray([False])
        basis = (
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        counts, _ = _swept_prism_counts(
            points,
            dynamic,
            basis,
            np.radians(np.asarray([-45.0, 45.0])),
            np.asarray([0.0, 2.0]),
            [(0.35, 1.35), (1.35, 2.05)],
            np.asarray([0.4, 0.28]),
        )
        self.assertEqual(0, counts[0, 0, 0])
        self.assertEqual(1, counts[0, 0, 1])

    def test_all_structural_canaries_pass(self) -> None:
        canaries = _structural_canaries()
        self.assertTrue(all(canaries.values()), canaries)


if __name__ == "__main__":
    unittest.main()
