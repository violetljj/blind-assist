from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import taro_r38_r32_arkitscenes_confirmation as subject


def frame(index: int) -> bonn.Frame:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = index * 0.05
    return bonn.Frame("p", float(index), Path(f"rgb-{index}"), Path(f"depth-{index}"), pose)


def pair(reference: bonn.Frame, candidate: bonn.Frame, translation: float) -> bonn.Pair:
    return bonn.Pair(reference, candidate, reference.timestamp_s - candidate.timestamp_s, translation, 0.0)


class R38ConfirmationTest(unittest.TestCase):
    def test_role_disjoint_filter_removes_every_reference_from_candidate_roles(self) -> None:
        f0, f1, f2, f3 = (frame(index) for index in range(4))
        rows = [
            bonn.ReferenceSupport(f2, (pair(f2, f0, 0.10), pair(f2, f1, 0.05)), (pair(f2, f1, 0.05),)),
            bonn.ReferenceSupport(f3, (pair(f3, f0, 0.15), pair(f3, f2, 0.05)), (pair(f3, f2, 0.05),)),
        ]
        identities, retained = subject.role_disjoint_identity_rows(rows)
        references = {row.reference.frame_id for row in retained}
        candidates = {value for row in identities for value in row["candidate_frame_ids"]}
        self.assertTrue(references.isdisjoint(candidates))
        self.assertEqual({f2.frame_id}, references)

    def test_role_disjoint_filter_updates_candidates_when_all_references_survive(self) -> None:
        f0, f1, f2, f3 = (frame(index) for index in range(4))
        rows = [
            bonn.ReferenceSupport(f2, (pair(f2, f0, 0.10),), (pair(f2, f0, 0.10),)),
            bonn.ReferenceSupport(
                f3,
                (pair(f3, f0, 0.15), pair(f3, f1, 0.05), pair(f3, f2, 0.05)),
                (pair(f3, f1, 0.05), pair(f3, f2, 0.05)),
            ),
        ]
        identities, retained = subject.role_disjoint_identity_rows(rows)
        references = {row.reference.frame_id for row in retained}
        candidates = {value for row in identities for value in row["candidate_frame_ids"]}
        self.assertEqual({f2.frame_id, f3.frame_id}, references)
        self.assertTrue(references.isdisjoint(candidates))

    def test_confirmation_checks_apply_frozen_opportunity_fraction(self) -> None:
        contract = {
            "minimum_evaluated_parent_count": 8,
            "minimum_evaluated_reference_count": 24,
            "minimum_opportunity_parent_count": 4,
            "minimum_strict_win_parent_count": 3,
            "minimum_strict_win_fraction_of_opportunity_parents": 0.5,
            "same_one_extra_frame_budget": True,
            "retention_failures_allowed": 0,
        }
        metrics = {
            "opportunity_parent_count": 8,
            "policy_strict_win_opportunity_parent_count": 3,
            "parent_macro": {"ranker": 10.0, "generic": 9.0, "passive": 8.0},
        }
        checks = subject.confirmation_checks(contract, metrics, 12, 60, True)
        self.assertFalse(checks["opportunity_denominated_strict_win_gate"])
        metrics["policy_strict_win_opportunity_parent_count"] = 4
        self.assertTrue(all(subject.confirmation_checks(contract, metrics, 12, 60, True).values()))


if __name__ == "__main__":
    unittest.main()
