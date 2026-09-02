from __future__ import annotations

import unittest

from l10_evidence_authority_lattice import Action, Authority, Evidence, EvidenceAuthorityLattice


def binding(**overrides: object) -> Evidence:
    values: dict[str, object] = {
        "directional_supported": True,
        "bearing": 0.2,
        "referent_id": "target-sign",
        "facade_id": "facade-7",
        "same_referent_continuity_supported": True,
        "sibling_exclusion_supported": True,
        "sign_facade_association_supported": True,
    }
    values.update(overrides)
    return Evidence(**values)


class EvidenceAuthorityLatticeTest(unittest.TestCase):
    def test_distance_never_changes_authority_or_action(self) -> None:
        far = EvidenceAuthorityLattice().step(
            Evidence(directional_supported=True, bearing=-0.3, observed_distance_m=50.0)
        )
        near = EvidenceAuthorityLattice().step(
            Evidence(directional_supported=True, bearing=-0.3, observed_distance_m=0.5)
        )
        self.assertEqual((far.authority, far.action), (near.authority, near.action))
        self.assertEqual((far.authority, far.action), (Authority.DIRECTIONAL, Action.ORIENT_LEFT))

    def test_binding_requires_exact_continuity_and_sibling_exclusion(self) -> None:
        lattice = EvidenceAuthorityLattice()
        incomplete = lattice.step(binding(sibling_exclusion_supported=False))
        bound = lattice.step(binding())
        self.assertEqual(incomplete.authority, Authority.DIRECTIONAL)
        self.assertEqual(bound.authority, Authority.BINDING)
        self.assertEqual(bound.action, Action.TRACK)

    def test_terminal_requires_owned_visible_endpoint_pose_and_commit_or_handoff(self) -> None:
        lattice = EvidenceAuthorityLattice()
        candidate = lattice.step(
            binding(
                entrance_id="entrance-2",
                entrance_ownership_supported=True,
                endpoint_visible=True,
                terminal_pose_supported=True,
                terminal_commit_supported=False,
                handoff_supported=False,
            )
        )
        commit = lattice.step(
            binding(
                entrance_id="entrance-2",
                entrance_ownership_supported=True,
                endpoint_visible=True,
                terminal_pose_supported=True,
                terminal_commit_supported=True,
            )
        )
        handoff = lattice.step(
            binding(
                entrance_id="entrance-2",
                entrance_ownership_supported=True,
                endpoint_visible=True,
                terminal_pose_supported=True,
                handoff_supported=True,
            )
        )
        self.assertEqual((candidate.authority, candidate.action), (Authority.BINDING, Action.TRACK))
        self.assertEqual((commit.authority, commit.action), (Authority.TERMINAL, Action.COMMIT))
        self.assertEqual((handoff.authority, handoff.action), (Authority.TERMINAL, Action.HANDOFF))

    def test_conflict_and_evidence_loss_are_reversible(self) -> None:
        lattice = EvidenceAuthorityLattice()
        self.assertEqual(lattice.step(binding()).authority, Authority.BINDING)
        conflict = lattice.step(binding(referent_id="sibling-sign"))
        self.assertEqual((conflict.authority, conflict.action), (Authority.UNKNOWN, Action.SEARCH))

        lattice.step(binding())
        terminal = lattice.step(
            binding(
                entrance_id="entrance-2",
                entrance_ownership_supported=True,
                endpoint_visible=True,
                terminal_pose_supported=True,
                terminal_commit_supported=True,
                handoff_supported=True,
            )
        )
        regressed = lattice.step(binding())
        unknown = lattice.step(Evidence())
        self.assertEqual(terminal.authority, Authority.TERMINAL)
        self.assertEqual(regressed.authority, Authority.BINDING)
        self.assertEqual((unknown.authority, unknown.action), (Authority.UNKNOWN, Action.SEARCH))


if __name__ == "__main__":
    unittest.main()
