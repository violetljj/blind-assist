from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from scripts.research.goal_copilot_bridge.p0_s0_materialization.candidate_generator_admission import (
    validate_admission,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
RECORD_PATH = REPO_ROOT / "docs/research/goal-copilot/p0_s0_visual_candidate_generator_admission_v0.json"


class CandidateGeneratorAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))

    def test_frozen_non_admission_is_internally_valid(self) -> None:
        self.assertEqual([], validate_admission(self.record))

    def test_unresolved_provenance_cannot_be_promoted(self) -> None:
        promoted = copy.deepcopy(self.record)
        promoted["verdict"] = "P0_S0_VISUAL_CANDIDATE_GENERATOR_ADMITTED_WITH_CONSTRAINTS"
        promoted["execution_authorized"] = True
        self.assertIn("TRAINING_PROVENANCE_NOT_ADMISSIBLE", validate_admission(promoted))
        self.assertIn("UNRESOLVED_BLOCKERS", validate_admission(promoted))

    def test_generator_cannot_acquire_truth_authority(self) -> None:
        escalated = copy.deepcopy(self.record)
        escalated["lineage_independence"]["candidate_can_promote_silver_a_primary"] = True
        self.assertIn("PROPOSAL_AUTHORITY_ESCALATION", validate_admission(escalated))


if __name__ == "__main__":
    unittest.main()
