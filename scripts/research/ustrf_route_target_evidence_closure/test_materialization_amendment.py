from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import validate_replacement_holdout as validator


class MaterializationAmendmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.replacement_path = (
            cls.repo / "configs/ustrf_route_target_evidence_closure_r1_replacement_holdout.json"
        )
        cls.replacement = json.loads(cls.replacement_path.read_text(encoding="utf-8"))
        cls.amendment = json.loads(
            (
                cls.repo
                / "configs/ustrf_route_target_evidence_closure_r1_materialization_amendment.json"
            ).read_text(encoding="utf-8")
        )
        cls.binding = next(
            row
            for row in cls.replacement["implementation_bindings"]
            if Path(row["path"]).name == "materialize_crowdbot_holdout_sources.py"
        )

    def test_exact_candidate_blind_amendment_accepts_materializer(self) -> None:
        path = validator.resolve_materializer_amendment(
            self.repo,
            self.binding,
            self.amendment,
            validator.sha256_file(self.replacement_path),
        )
        self.assertEqual(path, self.repo / self.binding["path"])

    def test_materializer_drift_without_amendment_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "without materialization amendment"):
            validator.resolve_materializer_amendment(
                self.repo,
                self.binding,
                None,
                validator.sha256_file(self.replacement_path),
            )

    def test_candidate_output_exposure_in_amendment_fails_closed(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["candidate_outputs_executed_before_amendment"] = True
        with self.assertRaisesRegex(RuntimeError, "candidate output exposure"):
            validator.resolve_materializer_amendment(
                self.repo,
                self.binding,
                changed,
                validator.sha256_file(self.replacement_path),
            )

    def test_transport_lineage_is_hash_bound_and_complete(self) -> None:
        hashes = validator.validate_transport_lineage(
            self.repo, self.amendment["transport_lineage_tip"]
        )
        self.assertEqual(len(hashes), 7)


if __name__ == "__main__":
    unittest.main()
