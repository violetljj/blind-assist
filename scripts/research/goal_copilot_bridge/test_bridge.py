from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from evaluator import CandidateContractError  # noqa: E402
from sky_bridge import (  # noqa: E402
    CANDIDATE_KIND,
    CANDIDATE_PAYLOADS,
    PROTOCOL_ID,
    BundleError,
    canonical_json,
    export_task,
    import_candidate,
    payload_checksums,
    sha256,
    validate_candidate,
    verify_task_bundle,
)


class BridgeTest(unittest.TestCase):
    def _candidate(
        self,
        root: Path,
        task_bundle: Path,
        *,
        policy: Path | None = None,
        source_digest: str | None = None,
        protocol_id: str = PROTOCOL_ID,
        search_fitness: float = 1.0,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        task_manifest = json.loads((task_bundle / "manifest.json").read_text(encoding="utf-8"))
        source_digest = source_digest or task_manifest["bundle_digest"]
        staging = root / f"candidate-staging-{len(list(root.glob('candidate-staging-*')))}"
        (staging / "candidate").mkdir(parents=True)
        shutil.copy2(policy or (task_bundle / "initial_policy.py"), staging / "candidate/policy.py")
        candidate_id = sha256(staging / "candidate/policy.py")
        (staging / "provenance.json").write_bytes(
            canonical_json(
                {
                    "source_search_task_bundle_digest": source_digest,
                    "skydiscover_repository_commit": "b" * 40,
                    "search_configuration": "mock-v0",
                    "parent_candidate": None,
                    "generation": 0,
                    "iteration": 0,
                    "model_provider_identity": "none/mock",
                    "resource_usage": {"model_calls": 0},
                }
            )
        )
        (staging / "search_metrics.json").write_bytes(
            canonical_json(
                {
                    "authority": "PROVENANCE_ONLY_NOT_ACCEPTANCE",
                    "ranking_fitness": search_fitness,
                }
            )
        )
        (staging / "candidate_manifest.json").write_bytes(
            canonical_json(
                {
                    "schema_version": 1,
                    "kind": CANDIDATE_KIND,
                    "protocol_id": protocol_id,
                    "candidate_id": candidate_id,
                    "source_search_task_bundle_digest": source_digest,
                    "candidate_files": ["candidate/policy.py"],
                }
            )
        )
        checksums = payload_checksums(staging, CANDIDATE_PAYLOADS)
        (staging / "checksums.json").write_bytes(canonical_json(checksums))
        destination = root / candidate_id
        staging.rename(destination)
        return destination

    def test_deterministic_export_checksums_and_hidden_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = export_task(root / "exports")
            first_manifest = (task / "manifest.json").read_bytes()
            self.assertEqual(task, export_task(root / "exports"))
            self.assertEqual(first_manifest, (task / "manifest.json").read_bytes())
            manifest = verify_task_bundle(task)
            self.assertEqual(task.name, manifest["bundle_digest"])
            self.assertEqual("PROPOSAL_ONLY", manifest["search_authority"])
            self.assertEqual("BLINDASSIST_ONLY", manifest["acceptance_authority"])
            members = {path.name for path in task.rglob("*") if path.is_file()}
            self.assertNotIn("evaluator.py", members)
            self.assertNotIn("sealed_scenarios.json", members)

    def test_mock_roundtrip_independent_acceptance_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = export_task(root / "exports")
            candidate = self._candidate(root, task, search_fitness=-100.0)
            imported = import_candidate(candidate, task, root / "imports")
            receipt = validate_candidate(imported, task, root / "validations")
            first = receipt.read_bytes()
            self.assertEqual(receipt, validate_candidate(imported, task, root / "validations"))
            self.assertEqual(first, receipt.read_bytes())
            assessment = json.loads(first)
            self.assertEqual("ACCEPT", assessment["status"])
            self.assertEqual(1.0, assessment["evaluation"]["metrics"]["goal_completion"])
            self.assertLessEqual(assessment["evaluation"]["metrics"]["tracking_continuity"], 1.0)
            self.assertEqual(
                "PROVENANCE_ONLY_NOT_ACCEPTANCE", assessment["sky_search_metrics_authority"]
            )

    def test_high_sky_score_does_not_override_ba_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = export_task(root / "exports")
            unsafe = root / "unsafe_policy.py"
            source = (task / "initial_policy.py").read_text(encoding="utf-8")
            before, marker, after = source.rpartition("    return proposals[0]")
            self.assertTrue(marker)
            source = before + '    return "FORWARD"' + after
            unsafe.write_text(source, encoding="utf-8")
            candidate = self._candidate(root, task, policy=unsafe, search_fitness=999999.0)
            imported = import_candidate(candidate, task, root / "imports")
            assessment = json.loads(
                validate_candidate(imported, task, root / "validations").read_text(encoding="utf-8")
            )
            self.assertEqual("REJECT", assessment["status"])

    def test_candidate_scope_violation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = export_task(root / "exports")
            candidate = self._candidate(root, task)
            (candidate / "evaluator.py").write_text("# forbidden\n", encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "members"):
                import_candidate(candidate, task, root / "imports")

    def test_candidate_surface_allowlist_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = export_task(root / "exports")
            bad_policy = root / "bad_policy.py"
            bad_policy.write_text(
                (task / "initial_policy.py").read_text(encoding="utf-8")
                + "\ndef modify_evaluator():\n    return True\n",
                encoding="utf-8",
            )
            candidate = self._candidate(root, task, policy=bad_policy)
            with self.assertRaises(CandidateContractError):
                import_candidate(candidate, task, root / "imports")

    def test_protocol_source_and_checksum_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = export_task(root / "exports")

            wrong_protocol = self._candidate(
                root / "wrong-protocol", task, protocol_id="GOAL-COPILOT-X"
            )
            with self.assertRaisesRegex(BundleError, "protocol"):
                import_candidate(wrong_protocol, task, root / "imports-a")

            wrong_source = self._candidate(root / "wrong-source", task, source_digest="0" * 64)
            with self.assertRaisesRegex(BundleError, "different SearchTaskBundle"):
                import_candidate(wrong_source, task, root / "imports-b")

            tampered = self._candidate(root / "tampered", task)
            with (tampered / "candidate/policy.py").open("a", encoding="utf-8") as stream:
                stream.write("\n# changed\n")
            with self.assertRaisesRegex(BundleError, "checksum"):
                import_candidate(tampered, task, root / "imports-c")


if __name__ == "__main__":
    unittest.main()
