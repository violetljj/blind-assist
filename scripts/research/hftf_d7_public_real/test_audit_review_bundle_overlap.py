import json
import tempfile
import unittest
from pathlib import Path

from audit_review_bundle_overlap import ROLES, run


class AuditReviewBundleOverlapTest(unittest.TestCase):
    def _write_bundle(self, root: Path, batch_id: str, rows: list[tuple[str, str]]) -> Path:
        batch = root / batch_id
        (batch / "manifests").mkdir(parents=True)
        candidate_ids = [candidate_id for candidate_id, _ in rows]
        (batch / "bundle_manifest.json").write_text(
            json.dumps({
                "batch_id": batch_id,
                "candidate_count": len(candidate_ids),
                "candidate_ids": candidate_ids,
            }),
            encoding="utf-8",
        )
        for role in ROLES:
            path = batch / "manifests" / f"{role}.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"candidate_id": candidate_id, "source_session_token": session}) + "\n"
                    for candidate_id, session in rows
                ),
                encoding="utf-8",
            )
        return batch

    def test_passes_when_candidate_and_session_sets_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self._write_bundle(root, "reference", [("a", "session-a")])
            candidate = self._write_bundle(root, "candidate", [("b", "session-b")])
            result = run(type("Args", (), {
                "audit_id": "audit-pass",
                "output_root": str(root),
                "reference_batch_root": str(reference),
                "candidate_batch_root": str(candidate),
            })())
            self.assertEqual(result["status"], "PASS_NO_MANIFEST_OVERLAP")
            self.assertEqual(result["candidate_overlap_count"], 0)
            self.assertEqual(result["source_session_identity_overlap_count"], 0)

    def test_fails_closed_on_candidate_or_session_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = self._write_bundle(root, "reference", [("a", "session-a")])
            candidate = self._write_bundle(root, "candidate", [("a", "session-a")])
            result = run(type("Args", (), {
                "audit_id": "audit-fail",
                "output_root": str(root),
                "reference_batch_root": str(reference),
                "candidate_batch_root": str(candidate),
            })())
            self.assertEqual(result["status"], "FAIL_OVERLAP")
            self.assertEqual(result["candidate_overlap_count"], 1)
            self.assertEqual(result["source_session_identity_overlap_count"], 1)


if __name__ == "__main__":
    unittest.main()
