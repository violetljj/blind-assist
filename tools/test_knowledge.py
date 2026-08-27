from __future__ import annotations

import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from tools import knowledge


class KnowledgeCliTest(unittest.TestCase):
    def run_cli(self, root: Path, *arguments: str) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = knowledge.main(["--root", str(root), *arguments])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_item_use_update_lifecycle(self) -> None:
        with TemporaryDirectory(prefix="blindassist-knowledge-test-") as temporary:
            root = Path(temporary) / "research" / "knowledge"
            (root / "items").mkdir(parents=True)
            (root / "uses").mkdir()

            result, _, stderr = self.run_cli(root, "validate")
            self.assertEqual(0, result, stderr)

            result, _, stderr = self.run_cli(
                root,
                "new-item",
                "--id",
                "project-smoke",
                "--kind",
                "project",
                "--title",
                "Smoke project",
                "--canonical-ref",
                "https://example.org/project",
                "--summary",
                "Temporary CLI smoke item.",
                "--mechanism-id",
                "smoke-mechanism",
                "--mechanism-name",
                "Smoke mechanism",
                "--mechanism-description",
                "Temporary mechanism.",
                "--mechanism-input",
                "input",
                "--mechanism-output",
                "output",
                "--mechanism-limitations",
                "Temporary only.",
                "--tag",
                "smoke",
            )
            self.assertEqual(0, result, stderr)

            result, _, stderr = self.run_cli(
                root,
                "new-use",
                "--id",
                "use-smoke-route",
                "--item",
                "project-smoke",
                "--route",
                "smoke-route",
                "--mechanism",
                "smoke-mechanism",
                "--source-scope",
                "Only the temporary mechanism.",
                "--project-application",
                "Exercise the update path.",
                "--modifications",
                "No functional modification.",
                "--expected-effect",
                "A valid mutable record.",
                "--claim-boundary",
                "No project claim.",
            )
            self.assertEqual(0, result, stderr)

            result, _, stderr = self.run_cli(
                root,
                "update-use",
                "use-smoke-route",
                "--state",
                "active",
                "--reproduction",
                "partial",
                "--verdict",
                "mixed",
                "--effect",
                "Mutation path completed.",
                "--metric",
                "smoke=pass",
                "--evidence",
                "external",
                "https://example.org/evidence",
                "Temporary external evidence.",
                "--note",
                "Exercised atomic route-use update.",
            )
            self.assertEqual(0, result, stderr)

            result, output, stderr = self.run_cli(root, "validate")
            self.assertEqual(0, result, stderr)
            self.assertIn("items=1 uses=1 routes=1", output)

            use = json.loads(
                (root / "uses" / "use-smoke-route.json").read_text(encoding="utf-8")
            )
            self.assertEqual("active", use["use_state"])
            self.assertEqual("partial", use["evaluation"]["reproduction_status"])
            self.assertEqual("mixed", use["evaluation"]["verdict"])
            self.assertEqual(["smoke=pass"], use["evaluation"]["metrics"])
            self.assertEqual("external", use["evidence"][0]["kind"])
            self.assertEqual(2, len(use["history"]))

            result, output, stderr = self.run_cli(
                root, "context", "--route", "smoke-route", "--json"
            )
            self.assertEqual(0, result, stderr)
            context = json.loads(output)
            self.assertEqual("smoke-route", context["route"])
            self.assertEqual(1, context["summary"]["route_total_uses"])
            self.assertEqual(1, context["summary"]["returned_uses"])
            self.assertEqual("project-smoke", context["entries"][0]["item"]["id"])
            self.assertEqual(
                "mixed",
                context["entries"][0]["use"]["evaluation"]["verdict"],
            )

            candidate_use = json.loads(json.dumps(use))
            candidate_use["id"] = "use-smoke-route-candidate"
            candidate_use["use_state"] = "candidate"
            candidate_use["usage"]["project_application"] = (
                "Candidate-only context phrase."
            )
            candidate_use["evaluation"].update(
                {
                    "reproduction_status": "not_attempted",
                    "verdict": "not_run",
                    "setup": "",
                    "effect": "",
                    "metrics": [],
                }
            )
            candidate_use["history"] = [
                {
                    "date": "2026-08-28",
                    "change": "Added a lower-priority route candidate.",
                }
            ]
            (root / "uses" / "use-smoke-route-candidate.json").write_text(
                json.dumps(candidate_use, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result, output, stderr = self.run_cli(
                root,
                "context",
                "--route",
                "smoke-route",
                "--limit",
                "1",
                "--json",
            )
            self.assertEqual(0, result, stderr)
            context = json.loads(output)
            self.assertEqual(2, context["summary"]["matched_uses"])
            self.assertEqual(1, context["summary"]["returned_uses"])
            self.assertEqual(1, context["summary"]["omitted_uses"])
            self.assertEqual("active", context["entries"][0]["use"]["use_state"])

            result, output, stderr = self.run_cli(
                root,
                "context",
                "--route",
                "smoke-route",
                "--query",
                "candidate-only",
                "--all",
                "--json",
            )
            self.assertEqual(0, result, stderr)
            context = json.loads(output)
            self.assertEqual(1, context["summary"]["matched_uses"])
            self.assertEqual(
                "use-smoke-route-candidate",
                context["entries"][0]["use"]["id"],
            )

            result, output, stderr = self.run_cli(
                root, "context", "--route", "smoke-route", "--limit", "0"
            )
            self.assertEqual(2, result)
            self.assertIn("--limit must be at least 1", stderr)

    def test_context_limit_keeps_a_representative_from_each_present_tier(self) -> None:
        entries = [
            {
                "item": {"title": "Active A"},
                "use": {
                    "id": "use-active-a",
                    "use_state": "active",
                    "evaluation": {"verdict": "not_run"},
                },
            },
            {
                "item": {"title": "Active B"},
                "use": {
                    "id": "use-active-b",
                    "use_state": "active",
                    "evaluation": {"verdict": "not_run"},
                },
            },
            {
                "item": {"title": "Candidate"},
                "use": {
                    "id": "use-candidate",
                    "use_state": "candidate",
                    "evaluation": {"verdict": "not_run"},
                },
            },
        ]
        entries.sort(key=knowledge._context_sort_key)
        selected = knowledge._select_context_entries(entries, 2)
        self.assertEqual(
            ["use-active-a", "use-candidate"],
            [entry["use"]["id"] for entry in selected],
        )

    def test_migration_receipt_requires_resolvable_alias_and_use(self) -> None:
        with TemporaryDirectory(prefix="blindassist-knowledge-migration-test-") as temporary:
            root = Path(temporary) / "research" / "knowledge"
            (root / "items").mkdir(parents=True)
            (root / "uses").mkdir()
            (root / "migrations").mkdir()

            result, _, stderr = self.run_cli(
                root,
                "new-item",
                "--id",
                "paper-migrated",
                "--kind",
                "paper",
                "--title",
                "Migrated paper",
                "--canonical-ref",
                "https://example.org/migrated",
                "--summary",
                "A migrated source.",
                "--mechanism-id",
                "migrated-mechanism",
                "--mechanism-name",
                "Migrated mechanism",
                "--mechanism-description",
                "A migrated mechanism.",
                "--mechanism-limitations",
                "No project claim.",
                "--alias",
                "legacy:P01",
            )
            self.assertEqual(0, result, stderr)
            result, _, stderr = self.run_cli(
                root,
                "new-use",
                "--id",
                "use-migrated-route",
                "--item",
                "paper-migrated",
                "--route",
                "migrated-route",
                "--mechanism",
                "migrated-mechanism",
                "--source-scope",
                "The migrated mechanism only.",
                "--project-application",
                "Retain the legacy route hypothesis.",
                "--modifications",
                "No implementation was imported.",
                "--expected-effect",
                "Make the old judgment searchable.",
                "--claim-boundary",
                "Migration is not reproduction.",
            )
            self.assertEqual(0, result, stderr)

            manifest = {
                "schema_version": 1,
                "id": "migration-smoke",
                "created_at": "2026-08-27",
                "scope": "Migration validator smoke test.",
                "source_groups": [
                    {
                        "id": "legacy-smoke",
                        "source_ref": "git:test:legacy.md",
                        "source_sha256": None,
                        "expected_entries": 1,
                        "mappings": [
                            {
                                "legacy_id": "legacy:P01",
                                "item_id": "paper-migrated",
                                "use_ids": ["use-migrated-route"],
                                "disposition": "migrated",
                                "note": "Legacy row is represented by one item and one use.",
                            }
                        ],
                    }
                ],
                "exclusions": [
                    {
                        "scope": "Incidental citations",
                        "reason": "They are not curated knowledge records.",
                    }
                ],
            }
            (root / "migrations" / "migration-smoke.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result, output, stderr = self.run_cli(root, "validate")
            self.assertEqual(0, result, stderr)
            self.assertIn("migrations=1 legacy_mappings=1", output)

            manifest["source_groups"][0]["mappings"][0]["legacy_id"] = "legacy:P02"
            (root / "migrations" / "migration-smoke.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result, _, stderr = self.run_cli(root, "validate")
            self.assertEqual(1, result)
            self.assertIn("does not carry alias legacy:P02", stderr)


if __name__ == "__main__":
    unittest.main()
