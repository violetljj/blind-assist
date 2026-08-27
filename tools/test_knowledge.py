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


if __name__ == "__main__":
    unittest.main()
