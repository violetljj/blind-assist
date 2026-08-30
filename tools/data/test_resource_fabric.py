from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import resource_fabric as fabric  # noqa: E402


class ResourceFabricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="blindassist-resource-fabric-")
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts.local"
        self.source = self.root / "source.json"
        self.source.write_text(
            json.dumps({"episodes": [2, 1], "status": "development"}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_cli(self, *arguments: str) -> dict:
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["--artifact-root", str(self.artifacts), *arguments]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = fabric.main(argv)
        self.assertEqual(0, code, stderr.getvalue())
        return json.loads(stdout.getvalue())

    def run_cli_failure(self, *arguments: str) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["--artifact-root", str(self.artifacts), *arguments]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = fabric.main(argv)
        self.assertEqual(2, code, stdout.getvalue())
        return stderr.getvalue()

    def ingest(self, name: str) -> dict:
        return self.run_cli(
            "ingest",
            str(self.source),
            "--name",
            name,
            "--kind",
            "data",
            "--storage-class",
            "sealed",
            "--route",
            "l10-r0",
            "--consumer",
            "fixture-evaluator",
            "--evidence-role",
            "public-metadata",
            "--dataset-id",
            "fixture-dataset-v1",
            "--source-unit-id",
            "fixture-unit-1",
            "--disjoint-key",
            "fixture-parent-1",
            "--owner",
            "resource-fabric-test",
            "--retention-reason",
            "end-to-end fixture",
            "--evidence-status",
            "fresh",
            "--storage-status",
            "active",
            "--reason",
            "fixture admitted",
            "--allowed-use",
            "development_replay",
            "--forbidden-use",
            "fresh_confirmation",
        )

    def test_end_to_end_reuse_loop_and_thin_experiment(self) -> None:
        first = self.ingest("fixture-source")
        second = self.ingest("fixture-source-alias")
        self.assertEqual(first["resource_id"], second["resource_id"])
        self.assertTrue(second["object_reused"])

        cache = self.run_cli("cache-json", first["resource_id"])
        first_cache_use = self.run_cli(
            "cache-use",
            cache["cache_key"],
            "--event-id",
            "fixture-evaluator-cache-hit",
            "--consumer",
            "fixture-experiment-v1",
            "--purpose",
            "normalized-input",
            "--experiment-id",
            "fixture-experiment-v1",
        )
        second_cache_use = self.run_cli(
            "cache-use",
            cache["cache_key"],
            "--event-id",
            "fixture-diagnostic-cache-hit",
            "--consumer",
            "fixture-diagnostic-v1",
            "--purpose",
            "failure-analysis",
        )
        self.assertEqual("cache_hit", first_cache_use["outcome"])
        self.assertEqual(cache["payload_bytes"], first_cache_use["payload_bytes"])
        self.assertNotEqual(
            first_cache_use["event_id"], second_cache_use["event_id"]
        )
        hard_case = self.run_cli(
            "hard-case",
            "--id",
            "missing-pixels-v1",
            "--route",
            "l10-r0",
            "--case-kind",
            "evidence_gap",
            "--failure-layer",
            "observation-payload",
            "--evidence-split",
            "development",
            "--source-id",
            first["resource_id"],
            "--cache-key",
            cache["cache_key"],
            "--selector-json",
            '{"episode_ids":[1,2]}',
            "--truth-authority",
            "metadata-only",
            "--selected-by",
            "fixture-selector-v1",
            "--observed-outcome",
            "pixel inference not run",
            "--claim-ceiling",
            "Development metadata mechanics only",
        )
        self.assertEqual("evidence_gap", hard_case["case_kind"])

        experiment = self.run_cli(
            "experiment-create",
            "--id",
            "fixture-experiment-v1",
            "--route",
            "l10-r0",
            "--question",
            "Can normalized metadata drive the action adapter?",
            "--evaluator",
            "fixture-evaluator.py",
            "--source-id",
            first["resource_id"],
            "--cache-key",
            cache["cache_key"],
            "--hard-case",
            "missing-pixels-v1",
            "--parameters-json",
            '{"episode_count":2}',
            "--boundary",
            "Development metadata mechanics only; no pixel or confirmation claim.",
        )
        result_source = self.root / "result.json"
        result_source.write_text('{"status":"NOT_RUN_INPUT_PENDING"}', encoding="utf-8")
        self.run_cli(
            "experiment-finalize",
            "--id",
            "fixture-experiment-v1",
            "--route",
            "l10-r0",
            "--result-json",
            str(result_source),
            "--status",
            "NOT_RUN_INPUT_PENDING",
        )
        self.run_cli(
            "transition",
            first["resource_id"],
            "--event-id",
            "fixture-development-consumed",
            "--evidence-status",
            "development_consumed",
            "--storage-status",
            "shared",
            "--reason",
            "fixture experiment completed",
            "--experiment-id",
            "fixture-experiment-v1",
        )

        verification = self.run_cli("verify", "--deep")
        self.assertEqual("PASS", verification["status"])
        report = self.run_cli("report", "--inventory-root", str(self.artifacts))
        self.assertEqual(1, report["unique_resources"])
        self.assertEqual(2, report["registrations"])
        self.assertEqual(1, report["shared_caches"])
        self.assertEqual(2, report["cache_access_events"])
        self.assertEqual(2, report["cache_access_consumers"])
        self.assertEqual(1, report["caches_with_recorded_reuse"])
        self.assertEqual(1, report["multi_consumer_caches"])
        self.assertEqual(2 * cache["payload_bytes"], report["avoided_recompute_bytes"])
        self.assertEqual(1, report["hard_cases"])
        self.assertEqual(1, report["thin_experiments"])
        self.assertEqual(0, report["unreferenced_resources"])

        experiment_dir = self.artifacts / experiment["experiment"]
        self.assertEqual(
            fabric.ALLOWED_EXPERIMENT_FILES,
            {path.name for path in experiment_dir.iterdir() if path.is_file()},
        )
        latest = fabric.current_lifecycle(self.artifacts)[first["resource_id"]]
        self.assertEqual("development_consumed", latest["evidence_status"])
        self.assertEqual("shared", latest["storage_status"])
        self.assertIn("training", latest["allowed_uses"])
        self.assertIn("fresh_confirmation", latest["forbidden_uses"])

        rejected = self.run_cli_failure(
            "transition",
            first["resource_id"],
            "--event-id",
            "illegal-fresh-reset",
            "--evidence-status",
            "fresh",
            "--storage-status",
            "active",
            "--reason",
            "attempt to reset consumed authority",
        )
        self.assertIn("cannot regain another evidence status", rejected)

        replayed = self.ingest("fixture-source")
        self.assertTrue(replayed["object_reused"])
        latest = fabric.current_lifecycle(self.artifacts)[first["resource_id"]]
        self.assertEqual("development_consumed", latest["evidence_status"])

    def test_directory_content_identity_is_path_independent(self) -> None:
        first_tree = self.root / "tree-a"
        second_tree = self.root / "tree-b"
        (first_tree / "nested").mkdir(parents=True)
        (second_tree / "nested").mkdir(parents=True)
        (first_tree / "nested" / "a.txt").write_text("same", encoding="utf-8")
        (second_tree / "nested" / "a.txt").write_text("same", encoding="utf-8")
        first = fabric.scan_payload(first_tree)
        second = fabric.scan_payload(second_tree)
        self.assertEqual(first["resource_id"], second["resource_id"])
        self.assertTrue(first["resource_id"].startswith("tree-sha256:"))


if __name__ == "__main__":
    unittest.main()
