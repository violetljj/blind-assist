from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import asset_catalog as catalog  # noqa: E402
import asset_runtime as runtime  # noqa: E402
import resource_fabric as fabric  # noqa: E402


def rule(kind: str, asset_class: str, *, exclude_children: list[str] | None = None) -> dict:
    value = {
        "asset_kind": kind,
        "asset_class": asset_class,
        "evidence_status": "diagnostic" if asset_class != "data" else "source_material",
        "storage_status": "shared",
        "owner": "asset-runtime-test",
        "retention_reason": "runtime fixture",
    }
    if exclude_children:
        value["exclude_children"] = exclude_children
    return value


class AssetRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="blindassist-asset-runtime-")
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts.local"
        self.database = self.artifacts / catalog.DEFAULT_DATABASE_RELATIVE
        self.policy = self.root / "asset-policy.json"
        self.policy.write_text(
            json.dumps(
                {
                    "schema": "blindassist-asset-management-policy-v1",
                    "catalog_unit": "direct_child",
                    "roots": {
                        "datasets": rule("dataset", "data"),
                        "evidence": rule(
                            "evidence",
                            "evidence",
                            exclude_children=["resource-fabric", "resource-store"],
                        ),
                        "work": rule("derived", "derived"),
                    },
                    "managed_assets": {},
                    "excluded_roots": {"tmp": "task-owned runtime staging"},
                    "top_level_file_exclude_globs": [],
                    "fallback": rule("legacy", "legacy"),
                }
            ),
            encoding="utf-8",
        )
        self.input = self.artifacts / "datasets" / "fixture" / "events.json"
        self.input.parent.mkdir(parents=True)
        self.input.write_text('{"episodes":[1,2,3]}', encoding="utf-8")
        self._discover()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _discover(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = catalog.main(
                [
                    "discover",
                    "--artifact-root",
                    str(self.artifacts),
                    "--database",
                    str(self.database),
                    "--policy",
                    str(self.policy),
                    "--repo-root",
                    str(self.root),
                ]
            )
        self.assertEqual(0, code, stderr.getvalue())

    def _write_spec(self, value: dict) -> Path:
        path = self.root / f"{value['id']}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _run(self, spec: dict) -> tuple[dict, int]:
        return runtime.run_spec(
            self._write_spec(spec),
            repo_root=self.root,
            artifact_root=self.artifacts,
            policy_path=self.policy,
        )

    def test_success_closes_asset_cache_hard_case_and_thin_result_loop(self) -> None:
        runner = self.root / "fixture_runner.py"
        runner.write_text(
            """from pathlib import Path
import json
import sys
source = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
Path(sys.argv[3]).write_text(json.dumps({'feature_count': len(source['episodes'])}), encoding='utf-8')
result = {
    'status': 'PASS',
    'episode_count': len(source['episodes']),
    'asset_lifecycle': {'hard_cases': [{
        'id': 'fixture-hard-case-v1',
        'failure_layer': 'fixture-selection',
        'selector': {'episode_ids': [2]},
        'observed_outcome': 'episode 2 retained for regression'
    }]}
}
Path(sys.argv[2]).write_text(json.dumps(result), encoding='utf-8')
""",
            encoding="utf-8",
        )
        experiment_result = (
            self.artifacts
            / "evidence"
            / "resource-fabric"
            / "experiments"
            / "dtr-r0"
            / "fixture-success"
            / "result.json"
        )
        feature_output = self.artifacts / "work" / "fixture-success" / "features.json"
        spec = {
            "schema": runtime.RUN_SCHEMA,
            "id": "fixture-success",
            "route": "dtr-r0",
            "question": "Does the runtime close the reusable asset loop?",
            "evaluator": "fixture_runner.py",
            "evidence_boundary": "Development fixture only.",
            "command": [
                sys.executable,
                str(runner),
                "{{input:events}}",
                "{{output:result}}",
                "{{output:features}}",
            ],
            "inputs": [
                {
                    "alias": "events",
                    "path": str(self.input),
                    "purpose": "fixture-event-stream",
                }
            ],
            "cache_inputs": [],
            "outputs": [
                {"alias": "result", "path": str(experiment_result), "role": "result"},
                {"alias": "features", "path": str(feature_output), "role": "features"},
            ],
            "result_output": "result",
            "cache_outputs": [
                {
                    "alias": "fixture-features-v1",
                    "output": "features",
                    "layer": "features",
                    "transform": "fixture-feature-extractor",
                    "transform_version": "v1",
                    "source_inputs": ["events"],
                    "mode": "hardlink",
                }
            ],
            "parameters": {"fixture": True},
        }

        result, code = self._run(spec)

        self.assertEqual(0, code)
        self.assertEqual("succeeded", result["state"])
        journal = fabric.read_json(self.artifacts / result["journal"])
        self.assertEqual("succeeded", journal["state"])
        self.assertEqual(1, len(result["created_caches"]))
        self.assertEqual(1, len(result["hard_cases"]))
        experiment_dir = experiment_result.parent
        self.assertEqual(
            fabric.ALLOWED_EXPERIMENT_FILES,
            {path.name for path in experiment_dir.iterdir() if path.is_file()},
        )
        manifest = fabric.read_json(experiment_dir / "manifest.json")
        self.assertEqual(1, len(manifest["asset_inputs"]))
        self.assertEqual(1, len(manifest["produced_cache_keys"]))
        cache = fabric.load_cache(self.artifacts, manifest["produced_cache_keys"][0])
        self.assertEqual(1, len(cache["asset_inputs"]))
        self.assertNotIn("alias", cache["asset_inputs"][0])
        hard_case = fabric.read_json(
            self.artifacts / result["hard_cases"][0]["hard_case"]
        )
        self.assertEqual(manifest["asset_inputs"][0]["asset_id"], hard_case["asset_inputs"][0]["asset_id"])
        connection = sqlite3.connect(self.database)
        try:
            usage_count = connection.execute(
                "SELECT COUNT(*) FROM usage_events WHERE consumer = 'fixture-success'"
            ).fetchone()[0]
            work_asset = connection.execute(
                "SELECT asset_id FROM assets WHERE locator = 'work/fixture-success'"
            ).fetchone()
            lineage = connection.execute(
                """
                SELECT COUNT(*) FROM derivation_inputs
                WHERE input_asset_id = ?
                """,
                (manifest["asset_inputs"][0]["asset_id"],),
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertGreaterEqual(usage_count, 1)
        self.assertIsNotNone(work_asset)
        self.assertEqual(1, lineage)
        with self.assertRaises(runtime.RuntimeError_):
            self._run(spec)

    def test_native_failure_is_terminal_and_reusable_without_rerun(self) -> None:
        runner = self.root / "failing_runner.py"
        runner.write_text("raise SystemExit(7)\n", encoding="utf-8")
        result_path = (
            self.artifacts
            / "evidence"
            / "resource-fabric"
            / "experiments"
            / "l10-r0"
            / "fixture-failure"
            / "result.json"
        )
        spec = {
            "schema": runtime.RUN_SCHEMA,
            "id": "fixture-failure",
            "route": "l10-r0",
            "question": "Is a failed run retained as reusable evidence?",
            "evaluator": "failing_runner.py",
            "evidence_boundary": "Execution receipt only.",
            "command": [sys.executable, str(runner), "{{input:events}}"],
            "inputs": [{"alias": "events", "path": str(self.input)}],
            "cache_inputs": [],
            "outputs": [{"alias": "result", "path": str(result_path), "role": "result"}],
            "result_output": "result",
            "cache_outputs": [],
            "parameters": {},
        }

        result, code = self._run(spec)

        self.assertEqual(7, code)
        self.assertEqual("failed", result["state"])
        self.assertTrue(result_path.is_file())
        self.assertEqual("FAILED_EXIT_7", fabric.read_json(result_path)["status"])
        self.assertEqual("failed", fabric.read_json(self.artifacts / result["journal"])["state"])
        self.assertEqual("execution", result["hard_cases"][0]["failure_layer"])
        with self.assertRaises(runtime.RuntimeError_):
            self._run(spec)


if __name__ == "__main__":
    unittest.main()
