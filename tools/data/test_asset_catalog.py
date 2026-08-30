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
import resource_fabric as fabric  # noqa: E402


def rule(kind: str, asset_class: str, *, exclude_children: list[str] | None = None) -> dict:
    value = {
        "asset_kind": kind,
        "asset_class": asset_class,
        "evidence_status": "unknown",
        "storage_status": "shared",
        "owner": "asset-catalog-test",
        "retention_reason": "test fixture",
    }
    if exclude_children:
        value["exclude_children"] = exclude_children
    return value


class AssetCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="blindassist-asset-catalog-")
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts.local"
        self.database = (
            self.artifacts
            / "evidence"
            / "resource-fabric"
            / "catalog"
            / "master-assets.sqlite3"
        )
        self.policy = self.root / "policy.json"
        self.policy.write_text(
            json.dumps(
                {
                    "schema": "blindassist-asset-management-policy-v1",
                    "catalog_unit": "direct_child",
                    "roots": {
                        "datasets": rule("dataset", "data"),
                        "models": rule("model", "model", exclude_children=["resource-store"]),
                        "evidence": rule(
                            "evidence",
                            "evidence",
                            exclude_children=["resource-fabric", "resource-store"],
                        ),
                    },
                    "excluded_roots": {"runtime": "mutable runtime"},
                    "top_level_file_exclude_globs": ["*.log", "temp_*"],
                    "fallback": rule("legacy", "legacy_unclassified"),
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_catalog(self, *arguments: str) -> dict:
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            arguments[0],
            "--artifact-root",
            str(self.artifacts),
            "--database",
            str(self.database),
            *arguments[1:],
        ]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = catalog.main(argv)
        self.assertEqual(0, code, stderr.getvalue())
        return json.loads(stdout.getvalue())

    def run_catalog_failure(self, *arguments: str) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            arguments[0],
            "--artifact-root",
            str(self.artifacts),
            "--database",
            str(self.database),
            *arguments[1:],
        ]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = catalog.main(argv)
        self.assertEqual(2, code, stdout.getvalue())
        return stderr.getvalue()

    def discover(self) -> dict:
        return self.run_catalog(
            "discover",
            "--policy",
            str(self.policy),
            "--repo-root",
            str(self.root),
        )

    def test_zero_copy_discovery_hash_resolve_consume_and_derive(self) -> None:
        (self.artifacts / "datasets" / "source").mkdir(parents=True)
        (self.artifacts / "datasets" / "copy").mkdir(parents=True)
        (self.artifacts / "datasets" / "source" / "a.txt").write_text("same", encoding="utf-8")
        (self.artifacts / "datasets" / "copy" / "a.txt").write_text("same", encoding="utf-8")
        (self.artifacts / "models").mkdir()
        (self.artifacts / "models" / "model.bin").write_bytes(b"model")
        (self.artifacts / "runtime").mkdir()
        (self.artifacts / "runtime" / "cache.bin").write_bytes(b"mutable")
        (self.artifacts / "custom" / "legacy").mkdir(parents=True)
        (self.artifacts / "custom" / "legacy" / "result.json").write_text("{}", encoding="utf-8")
        (self.root / "consumer.py").write_text(
            'SOURCE = "artifacts.local/datasets/source/a.txt"\n',
            encoding="utf-8",
        )

        discovered = self.discover()
        self.assertEqual("PASS", discovered["status"])
        self.assertEqual(1, discovered["repository_references"]["resolved_assets"])
        self.assertFalse((self.artifacts / "datasets" / "source").is_symlink())

        first_hash = self.run_catalog("hash", "datasets/source")
        second_hash = self.run_catalog("hash", "datasets/copy")
        self.assertEqual(first_hash["content_id"], second_hash["content_id"])
        self.assertEqual(1, len(second_hash["duplicate_assets"]))

        resolved = self.run_catalog(
            "resolve",
            "datasets/source",
            "--consumer",
            "fixture-experiment-v1",
            "--purpose",
            "training-input",
            "--experiment-id",
            "fixture-experiment-v1",
            "--evidence-effect",
            "development_consumed",
        )
        self.assertEqual("development_consumed", resolved["evidence_status"])
        rejected = self.run_catalog_failure(
            "transition",
            "datasets/source",
            "--evidence-status",
            "fresh",
            "--reason",
            "illegal reset",
        )
        self.assertIn("cannot regain", rejected)

        output = self.artifacts / "datasets" / "output"
        output.mkdir()
        (output / "derived.txt").write_text("derived", encoding="utf-8")
        self.discover()
        derived = self.run_catalog(
            "derive",
            "--output",
            "datasets/output",
            "--input",
            "datasets/source",
            "--transform",
            "fixture-transform",
            "--transform-version",
            "v1",
            "--producer",
            "test_asset_catalog.py",
            "--parameters-json",
            '{"frozen":true}',
        )
        self.assertTrue(derived["derivation_id"].startswith("derive:"))

        connection = sqlite3.connect(self.database)
        try:
            source = connection.execute(
                "SELECT identity_strength, content_id FROM assets WHERE locator = 'datasets/source'"
            ).fetchone()
            runtime_assets = connection.execute(
                "SELECT COUNT(*) FROM assets WHERE root_name = 'runtime'"
            ).fetchone()[0]
            file_hash = connection.execute(
                """
                SELECT f.sha256 FROM asset_files f JOIN assets a ON a.asset_id = f.asset_id
                WHERE a.locator = 'datasets/source'
                """
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual("content", source[0])
        self.assertIsNotNone(source[1])
        self.assertIsNotNone(file_hash)
        self.assertEqual(0, runtime_assets)

        report_dir = self.root / "report"
        report = self.run_catalog("report", "--output-dir", str(report_dir))
        self.assertGreaterEqual(report["assets"], 5)
        self.assertGreaterEqual(report["usage_events"], 2)
        self.assertEqual(1, report["derivations"])
        self.assertEqual(1, report["resolved_repository_references"])
        verification = self.run_catalog("verify", "--deep")
        self.assertEqual("PASS", verification["status"])
        self.assertEqual(2, verification["deep_checked"])

    def test_resource_fabric_is_imported_into_master_catalog(self) -> None:
        source = self.root / "source.json"
        source.write_text('{"episodes":[1,2]}', encoding="utf-8")
        fabric_stdout = io.StringIO()
        fabric_stderr = io.StringIO()
        with contextlib.redirect_stdout(fabric_stdout), contextlib.redirect_stderr(fabric_stderr):
            code = fabric.main(
                [
                    "--artifact-root",
                    str(self.artifacts),
                    "ingest",
                    str(source),
                    "--name",
                    "fixture-resource",
                    "--kind",
                    "data",
                    "--storage-class",
                    "sealed",
                    "--route",
                    "fixture",
                    "--consumer",
                    "fixture-consumer",
                    "--evidence-role",
                    "development-input",
                    "--owner",
                    "asset-catalog-test",
                    "--retention-reason",
                    "fabric integration fixture",
                    "--evidence-status",
                    "development_consumed",
                    "--storage-status",
                    "shared",
                    "--reason",
                    "fixture consumed",
                ]
            )
        self.assertEqual(0, code, fabric_stderr.getvalue())
        resource = json.loads(fabric_stdout.getvalue())
        cache_stdout = io.StringIO()
        with contextlib.redirect_stdout(cache_stdout), contextlib.redirect_stderr(fabric_stderr):
            code = fabric.main(
                [
                    "--artifact-root",
                    str(self.artifacts),
                    "cache-json",
                    resource["resource_id"],
                ]
            )
        self.assertEqual(0, code, fabric_stderr.getvalue())
        cache = json.loads(cache_stdout.getvalue())

        discovered = self.discover()
        self.assertEqual(1, discovered["resource_fabric"]["resources"])
        self.assertEqual(1, discovered["resource_fabric"]["caches"])
        resolved = self.run_catalog("resolve", resource["resource_id"])
        self.assertEqual("content", resolved["identity_strength"])
        self.assertEqual("development_consumed", resolved["evidence_status"])
        cache_resolved = self.run_catalog("resolve", f"cache:{cache['cache_key']}")
        self.assertEqual("shared", cache_resolved["storage_status"])
        verification = self.run_catalog("verify", "--deep")
        self.assertEqual("PASS", verification["status"])
        self.assertEqual(2, verification["deep_checked"])


if __name__ == "__main__":
    unittest.main()
