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
                    "managed_assets": {
                        "runtime/carla-assets": rule(
                            "simulator_asset_library", "data"
                        )
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

    def configure_authority_policy(self) -> None:
        policy = json.loads(self.policy.read_text(encoding="utf-8"))
        consumed_reference = "artifacts." + "local/datasets/consumed"
        policy["roots"]["experiments"] = rule("experiment_record", "evidence")
        policy["authority_classification"] = {
            "schema": "blindassist-evidence-authority-classification-v1",
            "current_authority_sources": ["docs/CURRENT.md"],
            "outcome_reference_markers": [
                "result",
                "report",
                "terminal",
                "receipt",
            ],
            "artifact_assertion_text_extensions": [".json", ".md"],
            "artifact_assertion_max_bytes": 1024 * 1024,
            "protected_authority_tokens": ["sealed_final"],
            "current_outcome_reference_rule": {
                "rule_id": "current-outcome",
                "evidence_status": "development_consumed",
                "claim_ceiling": "DEVELOPMENT_ONLY",
                "reason": "current authority cites the outcome",
            },
            "artifact_assertion_rules": [
                {
                    "rule_id": "artifact-consumed",
                    "evidence_status": "development_consumed",
                    "claim_ceiling": "DEVELOPMENT_ONLY",
                    "reason": "artifact explicitly says Development-only",
                    "tokens": ["development_only", "consumed"],
                },
                {
                    "rule_id": "artifact-diagnostic",
                    "evidence_status": "diagnostic",
                    "claim_ceiling": "DIAGNOSTIC_ONLY",
                    "reason": "artifact explicitly says diagnostic-only",
                    "tokens": ["diagnostic_only"],
                },
            ],
            "static_type_rules": [
                {
                    "rule_id": "source-material",
                    "asset_classes": ["data"],
                    "evidence_status": "source_material",
                    "claim_ceiling": "SOURCE_MATERIAL_ONLY",
                    "reason": "data is source material",
                },
                {
                    "rule_id": "not-applicable",
                    "asset_classes": ["model"],
                    "evidence_status": "not_applicable",
                    "claim_ceiling": "NOT_APPLICABLE",
                    "reason": "models do not carry cohort authority",
                },
            ],
            "exact_rules": [
                {
                    "rule_id": "exact-consumed-dataset",
                    "locator": "datasets/consumed",
                    "evidence_status": "development_consumed",
                    "claim_ceiling": "DEVELOPMENT_ONLY",
                    "reason": "current authority records the exact consumed input",
                    "authority_source": "docs/CURRENT.md",
                    "authority_anchors": [
                        consumed_reference,
                        "DATASET_CONSUMED_TERMINAL",
                    ],
                }
            ],
        }
        self.policy.write_text(json.dumps(policy), encoding="utf-8")

    def test_zero_copy_discovery_hash_resolve_consume_and_derive(self) -> None:
        policy = json.loads(self.policy.read_text(encoding="utf-8"))
        copy_override = rule("dataset", "data")
        copy_override["evidence_status"] = "diagnostic"
        copy_override["owner"] = "copy-owner"
        policy["roots"]["datasets"]["asset_overrides"] = {
            "copy": copy_override
        }
        semantic_profile = self.root / "data" / "profiles" / "carla-test.json"
        semantic_profile.parent.mkdir(parents=True)
        semantic_profile.write_text(
            json.dumps(
                {
                    "schema": "blindassist-asset-semantic-profile-v1",
                    "profile_id": "carla-test-profile-v1",
                    "asset_locator": "runtime/carla-assets",
                    "title": "CARLA fixture",
                    "summary": "A fixture scene registry inside a managed runtime asset.",
                    "components": [
                        {
                            "component_key": "scene-registry",
                            "logical_name": "Fixture scene registry",
                            "path": "scene.json",
                            "component_kind": "scene_registry",
                            "data_role": "scenario_definition",
                            "evidence_status": "diagnostic",
                            "claim_ceiling": "FIXTURE_ONLY",
                            "description": "One scene definition used by the catalog test.",
                            "facts": {"scenes": 1},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        policy["managed_assets"]["runtime/carla-assets"]["semantic_profile"] = (
            "data/profiles/carla-test.json"
        )
        self.policy.write_text(json.dumps(policy), encoding="utf-8")
        (self.artifacts / "datasets" / "source").mkdir(parents=True)
        (self.artifacts / "datasets" / "copy").mkdir(parents=True)
        (self.artifacts / "datasets" / "source" / "a.txt").write_text("same", encoding="utf-8")
        (self.artifacts / "datasets" / "copy" / "a.txt").write_text("same", encoding="utf-8")
        (self.artifacts / "models").mkdir()
        (self.artifacts / "models" / "model.bin").write_bytes(b"model")
        (self.artifacts / "runtime").mkdir()
        (self.artifacts / "runtime" / "cache.bin").write_bytes(b"mutable")
        (self.artifacts / "runtime" / "carla-assets").mkdir()
        (self.artifacts / "runtime" / "carla-assets" / "scene.json").write_text(
            "{}", encoding="utf-8"
        )
        (self.artifacts / "custom" / "legacy").mkdir(parents=True)
        (self.artifacts / "custom" / "legacy" / "result.json").write_text("{}", encoding="utf-8")
        artifact_reference = "artifacts." + "local/datasets/source/a.txt"
        (self.root / "consumer.py").write_text(
            f'SOURCE = "{artifact_reference}"\n', encoding="utf-8"
        )

        discovered = self.discover()
        self.assertEqual("PASS", discovered["status"])
        self.assertEqual(1, discovered["managed_assets_seen"])
        self.assertEqual(1, discovered["semantic_components_seen"])
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

        components = self.run_catalog("components", "runtime/carla-assets")
        self.assertEqual("carla-test-profile-v1", components["profile"]["profile_id"])
        self.assertEqual("scene-registry", components["components"][0]["component_key"])
        component = self.run_catalog(
            "resolve",
            "runtime/carla-assets#scene-registry",
            "--consumer",
            "fixture-scene-consumer",
            "--purpose",
            "scenario-definition",
        )
        self.assertEqual(
            str(self.artifacts / "runtime" / "carla-assets" / "scene.json"),
            component["path"],
        )
        self.assertEqual("scene-registry", component["component"]["component_key"])
        self.assertEqual("diagnostic", component["evidence_status"])

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
            copy_lifecycle = connection.execute(
                """
                SELECT evidence_status, owner FROM assets
                WHERE locator = 'datasets/copy'
                """
            ).fetchone()
            runtime_assets = connection.execute(
                "SELECT COUNT(*) FROM assets WHERE root_name = 'runtime'"
            ).fetchone()[0]
            managed_runtime = connection.execute(
                """
                SELECT state, file_count, metadata_json FROM assets
                WHERE locator = 'runtime/carla-assets'
                """
            ).fetchone()
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
        self.assertEqual(("diagnostic", "copy-owner"), copy_lifecycle)
        self.assertIsNotNone(file_hash)
        self.assertEqual(1, runtime_assets)
        self.assertEqual("present", managed_runtime[0])
        self.assertEqual(1, managed_runtime[1])
        self.assertEqual(
            "policy-managed-excluded-root",
            json.loads(managed_runtime[2])["discovery"],
        )

        report_dir = self.root / "report"
        report = self.run_catalog("report", "--output-dir", str(report_dir))
        self.assertGreaterEqual(report["assets"], 5)
        self.assertGreaterEqual(report["usage_events"], 2)
        self.assertEqual(1, report["derivations"])
        self.assertEqual(1, report["semantic_profiles"])
        self.assertEqual(1, report["semantic_components"])
        self.assertEqual(1, report["resolved_repository_references"])
        verification = self.run_catalog(
            "verify", "--repo-root", str(self.root), "--deep"
        )
        self.assertEqual("PASS", verification["status"])
        self.assertEqual(2, verification["deep_checked"])
        self.assertEqual(1, verification["semantic_profiles_checked"])
        self.assertEqual(1, verification["semantic_components_checked"])

    def test_reference_scan_normalizes_separators_and_accepts_present_excluded_paths(
        self,
    ) -> None:
        source = self.artifacts / "datasets" / "source"
        source.mkdir(parents=True)
        (source / "a.txt").write_text("source", encoding="utf-8")
        runtime_models = self.artifacts / "runtime" / "semantic-anchor-v1" / "models"
        runtime_models.mkdir(parents=True)
        (runtime_models / "ocr.bin").write_bytes(b"runtime")
        historical = self.artifacts / "evidence" / "device-regression" / "run-1"
        historical.mkdir(parents=True)
        (historical / "result.json").write_text("{}", encoding="utf-8")
        reference_prefix = "artifacts." + "local/"
        test_reference_prefix = "test-artifacts." + "local/"
        (self.root / "consumer.py").write_text(
            "\n".join(
                (
                    f'SOURCE = "{reference_prefix}datasets//source//a.txt"',
                    f'RUNTIME = "{reference_prefix}runtime/semantic-anchor-v1/models"',
                    f'MISSING = "{reference_prefix}runtime/not-installed"',
                    f'HISTORICAL = "{test_reference_prefix}device-regression/run-1/result.json"',
                )
            ),
            encoding="utf-8",
        )

        discovered = self.discover()

        self.assertEqual(2, discovered["repository_references"]["resolved_assets"])
        self.assertEqual(1, discovered["repository_references"]["root_scoped"])
        self.assertEqual(1, discovered["repository_references"]["missing_within_root"])
        connection = sqlite3.connect(self.database)
        try:
            rows = connection.execute(
                """
                SELECT raw_locator, resolution FROM asset_references
                ORDER BY line_number
                """
            ).fetchall()
        finally:
            connection.close()
        self.assertEqual(
            [
                ("datasets//source//a.txt", "asset"),
                ("runtime/semantic-anchor-v1/models", "root"),
                ("runtime/not-installed", "missing_within_root"),
                ("device-regression/run-1/result.json", "asset"),
            ],
            rows,
        )

    def test_authority_classification_is_auditable_and_fail_closed(self) -> None:
        self.configure_authority_policy()
        for locator, contents in (
            ("datasets/raw/source.txt", "raw"),
            ("datasets/consumed/source.txt", "opened"),
            ("models/weights/model.bin", "weights"),
            ("evidence/current/result.json", "{}"),
            ("evidence/asserted/report.json", '{"authority":"DEVELOPMENT_ONLY"}'),
            ("evidence/diagnostic/terminal-receipt.json", '{"authority":"DIAGNOSTIC_ONLY"}'),
            ("evidence/unresolved/notes.json", "{}"),
            ("experiments/open/result.json", '{"authority":"DEVELOPMENT_ONLY"}'),
        ):
            path = self.artifacts.joinpath(*locator.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        (self.artifacts / "evidence" / "empty").mkdir(parents=True)
        authority_reference = "artifacts." + "local/evidence/current/result.json"
        consumed_reference = "artifacts." + "local/datasets/consumed"
        current = self.root / "docs" / "CURRENT.md"
        current.parent.mkdir(parents=True)
        current.write_text(
            "\n".join(
                (
                    "# Current authority",
                    f"Outcome: {authority_reference}",
                    f"Input: {consumed_reference}",
                    "DATASET_CONSUMED_TERMINAL",
                )
            ),
            encoding="utf-8",
        )
        self.discover()

        dry_run = self.run_catalog(
            "classify-authority",
            "--policy",
            str(self.policy),
            "--repo-root",
            str(self.root),
        )
        self.assertEqual("DRY_RUN", dry_run["status"])
        self.assertEqual(7, dry_run["classified"])
        self.assertEqual(9, dry_run["unknown_before"])
        self.assertEqual(2, dry_run["projected_unknown_after"])

        applied = self.run_catalog(
            "classify-authority",
            "--policy",
            str(self.policy),
            "--repo-root",
            str(self.root),
            "--apply",
        )
        self.assertEqual("APPLIED", applied["status"])
        self.assertEqual(7, applied["applied"])
        self.assertEqual(2, applied["unknown_after"])
        self.assertEqual(
            {
                "development_consumed": 4,
                "diagnostic": 1,
                "not_applicable": 1,
                "source_material": 1,
            },
            applied["classified_by_status"],
        )

        connection = sqlite3.connect(self.database)
        try:
            statuses = dict(connection.execute(
                "SELECT locator, evidence_status FROM assets ORDER BY locator"
            ))
            classification_rows = connection.execute(
                """
                SELECT rule_id, source, reason, evidence_json
                FROM authority_classifications
                ORDER BY classification_id
                """
            ).fetchall()
            usage_events = connection.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual("source_material", statuses["datasets/raw"])
        self.assertEqual("development_consumed", statuses["datasets/consumed"])
        self.assertEqual("not_applicable", statuses["models/weights"])
        self.assertEqual("development_consumed", statuses["evidence/current"])
        self.assertEqual("development_consumed", statuses["evidence/asserted"])
        self.assertEqual("diagnostic", statuses["evidence/diagnostic"])
        self.assertEqual("unknown", statuses["evidence/unresolved"])
        self.assertEqual("unknown", statuses["evidence/empty"])
        self.assertEqual(7, len(classification_rows))
        self.assertTrue(all(all(value for value in row) for row in classification_rows))
        self.assertEqual(0, usage_events)

        report_dir = self.root / "report"
        report = self.run_catalog("report", "--output-dir", str(report_dir))
        self.assertEqual(7, report["authority_classification_events"])
        self.assertEqual(2, report["remaining_authority_queue_assets"])
        queue = json.loads(
            Path(report["authority_queue_json"]).read_text(encoding="utf-8")
        )
        self.assertEqual(2, queue["assets"])
        self.assertEqual(
            {
                "empty_asset_no_authority_evidence",
                "evidence_authority_not_cited_or_asserted",
            },
            set(queue["groups"]),
        )
        self.assertTrue(Path(report["authority_queue_markdown"]).is_file())

        repeated = self.run_catalog(
            "classify-authority",
            "--policy",
            str(self.policy),
            "--repo-root",
            str(self.root),
            "--apply",
        )
        self.assertEqual(0, repeated["applied"])
        rejected = self.run_catalog_failure(
            "transition",
            "evidence/current",
            "--evidence-status",
            "fresh",
            "--reason",
            "illegal reset",
        )
        self.assertIn("cannot regain", rejected)

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
        access_stdout = io.StringIO()
        with contextlib.redirect_stdout(access_stdout), contextlib.redirect_stderr(
            fabric_stderr
        ):
            code = fabric.main(
                [
                    "--artifact-root",
                    str(self.artifacts),
                    "cache-use",
                    cache["cache_key"],
                    "--event-id",
                    "fixture-cache-hit",
                    "--consumer",
                    "fixture-cache-consumer",
                    "--purpose",
                    "normalized-input",
                    "--experiment-id",
                    "fixture-cache-experiment",
                ]
            )
        self.assertEqual(0, code, fabric_stderr.getvalue())

        discovered = self.discover()
        self.assertEqual(1, discovered["resource_fabric"]["resources"])
        self.assertEqual(1, discovered["resource_fabric"]["caches"])
        self.assertEqual(2, discovered["resource_fabric"]["uses"])
        resolved = self.run_catalog("resolve", resource["resource_id"])
        self.assertEqual("content", resolved["identity_strength"])
        self.assertEqual("development_consumed", resolved["evidence_status"])
        cache_resolved = self.run_catalog("resolve", f"cache:{cache['cache_key']}")
        self.assertEqual("shared", cache_resolved["storage_status"])
        connection = sqlite3.connect(self.database)
        try:
            usage = connection.execute(
                """
                SELECT access_mode, consumer, purpose, experiment_id
                FROM usage_events
                WHERE event_id LIKE 'fabric-cache-access:%'
                """
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(
            (
                "cache_hit",
                "fixture-cache-consumer",
                "normalized-input",
                "fixture-cache-experiment",
            ),
            usage,
        )
        verification = self.run_catalog("verify", "--deep")
        self.assertEqual("PASS", verification["status"])
        self.assertEqual(2, verification["deep_checked"])


if __name__ == "__main__":
    unittest.main()
