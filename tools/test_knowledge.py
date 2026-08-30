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
                "--applicability",
                "Only the scoped smoke lifecycle with declared mechanism input and output.",
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
            self.assertIn("smoke lifecycle", use["usage"]["applicability"])
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
                root,
                "context",
                "--route",
                "smoke-route",
                "--query",
                "phrase context candidate-only",
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
            self.assertIn(
                "query relevance",
                context["summary"]["selection_policy"],
            )

            result, output, stderr = self.run_cli(
                root,
                "search",
                "context candidate-only phrase",
                "--json",
            )
            self.assertEqual(0, result, stderr)
            rows = json.loads(output)
            self.assertEqual(
                ["project-smoke"], [row["item"]["id"] for row in rows]
            )

            result, output, stderr = self.run_cli(
                root, "context", "--route", "smoke-route", "--limit", "0"
            )
            self.assertEqual(2, result)
            self.assertIn("--limit must be at least 1", stderr)

            result, _, stderr = self.run_cli(
                root,
                "new-use",
                "--id",
                "use-l10-alias",
                "--item",
                "project-smoke",
                "--route",
                "l10-r0",
                "--mechanism",
                "smoke-mechanism",
                "--source-scope",
                "Route normalization test.",
                "--project-application",
                "Store a stable route id.",
                "--modifications",
                "None.",
                "--expected-effect",
                "No new alias route is written.",
                "--claim-boundary",
                "Test only.",
            )
            self.assertEqual(0, result, stderr)
            normalized_use = json.loads(
                (root / "uses" / "use-l10-alias.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("ten-meter-copilot", normalized_use["route"])

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

    def test_promoted_use_requires_applicability_and_mechanism_io(self) -> None:
        item = {
            "id": "paper-promoted",
            "mechanisms": [
                {
                    "id": "promoted-mechanism",
                    "inputs": [],
                    "outputs": [],
                }
            ],
        }
        use = {
            "schema_version": 1,
            "id": "use-promoted",
            "item_id": "paper-promoted",
            "route": "smoke-route",
            "mechanism_ids": ["promoted-mechanism"],
            "use_state": "planned",
            "adoption_mode": "reference",
            "usage": {
                "source_scope": "Scoped source.",
                "project_application": "Scoped application.",
                "modifications": "None.",
                "expected_effect": "Observable effect.",
            },
            "evaluation": {
                "reproduction_status": "not_attempted",
                "verdict": "not_run",
                "setup": "",
                "effect": "",
                "metrics": [],
                "claim_boundary": "No route authority.",
            },
            "evidence": [],
            "history": [{"date": "2026-08-31", "change": "Planned."}],
            "added_at": "2026-08-31",
            "updated_at": "2026-08-31",
        }
        errors = knowledge._validate_use(
            use,
            "use promoted",
            {item["id"]: item},
            Path("."),
            set(),
        )
        self.assertTrue(any("applicability" in error for error in errors))
        self.assertTrue(any("non-empty inputs" in error for error in errors))
        self.assertTrue(any("non-empty outputs" in error for error in errors))

        use["usage"]["applicability"] = "Only the scoped planned experiment."
        item["mechanisms"][0]["inputs"] = ["input"]
        item["mechanisms"][0]["outputs"] = ["output"]
        self.assertEqual(
            [],
            knowledge._validate_use(
                use,
                "use promoted",
                {item["id"]: item},
                Path("."),
                set(),
            ),
        )

    def test_run_associations_merge_duplicate_rows_and_exact_links(self) -> None:
        rows = [
            {
                "id": "run-smoke",
                "decision": "SMOKE_DECISION",
                "commit": "abc123",
                "report": "reports/run-smoke.json",
            },
            {
                "id": "run-smoke",
                "decision": "SMOKE_DECISION",
                "commit": "abc123",
                "report": "reports/run-smoke.md",
            },
        ]
        uses = {
            "use-smoke": {
                "id": "use-smoke",
                "evidence": [
                    {"kind": "experiment", "ref": "run-smoke", "summary": "Exact."}
                ],
            }
        }
        terminals = [
            {"id": "terminal-smoke", "decision": "SMOKE_DECISION"}
        ]
        associations = knowledge._build_run_associations(rows, uses, terminals)
        self.assertEqual(1, len(associations))
        self.assertEqual("run-smoke", associations[0]["run_id"])
        self.assertEqual(["use-smoke"], associations[0]["use_ids"])
        self.assertEqual("terminal-smoke", associations[0]["decision_id"])
        self.assertEqual(2, associations[0]["source_rows"])
        self.assertEqual(
            ["reports/run-smoke.json", "reports/run-smoke.md"],
            associations[0]["artifact_refs"],
        )

    def test_register_experiment_writes_p1_and_refreshes_index(self) -> None:
        with TemporaryDirectory(prefix="blindassist-register-experiment-") as temporary:
            repo_root = Path(temporary) / "repo"
            root = repo_root / "research" / "knowledge"
            (root / "items").mkdir(parents=True)
            (root / "uses").mkdir()
            (root / "decision").mkdir()
            (repo_root / "experiments").mkdir()
            active = repo_root / "research" / "active" / "smoke"
            active.mkdir(parents=True)

            result, _, stderr = self.run_cli(
                root,
                "new-item",
                "--id",
                "paper-register-smoke",
                "--kind",
                "paper",
                "--title",
                "Register smoke",
                "--canonical-ref",
                "https://example.org/register-smoke",
                "--summary",
                "A registration smoke source.",
                "--mechanism-id",
                "register-mechanism",
                "--mechanism-name",
                "Register mechanism",
                "--mechanism-description",
                "A mechanism linked to one run.",
                "--mechanism-input",
                "input manifest",
                "--mechanism-output",
                "named metric",
                "--mechanism-limitations",
                "No project claim.",
            )
            self.assertEqual(0, result, stderr)
            result, _, stderr = self.run_cli(
                root,
                "new-use",
                "--id",
                "use-register-smoke",
                "--item",
                "paper-register-smoke",
                "--route",
                "smoke-route",
                "--mechanism",
                "register-mechanism",
                "--source-scope",
                "The named mechanism.",
                "--project-application",
                "Exercise P1 registration.",
                "--modifications",
                "None.",
                "--expected-effect",
                "One linked association.",
                "--claim-boundary",
                "Test only.",
            )
            self.assertEqual(0, result, stderr)

            source_config = (
                Path(knowledge.__file__).resolve().parents[1]
                / "research"
                / "knowledge"
                / "decision"
                / "config.json"
            )
            config = json.loads(source_config.read_text(encoding="utf-8"))
            config["mechanism_overrides"] = {}
            (root / "decision" / "config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            report_ref = "research/active/smoke/protocol.json"
            input_ref = "research/active/smoke/input-manifest.json"
            (repo_root / report_ref).write_text('{"status":"not_run"}\n', encoding="utf-8")
            (repo_root / input_ref).write_text('{"cohort":"smoke"}\n', encoding="utf-8")

            revision = "a" * 40
            result, output, stderr = self.run_cli(
                root,
                "register-experiment",
                "--id",
                "smoke-run-v1",
                "--status",
                "active",
                "--question",
                "Does the smoke change alter the named metric?",
                "--baseline",
                "Frozen smoke baseline.",
                "--change",
                "One smoke factor.",
                "--primary-metric",
                "smoke_metric",
                "--decision",
                "FROZEN_PROTOCOL / NOT_RUN",
                "--report",
                report_ref,
                "--source",
                "research/active/smoke",
                "--protocol-id",
                "smoke-protocol-v1",
                "--input",
                input_ref,
                "--use-id",
                "use-register-smoke",
                "--artifact-ref",
                "artifacts.local/evidence/smoke-run-v1",
                "--code-revision",
                revision,
            )
            self.assertEqual(0, result, stderr)
            self.assertIn("decision index refreshed", output)

            rows = knowledge._read_experiment_rows(repo_root)
            self.assertEqual(1, len(rows))
            row = rows[0]
            self.assertEqual(["use-register-smoke"], row["use_ids"])
            self.assertEqual("smoke-protocol-v1", row["protocol_id"])
            self.assertEqual(revision, row["code_revision"])
            self.assertEqual([input_ref], row["input_refs"])
            self.assertRegex(row["input_fingerprint"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                [report_ref, "artifacts.local/evidence/smoke-run-v1"],
                row["artifact_refs"],
            )
            self.assertIsNone(row["decision_id"])

            index = json.loads(
                (root / "decision" / "index.json").read_text(encoding="utf-8")
            )
            association = next(
                value
                for value in index["associations"]
                if value["run_id"] == "smoke-run-v1"
            )
            self.assertEqual(["use-register-smoke"], association["use_ids"])
            self.assertEqual("smoke-protocol-v1", association["protocol_id"])
            self.assertEqual(revision, association["code_revision"])
            self.assertEqual(row["input_fingerprint"], association["input_fingerprint"])

            result, _, stderr = self.run_cli(
                root, "build-decision-index", "--check"
            )
            self.assertEqual(0, result, stderr)

    def test_current_route_aliases_share_one_context_family(self) -> None:
        item = {
            "id": "paper-route-family",
            "kind": "paper",
            "title": "Route family",
            "canonical_ref": "https://example.org/route-family",
            "summary": "Route alias test item.",
            "aliases": [],
            "mechanisms": [],
        }
        base_use = {
            "item_id": item["id"],
            "mechanism_ids": [],
            "use_state": "candidate",
            "adoption_mode": "reference",
            "usage": {
                "source_scope": "Test only.",
                "project_application": "Test route-family lookup.",
                "modifications": "None.",
                "expected_effect": "Both route names find both records.",
            },
            "evaluation": {
                "reproduction_status": "not_attempted",
                "verdict": "not_run",
                "setup": "",
                "effect": "",
                "metrics": [],
                "claim_boundary": "Test only.",
            },
            "evidence": [],
            "history": [],
            "updated_at": "2026-08-31",
        }
        uses = {
            "use-canonical": {
                **base_use,
                "id": "use-canonical",
                "route": "ten-meter-copilot",
            },
            "use-alias": {
                **base_use,
                "id": "use-alias",
                "route": "l10-r0",
            },
        }

        canonical = knowledge._build_context(
            {item["id"]: item},
            uses,
            route="ten-meter-copilot",
            query=None,
            limit=None,
        )
        alias = knowledge._build_context(
            {item["id"]: item},
            uses,
            route="l10-r0",
            query=None,
            limit=None,
        )

        self.assertEqual("ten-meter-copilot", alias["route"])
        self.assertEqual("l10-r0", alias["requested_route"])
        self.assertEqual(2, alias["summary"]["route_total_uses"])
        self.assertEqual(
            [entry["use"]["id"] for entry in canonical["entries"]],
            [entry["use"]["id"] for entry in alias["entries"]],
        )

        decision_index = {
            "experiments": [
                {
                    "kind": "current_terminal",
                    "id": "terminal-route-family",
                    "status": "closed",
                    "decision": "ROUTE_FAMILY_CLOSED",
                    "question": "Current route-family terminal.",
                    "successor_requires": "Fresh causal evidence.",
                    "forbidden_repeats": ["alias-only rerun"],
                    "evidence": ["reports/terminal.md"],
                    "commit": "abc123",
                    "routes": ["ten-meter-copilot"],
                }
            ],
            "associations": [
                {
                    "run_id": "run-route-family",
                    "use_ids": ["use-canonical"],
                    "protocol_id": None,
                    "code_revision": "abc123",
                    "input_fingerprint": None,
                    "artifact_refs": ["reports/terminal.md"],
                    "decision_id": "terminal-route-family",
                    "source_rows": 1,
                }
            ],
        }
        compact = knowledge._build_context(
            {item["id"]: item},
            uses,
            route="l10-r0",
            query=None,
            limit=2,
            decision_index=decision_index,
        )
        self.assertEqual(2, compact["summary"]["returned_records"])
        self.assertEqual("terminal-route-family", compact["terminals"][0]["id"])
        self.assertEqual(
            "run-route-family",
            compact["terminals"][0]["association"]["run_id"],
        )

    def test_query_matching_requires_all_terms_and_normalizes_punctuation(self) -> None:
        searchable = (
            "Mesh z-buffer projection keeps a visible target door face. "
            "存在状态与可见性状态分别记录。"
        )
        self.assertGreater(
            knowledge._query_match_score("door visible z-buffer", searchable),
            0,
        )
        self.assertGreater(
            knowledge._query_match_score("可见性 存在状态", searchable),
            0,
        )
        exact_score = knowledge._query_match_score(
            "visible door transfer", "Visible Door Transfer"
        )
        scattered_score = knowledge._query_match_score(
            "visible door transfer",
            "A visible target supports one door transfer candidate.",
        )
        self.assertGreater(exact_score, scattered_score)
        self.assertEqual(
            0,
            knowledge._query_match_score("door missing-term", searchable),
        )

    def test_diagnose_builds_fast_decision_card_from_compiled_index(self) -> None:
        with TemporaryDirectory(prefix="blindassist-decision-test-") as temporary:
            root = Path(temporary) / "research" / "knowledge"
            (root / "decision").mkdir(parents=True)
            (root / "decision" / "config.json").write_text(
                "{}\n", encoding="utf-8"
            )
            experiment_template = {
                "hypothesis": "Temporal state is missing.",
                "baseline": "Frozen stateless baseline.",
                "single_change": "Add bounded temporal belief only.",
                "cohort": "Fresh occlusion sequence.",
                "primary_metric": "reacquisition coverage",
                "stop_conditions": ["future-frame access"],
                "not_evaluable_conditions": ["missing timestamps"],
                "claim_ceiling": "Replay-only temporal evidence.",
            }
            index = {
                "schema_version": 2,
                "engine_version": "decision-test",
                "source_fingerprint": knowledge._decision_source_fingerprint(root),
                "failure_layers": [
                    {
                        "id": "temporal_belief",
                        "name": "Temporal belief",
                        "description": "Occlusion and reacquisition failures.",
                        "symptom_signals": ["track dropout", "occlusion"],
                        "mechanism_signals": ["temporal", "track"],
                        "required_evidence": ["ordered timestamps"],
                        "experiment": experiment_template,
                    }
                ],
                "global_guardrails": [
                    "已消费 cohort cannot be reopened.",
                    "UNKNOWN must remain distinct.",
                    "只改变一个 information factor.",
                ],
                "associations": [],
                "mechanisms": [
                    {
                        "id": "paper-temporal#bounded-belief",
                        "item_id": "paper-temporal",
                        "mechanism_id": "bounded-belief",
                        "kind": "paper",
                        "title": "Bounded temporal belief",
                        "canonical_ref": "https://example.org/temporal",
                        "summary": "A bounded belief mechanism.",
                        "tags": ["temporal"],
                        "name": "Bounded temporal belief",
                        "description": "Preserve state through short gaps.",
                        "inputs": ["ordered observations"],
                        "outputs": ["belief state"],
                        "limitations": "Does not create identity.",
                        "signatures": ["track dropout"],
                        "layer_scores": {"temporal_belief": 20},
                        "routes": ["smoke-route"],
                        "uses": [],
                        "search_text": "track dropout temporal belief",
                    },
                    {
                        "id": "paper-tracker#causal-track",
                        "item_id": "paper-tracker",
                        "mechanism_id": "causal-track",
                        "kind": "paper",
                        "title": "Causal track",
                        "canonical_ref": "https://example.org/tracker",
                        "summary": "A causal tracker.",
                        "tags": ["track"],
                        "name": "Causal track",
                        "description": "Associate adjacent observations.",
                        "inputs": ["detections"],
                        "outputs": ["tracks"],
                        "limitations": "Short gaps only.",
                        "signatures": ["occlusion"],
                        "layer_scores": {"temporal_belief": 10},
                        "routes": [],
                        "uses": [],
                        "search_text": "occlusion causal track",
                    },
                ],
                "experiments": [
                    {
                        "kind": "current_terminal",
                        "id": "terminal-old-dropout",
                        "status": "not_evaluable",
                        "question": "Static observations did not bridge gaps.",
                        "decision": "STATIC_DROPOUT_NOT_EVALUABLE",
                        "report": "docs/result.md",
                        "commit": "deadbeef",
                        "routes": ["smoke-route"],
                        "layer_scores": {"temporal_belief": 30},
                        "successor_requires": "Causal temporal information.",
                        "forbidden_repeats": ["threshold sweep"],
                        "evidence": ["docs/result.md"],
                        "search_text": "track dropout static observations",
                    }
                ],
            }
            (root / "decision" / "index.json").write_text(
                json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result, output, stderr = self.run_cli(
                root,
                "diagnose",
                "--route",
                "smoke-route",
                "--symptom",
                "track dropout after occlusion",
                "--mechanism-limit",
                "2",
                "--json",
            )
            self.assertEqual(0, result, stderr)
            card = json.loads(output)
            self.assertEqual("temporal_belief", card["diagnosis"]["layers"][0]["id"])
            self.assertEqual(2, len(card["mechanisms"]))
            self.assertEqual(
                "paper-temporal#bounded-belief", card["mechanisms"][0]["id"]
            )
            self.assertEqual(
                "terminal-old-dropout", card["prior_attempts"][0]["id"]
            )
            self.assertEqual(
                "paper-temporal#bounded-belief",
                card["minimum_experiment"]["selected_mechanism"]["id"],
            )
            self.assertTrue(
                knowledge._experiment_plan_is_valid(card["minimum_experiment"])
            )

            index["mechanisms"][0]["routes"] = ["l10-r0"]
            index["experiments"][0]["routes"] = ["ten-meter-copilot"]
            (root / "decision" / "index.json").write_text(
                json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result, output, stderr = self.run_cli(
                root,
                "diagnose",
                "--route",
                "l10-r0",
                "--symptom",
                "track dropout after occlusion",
                "--mechanism-limit",
                "2",
                "--json",
            )
            self.assertEqual(0, result, stderr)
            card = json.loads(output)
            self.assertEqual("ten-meter-copilot", card["route"])
            self.assertEqual(
                "paper-temporal#bounded-belief", card["mechanisms"][0]["id"]
            )
            self.assertEqual(
                "terminal-old-dropout", card["prior_attempts"][0]["id"]
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
