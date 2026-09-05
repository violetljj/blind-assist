from __future__ import annotations

import copy
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from tools import knowledge, refresh_decision_templates as refresh


class DecisionTemplateRefreshTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(__file__).resolve().parents[1] / "artifacts.local" / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(prefix="decision-template-test-", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.root = self.repo / "research" / "knowledge"
        for directory in ("decision", "items", "uses"):
            (self.root / directory).mkdir(parents=True)
        (self.repo / "experiments").mkdir()
        # Deliberately invalid ledger JSON: a successful refresh must hash it
        # without parsing it or claiming that the ledger has been validated.
        (self.repo / "experiments" / "index.jsonl").write_text("unvalidated fixture ledger\n")
        (self.root / "items" / "fixture.json").write_text('{"fixture": "original"}\n')
        (self.root / "uses" / "fixture.json").write_text('{"outcome": "unknown"}\n')
        (self.root / "decision" / "terminals.json").write_text('{"terminal": "frozen"}\n')
        (self.root / "decision" / "inheritance.json").write_text('{"role": "NEGATIVE_CONTROL"}\n')
        self.previous = {
            "schema_version": knowledge.DECISION_SCHEMA_VERSION,
            "engine_version": "decision-old",
            "failure_layers": [self.layer("first"), self.layer("second")],
            "route_profiles": {"fixture-route": ["fixture-alias"]},
            "mechanism_overrides": {
                "fixture#mechanism": {"failure_layers": ["first"], "signatures": ["stable"]}
            },
            "global_guardrails": ["Consumed evidence is not fresh; UNKNOWN is not CLEAR."],
        }
        self.config_path = self.root / "decision" / "config.json"
        self.previous_path = self.repo / "previous.json"
        self.write_json(self.config_path, self.previous)
        self.previous_path.write_bytes(self.config_path.read_bytes())
        self.index_path = self.root / "decision" / "index.json"
        self.cached = {
            "schema_version": knowledge.DECISION_INDEX_SCHEMA_VERSION,
            "generated_at": "2000-01-01",
            "source_fingerprint": knowledge._decision_source_fingerprint(self.root),
            **{field: copy.deepcopy(self.previous[field]) for field in (
                "engine_version", "failure_layers", "route_profiles", "global_guardrails"
            )},
            "mechanisms": [{"id": "retained", "scores": {"first": 3}}],
            "experiments": [{"verdict": "not_evaluable", "result": {"metric": 7, "authority": "consumed"}}],
            "associations": [{"id": "old-run", "input_fingerprint": "0" * 64}],
            "counts": {"experiments": 1, "failure_layers": 2},
            "opaque_existing_metadata": {"preserve": [1, 2, 3]},
        }
        self.write_json(self.index_path, self.cached)
        self.current = copy.deepcopy(self.previous)
        self.current["engine_version"] = "decision-new"
        self.current["failure_layers"][0]["experiment"]["baseline"] = "updated template"
        self.current["global_guardrails"].append("Keep the actual evidence ceiling.")
        self.write_json(self.config_path, self.current)

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def layer(identifier: str) -> dict:
        return {
            "id": identifier, "name": identifier, "description": "fixture layer",
            "symptom_signals": [identifier], "mechanism_signals": ["stable"],
            "required_evidence": ["fixture evidence"],
            "experiment": {
                **{field: "fixture " + field for field in (
                    "hypothesis", "baseline", "single_change", "cohort", "primary_metric", "claim_ceiling"
                )},
                "stop_conditions": ["stop"], "not_evaluable_conditions": ["missing"],
            },
        }

    def test_success_preserves_all_cached_evidence_without_ledger_validation(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            status = refresh.main(["--root", str(self.root), "--previous-config", str(self.previous_path)])
        self.assertEqual(0, status)
        result = json.loads(self.index_path.read_text(encoding="utf-8"))
        mutable = {"engine_version", "failure_layers", "global_guardrails", "generated_at", "source_fingerprint"}
        for field in self.cached.keys() - mutable:
            self.assertEqual(self.cached[field], result[field], field)
        self.assertEqual(self.current["failure_layers"], result["failure_layers"])
        self.assertEqual(self.current["global_guardrails"], result["global_guardrails"])
        self.assertEqual(self.current["engine_version"], result["engine_version"])
        self.assertEqual(knowledge._decision_source_fingerprint(self.root), result["source_fingerprint"])
        self.assertFalse(result["template_only_refresh"]["experiment_ledger_validated"])
        self.assertFalse(result["template_only_refresh"]["experiment_outcomes_recomputed"])
        self.assertIn("Did NOT validate the experiment ledger", stdout.getvalue())

    def test_unrelated_source_drift_requires_full_rebuild(self) -> None:
        original = self.index_path.read_bytes()
        (self.root / "items" / "fixture.json").write_text('{"fixture": "changed"}\n')
        with self.assertRaisesRegex(knowledge.KnowledgeError, "unrelated source drift"):
            refresh.refresh_templates(self.root, self.previous_path)
        self.assertEqual(original, self.index_path.read_bytes())

    def test_retrieval_signal_route_override_and_layer_order_changes_are_rejected(self) -> None:
        original = self.index_path.read_bytes()
        variants = []
        for field in ("symptom_signals", "mechanism_signals", "required_evidence"):
            candidate = copy.deepcopy(self.current)
            candidate["failure_layers"][0][field].append("changed")
            variants.append((field, candidate))
        candidate = copy.deepcopy(self.current)
        candidate["failure_layers"].reverse()
        variants.append(("layer order", candidate))
        candidate = copy.deepcopy(self.current)
        candidate["route_profiles"]["fixture-route"].append("changed")
        variants.append(("route", candidate))
        candidate = copy.deepcopy(self.current)
        candidate["mechanism_overrides"]["fixture#mechanism"]["signatures"].append("changed")
        variants.append(("override", candidate))
        for name, config in variants:
            with self.subTest(change=name):
                self.write_json(self.config_path, config)
                with self.assertRaisesRegex(knowledge.KnowledgeError, "non-template config change"):
                    refresh.refresh_templates(self.root, self.previous_path)
                self.assertEqual(original, self.index_path.read_bytes())

    def test_concurrent_index_edit_is_not_overwritten(self) -> None:
        concurrent = b'{"concurrent": "must survive"}\n'
        validate = refresh._validate_template_change

        def edit_index(previous: dict, current: dict) -> None:
            validate(previous, current)
            self.index_path.write_bytes(concurrent)

        with mock.patch.object(refresh, "_validate_template_change", side_effect=edit_index):
            with self.assertRaisesRegex(knowledge.KnowledgeError, "cached index changed during refresh"):
                refresh.refresh_templates(self.root, self.previous_path)
        self.assertEqual(concurrent, self.index_path.read_bytes())

    def test_legacy_module_does_not_fingerprint_unrecognized_inheritance(self) -> None:
        legacy = SimpleNamespace(
            _decision_config_path=knowledge._decision_config_path,
            _decision_terminals_path=knowledge._decision_terminals_path,
        )
        with mock.patch.object(refresh, "knowledge", legacy):
            paths = [path for path, _ in refresh._source_snapshot(self.root)]
        self.assertIn("research/knowledge/decision/terminals.json", paths)
        self.assertNotIn("research/knowledge/decision/inheritance.json", paths)


if __name__ == "__main__":
    unittest.main()
