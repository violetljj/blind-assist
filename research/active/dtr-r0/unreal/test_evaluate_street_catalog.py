"""Frozen catalog and applied dwell tests; synthetic receipts, no UE execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import evaluate_street_closed_loop as evaluation
from street_scenarios import scenario_catalog


class FrozenCatalogTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[4] / "artifacts.local" / "tmp"
        parent.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix="catalog-evaluator-test-", dir=parent)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "evaluator/episodes").mkdir(parents=True)
        (self.root / "run.json").write_text('{"status":"COMPLETE"}', encoding="utf-8")

    def write(self, specs, catalog=None, mode="JOINT"):
        (self.root / "evaluator/scenarios.json").write_text(json.dumps(specs), encoding="utf-8")
        identity = {"controller_mode": mode}
        if catalog is not None:
            path = self.root / "evaluator/catalog-definition.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            identity["scenario_selection"] = {"catalog_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (self.root / "identity.json").write_text(json.dumps(identity), encoding="utf-8")

    def test_default_eight_requires_exact_geometry(self):
        specs = scenario_catalog()
        self.write(specs)
        result, _, _ = evaluation.evaluate(self.root)
        self.assertTrue(result["catalog_complete"])
        self.assertEqual(result["expected_pairs"], 8)
        specs[0]["actors"][0]["radius_m"] = 0.9
        self.write(specs)
        self.assertFalse(evaluation.evaluate(self.root)[0]["catalog_complete"])

    def test_custom_catalog_dynamic_denominator_and_subset(self):
        catalog = scenario_catalog()[:2]
        self.write(catalog, catalog, "DEPTH_ONLY")
        result, _, _ = evaluation.evaluate(self.root)
        self.assertEqual(result["expected_pairs"], 2)
        self.assertTrue(result["catalog_complete"])
        self.assertEqual(result["controller_mode"], "DEPTH_ONLY")
        self.assertIn("excluded from control", result["controller"])
        self.assertEqual(result["status"], "INCOMPLETE")  # Missing episodes stay missing.
        self.write(catalog[:1], catalog)
        self.assertFalse(evaluation.evaluate(self.root)[0]["catalog_complete"])
        changed = copy.deepcopy(catalog)
        changed[0]["duration_s"] += 1
        self.write(changed, catalog)
        self.assertFalse(evaluation.evaluate(self.root)[0]["catalog_complete"])

    def test_hash_binds_actual_file_bytes_and_missing_catalog(self):
        self.write(scenario_catalog(), scenario_catalog())
        path = self.root / "evaluator/catalog-definition.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            evaluation.evaluate(self.root)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing frozen catalog"):
            evaluation.evaluate(self.root)

    def test_legacy_identity_defaults_joint(self):
        self.write(scenario_catalog())
        (self.root / "identity.json").unlink()
        self.assertEqual(evaluation.evaluate(self.root)[0]["controller_mode"], "JOINT")

    def test_applied_dwell_uses_actual_intervals_not_next_commands(self):
        frames = [{"time_s": 0, "applied_command": {"action": "IGNORED"}},
                  {"time_s": 0.2, "applied_command": {"action": "WALK", "vx_mps": 1, "vy_mps": 0}},
                  {"time_s": 2.7, "applied_command": {"action": "WAIT", "vx_mps": 0, "vy_mps": 0}},
                  {"time_s": 5.2, "applied_command": {"action": "WAIT", "vx_mps": 0, "vy_mps": 0},
                   "response": {"command": {"action": "ARRIVED"}}}]
        result = evaluation.action_dwell(frames)
        self.assertEqual(result["action_dwell_s"], {"WALK": 0.2, "WAIT": 5.0})
        self.assertEqual(result["applied_stationary_s"], 5.0)
        self.assertEqual(len(result["applied_action_intervals"]), 2)
        self.assertEqual(result["applied_action_intervals"][-1]["duration_s"], 5.0)


if __name__ == "__main__":
    unittest.main()
