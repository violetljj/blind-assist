from __future__ import annotations

import copy
import ast
import json
from pathlib import Path
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    generator_geometry as producer,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_geometry_independent as validator,
)


class GeneratorGeometryP1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = producer.DEFAULT_OUTPUT
        manifest_path = cls.evidence / "all_seed_geometry_manifest.jsonl"
        cls.records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
        ]
        cls.main = [
            item for item in cls.records if item["record_type"] == "main_cluster"
        ]

    def test_seed_derivation_is_block_specific_and_deterministic(self) -> None:
        first = producer.derive_seed("MAIN", "ADVIO_13", 0)
        self.assertEqual(first, producer.derive_seed("MAIN", "ADVIO_13", 0))
        self.assertNotEqual(first, producer.derive_seed("MAIN", "ADVIO_14", 0))
        self.assertNotEqual(first, producer.derive_seed("CAL", "ADVIO_13", 0))

    def test_scene_is_deterministic_nonplanar_mesh_inventory(self) -> None:
        first = producer.build_scene("ADVIO_13", 0, "MAIN")
        second = producer.build_scene("ADVIO_13", 0, "MAIN")
        self.assertEqual(first, second)
        objects = first["world"]["objects"]
        self.assertGreaterEqual(len(objects), 12)
        self.assertGreaterEqual(len({item["plane_z_m"] for item in objects}), 3)
        self.assertTrue(all(item["triangles"] == [[0, 1, 2], [0, 2, 3]] for item in objects))

    def test_all_seed_identity_counts_are_frozen(self) -> None:
        guard = [
            item
            for item in self.records
            if item["record_type"] == "guardrail_cluster"
        ]
        self.assertEqual(80, len(self.main))
        self.assertEqual(480, sum(len(item["arms"]) for item in self.main))
        self.assertEqual(8, len(guard))
        self.assertEqual(16, sum(len(item["arms"]) for item in guard))

    def test_reference_geometry_passes_g01_g02(self) -> None:
        guard = [
            item
            for item in self.records
            if item["record_type"] == "guardrail_cluster"
        ]
        g01, g02 = validator.gate_g01_g02(self.main, guard)
        self.assertEqual("PASS", g01["status"])
        self.assertEqual("PASS", g02["status"])

    def test_pairing_mutation_fails_g11(self) -> None:
        mutated = copy.deepcopy(self.main)
        mutated[0]["arms"].pop()
        self.assertEqual("FAIL", validator.gate_g11(mutated)["status"])

    def test_quality_geometry_mutation_fails_g12(self) -> None:
        mutated = copy.deepcopy(self.main)
        mutated[0]["arms"][1]["depth_sha256"] = "0" * 64
        self.assertEqual("FAIL", validator.gate_g12(mutated)["status"])

    def test_replay_mutation_fails_g14(self) -> None:
        replay = json.loads(
            (self.evidence / "deterministic_replay_ledger.json").read_text(
                encoding="utf-8"
            )
        )
        replay["items"][0]["match"] = False
        self.assertEqual("FAIL", validator.gate_g14(replay)["status"])

    def test_independent_validator_does_not_import_producer(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(
            any(
                "generator_geometry" in name
                or "ecological_response" in name
                or "rcle_" in name.lower()
                for name in imported
            )
        )

    def test_runtime_firewall_is_p1_only(self) -> None:
        runtime = json.loads(
            (self.evidence / "runtime_manifest.json").read_text(encoding="utf-8")
        )
        self.assertIs(False, runtime["formal_execution_authorized"])
        self.assertIs(False, runtime["quality_calibration_authorized"])
        self.assertIs(False, runtime["rcle_imported_or_executed"])

    def test_frozen_g13_incompatibility_remains_fail_closed(self) -> None:
        receipt = json.loads(
            (
                self.evidence / "independent_geometry_validation_receipt.json"
            ).read_text(encoding="utf-8")
        )
        gate = next(item for item in receipt["gates"] if item["id"].startswith("G13_"))
        self.assertEqual("FAIL", gate["status"])
        self.assertLess(
            gate["frozen_25pct_endpoint_radial_rate_ceiling_per_s"], 0.05
        )
        self.assertEqual("INTERVENTION_NOT_EVALUABLE", receipt["terminal"])
        self.assertEqual("HOLD_P1", receipt["state"])


if __name__ == "__main__":
    unittest.main()
