from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dtr_carla_n1_natural_dynamics import (  # noqa: E402
    CARLA_MAP,
    REGISTRY_SCHEMA_VERSION,
    REQUIRED_DRIVING_PROFILES,
    REQUIRED_EVENT_TYPES,
    SUBSYSTEMS,
    RegistrySchemaError,
    compile_plan,
    load_json,
)


REGISTRY_PATH = HERE / "dtr_carla_n1_natural_dynamics_registry.json"
SCRIPT_PATH = HERE / "dtr_carla_n1_natural_dynamics.py"


class N1NaturalDynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_json(REGISTRY_PATH)

    def test_same_seed_compiles_byte_equivalent_plan(self) -> None:
        first = compile_plan(self.registry, 20260830)
        second = compile_plan(self.registry, 20260830)
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )

    def test_seed_perturbation_changes_independent_subsystems(self) -> None:
        first = compile_plan(self.registry, 20260830)
        perturbed = compile_plan(self.registry, 20260831)
        self.assertNotEqual(first["subsystem_seeds"], perturbed["subsystem_seeds"])
        self.assertNotEqual(
            first["plan_fingerprint_sha256"], perturbed["plan_fingerprint_sha256"]
        )
        self.assertNotEqual(first["vehicle_intents"], perturbed["vehicle_intents"])
        self.assertNotEqual(first["tail_events"], perturbed["tail_events"])

    def test_plan_covers_profiles_crowd_interactions_and_long_tail_events(self) -> None:
        plan = compile_plan(self.registry, 20260830)
        vehicle_profiles = {
            actor["behavior_profile"]
            for actor in plan["vehicle_intents"]
            if actor["actor_kind"] == "vehicle"
        }
        self.assertEqual(set(REQUIRED_DRIVING_PROFILES), vehicle_profiles)
        self.assertEqual(set(SUBSYSTEMS), set(plan["subsystem_seeds"]))
        self.assertEqual(CARLA_MAP, plan["environment"]["map"])
        self.assertFalse(plan["environment"]["requires_running_server_to_compile"])
        self.assertEqual("construction_zone", plan["focus"]["scenario_class"])
        self.assertEqual("c4_layout_05", plan["focus"]["source_scene_id"])
        self.assertGreaterEqual(len(plan["crowd_groups"]), 1)
        self.assertGreaterEqual(len(plan["group_reroute_interactions"]), 1)
        self.assertTrue(
            all(item["type"] == "group_reroute" for item in plan["group_reroute_interactions"])
        )
        event_types = {event["type"] for event in plan["tail_events"]}
        self.assertEqual(set(REQUIRED_EVENT_TYPES), event_types)
        for event in plan["tail_events"]:
            self.assertIn("trigger_time_s", event)
            self.assertIn("duration_s", event)
            self.assertIn("chosen_actor_id", event["selection_rule"])
            self.assertTrue(event["selection_rule"]["candidate_actor_ids"])
        for actor in plan["vehicle_intents"] + plan["walker_intents"]:
            self.assertIn("intent", actor)

    def test_rejects_illegal_registry_schema(self) -> None:
        mutations = []

        wrong_version = copy.deepcopy(self.registry)
        wrong_version["schema_version"] = "dtr-carla-n1-natural-dynamics-registry-v0"
        mutations.append(wrong_version)

        wrong_map = copy.deepcopy(self.registry)
        wrong_map["carla"]["map"] = "Carla/Maps/Town99"
        mutations.append(wrong_map)

        missing_profile = copy.deepcopy(self.registry)
        del missing_profile["traffic_profiles"]["assertive"]
        mutations.append(missing_profile)

        unknown_event = copy.deepcopy(self.registry)
        unknown_event["long_tail_events"][0]["type"] = "teleport"
        mutations.append(unknown_event)

        for mutation in mutations:
            with self.subTest(schema_version=mutation.get("schema_version")):
                with self.assertRaises(RegistrySchemaError):
                    compile_plan(mutation, 1)

    def test_json_cli_compiles_without_carla(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "n1-plan.json"
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    os.fspath(SCRIPT_PATH),
                    "--registry",
                    os.fspath(REGISTRY_PATH),
                    "--seed",
                    "20260830",
                    "--output",
                    os.fspath(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=environment,
            )
            summary = json.loads(completed.stdout)
            plan = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual("dtr-carla-n1-seed-20260830", summary["plan_id"])
            self.assertEqual(summary["plan_fingerprint_sha256"], plan["plan_fingerprint_sha256"])
            self.assertEqual(REGISTRY_SCHEMA_VERSION, plan["registry_schema_version"])


if __name__ == "__main__":
    unittest.main()
