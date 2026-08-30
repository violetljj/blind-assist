from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dtr_carla_c3_scene import (  # noqa: E402
    C3_EXPERIMENT_ID,
    analytical_walker_separation,
    compile_scene,
    load_json,
    sha256_json,
    validate_registry_bundle,
)
from join_dtr_carla_c3_dynamic_risk import (  # noqa: E402
    _c3_semantic_truth_paths,
    _visible_for_tracking_in_every_episode,
)


class C3SceneCompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assets = load_json(HERE / "dtr_carla_c3_asset_registry.json")
        cls.scenes = load_json(HERE / "dtr_carla_c3_scene_registry.json")
        cls.base = load_json(HERE / "dtr_carla_c2_rich_scene_protocol.json")

    def test_registry_bundle_and_compiler(self) -> None:
        validate_registry_bundle(self.assets, self.scenes)
        protocol, receipt = compile_scene(
            self.base,
            self.assets,
            self.scenes,
            "town10_dense_risk_canary",
        )
        layout = protocol["layouts"]["c3_town10_dense_risk_canary"]
        self.assertEqual(39, len(layout["assets"]))
        self.assertEqual(40, receipt["actor_counts"]["including_wearer"])
        self.assertEqual(16, receipt["actor_counts"]["dynamic_risk_targets"])
        self.assertEqual(C3_EXPERIMENT_ID, receipt["experiment_id"])
        self.assertEqual(receipt["compiled_protocol_sha256"], sha256_json(protocol))
        self.assertEqual([1280, 720], protocol["capture"]["resolution"])
        self.assertEqual("dx12", protocol["capture"]["render_backend"])
        self.assertEqual("Epic", protocol["capture"]["render_quality_level"])
        self.assertFalse(protocol["model_contract"]["include_current_actors"])
        self.assertFalse(protocol["wearer"]["collisions_enabled"])
        self.assertTrue(
            all(
                value["collisions_enabled"] is False
                for value in protocol["asset_templates"].values()
            )
        )

    def test_dynamic_risk_audit_requires_model_visibility(self) -> None:
        self.assertTrue(
            _visible_for_tracking_in_every_episode({"e01": 10, "e02": 11})
        )
        self.assertFalse(
            _visible_for_tracking_in_every_episode({"e01": 9, "e02": 50})
        )
        self.assertFalse(_visible_for_tracking_in_every_episode({}))

    def test_c3_truth_scan_catches_keys_and_string_values(self) -> None:
        self.assertTrue(_c3_semantic_truth_paths({"observed_outcome": "SAFE"}))
        self.assertTrue(_c3_semantic_truth_paths({"episode_id": "target_safe"}))
        self.assertFalse(
            _c3_semantic_truth_paths(
                {
                    "episode_id": "c3_town10_e01",
                    "authority": "ANCHOR_FORWARD_RIGHT",
                    "receipt_sha256": "a" * 64,
                }
            )
        )

    def test_contact_safe_override_uses_instance_key(self) -> None:
        protocol, _ = compile_scene(
            self.base,
            self.assets,
            self.scenes,
            "town10_dense_risk_canary",
        )
        target = next(
            value
            for value in protocol["layouts"]["c3_town10_dense_risk_canary"][
                "assets"
            ]
            if value["asset_key"] == "target_primary"
        )
        self.assertEqual("target_primary", target["trajectory_key"])
        self.assertEqual(
            {"target_primary": "c3_risk_01_child_contact"},
            protocol["scenarios"][0]["asset_trajectories"],
        )
        self.assertEqual(
            {"target_primary": "c3_risk_01_child_safe"},
            protocol["scenarios"][1]["asset_trajectories"],
        )

    def test_dynamic_asset_cannot_leave_risk_geometry(self) -> None:
        assets = copy.deepcopy(self.assets)
        dynamic_id = next(
            key
            for key, value in assets["assets"].items()
            if bool(value["risk_participation"])
        )
        assets["assets"][dynamic_id]["collision_relevant"] = False
        with self.assertRaisesRegex(ValueError, "not collision relevant"):
            validate_registry_bundle(assets, self.scenes)

    def test_non_target_walkers_have_collision_separation(self) -> None:
        result = analytical_walker_separation(
            self.assets, self.scenes, "town10_dense_risk_canary"
        )
        self.assertTrue(result["passed"], result["violations"])

    def test_formal_risk_corridor_threshold_is_frozen(self) -> None:
        scenes = copy.deepcopy(self.scenes)
        scenes["scenes"]["town10_dense_risk_canary"]["admission"][
            "risk_corridor_threshold_m"
        ] = 3.01
        with self.assertRaisesRegex(ValueError, "3.0 m risk corridor"):
            validate_registry_bundle(self.assets, scenes)

    def test_model_visible_episode_identifier_cannot_leak_outcome(self) -> None:
        scenes = copy.deepcopy(self.scenes)
        scenes["scenes"]["town10_dense_risk_canary"]["episodes"][0][
            "episode_id"
        ] = "c3_town10_contact"
        with self.assertRaisesRegex(ValueError, "leaks evaluator semantics"):
            validate_registry_bundle(self.assets, scenes)


if __name__ == "__main__":
    unittest.main()
