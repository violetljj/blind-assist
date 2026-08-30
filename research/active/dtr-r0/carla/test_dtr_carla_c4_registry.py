from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dtr_carla_c2_rich_scene import validate_protocol as validate_c2_protocol  # noqa: E402
from compile_dtr_carla_c4_multimap import main as compile_main  # noqa: E402
from dtr_carla_c4_scene import (  # noqa: E402
    C4_EXPERIMENT_ID,
    COMPILED_SCHEMA,
    PACKAGED_MAPS,
    REQUIRED_SCENE_CLASSES,
    compile_multimap,
    load_json,
    validate_registry_bundle,
)


class C4MultimapRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assets = load_json(HERE / "dtr_carla_c4_asset_registry.json")
        cls.scenes = load_json(HERE / "dtr_carla_c4_scene_registry.json")
        cls.source_c3 = load_json(HERE / "dtr_carla_c3_asset_registry.json")
        cls.base_c2 = load_json(HERE / "dtr_carla_c2_rich_scene_protocol.json")

    def test_materialized_bank_has_exact_requested_coverage(self) -> None:
        report = validate_registry_bundle(self.assets, self.scenes, self.source_c3)
        self.assertEqual(40, report["registered_asset_count"])
        self.assertEqual(16, report["registered_dynamic_asset_types"])
        self.assertEqual(23, report["registered_static_asset_types"])
        self.assertEqual(8, report["layout_count"])
        self.assertEqual(6, report["map_count"])
        self.assertEqual(PACKAGED_MAPS, set(report["maps"]))
        self.assertEqual(REQUIRED_SCENE_CLASSES, set(report["scene_classes"]))
        self.assertEqual(16, report["episode_count"])
        self.assertEqual(68, report["dynamic_target_placements"])
        self.assertEqual(43, report["static_support_placements"])
        self.assertEqual(119, report["actor_placements_including_wearers"])
        self.assertEqual(136, report["risk_corridor_dynamic_episode_checks"])
        self.assertEqual(8, report["contact_geometry_checks"])
        self.assertEqual(8, report["safe_geometry_checks"])
        self.assertLessEqual(report["maximum_contact_terminal_separation_m"], 0.5)
        self.assertGreaterEqual(report["minimum_safe_separation_m"], 1.0)
        self.assertLessEqual(
            report["maximum_contact_primary_footprint_clearance_m"], 0.0
        )
        self.assertGreater(
            report["minimum_safe_primary_footprint_clearance_m"], 0.05
        )
        self.assertGreater(
            report["minimum_nonprimary_footprint_clearance_m"], 0.05
        )
        self.assertGreaterEqual(
            report["minimum_background_pair_center_clearance_m"], 1.0
        )
        for scene in self.scenes["scenes"].values():
            expected_minimum = 12 if scene["scenario_class"] == "crowded_pedestrians" else 8
            self.assertGreaterEqual(scene["admission"]["dynamic_target_count"], expected_minimum)
            self.assertEqual(10, scene["admission"]["minimum_visible_frames_per_dynamic_target_per_episode"])

    def test_every_map_compiles_as_an_independent_c2_protocol(self) -> None:
        protocols, index, receipt = compile_multimap(
            self.base_c2, self.source_c3, self.assets, self.scenes
        )
        self.assertEqual(6, len(protocols))
        self.assertEqual(
            {value["carla_map"] for value in index["protocols"]}, PACKAGED_MAPS
        )
        self.assertEqual("DTR_CARLA_C4_MULTIMAP_WORLD_PACK_V1", C4_EXPERIMENT_ID)
        self.assertEqual("dtr-c4-per-map-c2-protocol-index-v1", COMPILED_SCHEMA)
        self.assertEqual([1280, 720], index["capture"]["resolution"])
        self.assertEqual(24, index["admission"]["expected_shard_count"])
        for protocol_id, protocol in protocols.items():
            with self.subTest(protocol_id=protocol_id):
                validate_c2_protocol(protocol)
                self.assertEqual([1280, 720], protocol["capture"]["resolution"])
                self.assertEqual([1280, 720], protocol["admission"]["required_resolution"])
                self.assertEqual("dx12", protocol["capture"]["render_backend"])
                self.assertEqual("Epic", protocol["capture"]["render_quality_level"])
                self.assertFalse(protocol["model_contract"]["include_current_actors"])
                self.assertTrue(protocol["wearer"]["scripted_invincible"])
                self.assertEqual(
                    "frozen_c4_weather_parameters_materialized_by_capture",
                    protocol["c4_compatibility"]["weather_override_authority"],
                )
        self.assertTrue(all(receipt["checks"].values()))

    def test_static_index_contract_is_exact_and_episode_metadata_is_derived(self) -> None:
        _, index, _ = compile_multimap(
            self.base_c2, self.source_c3, self.assets, self.scenes
        )
        self.assertEqual(
            {
                "schema_version",
                "experiment_id",
                "capture",
                "registries",
                "protocols",
                "admission",
                "model_visible_index",
                "evaluator_outcomes",
                "claim_boundary",
            },
            set(index),
        )
        protocol_keys = {
            "protocol_id",
            "protocol_path",
            "protocol_sha256",
            "carla_map",
            "startup_map_argument",
            "engine_ini_map_object_path",
            "cold_start_status",
            "layout_ids",
            "episodes",
            "layout_count",
            "episode_count",
            "unique_registered_blueprints_in_protocol",
            "layouts",
        }
        assets_by_id = {value["asset_id"]: value for value in self.assets["assets"]}
        for entry in index["protocols"]:
            self.assertEqual(protocol_keys, set(entry))
            self.assertFalse(Path(entry["protocol_path"]).is_absolute())
            self.assertEqual(entry["layout_count"], len(entry["layout_ids"]))
            self.assertEqual(entry["episode_count"], len(entry["episodes"]))
            for episode in entry["episodes"]:
                self.assertEqual(
                    {
                        "episode_id",
                        "layout_id",
                        "dynamic_target_ids",
                        "minimum_visible_frames_per_dynamic_target",
                        "risk_corridor_threshold_m",
                    },
                    set(episode),
                )
                self.assertEqual(10, episode["minimum_visible_frames_per_dynamic_target"])
                self.assertEqual(3.0, episode["risk_corridor_threshold_m"])
                scene = self.scenes["scenes"][episode["layout_id"]]
                expected_dynamic_ids = [
                    actor["instance_id"]
                    for actor in scene["actors"]
                    if assets_by_id[actor["asset_id"]]["mobility"] == "dynamic"
                ]
                self.assertEqual(expected_dynamic_ids, episode["dynamic_target_ids"])

    def test_cli_materializes_a_self_contained_static_bundle(self) -> None:
        def sha256_file(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest().upper()

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "c4-static-bundle"
            argv = [
                "compile_dtr_carla_c4_multimap.py",
                "--base-c2-protocol",
                str(HERE / "dtr_carla_c2_rich_scene_protocol.json"),
                "--source-c3-asset-registry",
                str(HERE / "dtr_carla_c3_asset_registry.json"),
                "--asset-registry",
                str(HERE / "dtr_carla_c4_asset_registry.json"),
                "--scene-registry",
                str(HERE / "dtr_carla_c4_scene_registry.json"),
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, compile_main())
            expected_names = {
                "compiled-index.json",
                "compiler-receipt.json",
                "dtr_carla_c4_asset_registry.json",
                "dtr_carla_c4_scene_registry.json",
                *{f"{name}.c2-protocol.json" for name in (
                    "town01", "town02", "town03_opt", "town04", "town05", "town10hd_opt"
                )},
            }
            self.assertEqual(expected_names, {path.name for path in output_dir.iterdir()})
            index = load_json(output_dir / "compiled-index.json")
            receipt = load_json(output_dir / "compiler-receipt.json")
            for registry in index["registries"].values():
                registry_path = output_dir / registry["path"]
                self.assertTrue(registry_path.is_file())
                self.assertEqual(registry["sha256"], sha256_file(registry_path))
            for protocol in index["protocols"]:
                protocol_path = output_dir / protocol["protocol_path"]
                self.assertTrue(protocol_path.is_file())
                self.assertEqual(protocol["protocol_sha256"], sha256_file(protocol_path))
            self.assertEqual(
                receipt["compiled_index_sha256"],
                sha256_file(output_dir / "compiled-index.json"),
            )
            serialized_index = str(index).casefold()
            self.assertNotIn("evidence_path", serialized_index)
            self.assertNotIn("result_sha256", serialized_index)

    def test_engine_ini_map_binding_and_weather_overrides_are_frozen(self) -> None:
        _, index, _ = compile_multimap(
            self.base_c2, self.source_c3, self.assets, self.scenes
        )
        for entry in index["protocols"]:
            leaf = entry["carla_map"].rsplit("/", 1)[-1]
            self.assertEqual(
                f"/Game/Carla/Maps/{leaf}.{leaf}", entry["startup_map_argument"]
            )
            self.assertEqual(
                f"/Game/Carla/Maps/{leaf}.{leaf}",
                entry["engine_ini_map_object_path"],
            )
            self.assertEqual(
                "C2_C3_CAPTURED"
                if entry["carla_map"] == "Carla/Maps/Town10HD_Opt"
                else "TASK_OWNED_ENGINE_INI_PROBED",
                entry["cold_start_status"],
            )
            for layout in entry["layouts"]:
                self.assertEqual(
                    {
                        "cloudiness",
                        "precipitation",
                        "precipitation_deposits",
                        "wind_intensity",
                        "sun_azimuth_angle",
                        "sun_altitude_angle",
                        "fog_density",
                        "wetness",
                    },
                    set(layout["weather_parameters"]),
                )
                self.assertIn("source", layout["anchor"])
                source = layout["anchor"]["source"]
                if entry["carla_map"] == "Carla/Maps/Town10HD_Opt":
                    self.assertEqual("c3_captured_anchor", source["kind"])
                else:
                    self.assertEqual("opendrive_waypoint_receipt", source["kind"])
                    self.assertEqual(entry["carla_map"], source["map"])
                    self.assertEqual(5.0, source["sample_distance_m"])
                    self.assertEqual(64, len(source["xodr_sha256"]))
        backlight = next(
            layout
            for entry in index["protocols"]
            for layout in entry["layouts"]
            if layout["scene_class"] == "backlight"
        )
        self.assertEqual("ClearSunset", backlight["weather_preset"])
        self.assertEqual(5.0, backlight["weather_parameters"]["sun_azimuth_angle"])
        self.assertEqual(8.0, backlight["weather_parameters"]["sun_altitude_angle"])

    def test_model_visible_index_has_no_outcome_labels(self) -> None:
        _, index, _ = compile_multimap(
            self.base_c2, self.source_c3, self.assets, self.scenes
        )
        model_text = str(index["model_visible_index"]).casefold()
        self.assertNotIn("contact", model_text)
        self.assertNotIn("safe", model_text)
        self.assertEqual(16, len(index["evaluator_outcomes"]))
        outcomes_by_slot: dict[str, set[str]] = {}
        for episode_id, truth in index["evaluator_outcomes"].items():
            outcomes_by_slot.setdefault(episode_id.rsplit("_", 1)[-1], set()).add(
                truth["expected_outcome"]
            )
        self.assertTrue(
            all(values == {"CONTACT", "SAFE"} for values in outcomes_by_slot.values())
        )

    def test_asset_blueprint_must_match_captured_c3_source(self) -> None:
        assets = copy.deepcopy(self.assets)
        assets["assets"][1]["blueprint_id"] = "walker.pedestrian.9999"
        with self.assertRaisesRegex(ValueError, "differs from C3 source"):
            validate_registry_bundle(assets, self.scenes, self.source_c3)

        assets = copy.deepcopy(self.assets)
        assets["dynamic_footprint_receipt"]["local_bbox_by_asset_id"][
            "c3_dynamic_14_bus"
        ]["extent_forward_m"] = 0.5
        with self.assertRaisesRegex(ValueError, "footprint content differs"):
            validate_registry_bundle(assets, self.scenes, self.source_c3)

    def test_layout_identity_cannot_encode_outcome(self) -> None:
        scenes = copy.deepcopy(self.scenes)
        scenes["scenes"]["c4_layout_01"]["display_name"] = "contact alley"
        with self.assertRaisesRegex(ValueError, "leaks evaluator outcome"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

    def test_episode_suffix_cannot_become_an_outcome_label(self) -> None:
        scenes = copy.deepcopy(self.scenes)
        for scene_id, scene in scenes["scenes"].items():
            contact = next(value for value in scene["episodes"] if value["expected_outcome"] == "CONTACT")
            safe = next(value for value in scene["episodes"] if value["expected_outcome"] == "SAFE")
            contact["episode_id"] = f"{scene_id.replace('layout_', 'l')}_e01"
            safe["episode_id"] = f"{scene_id.replace('layout_', 'l')}_e02"
            scene["counterfactual_contract"]["a"] = scene["episodes"][0]["episode_id"]
            scene["counterfactual_contract"]["b"] = scene["episodes"][1]["episode_id"]
        with self.assertRaisesRegex(ValueError, "episode suffix leaks"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

    def test_rain_backlight_and_crowd_parameters_are_decisive(self) -> None:
        scenes = copy.deepcopy(self.scenes)
        scenes["scenes"]["c4_layout_06"]["weather"]["parameters"][
            "precipitation"
        ] = 10.0
        with self.assertRaisesRegex(ValueError, "rainy-night precipitation"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        scenes["scenes"]["c4_layout_07"]["weather"]["parameters"][
            "sun_azimuth_angle"
        ] = 180.0
        with self.assertRaisesRegex(ValueError, "backlight sun is not camera-aligned"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        scenes["scenes"]["c4_layout_08"]["class_parameters"][
            "minimum_dynamic_pedestrians"
        ] = 11
        with self.assertRaisesRegex(ValueError, "too few dynamic pedestrians"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

    def test_resolution_map_and_twin_prefix_gates_are_decisive(self) -> None:
        scenes = copy.deepcopy(self.scenes)
        scenes["capture_contract"]["resolution"] = [640, 360]
        with self.assertRaisesRegex(ValueError, "1280x720"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        scenes["scenes"]["c4_layout_01"]["map"] = "Carla/Maps/Town11"
        with self.assertRaisesRegex(ValueError, "unbound map"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        scenes["trajectories"]["c4_twin_ped_variant_b"]["start_forward_m"] = 0.6
        with self.assertRaisesRegex(ValueError, "target prefixes differ"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

    def test_dynamic_density_corridor_motion_overlap_and_contact_are_hard_gates(self) -> None:
        scenes = copy.deepcopy(self.scenes)
        scene = scenes["scenes"]["c4_layout_01"]
        scene["actors"] = [value for value in scene["actors"] if value["instance_id"] != "dynamic_08"]
        scene["admission"]["dynamic_target_count"] = 7
        scene["admission"]["total_actors_including_wearer"] = 13
        with self.assertRaisesRegex(ValueError, "at least 8 dynamic risk participants"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        scenes["trajectories"]["c4_risk_flow_01"]["start_right_m"] = 20.0
        with self.assertRaisesRegex(ValueError, "never enters the 3m risk corridor"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        scenes["trajectories"]["c4_risk_flow_01"]["segments"][0]["velocity_right_mps"] = 0.0
        with self.assertRaisesRegex(ValueError, "invalid dynamic trajectory"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        actors = scenes["scenes"]["c4_layout_01"]["actors"]
        next(value for value in actors if value["instance_id"] == "dynamic_04")["trajectory_ref"] = "c4_risk_flow_01"
        with self.assertRaisesRegex(ValueError, "completely overlapping dynamic trajectories"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        scenes["trajectories"]["c4_risk_flow_05"]["start_forward_m"] = 1.0
        with self.assertRaisesRegex(ValueError, "background trajectories are too close"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        scenes["trajectories"]["c4_twin_ped_variant_a"]["start_forward_m"] = 1.5
        with self.assertRaisesRegex(ValueError, "CONTACT geometry does not converge"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        scenes["scenes"]["c4_layout_01"]["admission"]["minimum_visible_frames_per_dynamic_target_per_episode"] = 9
        with self.assertRaisesRegex(ValueError, "visibility gate differs"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

    def test_opendrive_receipt_and_plan_geometry_are_hard_gates(self) -> None:
        scenes = copy.deepcopy(self.scenes)
        scenes["scenes"]["c4_layout_01"]["anchor"]["source"]["xodr_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "OpenDRIVE SHA-256 differs"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)

        scenes = copy.deepcopy(self.scenes)
        scenes["scenes"]["c4_layout_01"]["episodes"][0]["issued_plan"][
            "time_parameterized_waypoints"
        ][-1]["forward_m"] = -1.0
        with self.assertRaisesRegex(ValueError, "plan does not match wearer trajectory"):
            validate_registry_bundle(self.assets, scenes, self.source_c3)


if __name__ == "__main__":
    unittest.main()
