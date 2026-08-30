"""Compile the three-map N3 native-dynamics suite without starting CARLA."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from dtr_carla_n1_natural_dynamics import (
    REGISTRY_SCHEMA_VERSION,
    compile_plan,
    write_json_atomic,
)


REGISTRY_SCHEMA = "dtr-carla-n3-multitown-native-dynamics-registry-v1"
SUITE_SCHEMA = "dtr-carla-n3-multitown-native-dynamics-suite-v1"
SCENE_ORDER = (
    "town01_crowded_pedestrians",
    "town04_bus_stop",
    "town05_parking_lot",
)
EXPECTED_MAPS = (
    "Carla/Maps/Town01",
    "Carla/Maps/Town04",
    "Carla/Maps/Town05",
)
EXPECTED_CLASSES = ("crowded_pedestrians", "bus_stop", "parking_lot")
EXPECTED_EVENT_TYPES = (
    "occluded_jaywalk",
    "sudden_brake",
    "reverse_pullout",
    "door_open",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def validate_suite_registry(registry: dict[str, Any]) -> None:
    _require(registry.get("schema_version") == REGISTRY_SCHEMA, "registry schema differs")
    _require(registry.get("carla_version") == "0.9.16", "CARLA version differs")
    _require(float(registry.get("duration_seconds", 0.0)) >= 16.0, "duration is too short")
    common = registry.get("common")
    scenes = registry.get("scenes")
    _require(isinstance(common, dict), "common contract is missing")
    _require(isinstance(scenes, list) and len(scenes) == 3, "exactly three scenes required")
    _require(
        tuple(str(scene.get("scene_id")) for scene in scenes) == SCENE_ORDER,
        "scene order/identity differs",
    )
    _require(
        tuple(str(scene.get("map")) for scene in scenes) == EXPECTED_MAPS,
        "map roster/order differs",
    )
    _require(
        tuple(str(scene.get("scenario_class")) for scene in scenes) == EXPECTED_CLASSES,
        "scenario-class roster/order differs",
    )
    seeds = [int(scene.get("master_seed", -1)) for scene in scenes]
    _require(len(set(seeds)) == 3 and min(seeds) >= 0, "scene seeds must be unique")
    for scene in scenes:
        required = set(map(str, scene.get("required_native_vehicle_classes", [])))
        _require(
            required == {"heavy_vehicle", "two_wheeler"},
            f"native vehicle classes differ for {scene['scene_id']}",
        )
        profiles = scene.get("traffic_profiles")
        _require(
            isinstance(profiles, dict)
            and set(profiles) == {"cautious", "nominal", "assertive"},
            f"traffic profiles differ for {scene['scene_id']}",
        )
        blueprints = {
            str(blueprint)
            for profile in profiles.values()
            for blueprint in profile.get("blueprint_ids", [])
        }
        _require(
            any(
                blueprint
                in {
                    "vehicle.mitsubishi.fusorosa",
                    "vehicle.carlamotors.carlacola",
                    "vehicle.mercedes.sprinter",
                    "vehicle.volkswagen.t2_2021",
                }
                for blueprint in blueprints
            ),
            f"heavy vehicle blueprint missing for {scene['scene_id']}",
        )
        _require(
            any(
                token in blueprint
                for blueprint in blueprints
                for token in (
                    "crossbike",
                    "century",
                    "omafiets",
                    "zx125",
                    "low_rider",
                    "ninja",
                    "yzf",
                )
            ),
            f"two-wheeler blueprint missing for {scene['scene_id']}",
        )
        bindings = scene.get("event_actor_bindings")
        _require(
            isinstance(bindings, dict) and tuple(bindings) == EXPECTED_EVENT_TYPES,
            f"event actor bindings differ for {scene['scene_id']}",
        )
        _require(
            str(bindings["occluded_jaywalk"]).startswith("n1_pedestrian_"),
            f"jaywalk binding differs for {scene['scene_id']}",
        )
        vehicle_bindings = [str(bindings[event_type]) for event_type in EXPECTED_EVENT_TYPES[1:]]
        _require(
            len(set(vehicle_bindings)) == 1
            and vehicle_bindings[0].startswith("n1_vehicle_cautious_"),
            f"vehicle event bindings must share one cautious actor for {scene['scene_id']}",
        )
        _require(
            0.5 <= float(scene.get("maximum_wearer_speed_mps", 0.0)) <= 2.0,
            f"wearer speed constraint differs for {scene['scene_id']}",
        )
        _require(
            float(scene.get("maximum_event_view_range_m", 0.0))
            >= float(scene.get("route_view_distance_m", 0.0)),
            f"event view range differs for {scene['scene_id']}",
        )


def _scene_registry(registry: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    common = registry["common"]
    crowd_common = common["crowd"]
    grouping_common = common["grouping"]
    rerouting_common = common["rerouting"]
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_id": f"DTR_CARLA_N3_{str(scene['scene_id']).upper()}",
        "purpose": "seeded_synthetic_development_world_planning",
        "carla": {
            "version": registry["carla_version"],
            "map": scene["map"],
            "requires_running_server": False,
        },
        "focus": {
            "scenario_class": scene["scenario_class"],
            "source_registry": "dtr_carla_c4_scene_registry.json",
            "source_scene_id": scene["source_scene_id"],
            "anchor": copy.deepcopy(scene["anchor"]),
            "class_parameters": copy.deepcopy(scene["class_parameters"]),
        },
        "duration_seconds": float(registry["duration_seconds"]),
        "seed_contract": copy.deepcopy(common["seed_contract"]),
        "traffic_profiles": copy.deepcopy(scene["traffic_profiles"]),
        "crowd": {
            "count_range": copy.deepcopy(scene["crowd_count_range"]),
            "blueprint_ids": copy.deepcopy(crowd_common["blueprint_ids"]),
            "walking_speed_mps": copy.deepcopy(crowd_common["walking_speed_mps"]),
            "personal_space_m": copy.deepcopy(crowd_common["personal_space_m"]),
            "crossing_propensity": copy.deepcopy(crowd_common["crossing_propensity"]),
            "distraction_probability": float(crowd_common["distraction_probability"]),
            "origin_ids": copy.deepcopy(scene["crowd_origin_ids"]),
            "destination_ids": copy.deepcopy(scene["crowd_destination_ids"]),
        },
        "grouping": {
            **copy.deepcopy(grouping_common),
            "shared_destination_ids": copy.deepcopy(scene["shared_destination_ids"]),
        },
        "rerouting": {
            **copy.deepcopy(rerouting_common),
            "blocked_destination_id": scene["blocked_destination_id"],
            "alternative_destination_ids": copy.deepcopy(
                scene["alternative_destination_ids"]
            ),
        },
        "long_tail_events": copy.deepcopy(common["long_tail_events"]),
        "claim_boundary": copy.deepcopy(registry["claim_boundary"]),
    }


def _apply_scene_event_bindings(
    plan: dict[str, Any], bindings: dict[str, Any]
) -> None:
    """Freeze route-feasible actor identities without changing seeded event timing/effects."""

    actor_by_id = {
        str(actor["actor_id"]): actor
        for actor in [*plan["vehicle_intents"], *plan["walker_intents"]]
    }
    for actor in actor_by_id.values():
        actor["intent"]["scheduled_event_ids"] = []
    for event in plan["tail_events"]:
        event_type = str(event["type"])
        actor_id = str(bindings[event_type])
        _require(actor_id in actor_by_id, f"bound event actor is absent: {actor_id}")
        actor = actor_by_id[actor_id]
        expected_kind = "pedestrian" if event_type == "occluded_jaywalk" else "vehicle"
        _require(
            actor["actor_kind"] == expected_kind,
            f"bound actor kind differs for {event_type}: {actor_id}",
        )
        if expected_kind == "vehicle":
            _require(
                actor["behavior_profile"] == "cautious",
                f"bound vehicle profile differs for {event_type}: {actor_id}",
            )
        event["primary_actor_id"] = actor_id
        event["selection_rule"]["selector"] = (
            "scene_frozen_binding_after_route_feasibility_diagnostic"
        )
        event["selection_rule"]["chosen_actor_id"] = actor_id
        event["selection_rule"]["unused_candidate_rule_applied"] = False
        event["selection_rule"]["binding_authority"] = (
            "N3_V2_ROUTE_FEASIBILITY_DIAGNOSTIC"
        )
        actor["intent"]["scheduled_event_ids"].append(event["event_id"])


def compile_suite(registry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    validate_suite_registry(registry)
    plans: dict[str, dict[str, Any]] = {}
    scene_entries: list[dict[str, Any]] = []
    for scene in registry["scenes"]:
        scene_id = str(scene["scene_id"])
        plan = compile_plan(_scene_registry(registry, scene), int(scene["master_seed"]))
        plan.pop("plan_fingerprint_sha256")
        _apply_scene_event_bindings(plan, scene["event_actor_bindings"])
        plan["plan_id"] = f"dtr-carla-n3-{scene_id}-seed-{scene['master_seed']}"
        plan["suite_scene"] = {
            "suite_id": registry["registry_id"],
            "scene_id": scene_id,
            "scenario_class": scene["scenario_class"],
            "source_scene_id": scene["source_scene_id"],
            "required_native_vehicle_classes": copy.deepcopy(
                scene["required_native_vehicle_classes"]
            ),
            "route_view_distance_m": float(scene["route_view_distance_m"]),
            "maximum_event_view_range_m": float(scene["maximum_event_view_range_m"]),
            "maximum_wearer_speed_mps": float(scene["maximum_wearer_speed_mps"]),
            "event_actor_bindings": copy.deepcopy(scene["event_actor_bindings"]),
            "event_actor_binding_authority": "N3_V2_ROUTE_FEASIBILITY_DIAGNOSTIC",
        }
        plan["plan_fingerprint_sha256"] = canonical_sha256(plan)
        plans[scene_id] = plan
        scene_entries.append(
            {
                "ordinal": len(scene_entries),
                "scene_id": scene_id,
                "map": scene["map"],
                "scenario_class": scene["scenario_class"],
                "source_scene_id": scene["source_scene_id"],
                "master_seed": int(scene["master_seed"]),
                "plan_path": f"plans/{scene_id}.json",
                "plan_id": plan["plan_id"],
                "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
                "expected_actor_count": int(plan["coverage"]["actor_count"]),
                "expected_trace_frames": int(
                    round(float(plan["duration_seconds"]) / 0.05)
                )
                + 1,
                "required_native_vehicle_classes": copy.deepcopy(
                    scene["required_native_vehicle_classes"]
                ),
                "route_view_distance_m": float(scene["route_view_distance_m"]),
                "maximum_event_view_range_m": float(
                    scene["maximum_event_view_range_m"]
                ),
                "maximum_wearer_speed_mps": float(scene["maximum_wearer_speed_mps"]),
                "event_actor_bindings": copy.deepcopy(scene["event_actor_bindings"]),
            }
        )
    suite = {
        "schema_version": SUITE_SCHEMA,
        "suite_id": registry["registry_id"],
        "evidence_role": "synthetic_development_native_source_suite",
        "carla_version": registry["carla_version"],
        "fixed_delta_seconds": 0.05,
        "scene_count": len(scene_entries),
        "scene_order": list(SCENE_ORDER),
        "scenes": scene_entries,
        "claim_boundary": copy.deepcopy(registry["claim_boundary"]),
    }
    return suite, plans


def materialize_suite(registry_path: Path, output_root: Path) -> dict[str, Any]:
    registry_path = registry_path.resolve(strict=True)
    registry = read_json(registry_path)
    suite, plans = compile_suite(registry)
    output_root = output_root.resolve()
    _require(not output_root.exists(), f"refusing suite overwrite: {output_root}")
    (output_root / "plans").mkdir(parents=True)
    for scene in suite["scenes"]:
        scene_id = str(scene["scene_id"])
        plan_path = output_root / str(scene["plan_path"])
        write_json_atomic(plan_path, plans[scene_id])
        scene["plan_file_sha256"] = sha256_file(plan_path)
    suite["registry"] = {
        "path": os.fspath(registry_path),
        "sha256": sha256_file(registry_path),
    }
    suite["suite_fingerprint_sha256"] = canonical_sha256(suite)
    write_json_atomic(output_root / "suite_manifest.json", suite)
    return suite


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).with_name(
            "dtr_carla_n3_multitown_native_dynamics_registry.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suite = materialize_suite(args.registry, args.output_root)
    print(json.dumps(suite, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
