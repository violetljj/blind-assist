"""Registry validation and C2-compatible compilation for CARLA C3 scenes.

C3 keeps the sealed C2 capture/join implementation as a compatibility engine.
The authoritative C3 layer is the pair of immutable asset/scene registries and
the compiler receipt that binds them to the generated capture protocol.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dtr_carla_c2_rich_scene import (
    EXPERIMENT_ID as C2_EXPERIMENT_ID,
    canonical_json_bytes,
    sha256_file,
    validate_protocol as validate_c2_protocol,
)


C3_EXPERIMENT_ID = "DTR_CARLA_C3_DENSE_DYNAMIC_RISK_SOURCE_V1"
ASSET_SCHEMA = 1
SCENE_SCHEMA = "dtr-c3-scene-registry-v1"
COMPILER_SCHEMA = "dtr-c3-c2-compatibility-compiler-receipt-v1"
FORMAL_SCENE_STATUS = "FORMAL_CANARY"
DYNAMIC_ROLE_FAMILIES = {
    "adult",
    "bicycle",
    "bus",
    "child",
    "emergency",
    "hgv",
    "motorcycle",
    "police",
    "sedan",
    "van",
}
MODEL_FORBIDDEN_KEY_PARTS = {
    "actor",
    "asset",
    "blueprint",
    "contact",
    "evaluator",
    "occluder",
    "outcome",
    "registry",
    "responsible",
    "role",
    "scenario",
    "target",
    "truth",
    "twin",
    "visibility",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def _require_exact_keys(
    value: dict[str, Any], required: set[str], optional: set[str], label: str
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing or extra:
        raise ValueError(f"{label} keys differ: missing={missing} extra={extra}")


def _has_nonzero_motion(trajectory: dict[str, Any]) -> bool:
    return any(
        abs(float(segment["velocity_forward_mps"])) > 1e-9
        or abs(float(segment["velocity_right_mps"])) > 1e-9
        for segment in trajectory.get("segments", [])
    )


def validate_asset_registry(registry: dict[str, Any]) -> None:
    _require_exact_keys(
        registry,
        {
            "schema_version",
            "registry_id",
            "carla_version",
            "scripted_actor_policy",
            "assets",
        },
        {"description", "claim_boundary"},
        "asset registry",
    )
    if registry["schema_version"] != ASSET_SCHEMA:
        raise ValueError("unexpected C3 asset registry schema")
    if str(registry["carla_version"]) != "0.9.16":
        raise ValueError("C3 asset registry must freeze CARLA 0.9.16")
    scripted_actor_policy = registry["scripted_actor_policy"]
    if not isinstance(scripted_actor_policy, dict) or set(scripted_actor_policy) != {
        "engine_collisions_enabled",
        "risk_contact_authority",
        "simulate_physics",
    }:
        raise ValueError("invalid C3 scripted actor policy")
    if bool(scripted_actor_policy["simulate_physics"]):
        raise ValueError("C3 scripted actor policy must disable physics")
    if bool(scripted_actor_policy["engine_collisions_enabled"]):
        raise ValueError("C3 scripted actor policy must disable engine collisions")
    if (
        str(scripted_actor_policy["risk_contact_authority"])
        != "evaluator_collision_polygons_xy"
    ):
        raise ValueError("C3 contact authority must be evaluator collision geometry")
    assets = registry["assets"]
    if not isinstance(assets, dict) or len(assets) != 40:
        raise ValueError("C3 formal asset registry must contain exactly 40 assets")
    required = {
        "attributes",
        "blueprint_id",
        "collision_relevant",
        "kind",
        "motion_policy",
        "physics_policy",
        "risk_participation",
        "role_family",
        "surface_policy",
        "trajectory_ref",
    }
    blueprint_ids: list[str] = []
    dynamic_ids: list[str] = []
    for asset_id, asset in assets.items():
        if not isinstance(asset, dict):
            raise ValueError(f"asset must be an object: {asset_id}")
        _require_exact_keys(asset, required, {"description"}, f"asset {asset_id}")
        blueprint_id = str(asset["blueprint_id"])
        if not blueprint_id.startswith(("walker.", "vehicle.", "static.prop.")):
            raise ValueError(f"unsupported blueprint namespace: {blueprint_id}")
        blueprint_ids.append(blueprint_id)
        surface = asset["surface_policy"]
        if not isinstance(surface, dict) or set(surface) != {
            "allow_lane_type_any",
            "allowed_lane_types",
            "projection",
            "surface_offset_m",
        }:
            raise ValueError(f"invalid surface policy: {asset_id}")
        if bool(surface["allow_lane_type_any"]):
            raise ValueError(f"formal C3 asset allows ambiguous LaneType.Any: {asset_id}")
        if not surface["allowed_lane_types"]:
            raise ValueError(f"formal C3 asset lacks an allowed lane type: {asset_id}")
        physics = asset["physics_policy"]
        if not isinstance(physics, dict) or bool(physics.get("simulate_physics", True)):
            raise ValueError(f"asset is not deterministic physics-off: {asset_id}")
        if bool(physics.get("autopilot", False)) or bool(
            physics.get("traffic_manager", False)
        ):
            raise ValueError(f"asset enables autonomous control: {asset_id}")
        if bool(asset["risk_participation"]):
            dynamic_ids.append(str(asset_id))
            if not bool(asset["collision_relevant"]):
                raise ValueError(f"dynamic risk asset is not collision relevant: {asset_id}")
            if str(asset["motion_policy"]) != "scripted_transform_each_fixed_tick":
                raise ValueError(f"dynamic risk asset lacks scripted motion: {asset_id}")
            if not str(asset["trajectory_ref"]):
                raise ValueError(f"dynamic risk asset lacks trajectory ref: {asset_id}")
    if len(set(blueprint_ids)) != len(blueprint_ids):
        raise ValueError("formal C3 registry requires one exact blueprint per asset")
    if len(dynamic_ids) != 16:
        raise ValueError(f"expected 16 dynamic risk assets, found {len(dynamic_ids)}")
    wearer_ids = [
        str(key)
        for key, value in assets.items()
        if str(value["role_family"]) == "wearer"
    ]
    if len(wearer_ids) != 1:
        raise ValueError("asset registry must contain exactly one wearer")
    if bool(assets[wearer_ids[0]]["risk_participation"]):
        raise ValueError("wearer must not be counted as a dynamic risk target")
    role_families = {str(assets[key]["role_family"]) for key in dynamic_ids}
    missing_roles = sorted(DYNAMIC_ROLE_FAMILIES - role_families)
    if missing_roles:
        raise ValueError(f"dynamic role-family coverage is incomplete: {missing_roles}")


def validate_scene_registry(
    registry: dict[str, Any], asset_registry: dict[str, Any]
) -> None:
    _require_exact_keys(
        registry,
        {"schema_version", "registry_id", "asset_registry_id", "trajectories", "scenes"},
        {"description", "claim_boundary"},
        "scene registry",
    )
    if registry["schema_version"] != SCENE_SCHEMA:
        raise ValueError("unexpected C3 scene registry schema")
    if registry["asset_registry_id"] != asset_registry["registry_id"]:
        raise ValueError("scene registry points at a different asset registry")
    trajectories = registry["trajectories"]
    if not isinstance(trajectories, dict) or not trajectories:
        raise ValueError("scene registry has no trajectories")
    for name, trajectory in trajectories.items():
        if not isinstance(trajectory, dict) or not trajectory.get("segments"):
            raise ValueError(f"invalid trajectory: {name}")
        starts = [float(value["start_s"]) for value in trajectory["segments"]]
        if not starts or starts[0] != 0.0 or starts != sorted(set(starts)):
            raise ValueError(f"trajectory segment starts are invalid: {name}")
    scenes = registry["scenes"]
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("scene registry has no scenes")
    assets = asset_registry["assets"]
    for scene_id, scene in scenes.items():
        _require_exact_keys(
            scene,
            {
                "status",
                "map",
                "display_name",
                "weather",
                "duration_seconds",
                "showcase_time_s",
                "anchor",
                "witness",
                "actors",
                "episodes",
                "occlusion_contract",
                "twin_contract",
                "admission",
            },
            {"description"},
            f"scene {scene_id}",
        )
        if scene["status"] != FORMAL_SCENE_STATUS:
            continue
        actors = scene["actors"]
        if len(actors) != 39:
            raise ValueError(f"formal scene {scene_id} must have 39 non-wearer actors")
        instance_ids = [str(value["instance_id"]) for value in actors]
        track_ids = [str(value["track_id"]) for value in actors]
        if len(set(instance_ids)) != 39 or len(set(track_ids)) != 39:
            raise ValueError(f"formal scene {scene_id} instance/track IDs are not unique")
        asset_ids = [str(value["asset_id"]) for value in actors]
        unknown = sorted(set(asset_ids) - set(assets))
        if unknown:
            raise ValueError(f"formal scene {scene_id} references unknown assets: {unknown}")
        if "wearer" in asset_ids:
            raise ValueError(f"formal scene {scene_id} counts wearer among actors")
        dynamic_actors = []
        for actor in actors:
            asset = assets[str(actor["asset_id"])]
            if bool(actor.get("risk_participation")) != bool(
                asset["risk_participation"]
            ):
                raise ValueError(f"risk participation differs for {actor['instance_id']}")
            if bool(actor.get("collision_relevant")) != bool(
                asset["collision_relevant"]
            ):
                raise ValueError(f"collision policy differs for {actor['instance_id']}")
            pose_keys = {
                key
                for key in ("fixed_pose", "trajectory_ref", "trajectory_key")
                if key in actor
            }
            if len(pose_keys) != 1:
                raise ValueError(f"actor needs one pose source: {actor['instance_id']}")
            if bool(actor["risk_participation"]):
                dynamic_actors.append(actor)
                reference = str(actor.get("trajectory_ref") or "")
                if reference and reference not in trajectories:
                    raise ValueError(f"unknown trajectory {reference}")
                default_reference = str(asset["trajectory_ref"])
                if default_reference not in trajectories:
                    raise ValueError(f"unknown asset trajectory {default_reference}")
                if not _has_nonzero_motion(trajectories[default_reference]):
                    raise ValueError(f"dynamic asset trajectory has zero motion: {actor['asset_id']}")
        if len(dynamic_actors) != 16:
            raise ValueError(f"formal scene {scene_id} needs 16 dynamic risk actors")
        admission = scene["admission"]
        if int(admission["total_scene_actors_including_wearer"]) != 40:
            raise ValueError("formal C3 scene actor count is not 40")
        if int(admission["dynamic_risk_targets"]) != 16:
            raise ValueError("formal C3 scene dynamic target count is not 16")
        if int(admission["static_support_actors"]) != 23:
            raise ValueError("formal C3 scene static support count is not 23")
        if str(admission.get("render_backend")) != "dx12":
            raise ValueError("formal C3 scene must freeze the proven dx12 backend")
        if str(admission.get("render_quality_level")) != "Epic":
            raise ValueError("formal C3 scene must freeze Epic render quality")
        if not math.isclose(
            float(admission.get("risk_corridor_threshold_m", -1.0)),
            3.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("formal C3 scene must freeze the 3.0 m risk corridor")
        if (
            int(
                admission.get(
                    "minimum_model_visible_frames_per_dynamic_target_per_episode",
                    0,
                )
            )
            != 10
        ):
            raise ValueError(
                "formal C3 scene must require ten visible frames per dynamic target and episode"
            )
        episode_ids = [str(value["episode_id"]) for value in scene["episodes"]]
        if len(episode_ids) != 2 or len(set(episode_ids)) != 2:
            raise ValueError("formal C3 canary must contain one CONTACT/SAFE episode pair")
        for episode in scene["episodes"]:
            identifiers = {
                "episode_id": str(episode["episode_id"]),
                "navigation_session_id": str(episode["navigation_session_id"]),
            }
            issued_plan = episode.get("issued_plan")
            if isinstance(issued_plan, dict):
                identifiers["plan_id"] = str(issued_plan["plan_id"])
                identifiers["plan_session_id"] = str(issued_plan["session_id"])
            for label, identifier in identifiers.items():
                lowered = identifier.lower()
                leaked = sorted(
                    token for token in MODEL_FORBIDDEN_KEY_PARTS if token in lowered
                )
                if leaked:
                    raise ValueError(
                        f"model-visible {label} leaks evaluator semantics: {leaked}"
                    )


def validate_registry_bundle(
    asset_registry: dict[str, Any], scene_registry: dict[str, Any]
) -> None:
    validate_asset_registry(asset_registry)
    validate_scene_registry(scene_registry, asset_registry)


def dynamic_risk_instance_ids(
    asset_registry: dict[str, Any], scene: dict[str, Any]
) -> list[str]:
    assets = asset_registry["assets"]
    return sorted(
        str(actor["instance_id"])
        for actor in scene["actors"]
        if bool(assets[str(actor["asset_id"])]["risk_participation"])
    )


def analytical_walker_separation(
    asset_registry: dict[str, Any], scene_registry: dict[str, Any], scene_id: str
) -> dict[str, Any]:
    """Check scripted local-coordinate walker centers before CARLA capture."""

    scene = scene_registry["scenes"][scene_id]
    assets = asset_registry["assets"]
    trajectories = scene_registry["trajectories"]
    sample_seconds = 0.05
    samples = int(round(float(scene["duration_seconds"]) / sample_seconds)) + 1

    def trajectory_position(name: str, time_s: float) -> tuple[float, float]:
        trajectory = trajectories[name]
        forward = float(trajectory["start_forward_m"])
        right = float(trajectory["start_right_m"])
        segments = sorted(trajectory["segments"], key=lambda value: value["start_s"])
        for index, segment in enumerate(segments):
            start = float(segment["start_s"])
            if time_s <= start:
                break
            end = (
                min(time_s, float(segments[index + 1]["start_s"]))
                if index + 1 < len(segments)
                else time_s
            )
            elapsed = max(0.0, end - start)
            forward += elapsed * float(segment["velocity_forward_mps"])
            right += elapsed * float(segment["velocity_right_mps"])
            if end >= time_s:
                break
        return forward, right

    actor_by_id = {str(value["instance_id"]): value for value in scene["actors"]}
    walker_ids = sorted(
        actor_id
        for actor_id, actor in actor_by_id.items()
        if str(assets[str(actor["asset_id"])]["kind"]) == "walker"
    )
    rows: list[dict[str, Any]] = []
    for episode in scene["episodes"]:
        episode_id = str(episode["episode_id"])
        names: dict[str, str | None] = {}
        fixed: dict[str, tuple[float, float]] = {}
        for actor_id in walker_ids:
            actor = actor_by_id[actor_id]
            if "fixed_pose" in actor:
                fixed[actor_id] = (
                    float(actor["fixed_pose"]["forward_m"]),
                    float(actor["fixed_pose"]["right_m"]),
                )
                names[actor_id] = None
            elif "trajectory_ref" in actor:
                names[actor_id] = str(actor["trajectory_ref"])
            else:
                names[actor_id] = str(episode["asset_trajectories"][actor_id])
        names["wearer"] = str(episode["wearer_trajectory"])
        ids = sorted(names)
        for first_index, first in enumerate(ids):
            for second in ids[first_index + 1 :]:
                if {first, second} == {"target_primary", "wearer"}:
                    continue
                minimum = math.inf
                minimum_sample = -1
                for sample_index in range(samples):
                    time_s = sample_index * sample_seconds
                    first_position = (
                        fixed[first]
                        if first in fixed
                        else trajectory_position(str(names[first]), time_s)
                    )
                    second_position = (
                        fixed[second]
                        if second in fixed
                        else trajectory_position(str(names[second]), time_s)
                    )
                    distance = math.dist(first_position, second_position)
                    if distance < minimum:
                        minimum = distance
                        minimum_sample = sample_index
                rows.append(
                    {
                        "episode_id": episode_id,
                        "a": first,
                        "b": second,
                        "minimum_center_distance_m": round(minimum, 6),
                        "minimum_sample_index": minimum_sample,
                    }
                )
    closest = min(rows, key=lambda value: value["minimum_center_distance_m"])
    return {
        "minimum_required_center_distance_m": 0.75,
        "closest_pair": closest,
        "violations": [
            value for value in rows if value["minimum_center_distance_m"] < 0.75
        ],
        "passed": all(value["minimum_center_distance_m"] >= 0.75 for value in rows),
    }


def compile_scene(
    base_c2_protocol: dict[str, Any],
    asset_registry: dict[str, Any],
    scene_registry: dict[str, Any],
    scene_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_registry_bundle(asset_registry, scene_registry)
    if scene_id not in scene_registry["scenes"]:
        raise ValueError(f"unknown C3 scene: {scene_id}")
    scene = scene_registry["scenes"][scene_id]
    if scene["status"] != FORMAL_SCENE_STATUS:
        raise ValueError(f"scene is registered but not admitted for capture: {scene_id}")
    if base_c2_protocol.get("experiment_id") != C2_EXPERIMENT_ID:
        raise ValueError("C3 compatibility compiler requires the frozen C2 protocol family")

    protocol = copy.deepcopy(base_c2_protocol)
    layout_id = f"c3_{scene_id}"
    assets = asset_registry["assets"]
    protocol["objective"] = (
        "Capture a 1280x720 CARLA canary with 16 independently registered "
        "dynamic targets that all enter the evaluator risk geometry."
    )
    protocol["environment"]["map"] = str(scene["map"])
    protocol["capture"]["seed"] = int(scene["admission"]["capture_seed"])
    protocol["capture"]["render_backend"] = str(
        scene["admission"]["render_backend"]
    )
    protocol["capture"]["render_quality_level"] = str(
        scene["admission"]["render_quality_level"]
    )
    protocol["asset_templates"] = {}
    wearer_ids = [
        str(key)
        for key, value in assets.items()
        if str(value["role_family"]) == "wearer"
    ]
    wearer_asset = assets[wearer_ids[0]]
    collisions_enabled = bool(
        asset_registry["scripted_actor_policy"]["engine_collisions_enabled"]
    )
    protocol["wearer"] = {
        "asset_key": "wearer",
        "track_id": "m_00",
        "role": "wearer",
        "kind": str(wearer_asset["kind"]),
        "blueprint_candidates": [str(wearer_asset["blueprint_id"])],
        "surface_offset_m": float(
            wearer_asset["surface_policy"]["surface_offset_m"]
        ),
        "collisions_enabled": collisions_enabled,
        "collision_relevant": False,
    }
    layout_actors: list[dict[str, Any]] = []
    for actor in scene["actors"]:
        asset_id = str(actor["asset_id"])
        asset = assets[asset_id]
        protocol["asset_templates"][asset_id] = {
            "kind": str(asset["kind"]),
            "blueprint_candidates": [str(asset["blueprint_id"])],
            "surface_offset_m": float(
                asset["surface_policy"]["surface_offset_m"]
            ),
            "collision_relevant": bool(asset["collision_relevant"]),
            "collisions_enabled": collisions_enabled,
            "c3_registry_asset_id": asset_id,
            "c3_risk_participation": bool(asset["risk_participation"]),
        }
        compiled_actor = {
            "asset_key": str(actor["instance_id"]),
            "track_id": str(actor["track_id"]),
            "role": str(actor["role"]),
            "template": asset_id,
        }
        if "fixed_pose" in actor:
            compiled_actor["fixed_pose"] = copy.deepcopy(actor["fixed_pose"])
        elif "trajectory_ref" in actor:
            compiled_actor["trajectory"] = str(actor["trajectory_ref"])
        else:
            # C2 resolves trajectory_key through scenario.asset_trajectories.
            # C3 keeps the default trajectory reference in the registry, while
            # the compatibility key is the stable scene instance id.
            compiled_actor["trajectory_key"] = str(actor["instance_id"])
        layout_actors.append(compiled_actor)
    protocol["trajectory_library"] = copy.deepcopy(scene_registry["trajectories"])
    protocol["layouts"] = {
        layout_id: {
            "display_name": str(scene["display_name"]),
            "weather": str(scene["weather"]),
            "duration_seconds": float(scene["duration_seconds"]),
            "showcase_time_s": float(scene["showcase_time_s"]),
            "anchor": copy.deepcopy(scene["anchor"]),
            "witness": copy.deepcopy(scene["witness"]),
            "assets": layout_actors,
        }
    }
    protocol["scenarios"] = []
    for source in scene["episodes"]:
        episode = copy.deepcopy(source)
        episode["layout_id"] = layout_id
        protocol["scenarios"].append(episode)
    protocol["occlusion_contracts"] = [copy.deepcopy(scene["occlusion_contract"])]
    protocol["twin_contracts"] = [copy.deepcopy(scene["twin_contract"])]
    protocol["admission"] = {
        "expected_episode_count": 2,
        "expected_layout_count": 1,
        "minimum_active_assets_per_layout_excluding_wearer": 39,
        "minimum_unique_actual_blueprints_across_pack": 40,
        "required_resolution": [1280, 720],
        "required_model_modalities": ["wearable", "depth"],
        "required_evaluator_modalities": ["instance", "witness"],
        "require_zero_blueprint_fallbacks": True,
    }
    protocol["c3_compatibility"] = {
        "schema_version": COMPILER_SCHEMA,
        "experiment_id": C3_EXPERIMENT_ID,
        "scene_id": scene_id,
        "asset_registry_id": asset_registry["registry_id"],
        "scene_registry_id": scene_registry["registry_id"],
        "dynamic_risk_instance_ids": dynamic_risk_instance_ids(asset_registry, scene),
        "scripted_actor_policy": copy.deepcopy(
            asset_registry["scripted_actor_policy"]
        ),
        "truth_boundary": "C3 registries are evaluator-only and never copied into model/.",
    }
    protocol["claim_boundary"] = [
        "C3 is scripted-kinematics Development evidence, not a real-world safety claim.",
        "All 16 registered dynamic targets enter evaluator risk geometry; this does not prove a model detects them.",
        "The compatibility protocol intentionally retains the C2 experiment identity so the sealed 1280x720 capture/join engine remains unchanged.",
        "C3 asset and scene registries remain evaluator-only and are forbidden from the model root.",
    ]
    validate_c2_protocol(protocol)
    receipt = {
        "schema_version": COMPILER_SCHEMA,
        "experiment_id": C3_EXPERIMENT_ID,
        "scene_id": scene_id,
        "base_c2_protocol_sha256": sha256_json(base_c2_protocol),
        "asset_registry_sha256": sha256_json(asset_registry),
        "scene_registry_sha256": sha256_json(scene_registry),
        "compiled_protocol_sha256": sha256_json(protocol),
        "dynamic_risk_instance_ids": dynamic_risk_instance_ids(asset_registry, scene),
        "actor_counts": {
            "including_wearer": 40,
            "dynamic_risk_targets": 16,
            "static_support": 23,
        },
    }
    return protocol, receipt


def source_file_identity(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
