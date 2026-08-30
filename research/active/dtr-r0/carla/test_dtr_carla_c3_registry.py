from __future__ import annotations

import copy
import json
import re
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ASSET_REGISTRY_PATH = HERE / "dtr_carla_c3_asset_registry.json"
SCENE_REGISTRY_PATH = HERE / "dtr_carla_c3_scene_registry.json"

ASSET_REGISTRY_KEYS = {
    "schema_version",
    "registry_id",
    "carla_version",
    "assets",
}
ASSET_KEYS = {
    "blueprint_id",
    "kind",
    "role_family",
    "risk_participation",
    "collision_relevant",
    "motion_policy",
    "trajectory_ref",
    "physics_policy",
    "surface_policy",
    "attributes",
}
SCENE_REGISTRY_KEYS = {
    "schema_version",
    "registry_id",
    "asset_registry_id",
    "trajectories",
    "scenes",
}
SCENE_KEYS = {
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
}
ACTOR_KEYS = {"instance_id", "asset_id", "track_id", "role"}
PLACEMENT_KEYS = ("trajectory_ref", "trajectory_key", "fixed_pose")

REQUIRED_DYNAMIC_ROLE_FAMILIES = frozenset(
    {
        "child",
        "adult",
        "police",
        "bicycle",
        "motorcycle",
        "sedan",
        "van",
        "bus",
        "hgv",
        "emergency",
    }
)
EXPECTED_SCENE_ACTORS_EXCLUDING_WEARER = 39
EXPECTED_SCENE_ACTORS_INCLUDING_WEARER = 40
EXPECTED_DYNAMIC_TARGETS = 16

MODEL_ONLY_FORBIDDEN_TOKENS = (
    "registry",
    "actor",
    "role",
    "outcome",
    "contact",
)
MODEL_ONLY_FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "asset_id",
        "blueprint_id",
        "instance_id",
        "track_id",
        "trajectory_key",
        "trajectory_ref",
    }
)
BLUEPRINT_CANDIDATE_KEYS = frozenset(
    {
        "blueprint_candidates",
        "candidate_blueprints",
        "blueprint_options",
    }
)


class RegistryAuditError(ValueError):
    """Raised when a C3 registry violates a deployability admission gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryAuditError(message)


def _nonzero(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "none", "null", "disabled"}
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _normal(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    folded: dict[str, str] = {}
    for key, value in pairs:
        normalized = key.casefold()
        if normalized in folded:
            raise RegistryAuditError(
                "duplicate JSON key (case-insensitive): "
                f"{folded[normalized]!r} and {key!r}"
            )
        folded[normalized] = key
        output[key] = value
    return output


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    _require(
        path.is_file(),
        f"missing C3 {label}: {path}; materialize the fixed registry before "
        "claiming the C3 risk-density gate",
    )
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except RegistryAuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegistryAuditError(f"invalid C3 {label} {path}: {error}") from error
    _require(isinstance(value, dict), f"C3 {label} must be a JSON object: {path}")
    return value


def _fallback_is_active(key: str, value: Any) -> bool:
    normalized = _normal(key)
    if normalized.startswith("require_zero_") and value is True:
        return False
    if normalized.startswith(("allow_", "use_", "enable_")) and value is False:
        return False
    return _nonzero(value)


def fallback_paths(value: Any, path: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized = _normal(key)
            if normalized in BLUEPRINT_CANDIDATE_KEYS:
                failures.append(child_path)
            elif "fallback" in normalized and _fallback_is_active(key, child):
                failures.append(child_path)
            failures.extend(fallback_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(fallback_paths(child, f"{path}[{index}]"))
    return failures


def model_only_forbidden_paths(value: Any, path: str = "$") -> list[str]:
    """Find evaluator-only identity or truth keys in a model-only value."""

    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized = _normal(key)
            if (
                normalized in MODEL_ONLY_FORBIDDEN_IDENTITY_KEYS
                or any(token in normalized for token in MODEL_ONLY_FORBIDDEN_TOKENS)
            ):
                failures.append(child_path)
            failures.extend(model_only_forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(model_only_forbidden_paths(child, f"{path}[{index}]"))
    return failures


def assert_model_only_truth_blind(value: Any) -> None:
    failures = model_only_forbidden_paths(value)
    _require(
        not failures,
        "C3 model-only payload contains evaluator-only registry/actor/role/"
        f"outcome/contact truth: {failures[:10]}",
    )


def _require_fields(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    _require(not missing, f"{label} missing required fields: {missing}")


def _casefold_unique(values: list[str], label: str) -> None:
    counts = Counter(value.casefold() for value in values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    _require(not duplicates, f"{label} must be case-insensitively unique: {duplicates}")


def audit_asset_registry(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require_fields(value, ASSET_REGISTRY_KEYS, "asset registry")
    _require(_nonzero(value["schema_version"]), "asset registry schema_version is empty")
    _require(_nonzero(value["registry_id"]), "asset registry registry_id is empty")
    _require(_nonzero(value["carla_version"]), "asset registry carla_version is empty")

    assets = value["assets"]
    _require(isinstance(assets, dict) and assets, "asset registry assets must be a non-empty object")
    asset_ids = [str(asset_id) for asset_id in assets]
    _require(all(asset_id.strip() for asset_id in asset_ids), "asset registry has an empty asset id")
    _casefold_unique(asset_ids, "asset registry ids")

    blueprint_ids: list[str] = []
    for asset_id, asset in assets.items():
        label = f"asset {asset_id!r}"
        _require(isinstance(asset, dict), f"{label} must be an object")
        _require_fields(asset, ASSET_KEYS, label)
        blueprint_id = asset["blueprint_id"]
        _require(
            isinstance(blueprint_id, str) and blueprint_id.strip() == blueprint_id and blueprint_id,
            f"{label} blueprint_id must be one exact non-empty string",
        )
        _require(
            not any(token in blueprint_id for token in ("*", "?", "[", "]")),
            f"{label} blueprint_id must not be a wildcard: {blueprint_id!r}",
        )
        blueprint_ids.append(blueprint_id)
        _require(_nonzero(asset["kind"]), f"{label} kind is empty")
        _require(_nonzero(asset["role_family"]), f"{label} role_family is empty")
        _require(
            type(asset["risk_participation"]) is bool,
            f"{label} risk_participation must be a JSON boolean",
        )
        _require(
            type(asset["collision_relevant"]) is bool,
            f"{label} collision_relevant must be a JSON boolean",
        )
        for policy in ("motion_policy", "physics_policy", "surface_policy"):
            _require(_nonzero(asset[policy]), f"{label} {policy} is empty")
        _require(isinstance(asset["attributes"], dict), f"{label} attributes must be an object")

    _casefold_unique(blueprint_ids, "asset registry blueprint_id values")
    failures = fallback_paths(value)
    _require(not failures, f"asset registry contains blueprint fallback paths: {failures[:10]}")
    return assets


def _is_formal_canary(scene: dict[str, Any]) -> bool:
    status = _normal(scene.get("status", ""))
    return status == "formal" or status.startswith("formal_canary")


def _audit_actor_identity(scene_id: str, actors: list[dict[str, Any]]) -> None:
    instance_ids = [str(actor.get("instance_id", "")) for actor in actors]
    track_ids = [str(actor.get("track_id", "")) for actor in actors]
    _casefold_unique(instance_ids, f"scene {scene_id!r} actor instance_id values")
    _casefold_unique(track_ids, f"scene {scene_id!r} actor track_id values")


def audit_scene_registry(
    value: dict[str, Any],
    *,
    asset_registry_id: str,
    assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    _require_fields(value, SCENE_REGISTRY_KEYS, "scene registry")
    _require(_nonzero(value["schema_version"]), "scene registry schema_version is empty")
    _require(_nonzero(value["registry_id"]), "scene registry registry_id is empty")
    _require(
        str(value["registry_id"]).casefold() != str(asset_registry_id).casefold(),
        "asset and scene registry_id values must be distinct",
    )
    _require(
        value["asset_registry_id"] == asset_registry_id,
        "scene registry asset_registry_id does not resolve to the asset registry: "
        f"{value['asset_registry_id']!r} != {asset_registry_id!r}",
    )
    trajectories = value["trajectories"]
    scenes = value["scenes"]
    _require(isinstance(trajectories, dict), "scene registry trajectories must be an object")
    _require(isinstance(scenes, dict) and scenes, "scene registry scenes must be a non-empty object")
    scene_ids = [str(scene_id) for scene_id in scenes]
    _casefold_unique(scene_ids, "scene registry scene ids")

    formal_scenes: list[tuple[str, dict[str, Any]]] = []
    for scene_id, scene in scenes.items():
        label = f"scene {scene_id!r}"
        _require(isinstance(scene, dict), f"{label} must be an object")
        _require_fields(scene, SCENE_KEYS, label)
        _require(_nonzero(scene["status"]), f"{label} status is empty")
        _require(_nonzero(scene["map"]), f"{label} map is empty")
        _require(_nonzero(scene["display_name"]), f"{label} display_name is empty")
        _require(isinstance(scene["anchor"], dict), f"{label} anchor must be an object")
        _require(isinstance(scene["witness"], dict), f"{label} witness must be an object")
        _require(isinstance(scene["actors"], list), f"{label} actors must be an array")
        for contract_key in ("occlusion_contract", "twin_contract", "admission"):
            _require(isinstance(scene[contract_key], dict), f"{label} {contract_key} must be an object")
        if _is_formal_canary(scene):
            formal_scenes.append((str(scene_id), scene))

    _require(formal_scenes, "scene registry has no status=formal/formal_canary scene")
    formal_actor_count = sum(len(scene["actors"]) for _, scene in formal_scenes)
    _require(
        formal_actor_count == EXPECTED_SCENE_ACTORS_EXCLUDING_WEARER,
        "formal canary must have exactly 39 registered actors excluding the wearer "
        f"(40 including wearer), got {formal_actor_count}",
    )

    dynamic_actors: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for scene_id, scene in formal_scenes:
        actors = scene["actors"]
        _audit_actor_identity(scene_id, actors)
        for index, actor in enumerate(actors):
            label = f"scene {scene_id!r} actor[{index}]"
            _require(isinstance(actor, dict), f"{label} must be an object")
            _require_fields(actor, ACTOR_KEYS, label)
            for identity_key in ACTOR_KEYS:
                _require(_nonzero(actor[identity_key]), f"{label} {identity_key} is empty")
            _require(_normal(actor["role"]) != "wearer", f"{label} must exclude the wearer")
            placements = [key for key in PLACEMENT_KEYS if key in actor]
            _require(
                len(placements) == 1,
                f"{label} must declare exactly one of {PLACEMENT_KEYS}, got {placements}",
            )
            placement_key = placements[0]
            _require(_nonzero(actor[placement_key]), f"{label} {placement_key} is empty")
            asset_id = actor["asset_id"]
            _require(asset_id in assets, f"{label} references unknown asset_id {asset_id!r}")
            asset = assets[asset_id]
            if placement_key in {"trajectory_ref", "trajectory_key"}:
                trajectory_id = actor[placement_key]
                _require(
                    isinstance(trajectory_id, str) and trajectory_id in trajectories,
                    f"{label} {placement_key} does not resolve in top-level trajectories: "
                    f"{trajectory_id!r}",
                )
                dynamic_actors.append((label, actor, asset))

    _require(
        len(dynamic_actors) == EXPECTED_DYNAMIC_TARGETS,
        f"formal canary must have exactly 16 trajectory-defined dynamic targets, got {len(dynamic_actors)}",
    )

    role_families: list[str] = []
    for label, actor, asset in dynamic_actors:
        _require(
            actor.get("risk_participation") is True,
            f"{label} dynamic target must set risk_participation=true",
        )
        _require(
            actor.get("collision_relevant") is True,
            f"{label} dynamic target must set collision_relevant=true",
        )
        _require(
            asset["risk_participation"] is True,
            f"{label} referenced asset must set risk_participation=true",
        )
        _require(
            asset["collision_relevant"] is True,
            f"{label} referenced asset must set collision_relevant=true",
        )
        _require(
            _nonzero(asset["trajectory_ref"]),
            f"{label} referenced asset trajectory_ref must be non-zero",
        )
        motion_policy = _normal(asset["motion_policy"])
        _require(
            motion_policy not in {"static", "fixed", "none", "disabled", "immobile"},
            f"{label} dynamic target references static motion_policy {asset['motion_policy']!r}",
        )
        role_families.append(_normal(asset["role_family"]))

    actual_role_families = frozenset(role_families)
    missing_roles = sorted(REQUIRED_DYNAMIC_ROLE_FAMILIES - actual_role_families)
    unexpected_roles = sorted(actual_role_families - REQUIRED_DYNAMIC_ROLE_FAMILIES)
    _require(
        not missing_roles and not unexpected_roles,
        "dynamic role_family coverage must be exactly child/adult/police/bicycle/"
        "motorcycle/sedan/van/bus/hgv/emergency; "
        f"missing={missing_roles}, unexpected={unexpected_roles}",
    )

    failures = fallback_paths(value)
    _require(not failures, f"scene registry contains active fallback paths: {failures[:10]}")
    return {
        "scene_registry_id": value["registry_id"],
        "formal_scene_ids": [scene_id for scene_id, _ in formal_scenes],
        "formal_actor_count_excluding_wearer": formal_actor_count,
        "formal_actor_count_including_wearer": formal_actor_count + 1,
        "dynamic_target_count": len(dynamic_actors),
        "dynamic_role_families": sorted(actual_role_families),
    }


def audit_registries(
    asset_registry: dict[str, Any], scene_registry: dict[str, Any]
) -> dict[str, Any]:
    assets = audit_asset_registry(asset_registry)
    scene_report = audit_scene_registry(
        scene_registry,
        asset_registry_id=str(asset_registry["registry_id"]),
        assets=assets,
    )
    return {
        "asset_registry_id": asset_registry["registry_id"],
        "registered_asset_count": len(assets),
        **scene_report,
    }


def audit_registry_files(
    asset_path: Path = ASSET_REGISTRY_PATH,
    scene_path: Path = SCENE_REGISTRY_PATH,
) -> dict[str, Any]:
    asset_registry = _load_json_object(Path(asset_path), "asset registry")
    scene_registry = _load_json_object(Path(scene_path), "scene registry")
    return audit_registries(asset_registry, scene_registry)


def _valid_registry_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    role_families = [
        "child",
        "adult",
        "police",
        "bicycle",
        "motorcycle",
        "sedan",
        "van",
        "bus",
        "hgv",
        "emergency",
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
        "adult",
    ]
    assets: dict[str, dict[str, Any]] = {}
    actors: list[dict[str, Any]] = []
    trajectories: dict[str, Any] = {}
    for index in range(EXPECTED_SCENE_ACTORS_EXCLUDING_WEARER):
        asset_id = f"asset_{index + 1:02d}"
        dynamic = index < EXPECTED_DYNAMIC_TARGETS
        trajectory_id = f"trajectory_{index + 1:02d}"
        assets[asset_id] = {
            "blueprint_id": f"test.blueprint.{index + 1:02d}",
            "kind": "dynamic" if dynamic else "prop",
            "role_family": role_families[index] if dynamic else "street_furniture",
            "risk_participation": dynamic,
            "collision_relevant": dynamic,
            "motion_policy": "scripted" if dynamic else "static",
            "trajectory_ref": trajectory_id if dynamic else None,
            "physics_policy": "collision_enabled",
            "surface_policy": "project_to_surface",
            "attributes": {},
        }
        actor = {
            "instance_id": f"instance_{index + 1:02d}",
            "asset_id": asset_id,
            "track_id": f"track_{index + 1:02d}",
            "role": "dynamic_target" if dynamic else "context",
        }
        if dynamic:
            actor.update(
                {
                    "trajectory_key": trajectory_id,
                    "risk_participation": True,
                    "collision_relevant": True,
                }
            )
            trajectories[trajectory_id] = {"segments": [{"time_s": 0.0}]}
        else:
            actor["fixed_pose"] = {"forward_m": float(index), "right_m": 0.0}
        actors.append(actor)
    asset_registry = {
        "schema_version": 1,
        "registry_id": "dtr-c3-assets-test-v1",
        "carla_version": "0.9.16",
        "assets": assets,
    }
    scene_registry = {
        "schema_version": 1,
        "registry_id": "dtr-c3-scenes-test-v1",
        "asset_registry_id": asset_registry["registry_id"],
        "trajectories": trajectories,
        "scenes": {
            "canary_01": {
                "status": "formal_canary",
                "map": "Carla/Maps/Town10HD_Opt",
                "display_name": "C3 registry audit fixture",
                "weather": "ClearNoon",
                "duration_seconds": 4.0,
                "showcase_time_s": 1.25,
                "anchor": {"x": 0.0, "y": 0.0},
                "witness": {"x": 0.0, "y": 0.0, "z": 10.0},
                "actors": actors,
                "episodes": [],
                "occlusion_contract": {},
                "twin_contract": {},
                "admission": {},
            }
        },
    }
    return asset_registry, scene_registry


class RegistryAuditLogicTest(unittest.TestCase):
    def test_valid_formal_canary_meets_density_gate(self) -> None:
        report = audit_registries(*_valid_registry_pair())
        self.assertEqual(EXPECTED_SCENE_ACTORS_EXCLUDING_WEARER, report["formal_actor_count_excluding_wearer"])
        self.assertEqual(EXPECTED_SCENE_ACTORS_INCLUDING_WEARER, report["formal_actor_count_including_wearer"])
        self.assertEqual(EXPECTED_DYNAMIC_TARGETS, report["dynamic_target_count"])
        self.assertEqual(sorted(REQUIRED_DYNAMIC_ROLE_FAMILIES), report["dynamic_role_families"])

    def test_dynamic_metadata_gates_do_not_self_select_the_denominator(self) -> None:
        for field, value, message in (
            ("risk_participation", False, "risk_participation=true"),
            ("collision_relevant", False, "collision_relevant=true"),
            ("trajectory_ref", 0, "trajectory_ref must be non-zero"),
        ):
            with self.subTest(field=field):
                assets, scenes = _valid_registry_pair()
                assets["assets"]["asset_01"][field] = value
                with self.assertRaisesRegex(RegistryAuditError, message):
                    audit_registries(assets, scenes)

    def test_density_role_and_fallback_gates_are_decisive(self) -> None:
        assets, scenes = _valid_registry_pair()
        scenes["scenes"]["canary_01"]["actors"].pop()
        with self.assertRaisesRegex(RegistryAuditError, "exactly 39"):
            audit_registries(assets, scenes)

        assets, scenes = _valid_registry_pair()
        assets["assets"]["asset_01"]["role_family"] = "adult"
        with self.assertRaisesRegex(RegistryAuditError, r"missing=\['child'\]"):
            audit_registries(assets, scenes)

        assets, scenes = _valid_registry_pair()
        assets["assets"]["asset_01"]["fallback_blueprint_id"] = "test.fallback"
        with self.assertRaisesRegex(RegistryAuditError, "fallback"):
            audit_registries(assets, scenes)

    def test_registry_link_and_blueprints_are_unique(self) -> None:
        assets, scenes = _valid_registry_pair()
        scenes["asset_registry_id"] = "wrong-registry"
        with self.assertRaisesRegex(RegistryAuditError, "does not resolve"):
            audit_registries(assets, scenes)

        assets, scenes = _valid_registry_pair()
        assets["assets"]["asset_02"]["blueprint_id"] = assets["assets"]["asset_01"]["blueprint_id"].upper()
        with self.assertRaisesRegex(RegistryAuditError, "blueprint_id values"):
            audit_registries(assets, scenes)

    def test_model_only_contract_rejects_nested_truth_families(self) -> None:
        assert_model_only_truth_blind(
            {"wearable_rgb": {"path": "rgb/000001.png"}, "metric_depth": {"path": "depth/000001.png"}}
        )
        for forbidden_key in (
            "asset_registry_id",
            "current_actors",
            "scenario_role",
            "expected_outcome",
            "future_contact_within_horizon",
            "track_id",
        ):
            with self.subTest(forbidden_key=forbidden_key):
                with self.assertRaisesRegex(RegistryAuditError, "evaluator-only"):
                    assert_model_only_truth_blind({"nested": [{forbidden_key: "leak"}]})


class C3RegistryFilesTest(unittest.TestCase):
    def test_materialized_c3_registries_meet_formal_gate(self) -> None:
        try:
            report = audit_registry_files()
        except RegistryAuditError as error:
            self.fail(str(error))
        self.assertEqual(EXPECTED_SCENE_ACTORS_INCLUDING_WEARER, report["formal_actor_count_including_wearer"])
        self.assertEqual(EXPECTED_DYNAMIC_TARGETS, report["dynamic_target_count"])


if __name__ == "__main__":
    unittest.main()
