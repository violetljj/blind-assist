"""Join immutable per-map C2 roots into one truth-blind C4 multimap package.

The C4 compiler/runner owns the outer index.  This module deliberately treats
every child evidence root as immutable: it validates the child's C2 result,
protocol, live trees, exact model contract, and truth separation before copying
the child model/evaluator trees into group-scoped namespaces.

The final runtime bundle extends the CARLA-free runner input with an exact
``registries`` object containing two immutable links and, for every map group,
the completed child evidence path and result hash.  Scene class, episode, and
dynamic-target truth is derived only
from the frozen evaluator-side registries; it is never repeated in the model
package or trusted from an ad-hoc join argument.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

from dtr_carla_c2_rich_scene import (
    MODEL_TOP_LEVEL_ALLOWLIST,
    plan_waypoints_world,
    point_polygon_distance,
    sha256_file,
    write_json_atomic,
)
from dtr_carla_c3_scene import load_json, sha256_json
from join_dtr_carla_c3_dynamic_risk import (
    _c3_semantic_truth_paths,
    _model_schema_failures,
    _model_truth_failures,
    _read_jsonl,
    _safe_relative,
    _seal_tree,
    _sealed_tree_manifest_matches,
)


C4_EXPERIMENT_ID = "DTR_CARLA_C4_MULTIMAP_WORLD_PACK_V1"
C4_INDEX_SCHEMA = "dtr-carla-c4-multimap-compiled-v1"
C4_RESULT_SCHEMA = "dtr-c4-multimap-result-v1"
C4_LAYOUT_AUDIT_SCHEMA = "dtr-c4-multimap-layout-coverage-audit-v1"
C4_OCCLUSION_AUDIT_SCHEMA = "dtr-c4-multimap-pack-occlusion-audit-v1"
C4_RESULT_STATUS = "DTR_CARLA_C4_MULTIMAP_SOURCE_COMPLETE"
C2_EXPERIMENT_ID = "DTR_CARLA_C2_RICH_MULTILAYOUT_OCCLUSION_SOURCE_V2"
C2_RESULT_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"
C2_RESULT_NOT_EVALUABLE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_NOT_EVALUABLE"
C2_SHARD_STATUS = "DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE"
C2_PACK_LEVEL_CHECKS = frozenset(
    {
        "track_then_complete_physical_occlusion_contract_met",
        "contact_safe_outcome_pair_matches",
    }
)
C2_DEFERRED_OCCLUSION_CHECK = (
    "track_then_complete_physical_occlusion_contract_deferred_to_c4_final_join"
)
FORMAL_SENSORS = ("instance", "wearable", "depth", "witness")
ASSET_REGISTRY_SCHEMA = "dtr-c4-asset-registry-v1"
SCENE_REGISTRY_SCHEMA = "dtr-c4-scene-registry-v1"
WEARER_RADIUS_M = 0.45
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
ALLOWED_MAP_STARTUP_ARGUMENTS = {
    "Carla/Maps/Town01": "/Game/Carla/Maps/Town01.Town01",
    "Carla/Maps/Town02": "/Game/Carla/Maps/Town02.Town02",
    "Carla/Maps/Town03_Opt": "/Game/Carla/Maps/Town03_Opt.Town03_Opt",
    "Carla/Maps/Town04": "/Game/Carla/Maps/Town04.Town04",
    "Carla/Maps/Town05": "/Game/Carla/Maps/Town05.Town05",
    "Carla/Maps/Town10HD_Opt": "/Game/Carla/Maps/Town10HD_Opt.Town10HD_Opt",
}

INDEX_EXACT_KEYS = {
    "schema_version",
    "experiment_id",
    "registries",
    "capture",
    "admission",
    "map_layout_groups",
}
REGISTRY_LINK_EXACT_KEYS = {"path", "sha256"}
REGISTRIES_EXACT_KEYS = {"asset_registry", "scene_registry"}
CAPTURE_EXACT_KEYS = {"resolution", "sensor_order"}
ADMISSION_EXACT_KEYS = {
    "expected_map_count",
    "expected_protocol_count",
    "expected_layout_count",
    "expected_episode_count",
    "expected_sensor_count",
    "expected_shard_count",
}
GROUP_EXACT_KEYS = {
    "group_id",
    "map",
    "startup_map_argument",
    "layout_ids",
    "protocol_path",
    "protocol_sha256",
    "evidence_path",
    "evidence_result_sha256",
}
ASSET_REGISTRY_EXACT_KEYS = {
    "schema_version",
    "registry_id",
    "carla_version",
    "description",
    "source_registry",
    "scripted_actor_policy",
    "dynamic_footprint_receipt",
    "assets",
    "claim_boundary",
}
SCENE_REGISTRY_EXACT_KEYS = {
    "schema_version",
    "registry_id",
    "asset_registry_id",
    "description",
    "map_inventory",
    "required_scene_classes",
    "capture_contract",
    "trajectories",
    "scenes",
    "claim_boundary",
}
MODEL_ROOT_MANIFEST_EXACT_KEYS = {
    "schema_version",
    "experiment_id",
    "camera_calibration",
    "model_contract",
    "rgbd_alignment_receipt",
    "episodes",
}
MODEL_ROOT_LINK_EXACT_KEYS = {"path", "sha256"}
ALIGNMENT_ROOT_LINK_EXACT_KEYS = {"path", "receipt_sha256", "sha256"}
MODEL_ROOT_EPISODE_LINK_EXACT_KEYS = {
    "episode_id",
    "manifest_path",
    "manifest_sha256",
}
EPISODE_MANIFEST_EXACT_KEYS = {
    "schema_version",
    "episode_id",
    "frames",
    "observations_sha256",
    "rgb_payloads",
    "depth_payloads",
    "navigation_session_id",
    "rgbd_alignment",
    "issued_plan",
}
EPISODE_ALIGNMENT_EXACT_KEYS = {
    "authority",
    "receipt_path",
    "receipt_sha256",
    "depth_minus_wearable_source_world_frame_offset",
}
EPISODE_PLAN_EXACT_KEYS = {
    "authority",
    "path",
    "receipt_sha256",
    "file_sha256",
}
ALIGNMENT_RECEIPT_EXACT_KEYS = {
    "schema_version",
    "experiment_id",
    "authority",
    "world_frame_rule",
    "matching_keys",
    "verified_equal_fields",
    "episodes",
    "receipt_sha256",
}
ALIGNMENT_EPISODE_EXACT_KEYS = {
    "episode_id",
    "frames",
    "wearable_source_world_frame_first",
    "wearable_source_world_frame_last",
    "depth_source_world_frame_first",
    "depth_source_world_frame_last",
    "depth_minus_wearable_source_world_frame_offset",
    "alignment_projection_sha256",
}
PLAN_EXACT_KEYS = {
    "schema_version",
    "episode_id",
    "navigation_session_id",
    "layout_anchor",
    "issued_plan",
}
PLAN_ISSUED_EXACT_KEYS = {
    "authority",
    "receipt",
    "receipt_sha256",
    "world_coordinate_frame",
    "time_parameterized_waypoints_world",
}
LAYOUT_ANCHOR_EXACT_KEYS = {
    "world_center_xy_m",
    "world_forward_xy",
    "world_right_xy",
}
PLAN_RECEIPT_EXACT_KEYS = {
    "schema_version",
    "plan_id",
    "session_id",
    "issued_at_s",
    "expires_at_s",
    "coordinate_frame",
    "time_parameterized_waypoints",
    "receipt_sha256",
}
PLAN_WAYPOINT_EXACT_KEYS = {"time_s", "forward_m", "right_m"}
WORLD_WAYPOINT_EXACT_KEYS = {"time_s", "x_m", "y_m"}
MODEL_CONTRACT_EXACT_KEYS = {
    "schema_version",
    "current_actors_enabled",
    "dense_modalities",
    "evaluator_sibling_not_required",
    "rgbd_alignment",
    "record_top_level_allowlist",
}
MODEL_CONTRACT_ALIGNMENT_EXACT_KEYS = {
    "authority",
    "receipt_path",
    "receipt_sha256",
    "file_sha256",
    "world_frame_rule",
}
CAMERA_CALIBRATION_EXACT_KEYS = {
    "schema_version",
    "resolution",
    "fov_degrees",
    "K",
    "depth_codec",
    "wearable_rigid_extrinsic",
    "sensor_tick_seconds",
}
RIGID_EXTRINSIC_EXACT_KEYS = {
    "pitch_degrees",
    "roll_degrees",
    "x_m",
    "y_m",
    "yaw_degrees",
    "z_m",
}
DEPTH_CODEC_EXACT_KEYS = {"formula", "maximum_depth_m", "name"}
OUTER_MODEL_MANIFEST_EXACT_KEYS = {
    "schema_version",
    "experiment_id",
    "groups",
}
OUTER_MODEL_GROUP_EXACT_KEYS = {
    "group_id",
    "model_root",
    "child_model_root_manifest_sha256",
    "child_sealed_model_manifest_sha256",
    "model_file_count",
}


class C4ContractError(RuntimeError):
    """Raised before materialization when an immutable child contract fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} keys differ: expected={sorted(expected)} actual={actual}")


def _nonempty_identifier(value: Any, label: str) -> str:
    normalized = str(value).strip()
    if not SAFE_ID.fullmatch(normalized):
        raise ValueError(f"{label} is not a safe identifier: {normalized!r}")
    if normalized.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        raise ValueError(f"{label} is a reserved Windows device name")
    return normalized


def _load_json_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if not SHA256.fullmatch(digest):
        raise ValueError(f"{label} is not a SHA-256")
    return digest.lower()


def _model_visible_identifier(value: Any, label: str) -> str:
    identifier = _nonempty_identifier(value, label)
    if _c3_semantic_truth_paths(identifier):
        raise ValueError(f"{label} leaks evaluator semantics: {identifier!r}")
    return identifier


def _registry_contract(
    asset_registry: dict[str, Any], scene_registry: dict[str, Any]
) -> dict[str, Any]:
    _require_exact_keys(asset_registry, ASSET_REGISTRY_EXACT_KEYS, "C4 asset registry")
    _require_exact_keys(scene_registry, SCENE_REGISTRY_EXACT_KEYS, "C4 scene registry")
    if asset_registry.get("schema_version") != ASSET_REGISTRY_SCHEMA:
        raise ValueError("unexpected C4 asset registry schema")
    if scene_registry.get("schema_version") != SCENE_REGISTRY_SCHEMA:
        raise ValueError("unexpected C4 scene registry schema")
    if scene_registry.get("asset_registry_id") != asset_registry.get("registry_id"):
        raise ValueError("C4 scene registry is not bound to the asset registry")

    raw_assets = asset_registry.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("C4 asset registry assets must be a nonempty array")
    assets: dict[str, dict[str, Any]] = {}
    for index, asset in enumerate(raw_assets):
        if not isinstance(asset, dict):
            raise ValueError(f"asset[{index}] must be an object")
        asset_id = _nonempty_identifier(asset.get("asset_id"), f"asset[{index}].asset_id")
        if asset_id in assets:
            raise ValueError(f"duplicate C4 asset_id: {asset_id}")
        if not isinstance(asset.get("risk_participation"), bool):
            raise ValueError(f"asset {asset_id} risk_participation must be boolean")
        if (asset.get("mobility") == "dynamic") != bool(asset["risk_participation"]):
            raise ValueError(
                f"asset {asset_id} dynamic mobility and risk participation differ"
            )
        assets[asset_id] = asset

    raw_families = scene_registry.get("required_scene_classes")
    if not isinstance(raw_families, list) or len(raw_families) != 8:
        raise ValueError("C4 must declare exactly eight required scene classes")
    families = [_nonempty_identifier(value, "required_scene_class") for value in raw_families]
    if len(families) != len(set(families)):
        raise ValueError("C4 required scene classes contain duplicates")
    capture_contract = scene_registry.get("capture_contract")
    if (
        not isinstance(capture_contract, dict)
        or capture_contract.get("resolution") != [1280, 720]
        or capture_contract.get("sensor_order") != list(FORMAL_SENSORS)
    ):
        raise ValueError("C4 scene registry formal capture contract differs")
    raw_scenes = scene_registry.get("scenes")
    if not isinstance(raw_scenes, dict) or not raw_scenes:
        raise ValueError("C4 scene registry scenes must be a nonempty object")

    layouts: dict[str, dict[str, Any]] = {}
    global_episode_ids: set[str] = set()
    family_layouts = {family: [] for family in families}
    for raw_layout_id, scene in raw_scenes.items():
        layout_id = _model_visible_identifier(raw_layout_id, "layout_id")
        if not isinstance(scene, dict):
            raise ValueError(f"scene {layout_id} must be an object")
        family = _nonempty_identifier(scene.get("scenario_class"), f"{layout_id}.scenario_class")
        if family not in family_layouts:
            raise ValueError(f"{layout_id} has an undeclared scene class: {family}")
        map_name = str(scene.get("map", "")).strip()
        if not map_name:
            raise ValueError(f"{layout_id}.map must be nonempty")
        raw_actors = scene.get("actors")
        if not isinstance(raw_actors, list) or not raw_actors:
            raise ValueError(f"{layout_id}.actors must be a nonempty array")
        actor_ids: list[str] = []
        dynamic_target_ids: list[str] = []
        for actor_index, actor in enumerate(raw_actors):
            if not isinstance(actor, dict):
                raise ValueError(f"{layout_id}.actors[{actor_index}] must be an object")
            actor_id = _nonempty_identifier(
                actor.get("instance_id"), f"{layout_id}.actors[{actor_index}].instance_id"
            )
            asset_id = _nonempty_identifier(
                actor.get("asset_id"), f"{layout_id}.actors[{actor_index}].asset_id"
            )
            if asset_id not in assets:
                raise ValueError(f"{layout_id} references unknown asset {asset_id}")
            actor_ids.append(actor_id)
            if assets[asset_id].get("mobility") == "dynamic":
                dynamic_target_ids.append(actor_id)
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError(f"{layout_id} actor instance IDs contain duplicates")
        if not dynamic_target_ids:
            raise ValueError(f"{layout_id} has no dynamic risk targets")

        admission = scene.get("admission")
        if not isinstance(admission, dict):
            raise ValueError(f"{layout_id}.admission must be an object")
        if int(admission.get("dynamic_target_count", -1)) != len(dynamic_target_ids):
            raise ValueError(f"{layout_id} dynamic target count differs from registry truth")
        minimum_visible = int(
            admission.get("minimum_visible_frames_per_dynamic_target_per_episode", 0)
        )
        risk_threshold = float(admission.get("risk_corridor_threshold_m", 0.0))
        if minimum_visible < 1 or risk_threshold <= 0.0:
            raise ValueError(f"{layout_id} has invalid dynamic admission thresholds")

        raw_episodes = scene.get("episodes")
        if not isinstance(raw_episodes, list) or not raw_episodes:
            raise ValueError(f"{layout_id}.episodes must be a nonempty array")
        episodes: list[dict[str, str]] = []
        for episode_index, episode in enumerate(raw_episodes):
            if not isinstance(episode, dict):
                raise ValueError(f"{layout_id}.episodes[{episode_index}] must be an object")
            episode_id = _model_visible_identifier(
                episode.get("episode_id"), f"{layout_id}.episodes[{episode_index}].episode_id"
            )
            navigation_session_id = _model_visible_identifier(
                episode.get("navigation_session_id"),
                f"{layout_id}.episodes[{episode_index}].navigation_session_id",
            )
            issued_plan = episode.get("issued_plan")
            if not isinstance(issued_plan, dict):
                raise ValueError(f"{layout_id}.{episode_id}.issued_plan must be an object")
            plan_id = _model_visible_identifier(
                issued_plan.get("plan_id"), f"{layout_id}.{episode_id}.plan_id"
            )
            if episode_id in global_episode_ids:
                raise ValueError(f"duplicate C4 episode_id: {episode_id}")
            global_episode_ids.add(episode_id)
            episodes.append(
                {
                    "episode_id": episode_id,
                    "layout_id": layout_id,
                    "navigation_session_id": navigation_session_id,
                    "plan_id": plan_id,
                }
            )
        layouts[layout_id] = {
            "layout_id": layout_id,
            "layout_family_id": family,
            "map": map_name,
            "actor_ids": actor_ids,
            "dynamic_target_ids": dynamic_target_ids,
            "minimum_visible_frames": minimum_visible,
            "risk_corridor_threshold_m": risk_threshold,
            "episodes": episodes,
        }
        family_layouts[family].append(layout_id)

    missing_families = [family for family, values in family_layouts.items() if not values]
    if missing_families:
        raise ValueError(f"C4 scene classes lack layouts: {missing_families}")
    return {
        "assets": assets,
        "families": families,
        "family_layouts": family_layouts,
        "layouts": layouts,
        "episode_ids": global_episode_ids,
    }


def validate_multimap_index(
    index: dict[str, Any],
    asset_registry: dict[str, Any],
    scene_registry: dict[str, Any],
) -> dict[str, Any]:
    """Validate the final enriched runner bundle and return registry truth metadata."""

    _require_exact_keys(index, INDEX_EXACT_KEYS, "C4 compiled protocol")
    if index.get("schema_version") != C4_INDEX_SCHEMA:
        raise ValueError("unexpected C4 multimap compiled schema")
    if index.get("experiment_id") != C4_EXPERIMENT_ID:
        raise ValueError("unexpected C4 multimap experiment identity")
    capture = index.get("capture")
    _require_exact_keys(capture, CAPTURE_EXACT_KEYS, "capture")
    if capture.get("resolution") != [1280, 720]:
        raise ValueError("C4 capture resolution must be exactly 1280x720")
    if capture.get("sensor_order") != list(FORMAL_SENSORS):
        raise ValueError("C4 capture must use instance,wearable,depth,witness")
    registries = index.get("registries")
    _require_exact_keys(registries, REGISTRIES_EXACT_KEYS, "registries")
    for key in ("asset_registry", "scene_registry"):
        _require_exact_keys(registries.get(key), REGISTRY_LINK_EXACT_KEYS, key)
        _require_sha256(registries[key]["sha256"], f"{key}.sha256")
        if not isinstance(registries[key]["path"], str) or not registries[key]["path"].strip():
            raise ValueError(f"{key}.path must be nonempty")

    contract = _registry_contract(asset_registry, scene_registry)
    groups = index.get("map_layout_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("C4 compiled protocol requires map_layout_groups")
    group_ids: list[str] = []
    indexed_layouts: set[str] = set()
    indexed_maps: set[str] = set()
    for group_index, group in enumerate(groups):
        label = f"map_layout_groups[{group_index}]"
        _require_exact_keys(group, GROUP_EXACT_KEYS, label)
        group_id = _model_visible_identifier(group.get("group_id"), f"{label}.group_id")
        map_name = str(group.get("map", "")).strip()
        if map_name not in ALLOWED_MAP_STARTUP_ARGUMENTS:
            raise ValueError(f"{label}.map is not an installed C4 map")
        if group.get("startup_map_argument") != ALLOWED_MAP_STARTUP_ARGUMENTS[map_name]:
            raise ValueError(f"{label}.startup_map_argument does not bind its map")
        for path_key in ("protocol_path", "evidence_path"):
            if not isinstance(group.get(path_key), str) or not group[path_key].strip():
                raise ValueError(f"{label}.{path_key} must be nonempty")
        for hash_key in ("protocol_sha256", "evidence_result_sha256"):
            _require_sha256(group.get(hash_key), f"{label}.{hash_key}")
        raw_layout_ids = group.get("layout_ids")
        if not isinstance(raw_layout_ids, list) or not raw_layout_ids:
            raise ValueError(f"{label}.layout_ids must be a nonempty array")
        layout_ids = [
            _model_visible_identifier(value, f"{label}.layout_ids")
            for value in raw_layout_ids
        ]
        if len(layout_ids) != len(set(layout_ids)):
            raise ValueError(f"{label}.layout_ids contain duplicates")
        unknown = sorted(set(layout_ids) - set(contract["layouts"]))
        if unknown:
            raise ValueError(f"{label} has layouts absent from the scene registry: {unknown}")
        if any(contract["layouts"][layout_id]["map"] != map_name for layout_id in layout_ids):
            raise ValueError(f"{label} map differs from its registry layouts")
        overlap = sorted(indexed_layouts.intersection(layout_ids))
        if overlap:
            raise ValueError(f"layouts occur in multiple map groups: {overlap}")
        indexed_layouts.update(layout_ids)
        indexed_maps.add(map_name)
        group_ids.append(group_id)
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("C4 map group IDs must be unique")
    if indexed_layouts != set(contract["layouts"]):
        raise ValueError("C4 compiled and registry layout sets differ")
    if len(indexed_maps) < 4:
        raise ValueError("C4 requires at least four distinct maps")
    admission = index.get("admission")
    _require_exact_keys(admission, ADMISSION_EXACT_KEYS, "admission")
    for key, actual in (
        ("expected_map_count", len(indexed_maps)),
        ("expected_protocol_count", len(groups)),
        ("expected_layout_count", len(indexed_layouts)),
        ("expected_episode_count", len(contract["episode_ids"])),
        ("expected_sensor_count", len(FORMAL_SENSORS)),
        ("expected_shard_count", len(groups) * len(FORMAL_SENSORS)),
    ):
        if int(admission.get(key, -1)) != actual:
            raise ValueError(f"C4 admission {key} is inconsistent")
    return contract


def _resolve_index_path(index_path: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = index_path.parent / candidate
    return candidate.resolve(strict=True)


def _resolve_contained_protocol_path(index_path: Path, value: str) -> Path:
    candidate = _resolve_index_path(index_path, value)
    base = index_path.parent.resolve(strict=True)
    try:
        common = Path(os.path.commonpath((os.fspath(base), os.fspath(candidate))))
    except ValueError as exc:
        raise C4ContractError(f"protocol is outside the compiled bundle: {candidate}") from exc
    if os.path.normcase(os.fspath(common)) != os.path.normcase(os.fspath(base)):
        raise C4ContractError(f"protocol is outside the compiled bundle: {candidate}")
    if not candidate.is_file():
        raise C4ContractError(f"protocol is unavailable: {candidate}")
    return candidate


def _load_bound_registry(
    index_path: Path, index: dict[str, Any], key: str
) -> tuple[Path, dict[str, Any]]:
    registries = index.get("registries")
    _require_exact_keys(registries, REGISTRIES_EXACT_KEYS, "registries")
    link = registries.get(key)
    _require_exact_keys(link, REGISTRY_LINK_EXACT_KEYS, key)
    path = _resolve_index_path(index_path, str(link["path"]))
    if not path.is_file():
        raise C4ContractError(f"{key} is unavailable: {path}")
    if sha256_file(path).lower() != _require_sha256(link["sha256"], f"{key}.sha256"):
        raise C4ContractError(f"{key} hash differs from the compiled protocol")
    value = load_json(path)
    return path, value


def _rounded(value: Any, digits: int = 5) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _rounded(child, digits) for key, child in sorted(value.items())}
    if isinstance(value, list):
        return [_rounded(child, digits) for child in value]
    return value


def _model_file(model_root: Path, relative: str) -> Path:
    path = (model_root / relative).resolve()
    path.relative_to(model_root.resolve())
    return path


def _hash_link_failures(
    model_root: Path, link: Any, expected_keys: set[str], label: str
) -> list[str]:
    failures: list[str] = []
    if not isinstance(link, dict) or set(link) != expected_keys:
        return [f"{label}:keys"]
    try:
        path = _model_file(model_root, str(link["path"]))
        if not path.is_file() or sha256_file(path) != str(link["sha256"]):
            failures.append(f"{label}:sha256")
    except Exception as exc:
        failures.append(f"{label}:path:{exc}")
    return failures


def _plan_contract_failures(
    model_root: Path,
    episode_id: str,
    episode_manifest: dict[str, Any],
    observations: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    manifest_plan = episode_manifest.get("issued_plan")
    if not isinstance(manifest_plan, dict) or set(manifest_plan) != EPISODE_PLAN_EXACT_KEYS:
        return [f"{episode_id}:episode_manifest:issued_plan:keys"]
    try:
        plan_path = _model_file(model_root, str(manifest_plan["path"]))
        if not plan_path.is_file() or sha256_file(plan_path) != str(manifest_plan["file_sha256"]):
            failures.append(f"{episode_id}:plan:file_sha256")
        plan = load_json(plan_path)
    except Exception as exc:
        return [f"{episode_id}:plan:path:{exc}"]
    if set(plan) != PLAN_EXACT_KEYS:
        failures.append(f"{episode_id}:plan:keys")
        return failures
    if plan.get("schema_version") != "dtr-c2-model-plan-v1":
        failures.append(f"{episode_id}:plan:schema_version")
    if plan.get("episode_id") != episode_id:
        failures.append(f"{episode_id}:plan:episode_id")
    if plan.get("navigation_session_id") != episode_manifest.get("navigation_session_id"):
        failures.append(f"{episode_id}:plan:navigation_session_id")
    anchor = plan.get("layout_anchor")
    anchor_valid = False
    if not isinstance(anchor, dict) or set(anchor) != LAYOUT_ANCHOR_EXACT_KEYS:
        failures.append(f"{episode_id}:plan:layout_anchor:keys")
    else:
        center = anchor.get("world_center_xy_m", [])
        forward = anchor.get("world_forward_xy", [])
        right = anchor.get("world_right_xy", [])
        if not all(isinstance(value, list) and len(value) == 2 for value in (center, forward, right)):
            failures.append(f"{episode_id}:plan:layout_anchor:shape")
        elif (
            not math.isclose(math.hypot(*map(float, forward)), 1.0, abs_tol=1e-3)
            or not math.isclose(math.hypot(*map(float, right)), 1.0, abs_tol=1e-3)
            or not math.isclose(
                float(forward[0]) * float(right[0]) + float(forward[1]) * float(right[1]),
                0.0,
                abs_tol=1e-3,
            )
        ):
            failures.append(f"{episode_id}:plan:layout_anchor:axes")
        else:
            anchor_valid = True
    issued = plan.get("issued_plan")
    if not isinstance(issued, dict) or set(issued) != PLAN_ISSUED_EXACT_KEYS:
        failures.append(f"{episode_id}:plan:issued_plan:keys")
        return failures
    if issued.get("world_coordinate_frame") != "CARLA_WORLD_XY":
        failures.append(f"{episode_id}:plan:world_coordinate_frame")
    receipt = issued.get("receipt")
    if issued.get("authority") == "NO_PLAN":
        if receipt is not None or issued.get("receipt_sha256") is not None or issued.get(
            "time_parameterized_waypoints_world"
        ) != []:
            failures.append(f"{episode_id}:plan:no_plan")
    elif issued.get("authority") == "VALID":
        if not isinstance(receipt, dict) or set(receipt) != PLAN_RECEIPT_EXACT_KEYS:
            failures.append(f"{episode_id}:plan:receipt:keys")
        else:
            if (
                receipt.get("schema_version") != "dtr-c2-plan-receipt-v1"
                or receipt.get("coordinate_frame") != "LAYOUT_FORWARD_RIGHT"
            ):
                failures.append(f"{episode_id}:plan:receipt:identity")
            receipt_payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            if sha256_json(receipt_payload) != str(receipt["receipt_sha256"]):
                failures.append(f"{episode_id}:plan:receipt:self_hash")
            if issued.get("receipt_sha256") != receipt.get("receipt_sha256"):
                failures.append(f"{episode_id}:plan:receipt:issued_hash")
            if receipt.get("session_id") != plan.get("navigation_session_id"):
                failures.append(f"{episode_id}:plan:receipt:session")
            waypoints = receipt.get("time_parameterized_waypoints")
            if not isinstance(waypoints, list) or not waypoints or any(
                not isinstance(value, dict) or set(value) != PLAN_WAYPOINT_EXACT_KEYS
                for value in waypoints
            ):
                failures.append(f"{episode_id}:plan:receipt:waypoints")
            else:
                waypoint_times = [float(value["time_s"]) for value in waypoints]
                if (
                    waypoint_times != sorted(set(waypoint_times))
                    or float(receipt.get("expires_at_s", -1.0))
                    < float(receipt.get("issued_at_s", 0.0))
                    or any(
                        time_s < float(receipt["issued_at_s"])
                        or time_s > float(receipt["expires_at_s"])
                        for time_s in waypoint_times
                    )
                ):
                    failures.append(f"{episode_id}:plan:receipt:time_range")
            world_waypoints = issued.get("time_parameterized_waypoints_world")
            if not isinstance(world_waypoints, list) or any(
                not isinstance(value, dict) or set(value) != WORLD_WAYPOINT_EXACT_KEYS
                for value in world_waypoints
            ):
                failures.append(f"{episode_id}:plan:world_waypoints:keys")
            elif anchor_valid and _rounded(world_waypoints) != _rounded(
                plan_waypoints_world(receipt, {
                    "center_xy_m": anchor["world_center_xy_m"],
                    "forward_xy": anchor["world_forward_xy"],
                    "right_xy": anchor["world_right_xy"],
                })
            ):
                failures.append(f"{episode_id}:plan:world_waypoints:projection")
    else:
        failures.append(f"{episode_id}:plan:authority")
    if issued.get("receipt_sha256") != manifest_plan.get("receipt_sha256"):
        failures.append(f"{episode_id}:plan:manifest_receipt")
    for index, observation in enumerate(observations):
        navigation = observation.get("navigation", {})
        if navigation.get("navigation_session_id") != plan.get("navigation_session_id"):
            failures.append(f"{episode_id}:observations[{index}]:navigation_session")
        reference = navigation.get("issued_plan", {})
        if (
            reference.get("path") != manifest_plan.get("path")
            or reference.get("authority") != issued.get("authority")
            or reference.get("receipt_sha256") != issued.get("receipt_sha256")
        ):
            failures.append(f"{episode_id}:observations[{index}]:plan_reference")
    return failures


def _model_root_contract_failures(
    model_root: Path, expected_episode_ids: set[str]
) -> list[str]:
    """Audit root/episode manifests, hashes, plans, and RGB-D alignment."""

    failures = list(_model_schema_failures(model_root))
    calibration: dict[str, Any] = {}
    contract_alignment: dict[str, Any] = {}
    manifest_path = model_root / "manifest.json"
    if not manifest_path.is_file():
        return failures + ["model/manifest.json:missing"]
    manifest = load_json(manifest_path)
    if set(manifest) != MODEL_ROOT_MANIFEST_EXACT_KEYS:
        failures.append("model/manifest.json:keys")
        return failures
    if (
        manifest.get("schema_version") != "dtr-c2-model-root-manifest-v2"
        or manifest.get("experiment_id") != C2_EXPERIMENT_ID
    ):
        failures.append("model/manifest.json:identity")
    failures.extend(
        _hash_link_failures(
            model_root,
            manifest.get("camera_calibration"),
            MODEL_ROOT_LINK_EXACT_KEYS,
            "model/manifest.json:camera_calibration",
        )
    )
    failures.extend(
        _hash_link_failures(
            model_root,
            manifest.get("model_contract"),
            MODEL_ROOT_LINK_EXACT_KEYS,
            "model/manifest.json:model_contract",
        )
    )
    failures.extend(
        _hash_link_failures(
            model_root,
            manifest.get("rgbd_alignment_receipt"),
            ALIGNMENT_ROOT_LINK_EXACT_KEYS,
            "model/manifest.json:rgbd_alignment_receipt",
        )
    )
    try:
        calibration = load_json(
            _model_file(model_root, str(manifest["camera_calibration"]["path"]))
        )
        if set(calibration) != CAMERA_CALIBRATION_EXACT_KEYS:
            failures.append("camera_calibration.json:keys")
        if calibration.get("schema_version") != "dtr-c2-model-camera-contract-v1":
            failures.append("camera_calibration.json:schema_version")
        if calibration.get("resolution") != {"width": 1280, "height": 720}:
            failures.append("camera_calibration.json:resolution")
        expected_k = [
            [640.0, 0.0, 640.0],
            [0.0, 640.0, 360.0],
            [0.0, 0.0, 1.0],
        ]
        calibration_k = calibration.get("K")
        k_matches = (
            isinstance(calibration_k, list)
            and len(calibration_k) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in calibration_k)
            and all(
                math.isclose(
                    float(calibration_k[row][column]),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                for row, values in enumerate(expected_k)
                for column, expected in enumerate(values)
            )
        )
        if (
            float(calibration.get("fov_degrees", 0.0)) != 90.0
            or not k_matches
            or not isinstance(calibration.get("wearable_rigid_extrinsic"), dict)
            or set(calibration["wearable_rigid_extrinsic"])
            != RIGID_EXTRINSIC_EXACT_KEYS
            or not isinstance(calibration.get("depth_codec"), dict)
            or set(calibration["depth_codec"]) != DEPTH_CODEC_EXACT_KEYS
            or calibration["depth_codec"].get("name")
            != "CARLA_RGB24_NORMALIZED_DEPTH"
            or float(calibration["depth_codec"].get("maximum_depth_m", 0.0))
            != 1000.0
            or calibration["depth_codec"].get("formula")
            != "meters=1000*(R+256*G+65536*B)/(16777215)"
            or float(calibration.get("sensor_tick_seconds", 0.0)) <= 0.0
        ):
            failures.append("camera_calibration.json:contract")
    except Exception as exc:
        failures.append(f"camera_calibration.json:{exc}")
    try:
        contract = load_json(_model_file(model_root, str(manifest["model_contract"]["path"])))
        if set(contract) != MODEL_CONTRACT_EXACT_KEYS:
            failures.append("model_contract.json:keys")
        if contract.get("schema_version") != "dtr-c2-model-contract-v2":
            failures.append("model_contract.json:schema_version")
        if contract.get("current_actors_enabled") is not False:
            failures.append("model_contract.json:current_actors_enabled")
        if contract.get("dense_modalities") != ["wearable_rgb", "metric_depth"]:
            failures.append("model_contract.json:dense_modalities")
        if contract.get("evaluator_sibling_not_required") is not True:
            failures.append("model_contract.json:evaluator_sibling_not_required")
        contract_alignment = contract.get("rgbd_alignment")
        if not isinstance(contract_alignment, dict) or set(
            contract_alignment
        ) != MODEL_CONTRACT_ALIGNMENT_EXACT_KEYS:
            failures.append("model_contract.json:rgbd_alignment:keys")
        if set(contract.get("record_top_level_allowlist", [])) != MODEL_TOP_LEVEL_ALLOWLIST:
            failures.append("model_contract.json:record_top_level_allowlist")
    except Exception as exc:
        failures.append(f"model_contract.json:{exc}")

    episode_links = manifest.get("episodes")
    if not isinstance(episode_links, list) or any(
        not isinstance(value, dict) or set(value) != MODEL_ROOT_EPISODE_LINK_EXACT_KEYS
        for value in episode_links
    ):
        failures.append("model/manifest.json:episodes:keys")
        return failures
    linked_episode_ids = {str(value["episode_id"]) for value in episode_links}
    if linked_episode_ids != expected_episode_ids or len(episode_links) != len(linked_episode_ids):
        failures.append("model/manifest.json:episodes:set")

    try:
        alignment_path = _model_file(
            model_root, str(manifest["rgbd_alignment_receipt"]["path"])
        )
        alignment = load_json(alignment_path)
    except Exception as exc:
        return failures + [f"rgbd_alignment_receipt.json:{exc}"]
    if set(alignment) != ALIGNMENT_RECEIPT_EXACT_KEYS:
        failures.append("rgbd_alignment_receipt.json:keys")
        return failures
    if (
        alignment.get("schema_version")
        != "dtr-c2-model-rgbd-deterministic-replay-alignment-receipt-v1"
        or alignment.get("experiment_id") != C2_EXPERIMENT_ID
    ):
        failures.append("rgbd_alignment_receipt.json:identity")
    alignment_payload = {key: value for key, value in alignment.items() if key != "receipt_sha256"}
    if sha256_json(alignment_payload) != str(alignment["receipt_sha256"]):
        failures.append("rgbd_alignment_receipt.json:self_hash")
    if manifest["rgbd_alignment_receipt"].get("receipt_sha256") != alignment.get(
        "receipt_sha256"
    ):
        failures.append("model/manifest.json:rgbd_alignment_receipt:receipt_sha256")
    try:
        if (
            contract_alignment.get("authority") != alignment.get("authority")
            or contract_alignment.get("receipt_path")
            != manifest["rgbd_alignment_receipt"].get("path")
            or contract_alignment.get("receipt_sha256") != alignment.get("receipt_sha256")
            or contract_alignment.get("file_sha256")
            != manifest["rgbd_alignment_receipt"].get("sha256")
            or contract_alignment.get("world_frame_rule") != alignment.get("world_frame_rule")
        ):
            failures.append("model_contract.json:rgbd_alignment:binding")
    except AttributeError:
        failures.append("model_contract.json:rgbd_alignment:binding")
    if (
        alignment.get("authority") != "DETERMINISTIC_REPLAY_ALIGNMENT_VERIFIED"
        or alignment.get("matching_keys") != ["episode_id", "sample_index", "time_s"]
        or alignment.get("verified_equal_fields")
        != ["camera_world_transform", "wearer_pose_current"]
    ):
        failures.append("rgbd_alignment_receipt.json:authority")
    alignment_episodes = alignment.get("episodes")
    if not isinstance(alignment_episodes, list) or any(
        not isinstance(value, dict) or set(value) != ALIGNMENT_EPISODE_EXACT_KEYS
        for value in alignment_episodes
    ):
        failures.append("rgbd_alignment_receipt.json:episodes:keys")
        return failures
    alignment_by_episode = {str(value["episode_id"]): value for value in alignment_episodes}
    if set(alignment_by_episode) != expected_episode_ids or len(alignment_by_episode) != len(
        alignment_episodes
    ):
        failures.append("rgbd_alignment_receipt.json:episodes:set")

    for link in episode_links:
        episode_id = str(link["episode_id"])
        try:
            episode_manifest_path = _model_file(model_root, str(link["manifest_path"]))
            if sha256_file(episode_manifest_path) != str(link["manifest_sha256"]):
                failures.append(f"{episode_id}:episode_manifest:root_hash")
            episode_manifest = load_json(episode_manifest_path)
        except Exception as exc:
            failures.append(f"{episode_id}:episode_manifest:path:{exc}")
            continue
        if set(episode_manifest) != EPISODE_MANIFEST_EXACT_KEYS:
            failures.append(f"{episode_id}:episode_manifest:keys")
            continue
        if (
            episode_manifest.get("schema_version")
            != "dtr-c2-model-episode-manifest-v2"
            or episode_manifest.get("episode_id") != episode_id
        ):
            failures.append(f"{episode_id}:episode_manifest:identity")
        observations_path = episode_manifest_path.parent / "observations.jsonl"
        observations = _read_jsonl(observations_path)
        if (
            sha256_file(observations_path) != str(episode_manifest["observations_sha256"])
            or int(episode_manifest["frames"]) != len(observations)
            or int(episode_manifest["rgb_payloads"]) != len(observations)
            or int(episode_manifest["depth_payloads"]) != len(observations)
        ):
            failures.append(f"{episode_id}:episode_manifest:observations")
        episode_alignment = episode_manifest.get("rgbd_alignment")
        if not isinstance(episode_alignment, dict) or set(
            episode_alignment
        ) != EPISODE_ALIGNMENT_EXACT_KEYS:
            failures.append(f"{episode_id}:episode_manifest:rgbd_alignment:keys")
            continue
        receipt_episode = alignment_by_episode.get(episode_id)
        if receipt_episode is None:
            failures.append(f"{episode_id}:alignment:missing")
            continue
        offset = int(receipt_episode["depth_minus_wearable_source_world_frame_offset"])
        wearable_frames = [int(value["wearable_rgb"]["source_world_frame"]) for value in observations]
        depth_frames = [int(value["metric_depth"]["source_world_frame"]) for value in observations]
        projection = [
            {
                "episode_id": episode_id,
                "sample_index": int(value["sample_index"]),
                "time_s": round(float(value["time_s"]), 8),
                "world_frame": int(value["world_frame"]),
                "camera_world_transform": _rounded(value["camera"]["world_transform"]),
                "wearer_pose_current": _rounded(value["wearer_pose_current"]),
            }
            for value in observations
        ]
        if not observations or any(
            right != left + 1 for left, right in zip(wearable_frames, wearable_frames[1:])
        ) or any(right != left + 1 for left, right in zip(depth_frames, depth_frames[1:])):
            failures.append(f"{episode_id}:alignment:contiguous_frames")
        if any(depth - wearable != offset for wearable, depth in zip(wearable_frames, depth_frames)):
            failures.append(f"{episode_id}:alignment:offset")
        if observations and (
            int(receipt_episode["frames"]) != len(observations)
            or int(receipt_episode["wearable_source_world_frame_first"]) != wearable_frames[0]
            or int(receipt_episode["wearable_source_world_frame_last"]) != wearable_frames[-1]
            or int(receipt_episode["depth_source_world_frame_first"]) != depth_frames[0]
            or int(receipt_episode["depth_source_world_frame_last"]) != depth_frames[-1]
            or str(receipt_episode["alignment_projection_sha256"]) != sha256_json(projection)
        ):
            failures.append(f"{episode_id}:alignment:receipt_projection")
        if (
            episode_alignment.get("authority") != alignment.get("authority")
            or episode_alignment.get("receipt_path")
            != manifest["rgbd_alignment_receipt"].get("path")
            or episode_alignment.get("receipt_sha256") != alignment.get("receipt_sha256")
            or int(episode_alignment.get("depth_minus_wearable_source_world_frame_offset"))
            != offset
        ):
            failures.append(f"{episode_id}:episode_manifest:rgbd_alignment")
        for observation_index, observation in enumerate(observations):
            if (
                observation.get("camera", {}).get("rigid_extrinsic")
                != calibration.get("wearable_rigid_extrinsic")
                or observation.get("camera", {}).get("K") != calibration.get("K")
                or observation.get("camera", {}).get("fov_degrees")
                != calibration.get("fov_degrees")
                or observation.get("metric_depth", {}).get("codec")
                != calibration.get("depth_codec")
            ):
                failures.append(
                    f"{episode_id}:observations[{observation_index}]:camera_contract"
                )
            frame_alignment = observation.get("frame_alignment", {})
            if (
                frame_alignment.get("authority") != alignment.get("authority")
                or frame_alignment.get("reference_modality") != "wearable_rgb"
                or frame_alignment.get("receipt_path")
                != manifest["rgbd_alignment_receipt"].get("path")
                or frame_alignment.get("receipt_sha256") != alignment.get("receipt_sha256")
                or int(frame_alignment.get("depth_minus_wearable_source_world_frame_offset"))
                != offset
            ):
                failures.append(f"{episode_id}:observations[{observation_index}]:alignment")
        failures.extend(
            _plan_contract_failures(model_root, episode_id, episode_manifest, observations)
        )
    return failures


def _formal_sensor_audit(
    root: Path,
    expected_episode_ids: set[str],
    expected_map: str,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    sensors: dict[str, Any] = {}
    for sensor in FORMAL_SENSORS:
        result_path = root / "shards" / sensor / "result.json"
        failures: list[str] = []
        if not result_path.is_file():
            failures.append("missing_result")
            result: dict[str, Any] = {}
        else:
            result = load_json(result_path)
            if result.get("status") != C2_SHARD_STATUS:
                failures.append("status")
            if result.get("sensor") != sensor:
                failures.append("sensor_identity")
            if result.get("map") != expected_map:
                failures.append("map_identity")
            if str(result.get("protocol_sha256", "")).lower() != expected_protocol_sha256.lower():
                failures.append("protocol_identity")
            if result.get("checks", {}).get("all_formal_payloads_are_1280x720") is not True:
                failures.append("resolution")
            calibration_path = result_path.parent / "camera_calibration.json"
            inventory_path = result_path.parent / "payload_inventory.json"
            try:
                calibration = load_json(calibration_path)
                if (
                    calibration.get("sensor") != sensor
                    or int(calibration.get("width", 0)) != 1280
                    or int(calibration.get("height", 0)) != 720
                    or sha256_file(calibration_path)
                    != str(result.get("calibration_sha256"))
                ):
                    failures.append("calibration")
            except Exception:
                failures.append("calibration")
            try:
                inventory = _load_json_value(inventory_path)
                if (
                    not isinstance(inventory, list)
                    or not inventory
                    or any(
                        int(value.get("width", 0)) != 1280
                        or int(value.get("height", 0)) != 720
                        for value in inventory
                        if isinstance(value, dict)
                    )
                    or any(not isinstance(value, dict) for value in inventory)
                    or sha256_file(inventory_path)
                    != str(result.get("payload_inventory_sha256"))
                    or int(result.get("payload_count", -1)) != len(inventory)
                    or {
                        str(value.get("episode_id"))
                        for value in inventory
                        if isinstance(value, dict)
                    }
                    != expected_episode_ids
                ):
                    failures.append("payload_inventory")
            except Exception:
                failures.append("payload_inventory")
        observed_episode_ids = {
            path.parent.name
            for path in (root / "shards" / sensor / "episodes").glob("*/frames.jsonl")
        }
        if observed_episode_ids != expected_episode_ids:
            failures.append("episode_set")
        else:
            for episode_id in sorted(expected_episode_ids):
                try:
                    rows = _read_jsonl(
                        root
                        / "shards"
                        / sensor
                        / "episodes"
                        / episode_id
                        / "frames.jsonl"
                    )
                    if not rows or any(
                        row.get("sensor") != sensor
                        or row.get("episode_id") != episode_id
                        for row in rows
                    ):
                        failures.append(f"frames:{episode_id}")
                except Exception:
                    failures.append(f"frames:{episode_id}")
        sensors[sensor] = {
            "result_path": _safe_relative(result_path, root) if result_path.is_file() else None,
            "episode_count": len(observed_episode_ids),
            "formal_resolution": [1280, 720],
            "failures": failures,
            "passed": not failures,
        }
    return {
        "required_sensors": list(FORMAL_SENSORS),
        "sensors": sensors,
        "passed": set(sensors) == set(FORMAL_SENSORS)
        and all(value["passed"] for value in sensors.values()),
    }


def _episode_dynamic_audit(
    evaluator_root: Path,
    episode_id: str,
    target_ids: list[str],
    minimum_visible_frames: int,
    risk_corridor_threshold_m: float,
    wearer_radius_m: float,
) -> dict[str, Any]:
    frames_path = evaluator_root / "episodes" / episode_id / "frames.jsonl"
    rows = _read_jsonl(frames_path)
    targets: dict[str, dict[str, Any]] = {
        target_id: {
            "actor_state_frames": 0,
            "risk_geometry_frames": 0,
            "observed_transform_motion_frames": 0,
            "visible_frames": 0,
            "risk_corridor_frames": 0,
            "minimum_clearance_m": None,
        }
        for target_id in target_ids
    }
    previous_xy: dict[str, tuple[float, float]] = {}
    row_identity_failures = 0
    for row in rows:
        if str(row.get("episode_id")) != episode_id:
            row_identity_failures += 1
        wearer = row["actors"]["wearer"]["transform"]
        wearer_xy = (float(wearer["x"]), float(wearer["y"]))
        visibility = row["instance_visibility"]
        polygons = row["truth"]["collision_polygons_xy"]
        for target_id, summary in targets.items():
            actor = row["actors"].get(target_id)
            if actor is not None:
                summary["actor_state_frames"] += 1
                transform = actor["transform"]
                xy = (float(transform["x"]), float(transform["y"]))
                previous = previous_xy.get(target_id)
                if previous is not None and math.dist(previous, xy) > 1e-4:
                    summary["observed_transform_motion_frames"] += 1
                previous_xy[target_id] = xy
            if bool(visibility.get(target_id, {}).get("visible")):
                summary["visible_frames"] += 1
            polygon = polygons.get(target_id)
            if polygon is not None:
                summary["risk_geometry_frames"] += 1
                clearance = max(
                    0.0,
                    point_polygon_distance(wearer_xy, polygon) - wearer_radius_m,
                )
                current = summary["minimum_clearance_m"]
                summary["minimum_clearance_m"] = (
                    clearance if current is None else min(float(current), clearance)
                )
                if clearance <= risk_corridor_threshold_m:
                    summary["risk_corridor_frames"] += 1
    for summary in targets.values():
        if summary["minimum_clearance_m"] is not None:
            summary["minimum_clearance_m"] = round(
                float(summary["minimum_clearance_m"]), 6
            )
        summary["checks"] = {
            "actor_state_every_frame": summary["actor_state_frames"] == len(rows),
            "observed_transform_motion": summary["observed_transform_motion_frames"] > 0,
            "visible_for_required_frames": summary["visible_frames"]
            >= minimum_visible_frames,
            "risk_geometry_every_frame": summary["risk_geometry_frames"] == len(rows),
            "entered_risk_corridor": summary["risk_corridor_frames"] > 0,
        }
        summary["passed"] = all(summary["checks"].values())
    return {
        "episode_id": episode_id,
        "frames": len(rows),
        "row_identity_failures": row_identity_failures,
        "dynamic_target_count": len(targets),
        "targets": targets,
        "passed": bool(rows)
        and row_identity_failures == 0
        and bool(targets)
        and all(value["passed"] for value in targets.values()),
    }


def build_layout_coverage_audit(
    index: dict[str, Any],
    registry_contract: dict[str, Any],
    child_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-episode dynamic evidence into eight layout families."""

    family_by_layout = {
        str(layout_id): str(layout["layout_family_id"])
        for layout_id, layout in registry_contract["layouts"].items()
    }
    family_summaries = {
        str(family): {
            "layout_ids": [
                str(value) for value in registry_contract["family_layouts"][family]
            ],
            "episodes": [],
            "passed": False,
        }
        for family in registry_contract["families"]
    }
    episodes: list[dict[str, Any]] = []
    for child in child_groups:
        group = child["index_group"]
        evaluator_root = child["evidence_root"] / "evaluator"
        for episode in child["episodes"]:
            episode_id = str(episode["episode_id"])
            layout_id = str(episode["layout_id"])
            layout_contract = registry_contract["layouts"][layout_id]
            audit = _episode_dynamic_audit(
                evaluator_root,
                episode_id,
                [str(value) for value in layout_contract["dynamic_target_ids"]],
                int(layout_contract["minimum_visible_frames"]),
                float(layout_contract["risk_corridor_threshold_m"]),
                WEARER_RADIUS_M,
            )
            audit.update(
                {
                    "group_id": str(group["group_id"]),
                    "map": str(group["map"]),
                    "layout_id": layout_id,
                    "layout_family_id": family_by_layout[layout_id],
                }
            )
            episodes.append(audit)
            family_summaries[family_by_layout[layout_id]]["episodes"].append(
                {
                    "group_id": str(group["group_id"]),
                    "episode_id": episode_id,
                    "layout_id": layout_id,
                    "passed": bool(audit["passed"]),
                }
            )
    for summary in family_summaries.values():
        summary["passed"] = bool(summary["episodes"]) and all(
            bool(value["passed"]) for value in summary["episodes"]
        )
    checks = {
        "exactly_eight_declared_layout_families": len(family_summaries) == 8,
        "every_layout_family_has_at_least_one_episode": all(
            bool(value["episodes"]) for value in family_summaries.values()
        ),
        "all_child_groups_have_formal_four_sensor_1280x720_capture": all(
            bool(child["sensor_audit"]["passed"]) for child in child_groups
        ),
        "all_episode_dynamic_targets_have_state_motion_visibility_and_risk_corridor": bool(
            episodes
        )
        and all(bool(value["passed"]) for value in episodes),
        "all_layout_families_pass": all(
            bool(value["passed"]) for value in family_summaries.values()
        ),
    }
    return {
        "schema_version": C4_LAYOUT_AUDIT_SCHEMA,
        "experiment_id": str(index["experiment_id"]),
        "required_layout_family_count": 8,
        "wearer_radius_m": WEARER_RADIUS_M,
        "layout_requirements": {
            layout_id: {
                "minimum_visible_frames_per_dynamic_target_per_episode": int(
                    layout["minimum_visible_frames"]
                ),
                "risk_corridor_threshold_m": float(layout["risk_corridor_threshold_m"]),
                "dynamic_target_ids": [
                    str(value) for value in layout["dynamic_target_ids"]
                ],
            }
            for layout_id, layout in registry_contract["layouts"].items()
        },
        "map_group_count": len(child_groups),
        "layout_count": len(family_by_layout),
        "episode_count": len(episodes),
        "layout_families": family_summaries,
        "episodes": episodes,
        "sensor_audits": {
            str(child["index_group"]["group_id"]): child["sensor_audit"]
            for child in child_groups
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _validate_child_result_gate(
    child_result: dict[str, Any], group_id: str
) -> dict[str, bool]:
    """Admit only failures whose authority is deliberately lifted to C4 pack level."""

    checks = child_result.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise C4ContractError(f"{group_id}: child C2 checks are missing")
    required_non_occlusion = {"contact_safe_outcome_pair_matches"}
    occlusion_check_names = {
        "track_then_complete_physical_occlusion_contract_met",
        C2_DEFERRED_OCCLUSION_CHECK,
    }
    missing = sorted(required_non_occlusion - set(checks))
    if not (occlusion_check_names & set(checks)):
        missing.append("one physical occlusion pack-level check")
    if missing:
        raise C4ContractError(
            f"{group_id}: child C2 pack-level checks are missing: {missing}"
        )
    non_boolean = sorted(key for key, value in checks.items() if not isinstance(value, bool))
    if non_boolean:
        raise C4ContractError(
            f"{group_id}: child C2 checks are not boolean: {non_boolean}"
        )
    admitted_pack_checks = C2_PACK_LEVEL_CHECKS | {C2_DEFERRED_OCCLUSION_CHECK}
    non_pack_failures = sorted(
        key for key, value in checks.items() if not value and key not in admitted_pack_checks
    )
    if non_pack_failures:
        raise C4ContractError(
            f"{group_id}: child C2 non-pack checks failed: {non_pack_failures}"
        )
    expected_status = (
        C2_RESULT_STATUS if all(checks.values()) else C2_RESULT_NOT_EVALUABLE_STATUS
    )
    if child_result.get("status") != expected_status:
        raise C4ContractError(
            f"{group_id}: child C2 status is inconsistent with its checks"
        )
    return {
        "contact_safe_outcome_pair_matches": bool(
            checks["contact_safe_outcome_pair_matches"]
        ),
        "track_then_complete_physical_occlusion_contract_met": bool(
            checks.get(
                "track_then_complete_physical_occlusion_contract_met",
                checks.get(C2_DEFERRED_OCCLUSION_CHECK, False),
            )
        ),
        "deferred_to_c4_final_join": bool(
            checks.get(C2_DEFERRED_OCCLUSION_CHECK, False)
        ),
    }


def _validate_child_occlusion_sources(
    *,
    group_id: str,
    group: dict[str, Any],
    protocol: dict[str, Any],
    child_result: dict[str, Any],
    evaluator_root: Path,
    evidence_manifest: list[dict[str, Any]],
    registry_contract: dict[str, Any],
) -> dict[str, Any]:
    """Bind a child's sealed physical report and result outcomes to its layouts."""

    report_relative_path = "evaluator/physical_occlusion_report.json"
    report_path = evaluator_root / "physical_occlusion_report.json"
    manifest_entry = next(
        (
            value
            for value in evidence_manifest
            if isinstance(value, dict) and value.get("path") == report_relative_path
        ),
        None,
    )
    if not report_path.is_file() or manifest_entry is None:
        raise C4ContractError(
            f"{group_id}: physical occlusion report is missing from sealed evidence"
        )
    report_sha256 = sha256_file(report_path)
    if (
        str(manifest_entry.get("sha256", "")).lower() != report_sha256.lower()
        or int(manifest_entry.get("bytes", -1)) != report_path.stat().st_size
    ):
        raise C4ContractError(
            f"{group_id}: physical occlusion report seal differs from live file"
        )
    reports = _load_json_value(report_path)
    if not isinstance(reports, list) or not reports:
        raise C4ContractError(f"{group_id}: physical occlusion report must be nonempty")
    if child_result.get("occlusion_reports") != reports:
        raise C4ContractError(
            f"{group_id}: physical occlusion report differs from child result.occlusion_reports"
        )

    layout_by_episode_pair: dict[frozenset[str], str] = {}
    expected_episode_ids: set[str] = set()
    for raw_layout_id in group["layout_ids"]:
        layout_id = str(raw_layout_id)
        episode_ids = frozenset(
            str(value["episode_id"])
            for value in registry_contract["layouts"][layout_id]["episodes"]
        )
        if len(episode_ids) != 2 or episode_ids in layout_by_episode_pair:
            raise C4ContractError(
                f"{group_id}: registry layout does not define one distinct episode pair: {layout_id}"
            )
        layout_by_episode_pair[episode_ids] = layout_id
        expected_episode_ids.update(episode_ids)

    raw_contracts = protocol.get("occlusion_contracts")
    if not isinstance(raw_contracts, list) or len(raw_contracts) != len(layout_by_episode_pair):
        raise C4ContractError(
            f"{group_id}: protocol occlusion contract count differs from its layouts"
        )
    contract_layouts: dict[str, tuple[str, frozenset[str]]] = {}
    for index, contract in enumerate(raw_contracts):
        if not isinstance(contract, dict):
            raise C4ContractError(f"{group_id}: occlusion contract[{index}] is not an object")
        contract_id = _nonempty_identifier(
            contract.get("contract_id"), f"{group_id} occlusion contract_id"
        )
        raw_episode_ids = contract.get("episodes")
        if not isinstance(raw_episode_ids, list):
            raise C4ContractError(f"{group_id}: {contract_id} episodes are not an array")
        episode_ids = frozenset(str(value) for value in raw_episode_ids)
        layout_id = layout_by_episode_pair.get(episode_ids)
        if len(raw_episode_ids) != 2 or len(episode_ids) != 2 or layout_id is None:
            raise C4ContractError(
                f"{group_id}: {contract_id} does not bind one registry episode pair"
            )
        if contract_id in contract_layouts or any(
            value[0] == layout_id for value in contract_layouts.values()
        ):
            raise C4ContractError(f"{group_id}: duplicate occlusion contract binding")
        contract_layouts[contract_id] = (layout_id, episode_ids)
    if {value[0] for value in contract_layouts.values()} != set(
        map(str, group["layout_ids"])
    ):
        raise C4ContractError(f"{group_id}: occlusion contracts do not cover every layout")

    report_bindings: list[dict[str, Any]] = []
    seen_contracts: set[str] = set()
    for index, report in enumerate(reports):
        label = f"{group_id} physical occlusion report[{index}]"
        if not isinstance(report, dict):
            raise C4ContractError(f"{label} is not an object")
        contract_id = _nonempty_identifier(report.get("contract_id"), f"{label}.contract_id")
        if contract_id not in contract_layouts or contract_id in seen_contracts:
            raise C4ContractError(f"{label} does not bind one protocol contract")
        layout_id, episode_ids = contract_layouts[contract_id]
        episode_reports = report.get("episodes")
        selected_indices = report.get("selected_indices")
        if not isinstance(episode_reports, dict) or set(episode_reports) != set(episode_ids):
            raise C4ContractError(f"{label}.episodes differ from its contract")
        if not isinstance(selected_indices, dict) or set(selected_indices) != set(
            episode_ids
        ):
            raise C4ContractError(f"{label}.selected_indices differ from its contract")
        report_bindings.append(
            {
                "layout_id": layout_id,
                "episode_ids": sorted(episode_ids),
                "report": report,
            }
        )
        seen_contracts.add(contract_id)
    if seen_contracts != set(contract_layouts):
        raise C4ContractError(f"{group_id}: sealed reports do not cover every layout contract")

    outcomes = child_result.get("outcomes")
    if not isinstance(outcomes, list):
        raise C4ContractError(f"{group_id}: child result.outcomes must be an array")
    outcomes_by_episode: dict[str, dict[str, Any]] = {}
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, dict):
            raise C4ContractError(f"{group_id}: child outcome[{index}] is not an object")
        episode_id = str(outcome.get("episode_id", ""))
        if episode_id not in expected_episode_ids or episode_id in outcomes_by_episode:
            raise C4ContractError(f"{group_id}: child outcome episode set differs")
        outcomes_by_episode[episode_id] = outcome
    if set(outcomes_by_episode) != expected_episode_ids:
        raise C4ContractError(f"{group_id}: child result.outcomes do not cover every episode")

    sample_seconds = protocol.get("environment", {}).get("sample_seconds")
    if isinstance(sample_seconds, bool) or not isinstance(sample_seconds, (int, float)):
        raise C4ContractError(f"{group_id}: protocol sample_seconds is not numeric")
    if not math.isfinite(float(sample_seconds)) or float(sample_seconds) <= 0.0:
        raise C4ContractError(f"{group_id}: protocol sample_seconds must be positive")
    return {
        "relative_path": report_relative_path,
        "sha256": report_sha256,
        "sealed_and_result_bound": True,
        "report_bindings": report_bindings,
        "outcomes_by_episode": outcomes_by_episode,
        "sample_seconds": float(sample_seconds),
    }


def _integer_indices(value: Any) -> list[int] | None:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        return None
    return list(value)


def _selected_occlusion_audit(
    selected: Any, echoed_indices: Any, sample_seconds: float
) -> dict[str, Any]:
    if not isinstance(selected, dict):
        checks = {
            "selected_run_present": False,
            "pre_track_at_least_ten_frames": False,
            "complete_occlusion_between_0_30_and_0_60_seconds": False,
            "post_reappearance_at_least_ten_frames": False,
            "selected_indices_match_report_echo": echoed_indices == [],
            "complete_occlusion_indices_are_consecutive": False,
            "selected_run_declares_passed": False,
        }
        return {
            "sample_indices": [],
            "duration_seconds": None,
            "pre_track_frames": 0,
            "post_reappearance_frames": 0,
            "checks": checks,
            "passed": False,
        }
    sample_indices = _integer_indices(selected.get("sample_indices"))
    pre_indices = _integer_indices(selected.get("pre_track_sample_indices"))
    post_indices = _integer_indices(selected.get("post_reappearance_sample_indices"))
    echoed = _integer_indices(echoed_indices)
    raw_duration = selected.get("duration_seconds")
    duration = (
        float(raw_duration)
        if not isinstance(raw_duration, bool)
        and isinstance(raw_duration, (int, float))
        and math.isfinite(float(raw_duration))
        else None
    )
    raw_pre_frames = selected.get("pre_track_frames")
    raw_post_frames = selected.get("post_reappearance_frames")
    pre_frames = (
        raw_pre_frames
        if isinstance(raw_pre_frames, int) and not isinstance(raw_pre_frames, bool)
        else -1
    )
    post_frames = (
        raw_post_frames
        if isinstance(raw_post_frames, int) and not isinstance(raw_post_frames, bool)
        else -1
    )
    sample_indices = sample_indices or []
    pre_indices = pre_indices or []
    post_indices = post_indices or []
    expected_duration = len(sample_indices) * sample_seconds
    checks = {
        "selected_run_present": True,
        "pre_track_at_least_ten_frames": pre_frames == len(pre_indices) >= 10,
        "complete_occlusion_between_0_30_and_0_60_seconds": 0.30 - 1e-9
        <= (duration if duration is not None else -1.0)
        <= 0.60 + 1e-9
        and duration is not None
        and math.isclose(duration, expected_duration, abs_tol=1e-9),
        "post_reappearance_at_least_ten_frames": post_frames
        == len(post_indices)
        >= 10,
        "selected_indices_match_report_echo": echoed is not None
        and sample_indices == echoed,
        "complete_occlusion_indices_are_consecutive": bool(sample_indices)
        and sample_indices == list(range(sample_indices[0], sample_indices[-1] + 1)),
        "selected_run_declares_passed": selected.get("passed") is True,
    }
    return {
        "sample_indices": sample_indices,
        "duration_seconds": duration,
        "pre_track_frames": pre_frames,
        "post_reappearance_frames": post_frames,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_pack_occlusion_audit(
    index: dict[str, Any],
    registry_contract: dict[str, Any],
    child_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate sealed child reports; require one valid CONTACT/SAFE pair pack-wide."""

    report_audits: list[dict[str, Any]] = []
    aggregated_layout_ids: list[str] = []
    for child in child_groups:
        group = child["index_group"]
        source = child["occlusion_source"]
        for binding in source["report_bindings"]:
            report = binding["report"]
            episode_audits: list[dict[str, Any]] = []
            for episode_id in binding["episode_ids"]:
                episode_report = report["episodes"][episode_id]
                outcome = source["outcomes_by_episode"][episode_id]
                selected_audit = _selected_occlusion_audit(
                    episode_report.get("selected")
                    if isinstance(episode_report, dict)
                    else None,
                    report["selected_indices"][episode_id],
                    float(source["sample_seconds"]),
                )
                episode_audits.append(
                    {
                        "episode_id": episode_id,
                        "expected_outcome": str(outcome.get("expected_outcome")),
                        "observed_outcome": str(outcome.get("observed_outcome")),
                        "episode_report_declares_passed": isinstance(
                            episode_report, dict
                        )
                        and episode_report.get("passed") is True,
                        "selected_occlusion": selected_audit,
                    }
                )
            selected_index_sets = {
                tuple(value["selected_occlusion"]["sample_indices"])
                for value in episode_audits
            }
            observed_outcomes = {
                str(value["observed_outcome"]) for value in episode_audits
            }
            checks = {
                "exactly_two_episodes": len(episode_audits) == 2,
                "both_episodes_have_pre_track_at_least_ten_frames": all(
                    value["selected_occlusion"]["checks"][
                        "pre_track_at_least_ten_frames"
                    ]
                    for value in episode_audits
                ),
                "both_episodes_have_complete_occlusion_0_30_to_0_60_seconds": all(
                    value["selected_occlusion"]["checks"][
                        "complete_occlusion_between_0_30_and_0_60_seconds"
                    ]
                    for value in episode_audits
                ),
                "both_episodes_have_post_reappearance_at_least_ten_frames": all(
                    value["selected_occlusion"]["checks"][
                        "post_reappearance_at_least_ten_frames"
                    ]
                    for value in episode_audits
                ),
                "both_episode_report_indices_match_selected_runs": all(
                    value["selected_occlusion"]["checks"][
                        "selected_indices_match_report_echo"
                    ]
                    for value in episode_audits
                ),
                "both_selected_runs_are_consecutive_and_declared_passed": all(
                    value["selected_occlusion"]["checks"][
                        "complete_occlusion_indices_are_consecutive"
                    ]
                    and value["selected_occlusion"]["checks"][
                        "selected_run_declares_passed"
                    ]
                    and value["episode_report_declares_passed"]
                    for value in episode_audits
                ),
                "both_episodes_use_identical_nonempty_occlusion_indices": len(
                    selected_index_sets
                )
                == 1
                and bool(next(iter(selected_index_sets), ())),
                "report_declares_identical_indices_and_pair_passed": report.get(
                    "pair_occlusion_indices_identical"
                )
                is True
                and report.get("passed") is True,
                "observed_outcomes_are_contact_and_safe": observed_outcomes
                == {"CONTACT", "SAFE"},
            }
            report_audits.append(
                {
                    "group_id": str(group["group_id"]),
                    "map": str(group["map"]),
                    "layout_id": str(binding["layout_id"]),
                    "contract_id": str(report["contract_id"]),
                    "source_report": {
                        "path": f"groups/{group['group_id']}/physical_occlusion_report.json",
                        "sha256": str(source["sha256"]),
                    },
                    "episodes": episode_audits,
                    "observed_outcome_set": sorted(observed_outcomes),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
            aggregated_layout_ids.append(str(binding["layout_id"]))
    qualifying_pairs = [
        {
            "group_id": value["group_id"],
            "layout_id": value["layout_id"],
            "contract_id": value["contract_id"],
            "episode_ids": [episode["episode_id"] for episode in value["episodes"]],
        }
        for value in report_audits
        if value["passed"]
    ]
    checks = {
        "all_child_report_files_are_sealed_and_match_child_results": all(
            bool(child["occlusion_source"]["sealed_and_result_bound"])
            for child in child_groups
        ),
        "all_eight_layout_reports_are_aggregated_once": len(report_audits) == 8
        and len(aggregated_layout_ids) == len(set(aggregated_layout_ids))
        and set(aggregated_layout_ids) == set(registry_contract["layouts"]),
        "at_least_one_contact_safe_physical_occlusion_pair_passes": bool(
            qualifying_pairs
        ),
    }
    return {
        "schema_version": C4_OCCLUSION_AUDIT_SCHEMA,
        "experiment_id": str(index["experiment_id"]),
        "scope": "all_eight_layouts_across_all_sealed_child_map_groups",
        "thresholds": {
            "minimum_pre_track_frames_per_episode": 10,
            "minimum_complete_occlusion_seconds": 0.30,
            "maximum_complete_occlusion_seconds": 0.60,
            "minimum_post_reappearance_frames_per_episode": 10,
            "required_observed_outcome_set": ["CONTACT", "SAFE"],
            "require_identical_occlusion_sample_indices": True,
            "minimum_qualifying_pair_count": 1,
        },
        "map_group_count": len(child_groups),
        "layout_report_count": len(report_audits),
        "child_sources": [
            {
                "group_id": str(child["index_group"]["group_id"]),
                "map": str(child["index_group"]["map"]),
                "child_result_status": str(child["child_result"]["status"]),
                "child_pack_level_checks": child["child_pack_checks"],
                "physical_occlusion_report": {
                    "path": str(child["occlusion_source"]["relative_path"]),
                    "sha256": str(child["occlusion_source"]["sha256"]),
                    "sealed_and_result_bound": bool(
                        child["occlusion_source"]["sealed_and_result_bound"]
                    ),
                },
                "report_count": len(child["occlusion_source"]["report_bindings"]),
            }
            for child in child_groups
        ],
        "reports": report_audits,
        "qualifying_pairs": qualifying_pairs,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _preflight_child_group(
    index_path: Path,
    group: dict[str, Any],
    registry_contract: dict[str, Any],
) -> dict[str, Any]:
    group_id = str(group["group_id"])
    protocol_path = _resolve_contained_protocol_path(
        index_path, str(group["protocol_path"])
    )
    evidence_root = _resolve_index_path(index_path, str(group["evidence_path"]))
    if not evidence_root.is_dir():
        raise C4ContractError(f"{group_id}: evidence_path is not a directory")
    if sha256_file(protocol_path).lower() != str(group["protocol_sha256"]).lower():
        raise C4ContractError(f"{group_id}: protocol hash differs from index")
    child_result_path = evidence_root / "result.json"
    if not child_result_path.is_file() or sha256_file(child_result_path).lower() != str(
        group["evidence_result_sha256"]
    ).lower():
        raise C4ContractError(f"{group_id}: child result hash differs from index")
    child_result = load_json(child_result_path)
    if child_result.get("experiment_id") != C2_EXPERIMENT_ID:
        raise C4ContractError(f"{group_id}: child C2 experiment identity differs")
    child_pack_checks = _validate_child_result_gate(child_result, group_id)
    frozen_protocol_path = evidence_root / "frozen_protocol.json"
    if (
        not frozen_protocol_path.is_file()
        or sha256_file(frozen_protocol_path).lower()
        != str(group["protocol_sha256"]).lower()
        or sha256_file(frozen_protocol_path).lower()
        != str(child_result.get("protocol_sha256")).lower()
    ):
        raise C4ContractError(f"{group_id}: frozen protocol identity differs")
    protocol = load_json(frozen_protocol_path)
    if protocol.get("experiment_id") != C2_EXPERIMENT_ID:
        raise C4ContractError(f"{group_id}: frozen protocol is not C2 compatible")
    if str(protocol.get("environment", {}).get("map")) != str(group["map"]):
        raise C4ContractError(f"{group_id}: protocol map differs from index")
    if protocol.get("capture", {}).get("resolution") != [1280, 720] or protocol.get(
        "capture", {}
    ).get("sensor_order") != list(FORMAL_SENSORS):
        raise C4ContractError(f"{group_id}: protocol formal capture contract differs")
    protocol_layout_ids = set(map(str, protocol.get("layouts", {})))
    if protocol_layout_ids != set(map(str, group["layout_ids"])):
        raise C4ContractError(f"{group_id}: protocol layout set differs from index")
    episodes = [
        episode
        for layout_id in group["layout_ids"]
        for episode in registry_contract["layouts"][str(layout_id)]["episodes"]
    ]
    protocol_episode_layout = {
        str(value["episode_id"]): str(value["layout_id"])
        for value in protocol.get("scenarios", [])
    }
    registry_episode_layout = {
        str(value["episode_id"]): str(value["layout_id"])
        for value in episodes
    }
    if protocol_episode_layout != registry_episode_layout:
        raise C4ContractError(f"{group_id}: protocol and registry episode/layout mapping differs")
    scenario_by_episode = {
        str(value["episode_id"]): value for value in protocol.get("scenarios", [])
    }
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        scenario = scenario_by_episode[episode_id]
        if (
            str(scenario.get("navigation_session_id"))
            != str(episode["navigation_session_id"])
            or not isinstance(scenario.get("issued_plan"), dict)
            or str(scenario["issued_plan"].get("plan_id")) != str(episode["plan_id"])
        ):
            raise C4ContractError(
                f"{group_id}: protocol and registry navigation/plan identity differs for {episode_id}"
            )
    for layout_id in group["layout_ids"]:
        layout_id = str(layout_id)
        layout = protocol["layouts"].get(layout_id)
        if not isinstance(layout, dict) or not isinstance(layout.get("assets"), list):
            raise C4ContractError(f"{group_id}: protocol layout assets missing for {layout_id}")
        compiled_actor_ids = {
            str(value.get("asset_key"))
            for value in layout["assets"]
            if isinstance(value, dict)
        }
        if compiled_actor_ids != set(registry_contract["layouts"][layout_id]["actor_ids"]):
            raise C4ContractError(
                f"{group_id}: protocol and registry actor sets differ for {layout_id}"
            )

    model_root = evidence_root / "model"
    evaluator_root = evidence_root / "evaluator"
    sealed_model_path = evidence_root / "sealed_model_manifest.json"
    sealed_evidence_path = evidence_root / "sealed_evidence_manifest.json"
    if sha256_file(sealed_model_path) != str(child_result["sealed_model_manifest_sha256"]):
        raise C4ContractError(f"{group_id}: sealed model manifest hash differs")
    if sha256_file(sealed_evidence_path) != str(
        child_result["sealed_evidence_manifest_sha256"]
    ):
        raise C4ContractError(f"{group_id}: sealed evidence manifest hash differs")
    if not _sealed_tree_manifest_matches(sealed_model_path, model_root, [model_root]):
        raise C4ContractError(f"{group_id}: sealed model entries differ from live tree")
    if not _sealed_tree_manifest_matches(
        sealed_evidence_path,
        evidence_root,
        [evidence_root / "shards", model_root, evaluator_root],
    ):
        raise C4ContractError(f"{group_id}: sealed evidence entries differ from live tree")
    evidence_file_manifest = _load_json_value(sealed_evidence_path)
    if not isinstance(evidence_file_manifest, list):
        raise C4ContractError(f"{group_id}: sealed evidence manifest is not an array")
    occlusion_source = _validate_child_occlusion_sources(
        group_id=group_id,
        group=group,
        protocol=protocol,
        child_result=child_result,
        evaluator_root=evaluator_root,
        evidence_manifest=evidence_file_manifest,
        registry_contract=registry_contract,
    )
    expected_episode_ids = set(registry_episode_layout)
    model_failures = _model_root_contract_failures(model_root, expected_episode_ids)
    if model_failures:
        raise C4ContractError(
            f"{group_id}: model exact contract failed: {model_failures[:10]}"
        )
    truth_failures = _model_truth_failures(model_root)
    if truth_failures:
        raise C4ContractError(
            f"{group_id}: model semantic truth audit failed: {truth_failures[:10]}"
        )
    sensor_audit = _formal_sensor_audit(
        evidence_root,
        expected_episode_ids,
        str(group["map"]),
        str(group["protocol_sha256"]),
    )
    if not sensor_audit["passed"]:
        raise C4ContractError(f"{group_id}: formal sensor audit failed")
    return {
        "index_group": group,
        "protocol_path": protocol_path,
        "evidence_root": evidence_root,
        "child_result": child_result,
        "child_pack_checks": child_pack_checks,
        "sensor_audit": sensor_audit,
        "occlusion_source": occlusion_source,
        "episodes": episodes,
        "model_file_manifest": _load_json_value(sealed_model_path),
        "evidence_file_manifest": evidence_file_manifest,
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing tree overwrite: {destination}")
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def _materialize_outer_package(
    index_path: Path,
    index_sha256: str,
    index: dict[str, Any],
    asset_registry_path: Path,
    scene_registry_path: Path,
    registry_contract: dict[str, Any],
    child_groups: list[dict[str, Any]],
    output_root: Path,
    layout_audit: dict[str, Any],
    occlusion_audit: dict[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing C4 output overwrite: {output_root}")
    model_root = output_root / "model"
    evaluator_root = output_root / "evaluator"
    model_groups_root = model_root / "groups"
    evaluator_groups_root = evaluator_root / "groups"
    receipts_root = evaluator_root / "child_receipts"
    model_groups_root.mkdir(parents=True)
    evaluator_groups_root.mkdir(parents=True)
    receipts_root.mkdir(parents=True)

    outer_model_groups: list[dict[str, Any]] = []
    evaluator_groups: list[dict[str, Any]] = []
    copy_failures: list[str] = []
    for child in child_groups:
        group = child["index_group"]
        group_id = str(group["group_id"])
        child_root = child["evidence_root"]
        destination_model = model_groups_root / group_id
        destination_evaluator = evaluator_groups_root / group_id
        _copy_tree(child_root / "model", destination_model)
        _copy_tree(child_root / "evaluator", destination_evaluator)
        if _seal_tree(destination_model, [destination_model]) != child["model_file_manifest"]:
            copy_failures.append(f"{group_id}:model_copy")
        expected_evaluator_manifest = [
            {
                **value,
                "path": str(value["path"])[len("evaluator/") :],
            }
            for value in child["evidence_file_manifest"]
            if str(value.get("path", "")).startswith("evaluator/")
        ]
        if (
            _seal_tree(destination_evaluator, [destination_evaluator])
            != expected_evaluator_manifest
        ):
            copy_failures.append(f"{group_id}:evaluator_copy")

        receipt_dir = receipts_root / group_id
        receipt_dir.mkdir()
        for source, name in (
            (child_root / "result.json", "result.json"),
            (child_root / "frozen_protocol.json", "frozen_protocol.json"),
            (child_root / "sealed_model_manifest.json", "sealed_model_manifest.json"),
            (child_root / "sealed_evidence_manifest.json", "sealed_evidence_manifest.json"),
        ):
            shutil.copy2(source, receipt_dir / name)
        if (
            sha256_file(receipt_dir / "result.json").lower()
            != str(group["evidence_result_sha256"]).lower()
            or sha256_file(receipt_dir / "frozen_protocol.json").lower()
            != str(group["protocol_sha256"]).lower()
            or sha256_file(receipt_dir / "sealed_model_manifest.json")
            != str(child["child_result"]["sealed_model_manifest_sha256"])
            or sha256_file(receipt_dir / "sealed_evidence_manifest.json")
            != str(child["child_result"]["sealed_evidence_manifest_sha256"])
        ):
            copy_failures.append(f"{group_id}:receipt_copy")
        child_model_manifest_sha = sha256_file(child_root / "model" / "manifest.json")
        child_sealed_model_sha = sha256_file(child_root / "sealed_model_manifest.json")
        outer_model_groups.append(
            {
                "group_id": group_id,
                "model_root": f"groups/{group_id}",
                "child_model_root_manifest_sha256": child_model_manifest_sha,
                "child_sealed_model_manifest_sha256": child_sealed_model_sha,
                "model_file_count": len(child["model_file_manifest"]),
            }
        )
        evaluator_groups.append(
            {
                "group_id": group_id,
                "map": str(group["map"]),
                "layout_ids": [str(value) for value in group["layout_ids"]],
                "evaluator_root": f"groups/{group_id}",
                "child_receipt_root": f"child_receipts/{group_id}",
                "source_result_sha256": str(group["evidence_result_sha256"]),
                "source_sealed_evidence_manifest_sha256": str(
                    child["child_result"]["sealed_evidence_manifest_sha256"]
                ),
            }
        )

    outer_model_manifest = {
        "schema_version": "dtr-c4-model-root-manifest-v1",
        "experiment_id": str(index["experiment_id"]),
        "groups": outer_model_groups,
    }
    write_json_atomic(model_root / "manifest.json", outer_model_manifest)
    shutil.copy2(index_path, evaluator_root / "c4_multimap_index.json")
    registry_root = evaluator_root / "registries"
    registry_root.mkdir()
    shutil.copy2(asset_registry_path, registry_root / "asset_registry.json")
    shutil.copy2(scene_registry_path, registry_root / "scene_registry.json")
    write_json_atomic(evaluator_root / "layout_coverage_audit.json", layout_audit)
    write_json_atomic(evaluator_root / "pack_occlusion_audit.json", occlusion_audit)
    evaluator_manifest = {
        "schema_version": "dtr-c4-evaluator-root-manifest-v1",
        "experiment_id": str(index["experiment_id"]),
        "index": {
            "path": "c4_multimap_index.json",
            "sha256": sha256_file(evaluator_root / "c4_multimap_index.json"),
        },
        "asset_registry": {
            "path": "registries/asset_registry.json",
            "sha256": sha256_file(registry_root / "asset_registry.json"),
        },
        "scene_registry": {
            "path": "registries/scene_registry.json",
            "sha256": sha256_file(registry_root / "scene_registry.json"),
        },
        "layout_coverage_audit": {
            "path": "layout_coverage_audit.json",
            "sha256": sha256_file(evaluator_root / "layout_coverage_audit.json"),
        },
        "pack_occlusion_audit": {
            "path": "pack_occlusion_audit.json",
            "sha256": sha256_file(evaluator_root / "pack_occlusion_audit.json"),
        },
        "groups": evaluator_groups,
    }
    write_json_atomic(evaluator_root / "manifest.json", evaluator_manifest)

    model_truth_failures = _model_truth_failures(model_root)
    model_manifest_shape_ok = (
        set(load_json(model_root / "manifest.json")) == OUTER_MODEL_MANIFEST_EXACT_KEYS
        and all(set(value) == OUTER_MODEL_GROUP_EXACT_KEYS for value in outer_model_groups)
    )
    model_manifest = _seal_tree(model_root, [model_root])
    write_json_atomic(output_root / "sealed_model_manifest.json", model_manifest)
    evidence_manifest = _seal_tree(output_root, [model_root, evaluator_root])
    write_json_atomic(output_root / "sealed_evidence_manifest.json", evidence_manifest)
    checks = {
        "all_child_c2_contracts_and_live_trees_verified": True,
        "all_child_model_and_evaluator_copies_byte_identical": not copy_failures,
        "frozen_registry_copies_match_compiled_hashes": (
            sha256_file(registry_root / "asset_registry.json").lower()
            == str(index["registries"]["asset_registry"]["sha256"]).lower()
            and sha256_file(registry_root / "scene_registry.json").lower()
            == str(index["registries"]["scene_registry"]["sha256"]).lower()
        ),
        "frozen_compiled_index_copy_matches_preflight_hash": (
            sha256_file(evaluator_root / "c4_multimap_index.json") == index_sha256
        ),
        "exactly_eight_layout_families_complete": bool(layout_audit["passed"]),
        "pack_level_contact_safe_physical_occlusion_pair_met": bool(
            occlusion_audit["passed"]
        ),
        "outer_model_root_manifest_exact": model_manifest_shape_ok,
        "outer_model_tree_has_no_evaluator_semantic_truth": not model_truth_failures,
        "outer_sealed_model_manifest_matches_live_tree": _sealed_tree_manifest_matches(
            output_root / "sealed_model_manifest.json", model_root, [model_root]
        ),
        "outer_sealed_evidence_manifest_matches_live_tree": _sealed_tree_manifest_matches(
            output_root / "sealed_evidence_manifest.json",
            output_root,
            [model_root, evaluator_root],
        ),
    }
    result = {
        "schema_version": C4_RESULT_SCHEMA,
        "status": C4_RESULT_STATUS if all(checks.values()) else "DTR_CARLA_C4_MULTIMAP_GATE_NOT_MET",
        "experiment_id": str(index["experiment_id"]),
        "checks": checks,
        "copy_failures": copy_failures,
        "model_truth_failures": model_truth_failures,
        "map_group_count": len(child_groups),
        "layout_family_count": len(registry_contract["families"]),
        "layout_count": len(registry_contract["layouts"]),
        "episode_count": len(registry_contract["episode_ids"]),
        "asset_registry_sha256": sha256_file(asset_registry_path),
        "scene_registry_sha256": sha256_file(scene_registry_path),
        "layout_coverage_audit": {
            "path": "evaluator/layout_coverage_audit.json",
            "sha256": sha256_file(evaluator_root / "layout_coverage_audit.json"),
        },
        "pack_occlusion_audit": {
            "path": "evaluator/pack_occlusion_audit.json",
            "sha256": sha256_file(evaluator_root / "pack_occlusion_audit.json"),
        },
        "index_sha256": index_sha256,
        "sealed_model_manifest_sha256": sha256_file(
            output_root / "sealed_model_manifest.json"
        ),
        "sealed_evidence_manifest_sha256": sha256_file(
            output_root / "sealed_evidence_manifest.json"
        ),
        "sealed_model_files": len(model_manifest),
        "sealed_evidence_files": len(evidence_manifest),
    }
    write_json_atomic(output_root / "result.json", result)
    return result


def join_multimap(index_path: Path, output_root: Path) -> dict[str, Any]:
    index_path = index_path.resolve(strict=True)
    index_sha256 = sha256_file(index_path)
    index = load_json(index_path)
    _require_exact_keys(index, INDEX_EXACT_KEYS, "C4 compiled protocol")
    asset_registry_path, asset_registry = _load_bound_registry(
        index_path, index, "asset_registry"
    )
    scene_registry_path, scene_registry = _load_bound_registry(
        index_path, index, "scene_registry"
    )
    registry_contract = validate_multimap_index(
        index, asset_registry, scene_registry
    )
    child_groups = [
        _preflight_child_group(index_path, group, registry_contract)
        for group in index["map_layout_groups"]
    ]
    resolved_output = output_root.resolve()
    for child in child_groups:
        try:
            resolved_output.relative_to(child["evidence_root"].resolve())
        except ValueError:
            continue
        raise C4ContractError(
            "C4 output root must not be nested inside an immutable child evidence root"
        )
    if sha256_file(index_path) != index_sha256:
        raise C4ContractError("compiled C4 index changed during preflight")
    layout_audit = build_layout_coverage_audit(
        index, registry_contract, child_groups
    )
    occlusion_audit = build_pack_occlusion_audit(
        index, registry_contract, child_groups
    )
    return _materialize_outer_package(
        index_path,
        index_sha256,
        index,
        asset_registry_path,
        scene_registry_path,
        registry_contract,
        child_groups,
        resolved_output,
        layout_audit,
        occlusion_audit,
    )


def main() -> int:
    args = parse_args()
    result = join_multimap(args.compiled_protocol, args.output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == C4_RESULT_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
