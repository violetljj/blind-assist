"""Attach C3 registry/risk evidence to a completed C2-compatible capture."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from dtr_carla_c2_rich_scene import (
    forbidden_model_paths,
    point_polygon_distance,
    sha256_file,
    validate_model_record,
    write_json_atomic,
)
from dtr_carla_c3_scene import (
    C3_EXPERIMENT_ID,
    dynamic_risk_instance_ids,
    load_json,
    sha256_json,
    validate_registry_bundle,
)


RESULT_STATUS = "DTR_CARLA_C3_DENSE_DYNAMIC_RISK_SOURCE_COMPLETE"
C3_FORBIDDEN_MODEL_TOKENS = (
    "actor",
    "asset",
    "collision",
    "contact",
    "instance",
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
    "witness",
)
OBSERVATION_NESTED_EXACT_KEYS = {
    "camera": {"fov_degrees", "height", "K", "rigid_extrinsic", "width", "world_transform"},
    "wearable_rgb": {"bytes", "height", "path", "sha256", "source_world_frame", "width"},
    "metric_depth": {"bytes", "codec", "height", "path", "sha256", "source_world_frame", "width"},
    "navigation": {"issued_plan", "navigation_session_id"},
    "frame_alignment": {
        "authority",
        "depth_minus_wearable_source_world_frame_offset",
        "receipt_path",
        "receipt_sha256",
        "reference_modality",
    },
    "wearer_pose_current": {"pitch", "roll", "x", "y", "yaw", "z"},
}
ISSUED_PLAN_EXACT_KEYS = {"authority", "path", "receipt_sha256"}
TRANSFORM_EXACT_KEYS = {"pitch", "roll", "x", "y", "yaw", "z"}
RIGID_EXTRINSIC_EXACT_KEYS = {
    "pitch_degrees",
    "roll_degrees",
    "x_m",
    "y_m",
    "yaw_degrees",
    "z_m",
}
PLAN_EXACT_KEYS = {
    "episode_id",
    "issued_plan",
    "layout_anchor",
    "navigation_session_id",
    "schema_version",
}
LAYOUT_ANCHOR_EXACT_KEYS = {
    "world_center_xy_m",
    "world_forward_xy",
    "world_right_xy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset-registry", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--compiler-receipt", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _seal_tree(root: Path, directories: list[Path]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for directory in directories:
        for path in sorted(value for value in directory.rglob("*") if value.is_file()):
            values.append(
                {
                    "path": _safe_relative(path, root),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return values


def _c3_semantic_truth_paths(value: Any, prefix: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            lowered = str(key).lower()
            allowed_disabled_actor_contract = (
                lowered == "current_actors_enabled" and child is False
            )
            if not allowed_disabled_actor_contract and any(
                token in lowered for token in C3_FORBIDDEN_MODEL_TOKENS
            ):
                failures.append(path)
            failures.extend(_c3_semantic_truth_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_c3_semantic_truth_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in C3_FORBIDDEN_MODEL_TOKENS):
            failures.append(prefix)
    return failures


def _model_relative_file(model_root: Path, relative: str) -> Path:
    path = (model_root / relative).resolve()
    path.relative_to(model_root.resolve())
    return path


def _model_schema_failures(model_root: Path) -> list[str]:
    failures: list[str] = []
    observation_paths = sorted(model_root.glob("episodes/*/observations.jsonl"))
    if not observation_paths:
        return ["no model observations"]
    for observation_path in observation_paths:
        episode_id = observation_path.parent.name
        for index, record in enumerate(_read_jsonl(observation_path)):
            label = f"{observation_path.relative_to(model_root).as_posix()}[{index}]"
            try:
                validate_model_record(record)
            except Exception as exc:
                failures.append(f"{label}:top-level:{exc}")
                continue
            for field, expected in OBSERVATION_NESTED_EXACT_KEYS.items():
                value = record.get(field)
                if not isinstance(value, dict) or set(value) != expected:
                    failures.append(f"{label}:{field}:keys")
            camera = record["camera"]
            if set(camera.get("world_transform", {})) != TRANSFORM_EXACT_KEYS:
                failures.append(f"{label}:camera.world_transform:keys")
            if set(camera.get("rigid_extrinsic", {})) != RIGID_EXTRINSIC_EXACT_KEYS:
                failures.append(f"{label}:camera.rigid_extrinsic:keys")
            issued_plan = record["navigation"]["issued_plan"]
            if not isinstance(issued_plan, dict) or set(issued_plan) != ISSUED_PLAN_EXACT_KEYS:
                failures.append(f"{label}:navigation.issued_plan:keys")
            if record.get("episode_id") != episode_id:
                failures.append(f"{label}:episode_id")
            if int(record.get("world_frame", -1)) != int(
                record["wearable_rgb"].get("source_world_frame", -2)
            ):
                failures.append(f"{label}:world_frame")
            if (
                int(camera.get("width", 0)) != 1280
                or int(camera.get("height", 0)) != 720
                or float(camera.get("fov_degrees", 0.0)) != 90.0
                or len(camera.get("K", [])) != 3
                or any(len(row) != 3 for row in camera.get("K", []))
                or any(
                    not math.isclose(
                        float(camera["K"][row][column]),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    for row, values in enumerate(
                        ((640.0, 0.0, 640.0), (0.0, 640.0, 360.0), (0.0, 0.0, 1.0))
                    )
                    for column, expected in enumerate(values)
                )
            ):
                failures.append(f"{label}:camera_calibration")
            for modality in ("wearable_rgb", "metric_depth"):
                payload = record[modality]
                if int(payload.get("width", 0)) != 1280 or int(payload.get("height", 0)) != 720:
                    failures.append(f"{label}:{modality}:resolution")
                try:
                    payload_path = _model_relative_file(model_root, str(payload["path"]))
                    if (
                        not payload_path.is_file()
                        or payload_path.stat().st_size != int(payload["bytes"])
                        or sha256_file(payload_path) != str(payload["sha256"])
                    ):
                        failures.append(f"{label}:{modality}:payload")
                except Exception as exc:
                    failures.append(f"{label}:{modality}:path:{exc}")
            codec = record["metric_depth"].get("codec")
            if (
                not isinstance(codec, dict)
                or set(codec) != {"formula", "maximum_depth_m", "name"}
                or codec.get("name") != "CARLA_RGB24_NORMALIZED_DEPTH"
                or float(codec.get("maximum_depth_m", 0.0)) != 1000.0
                or codec.get("formula")
                != "meters=1000*(R+256*G+65536*B)/(16777215)"
            ):
                failures.append(f"{label}:metric_depth:codec")
            try:
                plan_path = _model_relative_file(model_root, str(issued_plan["path"]))
                plan = load_json(plan_path)
                if str(plan["issued_plan"]["receipt_sha256"]) != str(
                    issued_plan["receipt_sha256"]
                ):
                    failures.append(f"{label}:navigation.issued_plan:receipt")
            except Exception as exc:
                failures.append(f"{label}:navigation.issued_plan:path:{exc}")
            try:
                alignment = record["frame_alignment"]
                receipt_path = _model_relative_file(
                    model_root, str(alignment["receipt_path"])
                )
                receipt = load_json(receipt_path)
                if str(receipt["receipt_sha256"]) != str(alignment["receipt_sha256"]):
                    failures.append(f"{label}:frame_alignment:receipt")
            except Exception as exc:
                failures.append(f"{label}:frame_alignment:path:{exc}")
    for plan_path in sorted((model_root / "plans").glob("*.json")):
        plan = load_json(plan_path)
        label = plan_path.relative_to(model_root).as_posix()
        if set(plan) != PLAN_EXACT_KEYS:
            failures.append(f"{label}:keys")
        if set(plan.get("layout_anchor", {})) != LAYOUT_ANCHOR_EXACT_KEYS:
            failures.append(f"{label}:layout_anchor:keys")
    return failures


def _sealed_tree_manifest_matches(
    manifest_path: Path, root: Path, directories: list[Path]
) -> bool:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, list):
        return False
    return manifest == _seal_tree(root, directories)


def _model_truth_failures(model_root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(value for value in model_root.rglob("*") if value.is_file()):
        relative = _safe_relative(path, model_root)
        lowered = relative.lower()
        if any(token in lowered for token in C3_FORBIDDEN_MODEL_TOKENS):
            failures.append(f"path:{relative}")
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if path.suffix.lower() == ".jsonl":
            values: list[Any] = _read_jsonl(path)
        else:
            values = [load_json(path)]
        for index, value in enumerate(values):
            if not isinstance(value, (dict, list)):
                continue
            for failure in forbidden_model_paths(value):
                failures.append(f"{relative}[{index}]:{failure}")
            for failure in _c3_semantic_truth_paths(value):
                failures.append(f"{relative}[{index}]:{failure}")
    return failures


def _visible_for_tracking_in_every_episode(
    counts: dict[str, int], minimum_frames: int = 10
) -> bool:
    return bool(counts) and all(
        int(value) >= minimum_frames for value in counts.values()
    )


def _dynamic_risk_audit(
    root: Path,
    asset_registry: dict[str, Any],
    scene_registry: dict[str, Any],
    scene_id: str,
) -> dict[str, Any]:
    scene = scene_registry["scenes"][scene_id]
    dynamic_ids = dynamic_risk_instance_ids(asset_registry, scene)
    actor_by_id = {str(value["instance_id"]): value for value in scene["actors"]}
    wearer_radius = 0.45
    threshold = float(scene["admission"]["risk_corridor_threshold_m"])
    minimum_visible_frames = int(
        scene["admission"][
            "minimum_model_visible_frames_per_dynamic_target_per_episode"
        ]
    )
    per_actor: dict[str, dict[str, Any]] = {}
    for actor_id in dynamic_ids:
        asset = asset_registry["assets"][str(actor_by_id[actor_id]["asset_id"])]
        per_actor[actor_id] = {
            "asset_id": str(actor_by_id[actor_id]["asset_id"]),
            "track_id": str(actor_by_id[actor_id]["track_id"]),
            "role": str(actor_by_id[actor_id]["role"]),
            "role_family": str(asset["role_family"]),
            "blueprint_id": str(asset["blueprint_id"]),
            "frames_expected": 0,
            "actor_state_frames": 0,
            "risk_geometry_frames": 0,
            "nonzero_command_velocity_frames": 0,
            "observed_transform_motion_frames": 0,
            "observed_transform_motion_frames_by_episode": {},
            "visible_frames": 0,
            "visible_frames_by_episode": {},
            "minimum_clearance_m": None,
            "risk_corridor_frames": 0,
            "risk_corridor_frames_by_episode": {},
            "contact_responsibility_frames": 0,
        }
    episode_summaries: list[dict[str, Any]] = []

    collision_policy_rows: list[dict[str, Any]] = []
    for sensor_name in ("instance", "wearable", "depth", "witness"):
        for episode in scene["episodes"]:
            episode_id = str(episode["episode_id"])
            manifest_path = (
                root
                / "shards"
                / sensor_name
                / "episodes"
                / episode_id
                / "asset_manifest.json"
            )
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            collision_policy_rows.extend(
                {
                    "sensor": sensor_name,
                    "episode_id": episode_id,
                    "asset_key": str(value["asset_key"]),
                    "collisions_enabled": value.get("collisions_enabled"),
                }
                for value in manifest
            )

    def frames_path_for(episode_id: str) -> Path:
        joined = root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        if joined.is_file():
            return joined
        raw_instance = (
            root
            / "shards"
            / "instance"
            / "episodes"
            / episode_id
            / "frames.jsonl"
        )
        if raw_instance.is_file():
            return raw_instance
        raise FileNotFoundError(f"no instance truth rows for {episode_id}")

    for episode in scene["episodes"]:
        episode_id = str(episode["episode_id"])
        frames_path = frames_path_for(episode_id)
        rows = _read_jsonl(frames_path)
        episode_summaries.append({"episode_id": episode_id, "frames": len(rows)})
        for summary in per_actor.values():
            summary["visible_frames_by_episode"][episode_id] = 0
            summary["observed_transform_motion_frames_by_episode"][episode_id] = 0
            summary["risk_corridor_frames_by_episode"][episode_id] = 0
        previous_actor_xy: dict[str, tuple[float, float]] = {}
        for row in rows:
            wearer = row["actors"]["wearer"]["transform"]
            wearer_xy = (float(wearer["x"]), float(wearer["y"]))
            polygons = row["truth"]["collision_polygons_xy"]
            responsible = {str(value) for value in row["truth"]["responsible_assets"]}
            visibility = row["instance_visibility"]
            for actor_id, summary in per_actor.items():
                summary["frames_expected"] += 1
                actor = row["actors"].get(actor_id)
                if actor is not None:
                    summary["actor_state_frames"] += 1
                    velocity = actor["command_velocity"]
                    if (
                        abs(float(velocity["x"])) > 1e-6
                        or abs(float(velocity["y"])) > 1e-6
                        or abs(float(velocity["z"])) > 1e-6
                    ):
                        summary["nonzero_command_velocity_frames"] += 1
                    transform = actor["transform"]
                    actor_xy = (float(transform["x"]), float(transform["y"]))
                    previous_xy = previous_actor_xy.get(actor_id)
                    if previous_xy is not None and math.hypot(
                        actor_xy[0] - previous_xy[0], actor_xy[1] - previous_xy[1]
                    ) > 1e-4:
                        summary["observed_transform_motion_frames"] += 1
                        summary["observed_transform_motion_frames_by_episode"][
                            episode_id
                        ] += 1
                    previous_actor_xy[actor_id] = actor_xy
                if bool(visibility.get(actor_id, {}).get("visible")):
                    summary["visible_frames"] += 1
                    summary["visible_frames_by_episode"][episode_id] += 1
                polygon = polygons.get(actor_id)
                if polygon is not None:
                    summary["risk_geometry_frames"] += 1
                    clearance = max(
                        0.0,
                        point_polygon_distance(wearer_xy, polygon) - wearer_radius,
                    )
                    current = summary["minimum_clearance_m"]
                    summary["minimum_clearance_m"] = (
                        clearance if current is None else min(float(current), clearance)
                    )
                    if clearance <= threshold:
                        summary["risk_corridor_frames"] += 1
                        summary["risk_corridor_frames_by_episode"][episode_id] += 1
                if actor_id in responsible:
                    summary["contact_responsibility_frames"] += 1
    for summary in per_actor.values():
        summary["minimum_clearance_m"] = (
            round(float(summary["minimum_clearance_m"]), 6)
            if summary["minimum_clearance_m"] is not None
            else None
        )
        summary["entered_risk_corridor_in_every_episode"] = all(
            int(value) > 0
            for value in summary["risk_corridor_frames_by_episode"].values()
        )
        summary["full_frame_risk_geometry"] = (
            summary["risk_geometry_frames"] == summary["frames_expected"]
        )
        summary["transform_motion_observed_in_every_episode"] = all(
            int(value) > 0
            for value in summary[
                "observed_transform_motion_frames_by_episode"
            ].values()
        )
        summary["visible_for_tracking_in_every_episode"] = (
            _visible_for_tracking_in_every_episode(
                summary["visible_frames_by_episode"], minimum_visible_frames
            )
        )
    checks = {
        "exactly_sixteen_registered_dynamic_risk_targets": len(per_actor) == 16,
        "all_dynamic_targets_have_actor_state_every_frame": all(
            value["actor_state_frames"] == value["frames_expected"]
            for value in per_actor.values()
        ),
        "all_dynamic_targets_enter_risk_geometry_every_frame": all(
            bool(value["full_frame_risk_geometry"]) for value in per_actor.values()
        ),
        "all_dynamic_targets_have_observed_transform_motion_in_every_episode": all(
            bool(value["transform_motion_observed_in_every_episode"])
            for value in per_actor.values()
        ),
        "all_dynamic_targets_enter_declared_risk_corridor_in_every_episode": all(
            bool(value["entered_risk_corridor_in_every_episode"])
            for value in per_actor.values()
        ),
        "all_dynamic_targets_are_model_visible_for_at_least_ten_frames_per_episode": all(
            bool(value["visible_for_tracking_in_every_episode"])
            for value in per_actor.values()
        ),
        "all_dynamic_target_blueprints_match_registry": all(
            all(
                row["actors"][actor_id]["actual_blueprint"]
                == per_actor[actor_id]["blueprint_id"]
                for row in _read_jsonl(frames_path_for(str(episode["episode_id"])))
            )
            for actor_id in per_actor
            for episode in scene["episodes"]
        ),
        "all_scripted_actors_disable_engine_collisions": (
            len(collision_policy_rows) == 4 * len(scene["episodes"]) * 40
            and all(
                value["collisions_enabled"] is False
                for value in collision_policy_rows
            )
        ),
    }
    return {
        "schema_version": "dtr-c3-dynamic-risk-audit-v1",
        "experiment_id": C3_EXPERIMENT_ID,
        "scene_id": scene_id,
        "risk_corridor_threshold_m": threshold,
        "minimum_model_visible_frames_per_dynamic_target_per_episode": (
            minimum_visible_frames
        ),
        "episode_summaries": episode_summaries,
        "dynamic_target_count": len(per_actor),
        "scripted_collision_policy_rows": len(collision_policy_rows),
        "per_actor": per_actor,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve(strict=True)
    asset_path = args.asset_registry.resolve(strict=True)
    scene_path = args.scene_registry.resolve(strict=True)
    receipt_path = args.compiler_receipt.resolve(strict=True)
    asset_registry = load_json(asset_path)
    scene_registry = load_json(scene_path)
    compiler_receipt = load_json(receipt_path)
    validate_registry_bundle(asset_registry, scene_registry)
    scene_id = str(args.scene_id)
    if scene_id not in scene_registry["scenes"]:
        raise ValueError(f"unknown scene: {scene_id}")
    c2_result_path = root / "result.json"
    c2_result = load_json(c2_result_path)
    if c2_result.get("status") != "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE":
        raise RuntimeError("C2 compatibility capture/join did not pass")
    if compiler_receipt["asset_registry_sha256"] != sha256_json(asset_registry):
        raise RuntimeError("compiler receipt asset registry hash differs")
    if compiler_receipt["scene_registry_sha256"] != sha256_json(scene_registry):
        raise RuntimeError("compiler receipt scene registry hash differs")
    frozen_protocol_path = root / "frozen_protocol.json"
    frozen_protocol_file_sha256 = sha256_file(frozen_protocol_path)
    frozen_protocol_canonical_sha256 = sha256_json(load_json(frozen_protocol_path))
    if (
        str(compiler_receipt.get("compiled_protocol_sha256"))
        != frozen_protocol_canonical_sha256
    ):
        raise RuntimeError("compiler receipt differs from captured frozen protocol")
    if str(c2_result.get("protocol_sha256")) != frozen_protocol_file_sha256:
        raise RuntimeError("C2 result differs from captured frozen protocol")
    if str(compiler_receipt.get("scene_id")) != scene_id:
        raise RuntimeError("compiler receipt scene differs from requested scene")
    if str(compiler_receipt.get("experiment_id")) != C3_EXPERIMENT_ID:
        raise RuntimeError("compiler receipt experiment identity differs")

    c2_evidence_manifest_path = root / "sealed_evidence_manifest.json"
    c2_model_manifest_path = root / "sealed_model_manifest.json"
    if sha256_file(c2_evidence_manifest_path) != str(
        c2_result["sealed_evidence_manifest_sha256"]
    ):
        raise RuntimeError("C2 sealed evidence manifest differs from its result")
    if sha256_file(c2_model_manifest_path) != str(
        c2_result["sealed_model_manifest_sha256"]
    ):
        raise RuntimeError("C2 sealed model manifest differs from its result")

    evaluator_root = root / "evaluator"
    model_root = root / "model"
    if not _sealed_tree_manifest_matches(
        c2_model_manifest_path, model_root, [model_root]
    ):
        raise RuntimeError("C2 sealed model manifest entries differ from live model tree")
    if not _sealed_tree_manifest_matches(
        c2_evidence_manifest_path,
        root,
        [root / "shards", model_root, evaluator_root],
    ):
        raise RuntimeError("C2 sealed evidence manifest entries differ from live tree")
    model_schema_failures = _model_schema_failures(model_root)
    if model_schema_failures:
        raise RuntimeError(
            f"model exact-schema audit failed: {model_schema_failures[:10]}"
        )
    c3_root = evaluator_root / "c3_registry"
    if c3_root.exists():
        raise FileExistsError(f"refusing C3 join overwrite: {c3_root}")
    c3_root.mkdir(parents=True)
    old_result = c3_root / "c2_compatibility_result.json"
    old_manifest = c3_root / "c2_compatibility_sealed_evidence_manifest.json"
    old_model_manifest = c3_root / "c2_compatibility_sealed_model_manifest.json"
    shutil.copy2(c2_result_path, old_result)
    shutil.copy2(c2_evidence_manifest_path, old_manifest)
    shutil.copy2(c2_model_manifest_path, old_model_manifest)
    frozen_asset = c3_root / "asset_registry.json"
    frozen_scene = c3_root / "scene_registry.json"
    frozen_receipt = c3_root / "compiler_receipt.json"
    shutil.copy2(asset_path, frozen_asset)
    shutil.copy2(scene_path, frozen_scene)
    shutil.copy2(receipt_path, frozen_receipt)
    audit = _dynamic_risk_audit(root, asset_registry, scene_registry, scene_id)
    audit_path = evaluator_root / "c3_dynamic_risk_audit.json"
    write_json_atomic(audit_path, audit)
    model_root_manifest_path = model_root / "manifest.json"
    model_root_manifest = load_json(model_root_manifest_path)
    model_root_manifest["experiment_id"] = C3_EXPERIMENT_ID
    write_json_atomic(model_root_manifest_path, model_root_manifest)
    write_json_atomic(
        root / "sealed_model_manifest.json", _seal_tree(model_root, [model_root])
    )
    model_failures = _model_truth_failures(model_root)
    registry_hashes = {
        sha256_file(frozen_asset).lower(),
        sha256_file(frozen_scene).lower(),
        sha256_file(frozen_receipt).lower(),
    }
    model_text = "\n".join(
        path.read_text(encoding="utf-8-sig", errors="ignore").lower()
        for path in model_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    checks = {
        "c2_compatibility_join_passed": all(bool(value) for value in c2_result["checks"].values()),
        "c2_compatibility_manifest_hashes_match_result": (
            sha256_file(old_manifest)
            == str(c2_result["sealed_evidence_manifest_sha256"])
            and sha256_file(old_model_manifest)
            == str(c2_result["sealed_model_manifest_sha256"])
        ),
        "c2_compatibility_manifest_entries_match_live_tree": True,
        "compiler_receipt_binds_captured_protocol": (
            str(compiler_receipt["compiled_protocol_sha256"])
            == frozen_protocol_canonical_sha256
            and str(c2_result["protocol_sha256"])
            == frozen_protocol_file_sha256
        ),
        "asset_registry_frozen_evaluator_only": sha256_file(frozen_asset)
        == sha256_file(asset_path),
        "scene_registry_frozen_evaluator_only": sha256_file(frozen_scene)
        == sha256_file(scene_path),
        "compiler_receipt_frozen_evaluator_only": sha256_file(frozen_receipt)
        == sha256_file(receipt_path),
        "dynamic_risk_audit_passed": bool(audit["passed"]),
        "model_root_manifest_has_c3_experiment_id": (
            load_json(model_root_manifest_path).get("experiment_id")
            == C3_EXPERIMENT_ID
        ),
        "model_observation_and_plan_exact_schema": not model_schema_failures,
        "model_tree_has_no_evaluator_truth_keys_or_paths": not model_failures,
        "model_tree_has_no_c3_registry_hashes": not any(
            value in model_text for value in registry_hashes
        ),
    }
    provisional_result = {
        "schema_version": "dtr-c3-dynamic-risk-result-v1",
        "status": RESULT_STATUS if all(checks.values()) else "DTR_CARLA_C3_GATE_NOT_MET",
        "experiment_id": C3_EXPERIMENT_ID,
        "scene_id": scene_id,
        "checks": checks,
        "dynamic_risk_audit": {
            "path": _safe_relative(audit_path, root),
            "sha256": sha256_file(audit_path),
        },
        "registries": {
            "asset": {
                "path": _safe_relative(frozen_asset, root),
                "sha256": sha256_file(frozen_asset),
            },
            "scene": {
                "path": _safe_relative(frozen_scene, root),
                "sha256": sha256_file(frozen_scene),
            },
            "compiler_receipt": {
                "path": _safe_relative(frozen_receipt, root),
                "sha256": sha256_file(frozen_receipt),
            },
        },
        "model_truth_failures": model_failures,
        "model_schema_failures": model_schema_failures,
        "model_manifest_sha256": sha256_file(root / "sealed_model_manifest.json"),
        "frozen_protocol_sha256": frozen_protocol_file_sha256,
        "frozen_protocol_canonical_sha256": frozen_protocol_canonical_sha256,
        "c2_compatibility_result_sha256": sha256_file(old_result),
        "c2_compatibility_model_manifest_sha256": sha256_file(old_model_manifest),
    }
    write_json_atomic(c2_result_path, provisional_result)
    evidence_manifest = _seal_tree(
        root, [root / "shards", model_root, evaluator_root]
    )
    write_json_atomic(root / "sealed_evidence_manifest.json", evidence_manifest)
    final_result = load_json(c2_result_path)
    final_result["sealed_evidence_manifest_sha256"] = sha256_file(
        root / "sealed_evidence_manifest.json"
    )
    final_result["sealed_evidence_files"] = len(evidence_manifest)
    write_json_atomic(c2_result_path, final_result)
    print(json.dumps(final_result, ensure_ascii=False, sort_keys=True))
    return 0 if final_result["status"] == RESULT_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
