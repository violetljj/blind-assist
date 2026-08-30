"""Build one three-map N4 replay bundle and trace-conditioned wearer routes."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
from typing import Any, Sequence

from dtr_carla_c2_rich_scene import write_json_atomic
from dtr_carla_n2_frozen_trace_replay import SOURCE_FILES, validate_protocol
from dtr_carla_n3_multitown_native_dynamics import SCENE_ORDER, sha256_file


BUNDLE_SCHEMA = "dtr-carla-n4-multitown-frozen-replay-bundle-v1"
RESULT_STATUS = "DTR_CARLA_N3_MULTITOWN_NATIVE_DYNAMICS_MATERIALIZED"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(bool(line.strip()), f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            _require(isinstance(value, dict), f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def event_pose(
    target: dict[str, Any], view_distance_m: float, lateral_m: float
) -> tuple[float, float, float]:
    transform = target["transform"]
    yaw_rad = math.radians(float(transform["yaw"]))
    forward = (math.cos(yaw_rad), math.sin(yaw_rad))
    right = (-forward[1], forward[0])
    target_x = float(transform["x"])
    target_y = float(transform["y"])
    wearer_x = target_x - view_distance_m * forward[0] + lateral_m * right[0]
    wearer_y = target_y - view_distance_m * forward[1] + lateral_m * right[1]
    wearer_yaw = math.degrees(math.atan2(target_y - wearer_y, target_x - wearer_x))
    return wearer_x, wearer_y, wearer_yaw


def bounded_step(
    current: tuple[float, float], desired: tuple[float, float], maximum_step_m: float
) -> tuple[float, float]:
    dx = desired[0] - current[0]
    dy = desired[1] - current[1]
    distance = math.hypot(dx, dy)
    if distance <= maximum_step_m:
        return desired
    scale = maximum_step_m / distance
    return current[0] + dx * scale, current[1] + dy * scale


def build_event_bearing_route(
    rows: list[dict[str, Any]],
    events: dict[str, Any],
    *,
    fixed_delta_seconds: float,
    view_distance_m: float,
    maximum_event_view_range_m: float,
    maximum_wearer_speed_mps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(bool(rows), "source trace is empty")
    _require(maximum_wearer_speed_mps > 0.0, "wearer speed must be positive")
    _require(
        maximum_event_view_range_m >= view_distance_m,
        "maximum event view range is smaller than the desired view distance",
    )
    active_poses: dict[int, tuple[float, float, float]] = {}
    active_targets: dict[int, dict[str, Any]] = {}
    active_bindings: list[dict[str, Any]] = []
    for ordinal, event in enumerate(
        sorted(events["tail_events"], key=lambda value: float(value["applied_time_s"]))
    ):
        start = max(0, int(math.ceil(float(event["applied_time_s"]) / fixed_delta_seconds - 1e-9)))
        end = min(
            len(rows) - 1,
            int(math.floor(float(event["ended_time_s"]) / fixed_delta_seconds - 1e-9)),
        )
        _require(start <= end, f"empty event window: {event['event_id']}")
        actor_id = str(event["primary_actor_id"])
        lateral_m = -0.8 if ordinal % 2 == 0 else 0.8
        for sample_index in range(start, end + 1):
            _require(sample_index not in active_poses, "tail-event view windows overlap")
            target = rows[sample_index]["actors"][actor_id]
            active_poses[sample_index] = event_pose(
                target, view_distance_m, lateral_m
            )
            active_targets[sample_index] = target["transform"]
        active_bindings.append(
            {
                "event_id": str(event["event_id"]),
                "type": str(event["type"]),
                "primary_actor_id": actor_id,
                "start_sample": start,
                "end_sample": end,
                "lateral_offset_m": lateral_m,
            }
        )
    keys = sorted(active_poses)
    _require(bool(keys), "no event-bearing route anchors were created")
    route: list[dict[str, Any]] = []
    maximum_step_limit_m = maximum_wearer_speed_mps * fixed_delta_seconds
    maximum_observed_step_m = 0.0
    route_path_length_m = 0.0
    current_xy: tuple[float, float] | None = None
    previous_yaw = float(active_poses[keys[0]][2])
    for sample_index in range(len(rows)):
        if sample_index in active_poses:
            desired = active_poses[sample_index]
            target = active_targets[sample_index]
        else:
            insertion = bisect.bisect_left(keys, sample_index)
            if insertion < len(keys):
                next_sample = keys[insertion]
                desired = active_poses[next_sample]
                target = active_targets[next_sample]
            else:
                _require(current_xy is not None, "route state was lost after final event")
                desired = (current_xy[0], current_xy[1], previous_yaw)
                target = None
        if current_xy is None:
            current_xy = (float(desired[0]), float(desired[1]))
        else:
            current_xy = bounded_step(
                current_xy,
                (float(desired[0]), float(desired[1])),
                maximum_step_limit_m,
            )
        if target is not None:
            yaw = math.degrees(
                math.atan2(
                    float(target["y"]) - current_xy[1],
                    float(target["x"]) - current_xy[0],
                )
            )
        else:
            yaw = previous_yaw
        previous_yaw = yaw
        row = {
            "sample_index": sample_index,
            "time_s": round(sample_index * fixed_delta_seconds, 8),
            "x_m": round(current_xy[0], 6),
            "y_m": round(current_xy[1], 6),
            "yaw_degrees": round(float(yaw), 6),
        }
        if route:
            step_m = math.hypot(
                row["x_m"] - route[-1]["x_m"], row["y_m"] - route[-1]["y_m"]
            )
            maximum_observed_step_m = max(maximum_observed_step_m, step_m)
            route_path_length_m += step_m
        route.append(row)

    maximum_bearing_error_degrees = 0.0
    minimum_event_range_m = math.inf
    maximum_event_range_m = 0.0
    event_range_audits: list[dict[str, Any]] = []
    for binding in active_bindings:
        actor_id = binding["primary_actor_id"]
        event_minimum_range_m = math.inf
        event_maximum_range_m = 0.0
        for sample_index in range(binding["start_sample"], binding["end_sample"] + 1):
            target = rows[sample_index]["actors"][actor_id]["transform"]
            wearer = route[sample_index]
            dx = float(target["x"]) - float(wearer["x_m"])
            dy = float(target["y"]) - float(wearer["y_m"])
            bearing = math.degrees(math.atan2(dy, dx))
            error = abs((bearing - float(wearer["yaw_degrees"]) + 180.0) % 360.0 - 180.0)
            maximum_bearing_error_degrees = max(maximum_bearing_error_degrees, error)
            distance = math.hypot(dx, dy)
            minimum_event_range_m = min(minimum_event_range_m, distance)
            maximum_event_range_m = max(maximum_event_range_m, distance)
            event_minimum_range_m = min(event_minimum_range_m, distance)
            event_maximum_range_m = max(event_maximum_range_m, distance)
        event_range_audits.append(
            {
                "event_id": binding["event_id"],
                "minimum_range_m": event_minimum_range_m,
                "maximum_range_m": event_maximum_range_m,
            }
        )
    maximum_route_speed_mps = maximum_observed_step_m / fixed_delta_seconds
    audit = {
        "schema_version": "dtr-carla-n4-bounded-event-bearing-route-audit-v1",
        "authority": "TRACE_CONDITIONED_FROZEN_BEFORE_REPLAY",
        "motion_model": "bounded_speed_planar_wearer_route",
        "route_frames": len(route),
        "event_count": len(active_bindings),
        "event_bindings": active_bindings,
        "event_range_audits": event_range_audits,
        "desired_event_view_distance_m": view_distance_m,
        "maximum_event_view_range_m": maximum_event_view_range_m,
        "maximum_wearer_speed_mps": maximum_wearer_speed_mps,
        "maximum_bearing_error_degrees": maximum_bearing_error_degrees,
        "minimum_event_range_m": minimum_event_range_m,
        "maximum_event_range_m": maximum_event_range_m,
        "maximum_route_step_m": maximum_observed_step_m,
        "maximum_route_speed_mps": maximum_route_speed_mps,
        "route_path_length_m": route_path_length_m,
        "checks": {
            "all_four_events_have_nonempty_view_windows": len(active_bindings) == 4,
            "event_actors_are_centered_in_horizontal_fov": maximum_bearing_error_degrees
            <= 1e-3,
            "event_view_range_is_local": maximum_event_range_m
            <= maximum_event_view_range_m + 1e-6,
            "wearer_route_speed_is_bounded": maximum_route_speed_mps
            <= maximum_wearer_speed_mps + 1e-4,
        },
    }
    _require(all(audit["checks"].values()), f"route audit failed: {audit}")
    return route, audit


def child_protocol(scene: dict[str, Any], route: list[dict[str, Any]]) -> dict[str, Any]:
    map_name = str(scene["map"])
    scenario_class = str(scene["scenario_class"])
    return {
        "schema_version": "dtr-carla-n2-frozen-trace-replay-protocol-v1",
        "experiment_id": "DTR_CARLA_N2_FROZEN_TRACE_C2_REPLAY_V1",
        "evidence_role": "synthetic_development_source_replay",
        "source": {
            "required_result_status": "DTR_CARLA_N1_NATURAL_DYNAMICS_MATERIALIZED",
            "plan_id": str(scene["plan_id"]),
            "plan_fingerprint_sha256": str(scene["plan_fingerprint_sha256"]),
            "expected_trace_frames": int(scene["expected_trace_frames"]),
            "expected_actor_count": int(scene["expected_actor_count"]),
            "files": {name: str(scene["files"][name]["sha256"]) for name in SOURCE_FILES},
        },
        "environment": {
            "carla_version": "0.9.16",
            "map": map_name,
            "fixed_delta_seconds": 0.05,
            "weather": "ClearNoon",
        },
        "capture": {
            "sensor_order": ["instance", "wearable", "depth", "witness"],
            "same_world_frame": True,
            "resolution": [1280, 720],
            "fov_degrees": 90.0,
            "wearer": {
                "observer_mode": "frozen_event_bearing_route",
                "route_authority": "TRACE_CONDITIONED_FROZEN_BEFORE_REPLAY",
                "motion_model": "bounded_speed_planar_wearer_route",
                "maximum_speed_mps": float(scene["maximum_wearer_speed_mps"]),
                "maximum_event_view_range_m": float(
                    scene["maximum_event_view_range_m"]
                ),
                "blueprint": "walker.pedestrian.0001",
                "surface_offset_m": 0.8,
                "body_radius_m": 0.45,
                "route": route,
            },
            "wearable_relative_transform": {
                "x_m": 0.08,
                "y_m": 0.0,
                "z_m": 0.65,
                "pitch_degrees": -5.0,
                "yaw_degrees": 0.0,
                "roll_degrees": 0.0,
            },
            "witness_transform_authority": "source_actor_manifest_camera",
            "camera_calibration": {
                "intrinsic_convention": "CARLA_PINHOLE_K_WIDTH_OVER_2",
                "principal_point": [640.0, 360.0],
                "focal_length_pixels": [640.0, 640.0],
                "depth_codec": {
                    "name": "CARLA_RGB24_NORMALIZED_DEPTH",
                    "maximum_depth_m": 1000.0,
                    "formula": "meters=1000*(R+256*G+65536*B)/(16777215)",
                },
            },
        },
        "episode": {
            "episode_id": f"n4_{scene['scene_id']}",
            "layout_id": f"n4_{scenario_class}",
            "navigation_session_id": f"n4_route_{scene['scene_id']}",
            "issued_plan_authority": "FROZEN_EVENT_BEARING_ROUTE",
            "future_contact_horizon_seconds": 3.0,
        },
        "model_contract": {
            "schema_version": "dtr-c2-model-contract-v2",
            "include_current_actors": False,
            "dense_modalities": ["wearable_rgb", "metric_depth"],
            "evaluator_modalities": ["instance_segmentation", "witness_rgb"],
        },
        "claim_boundary": [
            "This is one shard of a single frozen three-map synthetic Development replay.",
            "The wearer route was derived from evaluator-side source traces and frozen before sensor replay solely to make the authored events observable.",
            "The wearer route is bounded to the predeclared planar speed and local event-view range; it is still a synthetic authored trajectory, not natural wearer-motion evidence.",
            "It is not a natural-traffic distribution, obstacle-algorithm result, source-disjoint confirmation, real-world evidence, product benefit, or safety evidence.",
        ],
    }


def build_bundle(source_run_root: Path, output_root: Path) -> dict[str, Any]:
    source_run_root = source_run_root.resolve(strict=True)
    output_root = output_root.resolve()
    _require(not output_root.exists(), f"refusing N4 bundle overwrite: {output_root}")
    source_result = read_json(source_run_root / "result.json")
    _require(source_result.get("status") == RESULT_STATUS, "N3 source result is not complete")
    receipt = read_json(source_run_root / "source_suite_receipt.json")
    _require(
        receipt.get("authority") == "FROZEN_N3_MULTITOWN_NATIVE_SOURCE_SUITE_VERIFIED",
        "N3 source receipt authority differs",
    )
    _require(tuple(receipt.get("scene_order", [])) == SCENE_ORDER, "N3 scene order differs")
    output_root.mkdir(parents=True)
    protocol_root = output_root / "protocols"
    audit_root = output_root / "route-audits"
    protocol_root.mkdir()
    audit_root.mkdir()
    children: list[dict[str, Any]] = []
    for scene in receipt["scenes"]:
        scene_id = str(scene["scene_id"])
        source_root = source_run_root / str(scene["source_root"])
        rows = read_jsonl(source_root / "behavior_trace.jsonl")
        events = read_json(source_root / "event_receipts.json")
        route, audit = build_event_bearing_route(
            rows,
            events,
            fixed_delta_seconds=0.05,
            view_distance_m=float(scene["route_view_distance_m"]),
            maximum_event_view_range_m=float(scene["maximum_event_view_range_m"]),
            maximum_wearer_speed_mps=float(scene["maximum_wearer_speed_mps"]),
        )
        protocol = child_protocol(scene, route)
        validate_protocol(protocol)
        protocol_path = protocol_root / f"{scene_id}.json"
        audit_path = audit_root / f"{scene_id}.json"
        write_json_atomic(protocol_path, protocol)
        write_json_atomic(audit_path, audit)
        leaf = str(scene["map"]).rsplit("/", 1)[-1]
        children.append(
            {
                "ordinal": int(scene["ordinal"]),
                "scene_id": scene_id,
                "map": scene["map"],
                "scenario_class": scene["scenario_class"],
                "source_root": str(source_root.resolve()),
                "protocol_path": f"protocols/{scene_id}.json",
                "protocol_sha256": sha256_file(protocol_path),
                "route_audit_path": f"route-audits/{scene_id}.json",
                "route_audit_sha256": sha256_file(audit_path),
                "engine_map_object_path": f"/Game/Carla/Maps/{leaf}.{leaf}",
                "expected_trace_frames": int(scene["expected_trace_frames"]),
                "expected_actor_count": int(scene["expected_actor_count"]),
            }
        )
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "experiment_id": "DTR_CARLA_N4_MULTITOWN_FROZEN_FOUR_MODAL_REPLAY_V1",
        "evidence_role": "single_frozen_multitown_synthetic_development_replay",
        "source_run_root": str(source_run_root),
        "source_result_sha256": sha256_file(source_run_root / "result.json"),
        "source_suite_receipt_sha256": sha256_file(
            source_run_root / "source_suite_receipt.json"
        ),
        "scene_count": len(children),
        "scene_order": list(SCENE_ORDER),
        "replay_invocation_budget": 1,
        "children": children,
        "claim_boundary": [
            "The outer N4 run is the sole frozen four-modal replay; its three map shards are required because CARLA loads one map per world.",
            "No route, source, threshold, or sensor retry is authorized after replay pixels are observed.",
        ],
    }
    write_json_atomic(output_root / "bundle.json", bundle)
    return bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = build_bundle(args.source_run_root, args.output_root)
    print(json.dumps(bundle, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
