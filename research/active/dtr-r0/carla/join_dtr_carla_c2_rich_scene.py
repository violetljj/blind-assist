"""Join C2 sensor shards into truth-separated model/evaluator evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from dtr_carla_c2_rich_scene import (
    EXPERIMENT_ID,
    angle_delta_degrees,
    build_rgbd_alignment_receipt,
    build_plan_receipt,
    camera_intrinsics,
    contiguous_runs,
    forbidden_model_paths,
    line_intersects_polygon,
    materialize_layout_assets,
    plan_waypoints_world,
    sha256_file,
    validate_model_record,
    validate_protocol,
    write_json_atomic,
    write_jsonl,
)


SENSOR_ORDER = ["instance", "wearable", "depth", "witness"]
C4_EXPERIMENT_ID = "DTR_CARLA_C4_MULTIMAP_WORLD_PACK_V1"
C4_OCCLUSION_SCOPE = "C4_MULTIMAP_PACK_FINAL_JOIN"
ACTUAL_ALL_REPLAY = "ACTUAL_ALL_ACTORS"
AUTHORITY_SCOPED_WITNESS_REPLAY = (
    "AUTHORITATIVE_ACTUAL_PLUS_NONAUTHORITATIVE_SCRIPTED_COMMAND"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rounded(value: Any, digits: int = 5) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _rounded(child, digits) for key, child in sorted(value.items())}
    if isinstance(value, list):
        return [_rounded(child, digits) for child in value]
    return value


def replay_projection(
    row: dict[str, Any],
    *,
    mode: str = ACTUAL_ALL_REPLAY,
    authoritative_actor_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    actors: dict[str, Any] = {}
    for key, actor in sorted(row["actors"].items()):
        identity = {
            "track_id": actor["track_id"],
            "asset_key": actor["asset_key"],
            "role": actor["role"],
            "kind": actor["kind"],
            "actual_blueprint": actor["actual_blueprint"],
            "local_position": _rounded(actor["local_position"]),
            "command_velocity": _rounded(actor["command_velocity"]),
        }
        if mode == ACTUAL_ALL_REPLAY or key in authoritative_actor_keys:
            actors[key] = {
                **identity,
                "pose_authority": "ACTUAL",
                "transform": _rounded(actor["transform"]),
                "bounding_box": _rounded(actor["bounding_box"]),
            }
        elif mode == AUTHORITY_SCOPED_WITNESS_REPLAY:
            actors[key] = {
                **identity,
                "pose_authority": "SCRIPTED_COMMAND",
                "transform": _rounded(actor["scripted_command_transform"]),
            }
        else:
            raise ValueError(f"unknown replay projection mode: {mode}")
    truth = row["truth"]
    truth_projection = {
        "scenario_role": truth["scenario_role"],
        "twin_role": truth["twin_role"],
        "expected_outcome": truth["expected_outcome"],
        "expected_responsible_assets": sorted(
            truth["expected_responsible_assets"]
        ),
        "current_contact": bool(truth["current_contact"]),
        "responsible_assets": sorted(truth["responsible_assets"]),
        "future_contact_within_horizon": bool(
            truth["future_contact_within_horizon"]
        ),
        "realized_time_to_contact_seconds": (
            round(float(truth["realized_time_to_contact_seconds"]), 5)
            if truth["realized_time_to_contact_seconds"] is not None
            else None
        ),
    }
    if mode == ACTUAL_ALL_REPLAY:
        truth_projection.update(
            {
                "minimum_distance_m": round(float(truth["minimum_distance_m"]), 5),
                "collision_polygons_xy": _rounded(truth["collision_polygons_xy"]),
            }
        )
    return {
        "episode_id": row["episode_id"],
        "layout_id": row["layout_id"],
        "sample_index": int(row["sample_index"]),
        "time_s": round(float(row["time_s"]), 8),
        "layout_receipt_sha256": row["layout_receipt_sha256"],
        "plan_receipt_sha256": row["plan_receipt_sha256"],
        "wearer_transform": _rounded(row["wearer_transform"]),
        "actors": actors,
        "truth": truth_projection,
    }


def compare_replays(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    candidate_sensor: str,
    mode: str = ACTUAL_ALL_REPLAY,
    authoritative_actor_keys: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    if len(reference) != len(candidate):
        return [
            {
                "sensor": candidate_sensor,
                "reason": "frame_count",
                "reference": len(reference),
                "candidate": len(candidate),
            }
        ]
    failures: list[dict[str, Any]] = []
    for first, second in zip(reference, candidate, strict=True):
        if replay_projection(
            first,
            mode=mode,
            authoritative_actor_keys=authoritative_actor_keys,
        ) != replay_projection(
            second,
            mode=mode,
            authoritative_actor_keys=authoritative_actor_keys,
        ):
            failures.append(
                {
                    "sensor": candidate_sensor,
                    "episode_id": first["episode_id"],
                    "sample_index": first["sample_index"],
                    "reason": "actual_replay_state_mismatch",
                }
            )
            if len(failures) >= 50:
                break
    return failures


def verify_payload_inventory(shard_root: Path) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    values = read_json(shard_root / "payload_inventory.json")
    for value in values:
        path = shard_root / str(value["relative_path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        if path.stat().st_size != int(value["bytes"]):
            failures.append({"path": str(path), "reason": "bytes"})
        elif sha256_file(path) != str(value["sha256"]):
            failures.append({"path": str(path), "reason": "sha256"})
        if len(failures) >= 50:
            break
    return failures


def physical_occlusion_runs(
    rows: list[dict[str, Any]],
    *,
    target_key: str,
    occluder_key: str,
    fov_degrees: float,
    hidden_fraction: float,
) -> list[list[int]]:
    indices: list[int] = []
    for row in rows:
        camera = row["camera_transform"]
        target = row["actors"][target_key]["transform"]
        start = (float(camera["x"]), float(camera["y"]))
        end = (float(target["x"]), float(target["y"]))
        bearing = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
        in_fov = (
            angle_delta_degrees(bearing, float(camera["yaw"]))
            <= fov_degrees / 2.0 - 1.0
        )
        fraction = float(row["instance_visibility"][target_key]["pixel_fraction"])
        polygon = row["truth"]["collision_polygons_xy"][occluder_key]
        if (
            in_fov
            and fraction <= hidden_fraction + 1e-12
            and line_intersects_polygon(start, end, polygon)
        ):
            indices.append(int(row["sample_index"]))
    return contiguous_runs(indices)


def consecutive_trackable_before(
    rows: list[dict[str, Any]],
    *,
    end_index_exclusive: int,
    target_key: str,
    minimum_fraction: float,
) -> list[int]:
    values: list[int] = []
    index = end_index_exclusive - 1
    while index >= 0:
        fraction = float(rows[index]["instance_visibility"][target_key]["pixel_fraction"])
        if fraction < minimum_fraction:
            break
        values.append(index)
        index -= 1
    return sorted(values)


def consecutive_trackable_after(
    rows: list[dict[str, Any]],
    *,
    start_index_inclusive: int,
    target_key: str,
    minimum_fraction: float,
) -> list[int]:
    values: list[int] = []
    for index in range(start_index_inclusive, len(rows)):
        fraction = float(rows[index]["instance_visibility"][target_key]["pixel_fraction"])
        if fraction < minimum_fraction:
            break
        values.append(index)
    return values


def evaluate_occlusion_contract(
    protocol: dict[str, Any],
    contract: dict[str, Any],
    rows_by_episode: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    target = str(contract["target_asset"])
    occluder = str(contract["occluder_asset"])
    sample_s = float(protocol["environment"]["sample_seconds"])
    episode_reports: dict[str, Any] = {}
    selected_indices: dict[str, list[int]] = {}
    for episode_id in contract["episodes"]:
        rows = rows_by_episode[str(episode_id)]
        runs = physical_occlusion_runs(
            rows,
            target_key=target,
            occluder_key=occluder,
            fov_degrees=float(protocol["capture"]["fov_degrees"]),
            hidden_fraction=float(contract["complete_occlusion_pixel_fraction"]),
        )
        eligible: list[dict[str, Any]] = []
        for run in runs:
            pre = consecutive_trackable_before(
                rows,
                end_index_exclusive=run[0],
                target_key=target,
                minimum_fraction=float(contract["minimum_trackable_pixel_fraction"]),
            )
            post = consecutive_trackable_after(
                rows,
                start_index_inclusive=run[-1] + 1,
                target_key=target,
                minimum_fraction=float(contract["minimum_trackable_pixel_fraction"]),
            )
            duration = len(run) * sample_s
            passed = (
                len(pre) >= int(contract["minimum_pre_track_frames"])
                and len(post) >= int(contract["minimum_post_reappearance_frames"])
                and duration
                >= float(contract["minimum_complete_occlusion_seconds"]) - 1e-9
                and duration
                <= float(contract["maximum_complete_occlusion_seconds"]) + 1e-9
            )
            eligible.append(
                {
                    "sample_indices": run,
                    "duration_seconds": duration,
                    "pre_track_sample_indices": pre,
                    "post_reappearance_sample_indices": post,
                    "pre_track_frames": len(pre),
                    "post_reappearance_frames": len(post),
                    "passed": passed,
                }
            )
        selected = next((value for value in eligible if value["passed"]), None)
        selected_indices[str(episode_id)] = (
            list(selected["sample_indices"]) if selected else []
        )
        episode_reports[str(episode_id)] = {
            "runs": eligible,
            "selected": selected,
            "passed": selected is not None,
        }
    pair_identical = len({tuple(value) for value in selected_indices.values()}) == 1
    passed = all(value["passed"] for value in episode_reports.values()) and pair_identical
    return {
        "contract_id": str(contract["contract_id"]),
        "episodes": episode_reports,
        "pair_occlusion_indices_identical": pair_identical,
        "selected_indices": selected_indices,
        "passed": passed,
    }


def link_or_copy(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        method = "copy"
    return {
        "relative_path": str(destination).replace("\\", "/"),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "materialization": method,
    }


def safe_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        value = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"path escapes evidence root: {resolved}") from exc
    return str(value).replace("\\", "/")


def seal_tree(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        values.append(
            {
                "path": safe_relative(path, root),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return values


def scan_model_truth(model_root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(model_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        values = read_jsonl(path) if path.suffix.lower() == ".jsonl" else [read_json(path)]
        for index, value in enumerate(values):
            for forbidden in forbidden_model_paths(value):
                failures.append(f"{safe_relative(path, model_root)}[{index}]{forbidden}")
                if len(failures) >= 100:
                    return failures
    return failures


def render_summary_svg(
    protocol: dict[str, Any],
    reports: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">',
        '<rect width="1280" height="720" fill="#0f172a"/>',
        '<text x="50" y="70" font-family="sans-serif" font-size="34" fill="#f8fafc">CARLA C2 rich scene source</text>',
        '<text x="50" y="110" font-family="sans-serif" font-size="20" fill="#94a3b8">1280x720 RGB + depth model inputs; instance + witness evaluator only</text>',
    ]
    y = 175
    for layout_id, layout in protocol["layouts"].items():
        count = len(materialize_layout_assets(protocol, layout_id))
        lines.append(
            f'<rect x="50" y="{y - 35}" width="1180" height="105" rx="14" fill="#1e293b"/>'
        )
        lines.append(
            f'<text x="75" y="{y}" font-family="sans-serif" font-size="25" fill="#38bdf8">{layout_id}: {layout["display_name"]}</text>'
        )
        lines.append(
            f'<text x="75" y="{y + 38}" font-family="sans-serif" font-size="20" fill="#e2e8f0">{count} non-wearer assets | {layout["weather"]} | {layout["duration_seconds"]:.1f}s</text>'
        )
        y += 125
    report = reports[0]
    pair = report["selected_indices"]
    lines.append(
        f'<text x="50" y="590" font-family="sans-serif" font-size="22" fill="#4ade80">Physical complete occlusion: {pair}</text>'
    )
    lines.append(
        '<text x="50" y="635" font-family="sans-serif" font-size="20" fill="#fbbf24">74 actual blueprint types; zero fallback; CONTACT/SAFE outcomes scored evaluator-side</text>'
    )
    lines.append(
        '<text x="50" y="680" font-family="sans-serif" font-size="18" fill="#94a3b8">Synthetic Development source only; no real-world or safety claim.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    root = args.root.resolve(strict=True)
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    frozen = root / "frozen_protocol.json"
    if sha256_file(frozen) != sha256_file(protocol_path):
        raise RuntimeError("frozen protocol does not match the requested protocol")
    model_root = root / "model"
    evaluator_root = root / "evaluator"
    if model_root.exists() or evaluator_root.exists():
        raise FileExistsError("refusing to overwrite joined model/evaluator evidence")
    model_root.mkdir()
    evaluator_root.mkdir()

    results: dict[str, dict[str, Any]] = {}
    payload_failures: list[dict[str, Any]] = []
    for sensor in SENSOR_ORDER:
        shard_root = root / "shards" / sensor
        result = read_json(shard_root / "result.json")
        results[sensor] = result
        if result["status"] != "DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE":
            raise RuntimeError(f"{sensor} shard is not complete")
        payload_failures.extend(verify_payload_inventory(shard_root))

    identity_fields = ("protocol_sha256", "capture_script_sha256", "helper_module_sha256")
    identity_equal = all(
        len({str(results[sensor][field]) for sensor in SENSOR_ORDER}) == 1
        for field in identity_fields
    )
    episode_ids = [str(value["episode_id"]) for value in protocol["scenarios"]]
    rows_by_sensor: dict[str, dict[str, list[dict[str, Any]]]] = {
        sensor: {
            episode_id: read_jsonl(
                root
                / "shards"
                / sensor
                / "episodes"
                / episode_id
                / "frames.jsonl"
            )
            for episode_id in episode_ids
        }
        for sensor in SENSOR_ORDER
    }
    witness_replay_mode = str(
        protocol["capture"].get("witness_replay_projection", ACTUAL_ALL_REPLAY)
    )
    if witness_replay_mode not in {
        ACTUAL_ALL_REPLAY,
        AUTHORITY_SCOPED_WITNESS_REPLAY,
    }:
        raise ValueError(f"unknown witness replay projection: {witness_replay_mode}")
    authoritative_actor_keys = frozenset(
        str(asset["asset_key"])
        for layout_id in protocol["layouts"]
        for asset in materialize_layout_assets(protocol, str(layout_id))
        if bool(asset.get("scripted_pose_authority", False))
    )
    replay_failures: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        reference = rows_by_sensor["instance"][episode_id]
        for sensor in SENSOR_ORDER[1:]:
            replay_failures.extend(
                compare_replays(
                    reference,
                    rows_by_sensor[sensor][episode_id],
                    candidate_sensor=sensor,
                    mode=(
                        witness_replay_mode
                        if sensor == "witness"
                        else ACTUAL_ALL_REPLAY
                    ),
                    authoritative_actor_keys=authoritative_actor_keys,
                )
            )

    instance_rows = rows_by_sensor["instance"]
    occlusion_reports = [
        evaluate_occlusion_contract(protocol, value, instance_rows)
        for value in protocol["occlusion_contracts"]
    ]
    write_json_atomic(evaluator_root / "physical_occlusion_report.json", occlusion_reports)

    alignment_receipt = build_rgbd_alignment_receipt(
        rows_by_sensor["wearable"], rows_by_sensor["depth"]
    )
    alignment_receipt_path = model_root / "rgbd_alignment_receipt.json"
    write_json_atomic(alignment_receipt_path, alignment_receipt)
    alignment_receipt_file_sha256 = sha256_file(alignment_receipt_path)
    alignment_by_episode = {
        str(value["episode_id"]): value for value in alignment_receipt["episodes"]
    }

    width, height = map(int, protocol["capture"]["resolution"])
    fov = float(protocol["capture"]["fov_degrees"])
    calibration = {
        "schema_version": "dtr-c2-model-camera-contract-v1",
        "resolution": {"width": width, "height": height},
        "fov_degrees": fov,
        "K": camera_intrinsics(width, height, fov),
        "depth_codec": protocol["capture"]["camera_calibration"]["depth_codec"],
        "wearable_rigid_extrinsic": protocol["capture"][
            "wearable_relative_transform"
        ],
        "sensor_tick_seconds": float(protocol["environment"]["sample_seconds"]),
    }
    camera_calibration_path = model_root / "camera_calibration.json"
    write_json_atomic(camera_calibration_path, calibration)
    model_contract_path = model_root / "model_contract.json"
    write_json_atomic(
        model_contract_path,
        {
            "schema_version": "dtr-c2-model-contract-v2",
            "current_actors_enabled": False,
            "dense_modalities": ["wearable_rgb", "metric_depth"],
            "evaluator_sibling_not_required": True,
            "rgbd_alignment": {
                "authority": alignment_receipt["authority"],
                "receipt_path": safe_relative(alignment_receipt_path, model_root),
                "receipt_sha256": alignment_receipt["receipt_sha256"],
                "file_sha256": alignment_receipt_file_sha256,
                "world_frame_rule": alignment_receipt["world_frame_rule"],
            },
            "record_top_level_allowlist": sorted(
                {
                    "schema_version",
                    "episode_id",
                    "sample_index",
                    "world_frame",
                    "time_s",
                    "timestamp_s",
                    "wearable_rgb",
                    "metric_depth",
                    "camera",
                    "wearer_pose_current",
                    "navigation",
                    "frame_alignment",
                }
            ),
        },
    )

    model_records_total = 0
    model_payloads_total = 0
    model_camera_alignment = True
    model_plan_paths: list[Path] = []
    model_episode_manifest_paths: list[Path] = []
    outcome_summaries: list[dict[str, Any]] = []
    for scenario in protocol["scenarios"]:
        episode_id = str(scenario["episode_id"])
        layout = protocol["layouts"][scenario["layout_id"]]
        plan_receipt = build_plan_receipt(scenario.get("issued_plan"))
        plan_authority = "VALID" if plan_receipt is not None else "NO_PLAN"
        model_plan_path = model_root / "plans" / f"{episode_id}.json"
        model_plan_payload = {
            "schema_version": "dtr-c2-model-plan-v1",
            "episode_id": episode_id,
            "navigation_session_id": str(scenario["navigation_session_id"]),
            "layout_anchor": {
                "world_center_xy_m": [
                    float(value) for value in layout["anchor"]["center_xy_m"]
                ],
                "world_forward_xy": [
                    float(value) for value in layout["anchor"]["forward_xy"]
                ],
                "world_right_xy": [
                    float(value) for value in layout["anchor"]["right_xy"]
                ],
            },
            "issued_plan": {
                "authority": plan_authority,
                "receipt": plan_receipt,
                "receipt_sha256": (
                    plan_receipt["receipt_sha256"] if plan_receipt else None
                ),
                "world_coordinate_frame": "CARLA_WORLD_XY",
                "time_parameterized_waypoints_world": plan_waypoints_world(
                    plan_receipt, layout["anchor"]
                ),
            },
        }
        write_json_atomic(model_plan_path, model_plan_payload)
        model_plan_paths.append(model_plan_path)
        model_plan_relative_path = safe_relative(model_plan_path, model_root)
        wearable_rows = rows_by_sensor["wearable"][episode_id]
        depth_rows = rows_by_sensor["depth"][episode_id]
        instance_episode_rows = rows_by_sensor["instance"][episode_id]
        witness_rows = rows_by_sensor["witness"][episode_id]
        if not (
            len(wearable_rows)
            == len(depth_rows)
            == len(instance_episode_rows)
            == len(witness_rows)
        ):
            raise RuntimeError(f"modality frame count mismatch for {episode_id}")
        model_episode = model_root / "episodes" / episode_id
        evaluator_episode = evaluator_root / "episodes" / episode_id
        model_rows: list[dict[str, Any]] = []
        evaluator_rows: list[dict[str, Any]] = []
        episode_alignment = alignment_by_episode[episode_id]
        source_frame_offset = int(
            episode_alignment["depth_minus_wearable_source_world_frame_offset"]
        )
        for wearable, depth, instance, witness in zip(
            wearable_rows,
            depth_rows,
            instance_episode_rows,
            witness_rows,
            strict=True,
        ):
            sample_index = int(wearable["sample_index"])
            if not (
                sample_index
                == int(depth["sample_index"])
                == int(instance["sample_index"])
                == int(witness["sample_index"])
            ):
                raise RuntimeError(f"sample alignment failure in {episode_id}")
            rgb_destination = model_episode / "rgb" / f"{sample_index:06d}.png"
            depth_destination = model_episode / "depth" / f"{sample_index:06d}.png"
            instance_destination = (
                evaluator_episode / "instance" / f"{sample_index:06d}.png"
            )
            witness_destination = (
                evaluator_episode / "witness" / f"{sample_index:06d}.png"
            )
            rgb_link = link_or_copy(
                root / "shards" / "wearable" / wearable["sensor_path"],
                rgb_destination,
            )
            depth_link = link_or_copy(
                root / "shards" / "depth" / depth["sensor_path"],
                depth_destination,
            )
            instance_link = link_or_copy(
                root / "shards" / "instance" / instance["sensor_path"],
                instance_destination,
            )
            witness_link = link_or_copy(
                root / "shards" / "witness" / witness["sensor_path"],
                witness_destination,
            )
            rgb_link["path"] = safe_relative(rgb_destination, model_root)
            depth_link["path"] = safe_relative(depth_destination, model_root)
            instance_link["path"] = safe_relative(instance_destination, evaluator_root)
            witness_link["path"] = safe_relative(witness_destination, evaluator_root)
            camera_equal = _rounded(wearable["camera_transform"], 4) == _rounded(
                depth["camera_transform"], 4
            )
            model_camera_alignment = model_camera_alignment and camera_equal
            aligned_world_frame = int(wearable["world_frame"])
            wearable_source_world_frame = int(wearable["world_frame"])
            depth_source_world_frame = int(depth["world_frame"])
            if depth_source_world_frame - wearable_source_world_frame != source_frame_offset:
                raise RuntimeError(f"source-frame offset changed in {episode_id}")
            model_record = {
                "schema_version": "dtr-c2-model-observation-v2",
                "episode_id": episode_id,
                "sample_index": sample_index,
                "world_frame": aligned_world_frame,
                "time_s": float(wearable["time_s"]),
                "timestamp_s": float(wearable["time_s"]),
                "wearable_rgb": {
                    "path": rgb_link["path"],
                    "bytes": int(rgb_link["bytes"]),
                    "sha256": rgb_link["sha256"],
                    "width": width,
                    "height": height,
                    "source_world_frame": wearable_source_world_frame,
                },
                "metric_depth": {
                    "path": depth_link["path"],
                    "bytes": int(depth_link["bytes"]),
                    "sha256": depth_link["sha256"],
                    "width": width,
                    "height": height,
                    "codec": calibration["depth_codec"],
                    "source_world_frame": depth_source_world_frame,
                },
                "camera": {
                    "world_transform": wearable["camera_transform"],
                    "rigid_extrinsic": calibration["wearable_rigid_extrinsic"],
                    "width": width,
                    "height": height,
                    "fov_degrees": fov,
                    "K": calibration["K"],
                },
                "wearer_pose_current": wearable["wearer_transform"],
                "navigation": {
                    "navigation_session_id": str(
                        scenario["navigation_session_id"]
                    ),
                    "issued_plan": {
                        "authority": plan_authority,
                        "path": model_plan_relative_path,
                        "receipt_sha256": (
                            plan_receipt["receipt_sha256"]
                            if plan_receipt
                            else None
                        ),
                    },
                },
                "frame_alignment": {
                    "authority": alignment_receipt["authority"],
                    "reference_modality": "wearable_rgb",
                    "receipt_path": safe_relative(alignment_receipt_path, model_root),
                    "receipt_sha256": alignment_receipt["receipt_sha256"],
                    "depth_minus_wearable_source_world_frame_offset": (
                        source_frame_offset
                    ),
                },
            }
            validate_model_record(model_record)
            model_rows.append(model_record)
            evaluator_rows.append(
                {
                    "schema_version": "dtr-c2-evaluator-frame-v1",
                    "episode_id": episode_id,
                    "layout_id": instance["layout_id"],
                    "sample_index": sample_index,
                    "time_s": float(instance["time_s"]),
                    "instance": instance_link,
                    "witness": witness_link,
                    "camera_transform": instance["camera_transform"],
                    "actors": instance["actors"],
                    "instance_visibility": instance["instance_visibility"],
                    "truth": instance["truth"],
                }
            )
            model_payloads_total += 2
        write_jsonl(model_episode / "observations.jsonl", model_rows)
        write_jsonl(evaluator_episode / "frames.jsonl", evaluator_rows)
        model_episode_manifest_path = model_episode / "manifest.json"
        write_json_atomic(
            model_episode_manifest_path,
            {
                "schema_version": "dtr-c2-model-episode-manifest-v2",
                "episode_id": episode_id,
                "frames": len(model_rows),
                "observations_sha256": sha256_file(
                    model_episode / "observations.jsonl"
                ),
                "rgb_payloads": len(model_rows),
                "depth_payloads": len(model_rows),
                "navigation_session_id": str(scenario["navigation_session_id"]),
                "rgbd_alignment": {
                    "authority": alignment_receipt["authority"],
                    "receipt_path": safe_relative(alignment_receipt_path, model_root),
                    "receipt_sha256": alignment_receipt["receipt_sha256"],
                    "depth_minus_wearable_source_world_frame_offset": (
                        source_frame_offset
                    ),
                },
                "issued_plan": {
                    "authority": plan_authority,
                    "path": model_plan_relative_path,
                    "receipt_sha256": (
                        plan_receipt["receipt_sha256"] if plan_receipt else None
                    ),
                    "file_sha256": sha256_file(model_plan_path),
                },
            },
        )
        model_episode_manifest_paths.append(model_episode_manifest_path)
        model_records_total += len(model_rows)
        contact_rows = [
            row for row in instance_episode_rows if bool(row["truth"]["current_contact"])
        ]
        responsible = sorted(
            {
                str(key)
                for row in contact_rows
                for key in row["truth"]["responsible_assets"]
            }
        )
        outcome_summaries.append(
            {
                "episode_id": episode_id,
                "layout_id": str(scenario["layout_id"]),
                "expected_outcome": str(scenario["expected_outcome"]),
                "observed_outcome": "CONTACT" if contact_rows else "SAFE",
                "expected_responsible_assets": sorted(
                    str(value) for value in scenario["expected_responsible_assets"]
                ),
                "observed_responsible_assets": responsible,
                "first_contact_time_s": (
                    float(contact_rows[0]["time_s"]) if contact_rows else None
                ),
                "frames": len(instance_episode_rows),
                "active_assets_excluding_wearer": len(
                    materialize_layout_assets(protocol, str(scenario["layout_id"]))
                ),
            }
        )
    write_json_atomic(evaluator_root / "outcome_summary.json", outcome_summaries)

    instance_manifests = read_json(
        root / "shards" / "instance" / "episode_asset_manifests.json"
    )
    unique_actual_blueprints = sorted(
        {str(value["actual_blueprint"]) for value in instance_manifests}
    )
    write_json_atomic(
        evaluator_root / "asset_roster.json",
        {
            "schema_version": "dtr-c2-asset-roster-v1",
            "episode_asset_instances": len(instance_manifests),
            "unique_actual_blueprint_count": len(unique_actual_blueprints),
            "unique_actual_blueprints": unique_actual_blueprints,
            "zero_fallbacks": all(
                int(value["fallback_index"]) == 0 for value in instance_manifests
            ),
            "all_bbox_nonzero": all(
                bool(value["bbox_nonzero"]) for value in instance_manifests
            ),
        },
    )

    render_summary_svg(protocol, occlusion_reports, evaluator_root / "summary.svg")
    montage_sources: list[str] = []
    for scenario in protocol["scenarios"]:
        episode_id = str(scenario["episode_id"])
        layout = protocol["layouts"][scenario["layout_id"]]
        sample_index = int(
            round(
                float(layout["showcase_time_s"])
                / float(protocol["environment"]["sample_seconds"])
            )
        )
        montage_sources.extend(
            [
                str(model_root / "episodes" / episode_id / "rgb" / f"{sample_index:06d}.png"),
                str(
                    evaluator_root
                    / "episodes"
                    / episode_id
                    / "witness"
                    / f"{sample_index:06d}.png"
                ),
            ]
        )
    magick = shutil.which("magick")
    if magick is None:
        raise RuntimeError("ImageMagick is required to seal the C2 contact sheet")
    contact_sheet = evaluator_root / "contact_sheet.png"
    subprocess.run(
        [
            magick,
            "montage",
            *montage_sources,
            "-thumbnail",
            "640x360",
            "-tile",
            f"2x{len(protocol['scenarios'])}",
            "-geometry",
            "+8+8",
            str(contact_sheet),
        ],
        check=True,
    )

    model_root_manifest_path = model_root / "manifest.json"
    write_json_atomic(
        model_root_manifest_path,
        {
            "schema_version": "dtr-c2-model-root-manifest-v2",
            "experiment_id": EXPERIMENT_ID,
            "camera_calibration": {
                "path": "camera_calibration.json",
                "sha256": sha256_file(camera_calibration_path),
            },
            "model_contract": {
                "path": "model_contract.json",
                "sha256": sha256_file(model_contract_path),
            },
            "rgbd_alignment_receipt": {
                "path": safe_relative(alignment_receipt_path, model_root),
                "receipt_sha256": alignment_receipt["receipt_sha256"],
                "sha256": alignment_receipt_file_sha256,
            },
            "episodes": [
                {
                    "episode_id": path.parent.name,
                    "manifest_path": safe_relative(path, model_root),
                    "manifest_sha256": sha256_file(path),
                }
                for path in model_episode_manifest_paths
            ],
        },
    )
    model_truth_failures = scan_model_truth(model_root)
    model_manifest = seal_tree(model_root)
    write_json_atomic(root / "sealed_model_manifest.json", model_manifest)
    evidence_manifest = []
    for directory in (root / "shards", model_root, evaluator_root):
        for path in sorted(value for value in directory.rglob("*") if value.is_file()):
            evidence_manifest.append(
                {
                    "path": safe_relative(path, root),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json_atomic(root / "sealed_evidence_manifest.json", evidence_manifest)

    expected_records = sum(
        int(
            round(
                float(protocol["layouts"][scenario["layout_id"]]["duration_seconds"])
                / float(protocol["environment"]["sample_seconds"])
            )
        )
        + 1
        for scenario in protocol["scenarios"]
    )
    c4_compatibility = protocol.get("c4_compatibility", {})
    defer_occlusion_to_c4_pack = (
        isinstance(c4_compatibility, dict)
        and c4_compatibility.get("experiment_id") == C4_EXPERIMENT_ID
    )
    occlusion_contract_scope = (
        C4_OCCLUSION_SCOPE if defer_occlusion_to_c4_pack else "C2_LOCAL_PROTOCOL"
    )
    replay_check_name = (
        "cross_sensor_actual_replay_identical"
        if witness_replay_mode == ACTUAL_ALL_REPLAY
        else "cross_sensor_authority_scoped_replay_identical"
    )
    checks = {
        "all_four_fresh_server_shards_complete": all(
            results[sensor]["status"] == "DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE"
            for sensor in SENSOR_ORDER
        ),
        "cross_shard_code_and_protocol_identity": identity_equal,
        "raw_payload_inventories_verified": not payload_failures,
        replay_check_name: not replay_failures,
        "all_formal_sensors_1280x720": all(
            bool(results[sensor]["checks"]["all_formal_payloads_are_1280x720"])
            for sensor in SENSOR_ORDER
        ),
        "minimum_sixty_unique_actual_blueprints": len(unique_actual_blueprints)
        >= int(
            protocol["admission"]["minimum_unique_actual_blueprints_across_pack"]
        ),
        "zero_blueprint_fallbacks": all(
            bool(results[sensor]["checks"]["zero_blueprint_fallbacks"])
            for sensor in SENSOR_ORDER
        ),
        "all_spawned_assets_have_nonzero_bbox": all(
            bool(results[sensor]["checks"]["all_spawned_assets_have_nonzero_bbox"])
            for sensor in SENSOR_ORDER
        ),
        "contact_safe_outcome_pair_matches": all(
            value["expected_outcome"] == value["observed_outcome"]
            and value["expected_responsible_assets"]
            == value["observed_responsible_assets"]
            for value in outcome_summaries
        ),
        "dense_model_rgb_depth_complete": model_records_total == expected_records
        and model_payloads_total == expected_records * 2,
        "model_camera_transforms_align_across_rgb_depth": model_camera_alignment,
        "deterministic_rgb_depth_frame_alignment_materialized": (
            alignment_receipt_path.is_file()
            and len(alignment_by_episode) == len(protocol["scenarios"])
            and all(
                int(value["frames"])
                == len(rows_by_sensor["wearable"][str(value["episode_id"])])
                for value in alignment_receipt["episodes"]
            )
        ),
        "model_root_zero_actor_or_evaluator_truth_keys": not model_truth_failures,
        "model_contract_current_actors_disabled": not bool(
            protocol["model_contract"]["include_current_actors"]
        ),
        "truth_blind_model_root_manifest_materialized": (
            model_root_manifest_path.is_file()
            and len(model_episode_manifest_paths) == len(protocol["scenarios"])
        ),
        "immutable_plan_receipts_and_world_routes_materialized": (
            len(model_plan_paths) == len(protocol["scenarios"])
            and all(path.is_file() and path.stat().st_size > 0 for path in model_plan_paths)
        ),
        "camera_calibration_and_depth_codec_materialized": (
            calibration["resolution"] == {"width": 1280, "height": 720}
            and calibration["K"] == camera_intrinsics(1280, 720, fov)
            and bool(calibration["depth_codec"]["formula"])
        ),
        "visual_contact_sheet_and_summary_materialized": contact_sheet.is_file()
        and contact_sheet.stat().st_size > 0
        and (evaluator_root / "summary.svg").is_file(),
        "sealed_model_and_full_evidence_nonempty": bool(model_manifest)
        and bool(evidence_manifest),
    }
    checks["track_then_complete_physical_occlusion_contract_met"] = (
        True
        if defer_occlusion_to_c4_pack
        else all(bool(value["passed"]) for value in occlusion_reports)
    )
    if defer_occlusion_to_c4_pack:
        checks[
            "track_then_complete_physical_occlusion_contract_deferred_to_c4_final_join"
        ] = True
    status = (
        "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"
        if all(checks.values())
        else "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_NOT_EVALUABLE"
    )
    result = {
        "schema_version": "dtr-carla-c2-rich-scene-result-v2",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "checks": checks,
        "protocol_sha256": sha256_file(protocol_path),
        "capture_script_sha256": results["instance"]["capture_script_sha256"],
        "helper_module_sha256": results["instance"]["helper_module_sha256"],
        "join_script_sha256": sha256_file(Path(__file__).resolve()),
        "join_helper_module_sha256": sha256_file(
            Path(__file__).with_name("dtr_carla_c2_rich_scene.py")
        ),
        "rgbd_alignment_receipt_sha256": alignment_receipt["receipt_sha256"],
        "rgbd_alignment_receipt_file_sha256": alignment_receipt_file_sha256,
        "rgbd_source_frame_offsets": {
            str(value["episode_id"]): int(
                value["depth_minus_wearable_source_world_frame_offset"]
            )
            for value in alignment_receipt["episodes"]
        },
        "sensor_results": {
            sensor: {
                "status": results[sensor]["status"],
                "payload_count": int(results[sensor]["payload_count"]),
                "unique_actual_blueprint_count": int(
                    results[sensor]["unique_actual_blueprint_count"]
                ),
                "calibration_sha256": results[sensor]["calibration_sha256"],
            }
            for sensor in SENSOR_ORDER
        },
        "episode_count": len(episode_ids),
        "layout_count": len(protocol["layouts"]),
        "frames_per_sensor": expected_records,
        "total_raw_sensor_payloads": expected_records * len(SENSOR_ORDER),
        "model_observation_frames": model_records_total,
        "unique_actual_blueprint_count": len(unique_actual_blueprints),
        "outcomes": outcome_summaries,
        "occlusion_reports": occlusion_reports,
        "occlusion_contract_scope": occlusion_contract_scope,
        "local_occlusion_contracts_passed": sum(
            1 for value in occlusion_reports if bool(value["passed"])
        ),
        "local_occlusion_contracts_total": len(occlusion_reports),
        "payload_verification_failures": payload_failures,
        "replay_failures": replay_failures,
        "replay_contract": {
            "model_sensor_mode": ACTUAL_ALL_REPLAY,
            "witness_sensor_mode": witness_replay_mode,
            "authoritative_actor_keys": sorted(authoritative_actor_keys),
            "truth_projection": (
                "ACTUAL_EXACT"
                if witness_replay_mode == ACTUAL_ALL_REPLAY
                else "CONTACT_OUTCOME_EXACT_WITH_NONAUTHORITATIVE_GEOMETRY_EXCLUDED"
            ),
        },
        "model_truth_failures": model_truth_failures,
        "sealed_model_file_count": len(model_manifest),
        "sealed_evidence_file_count": len(evidence_manifest),
        "sealed_model_manifest_sha256": sha256_file(
            root / "sealed_model_manifest.json"
        ),
        "sealed_evidence_manifest_sha256": sha256_file(
            root / "sealed_evidence_manifest.json"
        ),
        "model_root": str(model_root),
        "evaluator_root": str(evaluator_root),
        "contact_sheet": str(contact_sheet),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
